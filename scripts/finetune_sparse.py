"""
Mask-fixed sparse fine-tuning.

Loads a pruned model, records which weights are zero (the OBS-optimized
sparsity mask), then fine-tunes with the mask held constant throughout:

  - Backward hooks zero gradients at masked positions so the optimizer
    never updates zero weights.
  - After each optimizer step, masked positions are explicitly re-zeroed
    to neutralise any numerical drift from AdamW weight decay.
  - The sparsity pattern is identical at the start and end — no L1
    penalty, no soft-threshold, no hard projection needed.

This preserves the second-order weight corrections computed by
OBS-cancel-block while recovering task accuracy through continued
language-model training on the surviving non-zero weights.

Usage
-----
CUDA_VISIBLE_DEVICES=0 python scripts/finetune_sparse.py \\
    --model_path exp/hgrn-1.3B-obs-cancel-block-50pct \\
    --model_type hgrn \\
    --output_path exp/hgrn-1.3B-obs-cancel-block-50pct-ft \\
    --total_steps 20000 \\
    --lr 2e-5

CUDA_VISIBLE_DEVICES=1 python scripts/finetune_sparse.py \\
    --model_path exp/hgrn-1.3B-obs_cancel-80pct \\
    --model_type hgrn \\
    --output_path exp/hgrn-1.3B-obs_cancel-80pct-ft \\
    --total_steps 20000 \\
    --lr 2e-5
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

try:
    import fla  # noqa
    from fla.models.transformer import TransformerConfig, TransformerForCausalLM
    from fla.models.hgrn import HGRNConfig, HGRNForCausalLM
    from fla.models.hgrn2 import HGRN2Config, HGRN2ForCausalLM
    AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
    AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)
    AutoConfig.register("hgrn", HGRNConfig, exist_ok=True)
    AutoModelForCausalLM.register(HGRNConfig, HGRNForCausalLM, exist_ok=True)
    AutoConfig.register("hgrn2", HGRN2Config, exist_ok=True)
    AutoModelForCausalLM.register(HGRN2Config, HGRN2ForCausalLM, exist_ok=True)
except Exception as e:
    print(f"Warning: could not register FLA types: {e}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloader(tokenizer, seq_len, batch_size, dataset="wikitext2"):
    if dataset == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(raw["text"])
    elif dataset == "c4":
        raw = load_dataset("allenai/c4", "en", split="train", streaming=True)
        text = " ".join(x["text"] for x in list(raw.take(50_000)))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = torch.tensor(ids, dtype=torch.long)

    n_full = len(ids) // (seq_len + 1)
    ids = ids[: n_full * (seq_len + 1)].reshape(n_full, seq_len + 1)
    inputs = ids[:, :-1]
    labels = ids[:, 1:]

    ds = torch.utils.data.TensorDataset(inputs, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def register_masks(model):
    """Record zero positions and wire up gradient + weight hooks.

    Returns a list of (weight_tensor, mask_tensor) pairs and a list of
    hook handles (keep alive for the training duration).
    """
    pairs = []
    handles = []

    for m in model.modules():
        if not isinstance(m, nn.Linear):
            continue
        mask = (m.weight.data != 0).to(dtype=m.weight.dtype, device=m.weight.device)
        pairs.append((m.weight, mask))

        # Zero gradients at masked-off positions before the optimizer step.
        def make_grad_hook(msk):
            def hook(grad):
                return grad * msk
            return hook

        handles.append(m.weight.register_hook(make_grad_hook(mask)))

    return pairs, handles


def enforce_masks(pairs):
    """Re-zero masked positions after each optimizer step (handles AdamW drift)."""
    with torch.no_grad():
        for weight, mask in pairs:
            weight.data.mul_(mask)


def current_sparsity(pairs):
    total = zeros = 0
    for weight, _ in pairs:
        total += weight.numel()
        zeros += (weight.data == 0).sum().item()
    return zeros / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# PPL evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, seq_len=512, n_tokens=500_000):
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(raw["text"])
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = torch.tensor(ids[:n_tokens], dtype=torch.long)

    n_chunks = len(ids) // seq_len
    ids = ids[: n_chunks * seq_len].reshape(n_chunks, seq_len)

    model.eval()
    total_loss = 0.0
    for chunk in ids:
        chunk = chunk.unsqueeze(0).to(device)
        out = model(chunk, labels=chunk)
        total_loss += out.loss.item()
    model.train()
    return math.exp(total_loss / n_chunks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_type", choices=["transformer", "llama", "hgrn"], required=True)
    p.add_argument("--output_path", required=True)

    p.add_argument("--total_steps", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--calib_dataset", choices=["wikitext2", "c4"], default="wikitext2")

    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"\n{'='*60}", flush=True)
    print(f"Model     : {args.model_path}", flush=True)
    print(f"Steps     : {args.total_steps}", flush=True)
    print(f"LR        : {args.lr}", flush=True)
    print(f"Output    : {args.output_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.train()

    # Record sparsity mask and wire hooks before any parameter updates.
    mask_pairs, hook_handles = register_masks(model)
    print(f"Loaded. Sparsity: {current_sparsity(mask_pairs)*100:.1f}%  "
          f"(mask fixed for entire run)", flush=True)

    loader = build_dataloader(
        tokenizer, args.seq_len, args.batch_size, dataset=args.calib_dataset
    )
    data_iter = iter(loader)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    pbar = tqdm(range(1, args.total_steps + 1), desc="finetune")
    for step in pbar:
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, labels = next(data_iter)

        inputs, labels = inputs.to(device), labels.to(device)

        out = model(input_ids=inputs, labels=labels)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()          # grad hooks zero masked positions here
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        enforce_masks(mask_pairs)  # re-zero any AdamW weight-decay drift

        if step % args.log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             sparsity=f"{current_sparsity(mask_pairs)*100:.1f}%")

        if step % args.eval_every == 0:
            ppl = evaluate_ppl(model, tokenizer, device)
            print(f"\nStep {step}: PPL={ppl:.2f}  "
                  f"sparsity={current_sparsity(mask_pairs)*100:.1f}%", flush=True)

    # Remove hooks before saving
    for h in hook_handles:
        h.remove()

    ppl = evaluate_ppl(model, tokenizer, device)
    final_sp = current_sparsity(mask_pairs)
    print(f"\nFinal PPL: {ppl:.4f}  sparsity={final_sp*100:.1f}%", flush=True)

    os.makedirs(args.output_path, exist_ok=True)
    _tied = getattr(model, "_tied_weights_keys", None)
    model._tied_weights_keys = None
    model.save_pretrained(args.output_path)
    model._tied_weights_keys = _tied
    tokenizer.save_pretrained(args.output_path)

    meta = {
        "base_model": args.model_path,
        "total_steps": args.total_steps,
        "lr": args.lr,
        "final_ppl": ppl,
        "final_sparsity": final_sp,
        "method": "mask-fixed",
    }
    with open(os.path.join(args.output_path, "finetune_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
