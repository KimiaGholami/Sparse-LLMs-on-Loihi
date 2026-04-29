"""
Dynamic-mask fine-tuning.

Unlike mask-fixed fine-tuning (which freezes the zero positions for the entire
run), this approach re-computes the sparsity mask after every optimizer step by
keeping the top-(1-sparsity) weights by magnitude per row.  This allows the
network to "rewire": weights that OBS retained but gradient descent finds
unhelpful can be zeroed, and weights that OBS killed can re-enter the active
set if they become useful.

Sparsity is maintained exactly at the target level at all times (semi-structured:
exactly s fraction zeroed per row of each Linear layer).

Usage
-----
CUDA_VISIBLE_DEVICES=0 python scripts/finetune_dynamic_mask.py \\
    --model_path  exp/hgrn-1.3B-obs_cancel-80pct \\
    --model_type  hgrn \\
    --output_path exp/hgrn-1.3B-80pct-dynamic-mask \\
    --sparsity    0.80 \\
    --total_steps 20000 \\
    --lr 2e-5
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

try:
    import fla
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
    print(f"Warning: {e}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class C4StreamDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len   = seq_len

    def __iter__(self):
        raw = load_dataset("allenai/c4", "en", split="train", streaming=True)
        buf = []
        for doc in raw:
            buf.extend(self.tokenizer.encode(doc["text"], add_special_tokens=False))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf   = buf[self.seq_len + 1:]
                yield (torch.tensor(chunk[:-1], dtype=torch.long),
                       torch.tensor(chunk[1:],  dtype=torch.long))


# ---------------------------------------------------------------------------
# Dynamic mask
# ---------------------------------------------------------------------------

def apply_dynamic_mask(model, sparsity):
    """
    Re-zero bottom-sparsity fraction by magnitude per row of each Linear layer.
    Returns current overall sparsity.
    """
    total = zeros = 0
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, nn.Linear):
                continue
            W = m.weight.data                        # (out_f, in_f)
            out_f, in_f = W.shape
            k = int(in_f * sparsity)
            if k == 0:
                continue
            # threshold = k-th smallest absolute value per row
            threshold = W.abs().kthvalue(k, dim=1).values.unsqueeze(1)  # (out_f, 1)
            mask = W.abs() > threshold
            W.mul_(mask.to(W.dtype))
            total += W.numel()
            zeros += (W == 0).sum().item()
    return zeros / total if total else 0.0


def measure_sparsity(model):
    total = zeros = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.numel()
            zeros += (m.weight.data == 0).sum().item()
    return zeros / total if total else 0.0


# ---------------------------------------------------------------------------
# PPL
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, seq_len=512):
    raw  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(raw["text"])
    ids  = torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
    n    = len(ids) // seq_len
    ids  = ids[:n * seq_len].reshape(n, seq_len)
    model.eval()
    loss = sum(model(c.unsqueeze(0).to(device), labels=c.unsqueeze(0).to(device)).loss.item()
               for c in ids) / n
    model.train()
    return math.exp(loss)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   required=True)
    p.add_argument("--model_type",   choices=["transformer", "llama", "hgrn"], required=True)
    p.add_argument("--output_path",  required=True)
    p.add_argument("--sparsity",     type=float, default=0.80,
                   help="Target sparsity maintained throughout training")

    p.add_argument("--total_steps",  type=int,   default=20_000)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--grad_clip",    type=float, default=1.0)

    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--seq_len",      type=int,   default=512)

    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--eval_every",   type=int,   default=2000)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"\n{'='*60}", flush=True)
    print(f"Model     : {args.model_path}", flush=True)
    print(f"Sparsity  : {args.sparsity:.0%}  (dynamic — mask rewires each step)", flush=True)
    print(f"Steps     : {args.total_steps}", flush=True)
    print(f"LR        : {args.lr}", flush=True)
    print(f"Output    : {args.output_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.train()

    # Apply initial mask to ensure we start at exactly target sparsity
    sp = apply_dynamic_mask(model, args.sparsity)
    print(f"Initial sparsity after masking: {sp*100:.2f}%", flush=True)

    baseline_ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Baseline PPL: {baseline_ppl:.2f}\n", flush=True)

    loader    = DataLoader(C4StreamDataset(tokenizer, args.seq_len),
                           batch_size=args.batch_size, num_workers=2)
    data_iter = iter(loader)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.total_steps
    )

    best_ppl = baseline_ppl
    os.makedirs(args.output_path, exist_ok=True)

    pbar = tqdm(range(1, args.total_steps + 1), desc="dynamic-mask FT")
    for step in pbar:
        try:
            inp, lbl = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inp, lbl = next(data_iter)
        inp, lbl = inp.to(device), lbl.to(device)

        out  = model(input_ids=inp, labels=lbl)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Re-apply magnitude mask — allows rewiring
        sp = apply_dynamic_mask(model, args.sparsity)

        if step % args.log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}", sp=f"{sp*100:.1f}%")

        if step % args.eval_every == 0:
            ppl = evaluate_ppl(model, tokenizer, device)
            print(f"\nStep {step}: PPL={ppl:.2f}  sparsity={sp*100:.1f}%  "
                  f"(best={best_ppl:.2f})", flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                model.save_pretrained(os.path.join(args.output_path, "best"))
                tokenizer.save_pretrained(os.path.join(args.output_path, "best"))
                print(f"  -> saved best checkpoint", flush=True)

    final_ppl = evaluate_ppl(model, tokenizer, device)
    final_sp  = measure_sparsity(model)
    print(f"\nFinal PPL: {final_ppl:.4f}  sparsity={final_sp*100:.1f}%", flush=True)

    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    meta = {
        "base_model": args.model_path,
        "sparsity_target": args.sparsity,
        "final_sparsity": final_sp,
        "total_steps": args.total_steps,
        "lr": args.lr,
        "baseline_ppl": baseline_ppl,
        "final_ppl": final_ppl,
        "best_ppl": best_ppl,
        "method": "dynamic-mask",
    }
    with open(os.path.join(args.output_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
