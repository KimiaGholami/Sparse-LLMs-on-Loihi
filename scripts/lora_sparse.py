"""
LoRA fine-tuning on a sparse model.

The sparse base weights are frozen entirely; only the low-rank adapter
matrices (A, B) are trained.  This adds ~2-4M trainable parameters on top
of the 1.3B sparse backbone, restoring capacity without altering sparsity.

Usage
-----
CUDA_VISIBLE_DEVICES=0 python scripts/lora_sparse.py \\
    --model_path  exp/hgrn-1.3B-obs-cancel-block-50pct-distill/best \\
    --model_type  hgrn \\
    --output_path exp/hgrn-1.3B-50pct-lora \\
    --lora_r      16 \\
    --total_steps 20000 \\
    --lr 3e-4
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
from peft import LoraConfig, get_peft_model, TaskType
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

    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--total_steps",  type=int,   default=20_000)
    p.add_argument("--lr",           type=float, default=3e-4)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Model     : {args.model_path}", flush=True)
    print(f"LoRA r    : {args.lora_r}  alpha={args.lora_alpha}", flush=True)
    print(f"Steps     : {args.total_steps}", flush=True)
    print(f"LR        : {args.lr}", flush=True)
    print(f"Output    : {args.output_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # Check sparsity of base model
    total = zeros = 0
    for m in base_model.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.numel()
            zeros += (m.weight.data == 0).sum().item()
    print(f"Base model sparsity: {zeros/total*100:.1f}%", flush=True)

    # Identify target modules (all Linear projections)
    target_modules = []
    for name, mod in base_model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            # Get the leaf name (e.g. "i_proj", "gate_proj")
            leaf = name.split(".")[-1]
            if leaf not in target_modules:
                target_modules.append(leaf)
    print(f"LoRA target modules: {target_modules}", flush=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    baseline_ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Baseline PPL: {baseline_ppl:.2f}\n", flush=True)

    loader    = DataLoader(C4StreamDataset(tokenizer, args.seq_len),
                           batch_size=args.batch_size, num_workers=2)
    data_iter = iter(loader)

    # Only train LoRA parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.total_steps
    )

    best_ppl = baseline_ppl
    os.makedirs(args.output_path, exist_ok=True)

    pbar = tqdm(range(1, args.total_steps + 1), desc="LoRA FT")
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

        if step % args.log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        if step % args.eval_every == 0:
            ppl = evaluate_ppl(model, tokenizer, device)
            print(f"\nStep {step}: PPL={ppl:.2f}  (best={best_ppl:.2f})", flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                model.save_pretrained(os.path.join(args.output_path, "best"))
                tokenizer.save_pretrained(os.path.join(args.output_path, "best"))
                print(f"  -> saved best checkpoint", flush=True)

    final_ppl = evaluate_ppl(model, tokenizer, device)
    print(f"\nFinal PPL: {final_ppl:.4f}", flush=True)

    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    meta = {
        "base_model": args.model_path,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "total_steps": args.total_steps,
        "lr": args.lr,
        "baseline_ppl": baseline_ppl,
        "final_ppl": final_ppl,
        "best_ppl": best_ppl,
        "base_sparsity": zeros / total,
    }
    with open(os.path.join(args.output_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
