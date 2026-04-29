"""
Distillation-based mask-fixed sparse fine-tuning.

Uses the dense model as a frozen teacher. Loss is a weighted sum of:
  - KL divergence between student and teacher output distributions (main signal)
  - Cross-entropy against hard labels (keeps the student grounded)

  loss = alpha * CE(student, labels) + (1 - alpha) * KL(student || teacher) * T^2

The sparsity mask is recorded at startup and held fixed throughout:
  - Gradient hooks zero updates at masked (zero-weight) positions
  - After each optimizer step masked positions are re-zeroed

Data: C4 streaming — effectively unlimited, diverse, close to pretraining.

Usage
-----
# 50% sparse HGRN
CUDA_VISIBLE_DEVICES=0 python scripts/distill_sparse.py \
    --teacher_path exp/hgrn-1.3B-dense-baseline \
    --student_path exp/hgrn-1.3B-obs-cancel-block-50pct \
    --model_type hgrn \
    --output_path exp/hgrn-1.3B-obs-cancel-block-50pct-distill \
    --total_steps 200000 \
    --lr 2e-5

# 80% sparse HGRN
CUDA_VISIBLE_DEVICES=1 python scripts/distill_sparse.py \
    --teacher_path exp/hgrn-1.3B-dense-baseline \
    --student_path exp/hgrn-1.3B-obs_cancel-80pct \
    --model_type hgrn \
    --output_path exp/hgrn-1.3B-obs_cancel-80pct-distill \
    --total_steps 200000 \
    --lr 2e-5
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Streaming C4 dataset
# ---------------------------------------------------------------------------

class C4StreamDataset(IterableDataset):
    """Tokenises C4 on-the-fly and yields (seq_len,) input/label pairs."""

    def __init__(self, tokenizer, seq_len, buffer_tokens=1_000_000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer_tokens = buffer_tokens

    def __iter__(self):
        raw = load_dataset("allenai/c4", "en", split="train", streaming=True)
        buf = []
        for doc in raw:
            buf.extend(self.tokenizer.encode(doc["text"], add_special_tokens=False))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf   = buf[self.seq_len + 1 :]
                inp   = torch.tensor(chunk[:-1], dtype=torch.long)
                lbl   = torch.tensor(chunk[1:],  dtype=torch.long)
                yield inp, lbl


def build_c4_loader(tokenizer, seq_len, batch_size, num_workers=2):
    ds = C4StreamDataset(tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Mask utilities (identical to finetune_sparse.py)
# ---------------------------------------------------------------------------

def register_masks(model):
    pairs, handles = [], []
    for m in model.modules():
        if not isinstance(m, nn.Linear):
            continue
        mask = (m.weight.data != 0).to(dtype=m.weight.dtype, device=m.weight.device)
        pairs.append((m.weight, mask))

        def make_hook(msk):
            def hook(grad):
                return grad * msk
            return hook

        handles.append(m.weight.register_hook(make_hook(mask)))
    return pairs, handles


def enforce_masks(pairs):
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
# PPL evaluation on WikiText-2 test set
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, seq_len=512):
    raw  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(raw["text"])
    ids  = tokenizer.encode(text, add_special_tokens=False)
    ids  = torch.tensor(ids, dtype=torch.long)
    n    = len(ids) // seq_len
    ids  = ids[: n * seq_len].reshape(n, seq_len)

    model.eval()
    total_loss = 0.0
    for chunk in ids:
        out = model(chunk.unsqueeze(0).to(device), labels=chunk.unsqueeze(0).to(device))
        total_loss += out.loss.item()
    model.train()
    return math.exp(total_loss / n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_path", required=True)
    p.add_argument("--student_path", required=True)
    p.add_argument("--model_type", choices=["transformer", "llama", "hgrn"], required=True)
    p.add_argument("--output_path", required=True)

    # Distillation
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Weight on CE loss; (1-alpha) goes to KL distillation loss")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Softmax temperature for distillation")

    # Optimiser
    p.add_argument("--total_steps",  type=int,   default=200_000)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int,   default=2_000)
    p.add_argument("--grad_clip",    type=float, default=1.0)

    # Data
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--seq_len",      type=int, default=512)
    p.add_argument("--num_workers",  type=int, default=2)

    # Logging
    p.add_argument("--log_every",  type=int, default=100)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--save_every", type=int, default=50000)

    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--teacher_device", type=str, default=None,
                   help="Separate device for teacher (e.g. cuda:3). Defaults to --device.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the student for faster forward/backward.")
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="Enable gradient checkpointing on student to save memory.")
    return p.parse_args()


def main():
    args           = parse_args()
    device         = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device if args.teacher_device else args.device)

    print(f"\n{'='*60}", flush=True)
    print(f"Teacher   : {args.teacher_path}  (device: {teacher_device})", flush=True)
    print(f"Student   : {args.student_path}  (device: {device})", flush=True)
    print(f"Alpha     : {args.alpha}  (CE={args.alpha:.0%}, KL={1-args.alpha:.0%})", flush=True)
    print(f"Temp      : {args.temperature}", flush=True)
    print(f"Steps     : {args.total_steps}", flush=True)
    print(f"Compile   : {args.compile}", flush=True)
    print(f"Output    : {args.output_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.student_path, trust_remote_code=True)

    # Teacher: frozen, no gradients, optionally on a separate GPU
    print("Loading teacher ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(teacher_device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student: sparse, mask-fixed
    print("Loading student ...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.student_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    if args.grad_checkpoint:
        student.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.", flush=True)

    if args.compile:
        print("Compiling student with torch.compile ...", flush=True)
        student = torch.compile(student)
        print("Compiled.", flush=True)

    student.train()

    mask_pairs, hook_handles = register_masks(student)
    print(f"Student sparsity: {current_sparsity(mask_pairs)*100:.1f}%  (mask fixed)", flush=True)

    # Baseline PPL before any training
    baseline_ppl = evaluate_ppl(student, tokenizer, device, args.seq_len)
    print(f"Baseline PPL (before distillation): {baseline_ppl:.2f}\n", flush=True)

    loader    = build_c4_loader(tokenizer, args.seq_len, args.batch_size, args.num_workers)
    data_iter = iter(loader)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    T     = args.temperature
    alpha = args.alpha
    kl    = nn.KLDivLoss(reduction="batchmean")

    best_ppl  = baseline_ppl
    os.makedirs(args.output_path, exist_ok=True)

    pbar = tqdm(range(1, args.total_steps + 1), desc="distill")
    for step in pbar:
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs, labels = next(data_iter)

        inputs  = inputs.to(device)
        labels  = labels.to(device)

        # Teacher forward (no grad, may be on a different device)
        with torch.no_grad():
            t_logits = teacher(input_ids=inputs.to(teacher_device)).logits.to(device)

        # Student forward
        s_out    = student(input_ids=inputs, labels=labels)
        s_logits = s_out.logits                            # (B, L, V)
        ce_loss  = s_out.loss

        # KL distillation loss — flatten to (B*L, V)
        kl_loss = kl(
            F.log_softmax(s_logits.float().reshape(-1, s_logits.size(-1)) / T, dim=-1),
            F.softmax(t_logits.float().reshape(-1, t_logits.size(-1)) / T, dim=-1),
        ) * (T * T)

        loss = alpha * ce_loss + (1.0 - alpha) * kl_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        enforce_masks(mask_pairs)

        if step % args.log_every == 0:
            pbar.set_postfix(
                ce=f"{ce_loss.item():.3f}",
                kl=f"{kl_loss.item():.3f}",
                sp=f"{current_sparsity(mask_pairs)*100:.1f}%",
            )

        if step % args.eval_every == 0:
            ppl = evaluate_ppl(student, tokenizer, device, args.seq_len)
            sp  = current_sparsity(mask_pairs)
            print(f"\nStep {step:>7d}: PPL={ppl:.2f}  sparsity={sp*100:.1f}%  "
                  f"(best={best_ppl:.2f})", flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                student.save_pretrained(os.path.join(args.output_path, "best"))
                tokenizer.save_pretrained(os.path.join(args.output_path, "best"))
                print(f"  -> saved best checkpoint", flush=True)

        if step % args.save_every == 0:
            ckpt = os.path.join(args.output_path, f"step-{step}")
            student.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  -> saved checkpoint {ckpt}", flush=True)

    # Final save
    for h in hook_handles:
        h.remove()

    final_ppl = evaluate_ppl(student, tokenizer, device, args.seq_len)
    final_sp  = current_sparsity(mask_pairs)
    print(f"\nFinal PPL: {final_ppl:.4f}  sparsity={final_sp*100:.1f}%", flush=True)

    student.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    meta = {
        "teacher": args.teacher_path,
        "student": args.student_path,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "total_steps": args.total_steps,
        "lr": args.lr,
        "final_ppl": final_ppl,
        "baseline_ppl": baseline_ppl,
        "best_ppl": best_ppl,
        "final_sparsity": final_sp,
    }
    with open(os.path.join(args.output_path, "distill_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
