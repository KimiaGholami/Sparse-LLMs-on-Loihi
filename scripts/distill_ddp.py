"""
Multi-GPU knowledge distillation using DistributedDataParallel (DDP).

Works with any pre-pruned model.  Launch with torchrun:

    torchrun --nproc_per_node=4 scripts/distill_ddp.py \\
        --teacher_path exp/hgrn-1.3B-dense-baseline \\
        --student_path exp/hgrn-1.3B-80pct-no-lmhead/pruned \\
        --model_type   hgrn \\
        --output_path  exp/hgrn-1.3B-80pct-no-lmhead \\
        --total_steps  25000 \\
        --lr 2e-5

Design
------
- Each rank loads both student (DDP-wrapped) and teacher (frozen, local copy).
  Teacher is local to each rank so no cross-GPU logit broadcast is needed.
- Sparsity mask is registered on the underlying module (not the DDP wrapper),
  so gradient hooks fire correctly on every rank before the DDP all-reduce.
- C4 streaming dataset: each rank shuffles with a different seed so they see
  different data.
- Only rank 0 evaluates PPL, saves checkpoints, and prints logs.
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
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
    def __init__(self, tokenizer, seq_len, rank=0):
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.rank      = rank

    def __iter__(self):
        # Different shuffle seed per rank → different data per GPU
        raw = load_dataset("allenai/c4", "en", split="train", streaming=True)
        raw = raw.shuffle(seed=42 + self.rank, buffer_size=10_000)
        buf = []
        for doc in raw:
            buf.extend(self.tokenizer.encode(doc["text"], add_special_tokens=False))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]
                buf   = buf[self.seq_len + 1:]
                yield (torch.tensor(chunk[:-1], dtype=torch.long),
                       torch.tensor(chunk[1:],  dtype=torch.long))


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def register_masks(model):
    """Register on the underlying module, not the DDP wrapper."""
    base = model.module if isinstance(model, DDP) else model
    pairs, handles = [], []
    for m in base.modules():
        if not isinstance(m, nn.Linear):
            continue
        mask = (m.weight.data != 0).to(dtype=m.weight.dtype, device=m.weight.device)
        pairs.append((m.weight, mask))
        def make_hook(msk):
            def hook(grad): return grad * msk
            return hook
        handles.append(m.weight.register_hook(make_hook(mask)))
    return pairs, handles


def enforce_masks(pairs):
    with torch.no_grad():
        for w, m in pairs:
            w.data.mul_(m)


def current_sparsity(pairs):
    total = zeros = 0
    for w, _ in pairs:
        total += w.numel()
        zeros += (w.data == 0).sum().item()
    return zeros / total if total else 0.0


# ---------------------------------------------------------------------------
# PPL (rank-0 only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, seq_len=512):
    base = model.module if isinstance(model, DDP) else model
    raw  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(raw["text"])
    ids  = torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
    n    = len(ids) // seq_len
    ids  = ids[:n * seq_len].reshape(n, seq_len)
    base.eval()
    loss = sum(base(c.unsqueeze(0).to(device), labels=c.unsqueeze(0).to(device)).loss.item()
               for c in ids) / n
    base.train()
    return math.exp(loss)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_path",   required=True)
    p.add_argument("--student_path",   required=True)
    p.add_argument("--model_type",     choices=["transformer", "llama", "hgrn"], required=True)
    p.add_argument("--output_path",    required=True)

    p.add_argument("--total_steps",    type=int,   default=25_000)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--warmup_steps",   type=int,   default=1_000)
    p.add_argument("--alpha",          type=float, default=0.1)
    p.add_argument("--temperature",    type=float, default=2.0)
    p.add_argument("--batch_size",     type=int,   default=16,
                   help="Per-GPU batch size; effective = batch_size × n_gpus")
    p.add_argument("--seq_len",        type=int,   default=512)
    p.add_argument("--grad_clip",      type=float, default=1.0)

    p.add_argument("--log_every",      type=int,   default=100)
    p.add_argument("--eval_every",     type=int,   default=2500)
    return p.parse_args()


def main():
    args = parse_args()

    # DDP init
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main    = (rank == 0)

    if is_main:
        print(f"\n{'='*60}", flush=True)
        print(f"World size     : {world_size} GPUs", flush=True)
        print(f"Student        : {args.student_path}", flush=True)
        print(f"Teacher        : {args.teacher_path}", flush=True)
        print(f"Steps          : {args.total_steps}", flush=True)
        print(f"Per-GPU batch  : {args.batch_size}  (effective: {args.batch_size * world_size})", flush=True)
        print(f"Output         : {args.output_path}", flush=True)
        print(f"{'='*60}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.student_path, trust_remote_code=True)

    # Teacher: frozen, local copy on each rank
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student: DDP-wrapped
    student = AutoModelForCausalLM.from_pretrained(
        args.student_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    student.train()
    student_ddp = DDP(student, device_ids=[local_rank], output_device=local_rank)

    # Masks on the underlying module
    mask_pairs, hook_handles = register_masks(student_ddp)
    if is_main:
        print(f"Sparsity : {current_sparsity(mask_pairs)*100:.2f}%", flush=True)

    # Data: each rank gets a differently-shuffled C4 stream
    loader    = DataLoader(C4StreamDataset(tokenizer, args.seq_len, rank=rank),
                           batch_size=args.batch_size, num_workers=2)
    data_iter = iter(loader)

    optimizer = torch.optim.AdamW(student_ddp.parameters(),
                                  lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.total_steps
    )
    kl = nn.KLDivLoss(reduction="batchmean")

    os.makedirs(args.output_path, exist_ok=True)
    best_ppl = float("inf")
    if is_main:
        best_ppl = evaluate_ppl(student_ddp, tokenizer, device)
        print(f"PPL before distillation: {best_ppl:.2f}\n", flush=True)
    dist.barrier()

    T     = args.temperature
    alpha = args.alpha

    pbar = tqdm(range(1, args.total_steps + 1), desc="distill", disable=not is_main)
    for step in pbar:
        try:
            inp, lbl = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inp, lbl = next(data_iter)
        inp, lbl = inp.to(device), lbl.to(device)

        with torch.no_grad():
            t_logits = teacher(input_ids=inp).logits   # teacher is local

        s_out    = student_ddp(input_ids=inp, labels=lbl)
        s_logits = s_out.logits
        ce_loss  = s_out.loss

        kl_loss = kl(
            F.log_softmax(s_logits.float().reshape(-1, s_logits.size(-1)) / T, dim=-1),
            F.softmax(t_logits.float().reshape(-1, t_logits.size(-1)) / T, dim=-1),
        ) * T * T
        loss = alpha * ce_loss + (1 - alpha) * kl_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student_ddp.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        enforce_masks(mask_pairs)

        if is_main and step % args.log_every == 0:
            pbar.set_postfix(ce=f"{ce_loss.item():.3f}",
                             kl=f"{kl_loss.item():.3f}",
                             sp=f"{current_sparsity(mask_pairs)*100:.1f}%")

        if is_main and step % args.eval_every == 0:
            ppl = evaluate_ppl(student_ddp, tokenizer, device)
            print(f"\nStep {step}: PPL={ppl:.2f} (best={best_ppl:.2f})", flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                student.save_pretrained(os.path.join(args.output_path, "best"))
                tokenizer.save_pretrained(os.path.join(args.output_path, "best"))
                print(f"  -> saved best checkpoint", flush=True)
        dist.barrier()

    # Cleanup
    for h in hook_handles:
        h.remove()

    if is_main:
        final_ppl = evaluate_ppl(student_ddp, tokenizer, device)
        print(f"\nFinal PPL: {final_ppl:.2f}  sparsity={current_sparsity(mask_pairs)*100:.1f}%", flush=True)
        student.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        meta = {
            "teacher": args.teacher_path,
            "student": args.student_path,
            "total_steps": args.total_steps,
            "effective_batch": args.batch_size * world_size,
            "lr": args.lr,
            "final_ppl": final_ppl,
            "best_ppl": best_ppl,
            "n_gpus": world_size,
        }
        with open(os.path.join(args.output_path, "distill_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved to {args.output_path}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
