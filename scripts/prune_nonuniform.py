"""
Non-uniform layer-wise sparsity pruning + distillation.

Assigns higher sparsity to middle layers and lower sparsity to the first/last
layers, which are most sensitive to pruning errors (early layers accumulate
errors; final layers directly compute logits).

Default profile for 24-layer HGRN-1.3B at ~80% overall:
  Layer 0, 23   : 65%   (input/output boundaries)
  Layer 1, 22   : 70%
  Layer 2, 21   : 75%
  Layer 3, 20   : 80%
  Layers 4-19   : 85%   (middle bulk)
  lm_head       : 60%

Computed overall: ~80.0% sparsity.

After pruning the script runs mask-fixed knowledge distillation using the
original dense model as a frozen teacher (same approach as distill_sparse.py).

Usage
-----
CUDA_VISIBLE_DEVICES=2,3 python scripts/prune_nonuniform.py \\
    --dense_path   exp/hgrn-1.3B-dense-baseline \\
    --model_type   hgrn \\
    --output_path  exp/hgrn-1.3B-nonuniform-80pct \\
    --device       cuda:0 \\
    --teacher_device cuda:1 \\
    --total_steps  50000
"""

import argparse
import json
import math
import os
import re
import sys

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

sys.path.insert(0, os.path.dirname(__file__))
from prune_obs_cancel import (build_calib_batches, collect_covariance_stats,
                               obs_cancel_block_prune_layer)

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
# Per-layer sparsity schedule
# ---------------------------------------------------------------------------

def layer_sparsity(name: str, n_layers: int, profile: dict) -> float:
    """Return the target sparsity for a named linear module."""
    if "lm_head" in name:
        return profile.get("lm_head", 0.60)

    m = re.search(r"layers\.(\d+)\.", name)
    if m is None:
        return 0.0  # skip embeddings / unknown

    idx = int(m.group(1))
    dist = min(idx, n_layers - 1 - idx)  # distance from nearest boundary

    boundaries = sorted(profile["boundary_sparsities"].keys())
    for d in boundaries:
        if dist <= d:
            return profile["boundary_sparsities"][d]
    return profile.get("middle_sparsity", 0.85)


def build_default_profile(n_layers: int) -> dict:
    """U-shaped sparsity profile targeting ~80% overall for 24-layer model."""
    return {
        "boundary_sparsities": {
            0: 0.65,  # first/last layer
            1: 0.70,
            2: 0.75,
            3: 0.80,
        },
        "middle_sparsity": 0.85,
        "lm_head": 0.60,
    }


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_nonuniform(model, tokenizer, n_layers, profile, device,
                     n_calib=64, batch_size=4, seq_len=512):
    print("  Collecting calibration stats ...", flush=True)
    batches = build_calib_batches(tokenizer, n_calib, batch_size, seq_len)
    stats   = collect_covariance_stats(model, batches, device)

    total_w = total_p = 0
    layer_log = {}

    for name, module in tqdm(model.named_modules(), desc="non-uniform OBS pruning"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue

        sp = layer_sparsity(name, n_layers, profile)
        if sp <= 0.0:
            continue

        W = module.weight.data
        out_f, in_f = W.shape
        if int(in_f * sp) == 0:
            continue

        Sigma  = stats[name].second_moment().to(device)
        W_new  = obs_cancel_block_prune_layer(W.float(), Sigma, sp)
        module.weight.data.copy_(W_new.to(W.dtype))

        n_pruned = (W_new == 0).sum().item()
        total_w += W.numel()
        total_p += n_pruned
        layer_log[name] = {"sparsity_target": sp,
                            "sparsity_actual": n_pruned / W.numel()}

        del Sigma, W_new
        torch.cuda.empty_cache()

    del stats, batches
    torch.cuda.empty_cache()

    achieved = total_p / max(total_w, 1)
    print(f"  Overall sparsity achieved: {achieved*100:.2f}%", flush=True)
    return achieved, layer_log


# ---------------------------------------------------------------------------
# Data / masks / PPL  (same as distill_sparse.py)
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


def register_masks(model):
    pairs, handles = [], []
    for m in model.modules():
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
# Distillation loop
# ---------------------------------------------------------------------------

def distill(student, teacher, tokenizer, n_steps, lr, device, teacher_device,
            batch_size=16, seq_len=512, warmup=2000,
            alpha=0.1, T=2.0, log_every=200, eval_every=5000,
            output_path=None):

    mask_pairs, hook_handles = register_masks(student)
    sp = current_sparsity(mask_pairs)
    print(f"  Mask registered. Sparsity: {sp*100:.1f}%", flush=True)

    loader    = DataLoader(C4StreamDataset(tokenizer, seq_len),
                           batch_size=batch_size, num_workers=2)
    data_iter = iter(loader)

    opt   = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, warmup, n_steps)
    kl    = nn.KLDivLoss(reduction="batchmean")

    best_ppl = evaluate_ppl(student, tokenizer, device)
    print(f"  PPL before distillation: {best_ppl:.2f}", flush=True)

    pbar = tqdm(range(1, n_steps + 1), desc="distill")
    for step in pbar:
        try:
            inp, lbl = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inp, lbl = next(data_iter)
        inp, lbl = inp.to(device), lbl.to(device)

        with torch.no_grad():
            t_logits = teacher(input_ids=inp.to(teacher_device)).logits.to(device)

        s_out    = student(input_ids=inp, labels=lbl)
        s_logits = s_out.logits
        ce_loss  = s_out.loss

        kl_loss = kl(
            F.log_softmax(s_logits.float().reshape(-1, s_logits.size(-1)) / T, dim=-1),
            F.softmax(t_logits.float().reshape(-1, t_logits.size(-1)) / T, dim=-1),
        ) * T * T
        loss = alpha * ce_loss + (1 - alpha) * kl_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        enforce_masks(mask_pairs)

        if step % log_every == 0:
            pbar.set_postfix(ce=f"{ce_loss.item():.3f}",
                             kl=f"{kl_loss.item():.3f}",
                             sp=f"{current_sparsity(mask_pairs)*100:.1f}%")

        if step % eval_every == 0:
            ppl = evaluate_ppl(student, tokenizer, device)
            print(f"\n  Step {step}: PPL={ppl:.2f} (best={best_ppl:.2f})", flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                if output_path:
                    student.save_pretrained(os.path.join(output_path, "best"))
                    tokenizer.save_pretrained(os.path.join(output_path, "best"))
                    print(f"  -> saved best checkpoint", flush=True)

    for h in hook_handles:
        h.remove()

    final_ppl = evaluate_ppl(student, tokenizer, device)
    print(f"  Final PPL: {final_ppl:.2f}", flush=True)
    return final_ppl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dense_path",    required=True)
    p.add_argument("--model_type",    choices=["transformer", "llama", "hgrn"], required=True)
    p.add_argument("--output_path",   required=True)

    # Sparsity profile overrides
    p.add_argument("--lm_head_sparsity",  type=float, default=0.60)
    p.add_argument("--boundary_0",        type=float, default=0.65,
                   help="Sparsity for first/last layer")
    p.add_argument("--boundary_1",        type=float, default=0.70)
    p.add_argument("--boundary_2",        type=float, default=0.75)
    p.add_argument("--boundary_3",        type=float, default=0.80)
    p.add_argument("--middle_sparsity",   type=float, default=0.85)

    # Distillation
    p.add_argument("--total_steps",   type=int,   default=50_000)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--warmup_steps",  type=int,   default=2_000)
    p.add_argument("--alpha",         type=float, default=0.1)
    p.add_argument("--temperature",   type=float, default=2.0)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--seq_len",       type=int,   default=512)
    p.add_argument("--skip_distill",  action="store_true",
                   help="Only prune, skip distillation")

    p.add_argument("--device",        default="cuda")
    p.add_argument("--teacher_device", default=None)
    p.add_argument("--n_calib",       type=int, default=64)
    return p.parse_args()


def main():
    args           = parse_args()
    device         = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device or args.device)

    profile = {
        "boundary_sparsities": {
            0: args.boundary_0,
            1: args.boundary_1,
            2: args.boundary_2,
            3: args.boundary_3,
        },
        "middle_sparsity": args.middle_sparsity,
        "lm_head": args.lm_head_sparsity,
    }

    print(f"\n{'='*60}", flush=True)
    print(f"Dense model    : {args.dense_path}", flush=True)
    print(f"Sparsity profile: boundary={[profile['boundary_sparsities'][k] for k in sorted(profile['boundary_sparsities'])]}, "
          f"middle={args.middle_sparsity:.0%}, lm_head={args.lm_head_sparsity:.0%}", flush=True)
    print(f"Distill steps  : {args.total_steps}", flush=True)
    print(f"Devices        : student={device}, teacher={teacher_device}", flush=True)
    print(f"Output         : {args.output_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.dense_path, trust_remote_code=True)

    print("Loading model ...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.dense_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    n_layers = len(student.model.layers)
    print(f"Model has {n_layers} transformer layers.", flush=True)

    os.makedirs(args.output_path, exist_ok=True)

    # Non-uniform pruning
    achieved_sp, layer_log = prune_nonuniform(
        student, tokenizer, n_layers, profile, device,
        n_calib=args.n_calib, seq_len=args.seq_len
    )

    ppl_after_prune = evaluate_ppl(student, tokenizer, device)
    print(f"\nPPL after non-uniform pruning: {ppl_after_prune:.2f}", flush=True)
    print(f"Overall sparsity: {achieved_sp*100:.2f}%\n", flush=True)

    # Save pruned-only checkpoint
    pruned_path = os.path.join(args.output_path, "pruned")
    student.save_pretrained(pruned_path)
    tokenizer.save_pretrained(pruned_path)
    print(f"Saved pruned model to {pruned_path}", flush=True)

    meta = {
        "dense_path": args.dense_path,
        "profile": {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
                    for k, v in profile.items()},
        "achieved_sparsity": achieved_sp,
        "ppl_after_prune": ppl_after_prune,
        "layer_log": layer_log,
    }

    if args.skip_distill:
        with open(os.path.join(args.output_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print("Done (pruning only).", flush=True)
        return

    # Load teacher (frozen dense)
    print("Loading teacher ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.dense_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(teacher_device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student.train()
    final_ppl = distill(
        student, teacher, tokenizer, args.total_steps,
        lr=args.lr, device=device, teacher_device=teacher_device,
        batch_size=args.batch_size, seq_len=args.seq_len,
        warmup=args.warmup_steps,
        alpha=args.alpha, T=args.temperature,
        output_path=args.output_path,
    )

    student.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    meta["final_ppl"] = final_ppl
    meta["total_steps"] = args.total_steps
    with open(os.path.join(args.output_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Final PPL: {final_ppl:.2f}  Saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
