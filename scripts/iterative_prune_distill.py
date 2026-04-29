"""
Iterative pruning + distillation for high sparsity.

Prunes the dense model in stages, running knowledge distillation between each
stage using the original dense model as a frozen teacher.  Each pruning step
uses OBS-cancel-block on the current (adapted) model weights, so the mask
evolves to fit the weights rather than being computed once from the dense model.

Stages (default: 30% -> 50% -> 65% -> 80%)
-------------------------------------------
  1. Dense model pruned to stage_1 sparsity with OBS-cancel-block
  2. Distill for steps_per_stage steps (mask fixed)
  3. Re-prune surviving weights to stage_2 sparsity
  4. Distill ...  repeat until final sparsity

Usage
-----
CUDA_VISIBLE_DEVICES=0,1 python scripts/iterative_prune_distill.py \\
    --dense_path  exp/hgrn-1.3B-dense-baseline \\
    --model_type  hgrn \\
    --output_path exp/hgrn-1.3B-80pct-iterative \\
    --stages      0.3,0.5,0.65,0.8 \\
    --steps_per_stage 20000,20000,20000,50000 \\
    --lr 2e-5 \\
    --device      cuda:0 \\
    --teacher_device cuda:1
"""

import argparse, json, math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)
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
# Mask utilities
# ---------------------------------------------------------------------------

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
        for w, m in pairs: w.data.mul_(m)

def sparsity_of(model):
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
# Pruning step (incremental)
# ---------------------------------------------------------------------------

def prune_to_sparsity(model, tokenizer, target_sp, device,
                      n_calib=64, batch_size=4, seq_len=512):
    """Re-prune the current model weights to target_sp using OBS-cancel-block."""
    print(f"  Collecting calibration stats (target {target_sp*100:.0f}%) ...", flush=True)
    batches = build_calib_batches(tokenizer, n_calib, batch_size, seq_len)
    stats   = collect_covariance_stats(model, batches, device)

    total_w = total_p = 0
    for name, module in tqdm(model.named_modules(), desc="OBS-cancel-block pruning"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        W     = module.weight.data
        out_f, in_f = W.shape
        if int(in_f * target_sp) == 0:
            continue
        Sigma  = stats[name].second_moment().to(device)
        W_new  = obs_cancel_block_prune_layer(W.float(), Sigma, target_sp)
        module.weight.data.copy_(W_new.to(W.dtype))
        total_w += W.numel()
        total_p += (W_new == 0).sum().item()
        del Sigma, W_new
        torch.cuda.empty_cache()

    del stats, batches
    torch.cuda.empty_cache()
    return total_p / max(total_w, 1)


# ---------------------------------------------------------------------------
# Distillation loop
# ---------------------------------------------------------------------------

def distill(student, teacher, tokenizer, n_steps, lr, device, teacher_device,
            batch_size=16, seq_len=512, warmup=500,
            alpha=0.1, T=2.0, log_every=200, eval_every=2000):

    mask_pairs, hook_handles = register_masks(student)
    print(f"  Mask registered. Sparsity: {sparsity_of(student)*100:.1f}%", flush=True)

    loader    = DataLoader(C4StreamDataset(tokenizer, seq_len),
                           batch_size=batch_size, num_workers=2)
    data_iter = iter(loader)

    opt  = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, warmup, n_steps)
    kl   = nn.KLDivLoss(reduction="batchmean")
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
                             sp=f"{sparsity_of(student)*100:.1f}%")
        if step % eval_every == 0:
            ppl = evaluate_ppl(student, tokenizer, device)
            if ppl < best_ppl:
                best_ppl = ppl
            print(f"\n  Step {step}: PPL={ppl:.2f} (best={best_ppl:.2f})", flush=True)

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
    p.add_argument("--dense_path",  required=True)
    p.add_argument("--model_type",  choices=["transformer", "llama", "hgrn"], required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--stages",      default="0.3,0.5,0.65,0.8",
                   help="Comma-separated sparsity targets per stage")
    p.add_argument("--steps_per_stage", default="20000,20000,20000,50000",
                   help="Distillation steps per stage (comma-separated)")
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--seq_len",     type=int,   default=512)
    p.add_argument("--alpha",       type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--teacher_device", default=None)
    return p.parse_args()


def main():
    args           = parse_args()
    device         = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device or args.device)
    stages         = [float(s) for s in args.stages.split(",")]
    steps          = [int(s)   for s in args.steps_per_stage.split(",")]
    assert len(stages) == len(steps), "--stages and --steps_per_stage must have same length"

    print(f"\n{'='*60}", flush=True)
    print(f"Dense model : {args.dense_path}", flush=True)
    print(f"Stages      : {stages}", flush=True)
    print(f"Steps/stage : {steps}", flush=True)
    print(f"Output      : {args.output_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.dense_path, trust_remote_code=True)

    print("Loading teacher (frozen dense model) ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.dense_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(teacher_device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("Loading student (starts as dense) ...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.dense_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    os.makedirs(args.output_path, exist_ok=True)
    log = {"stages": []}

    for i, (target_sp, n_steps) in enumerate(zip(stages, steps)):
        print(f"\n{'='*60}", flush=True)
        print(f"Stage {i+1}/{len(stages)}: pruning to {target_sp*100:.0f}%  "
              f"then distilling {n_steps} steps", flush=True)
        print(f"{'='*60}", flush=True)

        actual_sp = prune_to_sparsity(
            student, tokenizer, target_sp, device
        )
        print(f"  Achieved sparsity: {actual_sp*100:.2f}%", flush=True)

        ppl_after_prune = evaluate_ppl(student, tokenizer, device)
        print(f"  PPL after pruning: {ppl_after_prune:.2f}", flush=True)

        final_ppl = distill(
            student, teacher, tokenizer, n_steps,
            lr=args.lr, device=device, teacher_device=teacher_device,
            batch_size=args.batch_size, seq_len=args.seq_len,
            warmup=min(500, n_steps // 10),
            alpha=args.alpha, T=args.temperature,
        )

        ckpt = os.path.join(args.output_path, f"stage-{i+1}-sp{target_sp:.0%}")
        student.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"  Saved stage checkpoint: {ckpt}", flush=True)

        log["stages"].append({
            "stage": i + 1,
            "target_sparsity": target_sp,
            "actual_sparsity": actual_sp,
            "ppl_after_prune": ppl_after_prune,
            "ppl_after_distill": final_ppl,
            "steps": n_steps,
        })

    student.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    with open(os.path.join(args.output_path, "iterative_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. Final model saved to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
