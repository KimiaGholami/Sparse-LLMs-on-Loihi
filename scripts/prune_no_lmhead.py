"""
OBS-cancel-block pruning with lm_head excluded, followed by distillation.

The lm_head maps hidden states to all 32k vocabulary logits.  At 80% sparsity
each logit only uses ~410 of 2048 features, which flattens the output
distribution and collapses benchmark accuracy.  Excluding lm_head (5% of
total params) and pruning the remaining layers slightly harder gives the same
overall sparsity while keeping vocabulary predictions coherent.

Default targets (--overall_sparsity 0.80):
  lm_head      : 0% pruned  (fully dense)
  all other     : 84.25% pruned  →  overall ≈ 80%

Usage
-----
CUDA_VISIBLE_DEVICES=0,1 python scripts/prune_no_lmhead.py \\
    --dense_path     exp/hgrn-1.3B-dense-baseline \\
    --model_type     hgrn \\
    --output_path    exp/hgrn-1.3B-80pct-no-lmhead \\
    --overall_sparsity 0.80 \\
    --device         cuda:0 \\
    --teacher_device cuda:1 \\
    --total_steps    50000
"""

import argparse
import json
import math
import os
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
# Sparsity arithmetic
# ---------------------------------------------------------------------------

def compute_layer_sparsity(model, overall_sparsity):
    """
    Return the uniform per-layer sparsity needed (excluding lm_head) to hit
    overall_sparsity across all Linear parameters.
    """
    total = lmhead = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        total += mod.weight.numel()
        if name == "lm_head":
            lmhead += mod.weight.numel()
    other = total - lmhead
    zeros_needed = overall_sparsity * total          # target zeros overall
    # lm_head contributes 0 zeros
    sp = zeros_needed / other if other > 0 else overall_sparsity
    return min(sp, 0.99), total, lmhead, other


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_model(model, tokenizer, layer_sparsity, device,
                n_calib=64, batch_size=4, seq_len=512):
    """Prune all Linear layers except lm_head to layer_sparsity."""
    print(f"  Collecting calibration stats ...", flush=True)
    batches = build_calib_batches(tokenizer, n_calib, batch_size, seq_len)
    stats   = collect_covariance_stats(model, batches, device)

    total_w = total_p = 0
    for name, module in tqdm(model.named_modules(), desc="OBS pruning (no lm_head)"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        if name == "lm_head":
            print(f"  Skipping lm_head (kept dense)", flush=True)
            continue

        W = module.weight.data
        out_f, in_f = W.shape
        if int(in_f * layer_sparsity) == 0:
            continue

        Sigma = stats[name].second_moment().to(device)
        W_new = obs_cancel_block_prune_layer(W.float(), Sigma, layer_sparsity)
        module.weight.data.copy_(W_new.to(W.dtype))

        total_w += W.numel()
        total_p += (W_new == 0).sum().item()
        del Sigma, W_new
        torch.cuda.empty_cache()

    del stats, batches
    torch.cuda.empty_cache()
    return total_p / max(total_w, 1)


# ---------------------------------------------------------------------------
# Data / masks / PPL
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
# Distillation
# ---------------------------------------------------------------------------

def distill(student, teacher, tokenizer, n_steps, lr, device, teacher_device,
            batch_size=16, seq_len=512, warmup=2000,
            alpha=0.1, T=2.0, log_every=200, eval_every=5000,
            output_path=None):

    mask_pairs, hook_handles = register_masks(student)
    print(f"  Sparsity after masking: {current_sparsity(mask_pairs)*100:.2f}%", flush=True)

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
    p.add_argument("--dense_path",       required=True)
    p.add_argument("--model_type",       choices=["transformer", "llama", "hgrn"], required=True)
    p.add_argument("--output_path",      required=True)
    p.add_argument("--overall_sparsity", type=float, default=0.80,
                   help="Target overall sparsity (lm_head excluded from pruning)")

    p.add_argument("--total_steps",    type=int,   default=50_000)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--warmup_steps",   type=int,   default=2_000)
    p.add_argument("--alpha",          type=float, default=0.1)
    p.add_argument("--temperature",    type=float, default=2.0)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--seq_len",        type=int,   default=512)
    p.add_argument("--n_calib",        type=int,   default=64)
    p.add_argument("--skip_distill",   action="store_true")

    p.add_argument("--device",         default="cuda")
    p.add_argument("--teacher_device", default=None)
    return p.parse_args()


def main():
    args           = parse_args()
    device         = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device or args.device)

    print(f"\n{'='*60}", flush=True)
    print(f"Dense model      : {args.dense_path}", flush=True)
    print(f"Overall sparsity : {args.overall_sparsity:.0%}", flush=True)
    print(f"lm_head          : KEPT DENSE", flush=True)
    print(f"Distill steps    : {args.total_steps}", flush=True)
    print(f"Devices          : student={device}, teacher={teacher_device}", flush=True)
    print(f"Output           : {args.output_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.dense_path, trust_remote_code=True)

    print("Loading model ...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.dense_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    layer_sp, total_params, lmhead_params, other_params = compute_layer_sparsity(
        student, args.overall_sparsity
    )
    print(f"Total params     : {total_params:,}", flush=True)
    print(f"lm_head params   : {lmhead_params:,}  ({lmhead_params/total_params*100:.1f}%)", flush=True)
    print(f"Other params     : {other_params:,}", flush=True)
    print(f"Required layer sp: {layer_sp*100:.2f}% on non-lm_head layers", flush=True)
    print(f"Expected overall : {layer_sp*other_params/total_params*100:.2f}%\n", flush=True)

    os.makedirs(args.output_path, exist_ok=True)

    # Prune
    achieved_other_sp = prune_model(
        student, tokenizer, layer_sp, device,
        n_calib=args.n_calib, seq_len=args.seq_len
    )
    # Verify lm_head is still dense
    lm_sp = (student.lm_head.weight.data == 0).sum().item() / student.lm_head.weight.numel()
    print(f"\nlm_head sparsity : {lm_sp*100:.1f}%  (should be 0%)", flush=True)

    # Compute true overall sparsity
    total_zeros = sum(
        (mod.weight.data == 0).sum().item()
        for mod in student.modules() if isinstance(mod, nn.Linear)
    )
    overall_achieved = total_zeros / total_params
    print(f"Overall sparsity : {overall_achieved*100:.2f}%", flush=True)

    ppl_after_prune = evaluate_ppl(student, tokenizer, device)
    print(f"PPL after pruning: {ppl_after_prune:.2f}\n", flush=True)

    pruned_path = os.path.join(args.output_path, "pruned")
    student.save_pretrained(pruned_path)
    tokenizer.save_pretrained(pruned_path)
    print(f"Saved pruned model to {pruned_path}", flush=True)

    meta = {
        "dense_path": args.dense_path,
        "overall_sparsity_target": args.overall_sparsity,
        "layer_sparsity": layer_sp,
        "overall_sparsity_achieved": overall_achieved,
        "lm_head_sparsity": lm_sp,
        "ppl_after_prune": ppl_after_prune,
    }

    if args.skip_distill:
        with open(os.path.join(args.output_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print("Done (pruning only).", flush=True)
        return

    # Teacher
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
        warmup=args.warmup_steps, alpha=args.alpha, T=args.temperature,
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
