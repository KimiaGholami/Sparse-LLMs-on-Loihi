"""
LLaMA pruning wrapper.

Runs prune_sparsegpt or prune_obs_cancel on LLaMA-family models, which use
a SentencePiece tokenizer requiring LlamaTokenizer instead of AutoTokenizer.

Usage
-----
python scripts/prune_llama.py \\
    --method sparsegpt|obs_cancel \\
    --model_path /path/to/llama \\
    --sparsity 0.5 \\
    --n_calib_batches 32 \\
    --batch_size 4 \\
    --seq_len 512 \\
    --output_path exp/llama-7b-sparsegpt-50pct \\
    --eval_ppl
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Import pruning functions from existing scripts
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))
from prune_sparsegpt import (
    CovarianceStats, build_calib_batches, collect_covariance_stats,
    evaluate_ppl, sparsegpt_prune_layer,
)
from prune_obs_cancel import obs_cancel_prune_layer, obs_cancel_block_prune_layer


# ---------------------------------------------------------------------------
# Model pruning
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_model(model, stats, sparsity, method, device, damp=0.01, block_size=128):
    total_w, total_p = 0, 0
    layer_info = {}
    _labels = {"sparsegpt": "SparseGPT", "obs_cancel": "OBS-cancel",
               "obs_cancel_block": "OBS-cancel-block"}
    label = _labels.get(method, method)
    _fns   = {"sparsegpt": sparsegpt_prune_layer,
               "obs_cancel": obs_cancel_prune_layer,
               "obs_cancel_block": obs_cancel_block_prune_layer}
    prune_fn = _fns[method]

    for name, module in tqdm(model.named_modules(), desc=f"{label} pruning"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue

        W = module.weight.data
        out_f, in_f = W.shape
        if int(in_f * sparsity) == 0:
            continue

        # Move Sigma to the same device as this layer's weights
        layer_device = W.device
        Sigma = stats[name].second_moment().to(layer_device)

        W_corr = prune_fn(W.float(), Sigma, sparsity,
                          damp=damp, block_size=block_size)

        module.weight.data.copy_(W_corr.to(W.dtype))

        n_pruned = (W_corr == 0).sum().item()
        total_w += out_f * in_f
        total_p += n_pruned
        layer_info[name] = {"out_features": out_f, "in_features": in_f,
                            "n_pruned": n_pruned}

        del Sigma, W_corr
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return total_p / max(total_w, 1), layer_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",
                   choices=["sparsegpt", "obs_cancel", "obs_cancel_block"],
                   required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--sparsity", type=float, default=0.5)
    p.add_argument("--n_calib_batches", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--output_path", type=str, default=None)
    p.add_argument("--eval_ppl", action="store_true")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading model from {args.model_path} ...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    # device_map="auto" distributes layers across GPU/CPU as available.
    # For pruning we iterate layer-by-layer, so this works correctly.
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {n_params:.2f}B parameters")
    print(f"  Device map: {model.hf_device_map}")

    # With device_map="auto", inputs go to the first device
    input_device = torch.device(next(iter(model.hf_device_map.values())))

    ppl_before = None
    if args.eval_ppl:
        ppl_before = evaluate_ppl(model, tokenizer, input_device, args.seq_len)
        print(f"Perplexity before: {ppl_before:.4f}")

    print(f"\nBuilding calibration batches ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches,
                                  args.batch_size, args.seq_len)
    stats = collect_covariance_stats(model, batches, input_device)

    print(f"\n{args.method} pruning at {args.sparsity * 100:.0f}% sparsity ...")
    actual_sparsity, layer_info = prune_model(
        model, stats, args.sparsity, args.method, input_device,
        damp=args.damp, block_size=args.block_size,
    )
    print(f"Actual sparsity: {actual_sparsity * 100:.2f}%")

    ppl_after = None
    if args.eval_ppl:
        ppl_after = evaluate_ppl(model, tokenizer, input_device, args.seq_len)
        print(f"Perplexity after:  {ppl_after:.4f}  (Δ = {ppl_after - ppl_before:+.4f})")

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        model.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        summary = {
            "method": args.method,
            "model": args.model_path,
            "sparsity_target": args.sparsity,
            "sparsity_actual": actual_sparsity,
            "ppl_before": ppl_before,
            "ppl_after": ppl_after,
            "layer_info": layer_info,
        }
        with open(os.path.join(args.output_path, "pruning_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
