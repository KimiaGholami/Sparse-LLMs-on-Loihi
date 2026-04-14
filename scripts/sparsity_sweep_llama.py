"""
LLaMA-7B sparsity sweep: SparseGPT vs OBS-cancel-block at 30–80% sparsity.

Loads the model and collects calibration stats once, then for each
(method, sparsity) pair: restores dense weights → prunes → evaluates PPL.
Intermediate results are flushed after each run.

Usage
-----
python scripts/sparsity_sweep_llama.py \\
    --model_path /path/to/open_llama_7b \\
    --output results/sparsity_sweep_llama.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from prune_sparsegpt import (
    CovarianceStats, build_calib_batches, collect_covariance_stats,
    evaluate_ppl, sparsegpt_prune_layer,
)
from prune_obs_cancel import obs_cancel_block_prune_layer


SPARSITIES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
METHODS    = ["sparsegpt", "obs_cancel_block"]


@torch.no_grad()
def prune_and_eval(model, dense_state, stats, sparsity, method,
                   input_device, tokenizer, seq_len, damp, block_size):
    """Restore dense weights, prune at sparsity, return PPL."""
    model.load_state_dict(dense_state, strict=False)

    _fn    = (obs_cancel_block_prune_layer if method == "obs_cancel_block"
              else sparsegpt_prune_layer)
    label  = "OBS-cancel-block" if method == "obs_cancel_block" else "SparseGPT"

    for name, module in tqdm(model.named_modules(),
                              desc=f"{label} {sparsity*100:.0f}%"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        W = module.weight.data
        if int(W.shape[1] * sparsity) == 0:
            continue
        layer_device = W.device
        Sigma  = stats[name].second_moment().to(layer_device)
        W_corr = _fn(W.float(), Sigma, sparsity, damp=damp, block_size=block_size)
        module.weight.data.copy_(W_corr.to(W.dtype))
        del Sigma, W_corr
        torch.cuda.empty_cache()

    return evaluate_ppl(model, tokenizer, input_device, seq_len)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--n_calib_batches", type=int, default=32)
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--seq_len",         type=int, default=512)
    p.add_argument("--damp",            type=float, default=0.01)
    p.add_argument("--block_size",      type=int, default=128)
    p.add_argument("--output", default="results/sparsity_sweep_llama.json")
    args = p.parse_args()

    print(f"Loading model from {args.model_path} ...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    input_device = torch.device(next(iter(model.hf_device_map.values())))

    # Dense baseline PPL
    print("\nEvaluating dense baseline ...")
    dense_ppl = evaluate_ppl(model, tokenizer, input_device, args.seq_len)
    print(f"Dense PPL: {dense_ppl:.4f}")

    # Snapshot dense weights on CPU
    print("Snapshotting dense weights (CPU) ...")
    dense_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Collect calibration stats once
    print("\nCollecting calibration stats ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches,
                                  args.batch_size, args.seq_len)
    stats = collect_covariance_stats(model, batches, input_device)

    # Load existing partial results if present
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Resuming from {args.output}")
    else:
        results = {"dense_ppl": dense_ppl, "sparsegpt": {}, "obs_cancel_block": {}}

    for method in METHODS:
        for sparsity in SPARSITIES:
            key = f"{sparsity:.2f}"
            if key in results.get(method, {}):
                print(f"[{method}] {sparsity*100:.0f}% already done (PPL={results[method][key]:.4f}), skipping.")
                continue
            print(f"\n[{method}] sparsity={sparsity*100:.0f}% ...")
            ppl = prune_and_eval(
                model, dense_state, stats, sparsity, method,
                input_device, tokenizer, args.seq_len,
                damp=args.damp, block_size=args.block_size,
            )
            results.setdefault(method, {})[key] = ppl
            print(f"  PPL: {ppl:.4f}")
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Dense baseline: {results['dense_ppl']:.2f}")
    print(f"\n{'Sparsity':>10}  {'SparseGPT':>12}  {'OBS-cancel-block':>18}")
    for sparsity in SPARSITIES:
        key = f"{sparsity:.2f}"
        sg  = results["sparsegpt"].get(key, "-")
        oc  = results["obs_cancel_block"].get(key, "-")
        sg_s = f"{sg:.2f}" if isinstance(sg, float) else sg
        oc_s = f"{oc:.2f}" if isinstance(oc, float) else oc
        print(f"{sparsity*100:>9.0f}%  {sg_s:>12}  {oc_s:>18}")
    print(f"\nFull results: {args.output}")


if __name__ == "__main__":
    main()
