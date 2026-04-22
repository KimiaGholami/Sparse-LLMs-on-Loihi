"""
LLaMA-7B AWP sparsity sweep: adds AWP results to sparsity_sweep_llama.json.

Loads the model once, collects covariance stats, then for each sparsity:
restores dense weights → AWP prune → evaluate PPL.
Writes results into results/sparsity_sweep_llama.json under key "awp".

Usage
-----
python scripts/sparsity_sweep_awp_llama.py \\
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
from prune_awp import CovarianceStats, build_calib_batches, collect_stats, prune_awp, evaluate_ppl


SPARSITIES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--n_calib_batches", type=int, default=32)
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--seq_len",         type=int, default=512)
    p.add_argument("--max_iter",        type=int, default=200)
    p.add_argument("--tol",             type=float, default=1e-4)
    p.add_argument("--output", default="results/sparsity_sweep_llama.json")
    args = p.parse_args()

    print(f"Loading model from {args.model_path} ...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    input_device = torch.device(next(iter(model.hf_device_map.values())))

    # Dense baseline
    print("\nEvaluating dense baseline ...")
    dense_ppl = evaluate_ppl(model, tokenizer, input_device, args.seq_len)
    print(f"Dense PPL: {dense_ppl:.4f}")

    # Snapshot dense weights
    print("Snapshotting dense weights (CPU) ...")
    dense_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Collect covariance stats once
    print("\nCollecting calibration stats ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches,
                                  args.batch_size, args.seq_len)
    stats = collect_stats(model, batches, input_device)

    # Load or init results file
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Loaded existing results from {args.output}")
    else:
        results = {"dense_ppl": dense_ppl}

    results.setdefault("awp", {})

    for sparsity in SPARSITIES:
        canon_key = f"{sparsity:.2f}"
        if canon_key in results["awp"]:
            print(f"[awp] {sparsity*100:.0f}% already done "
                  f"(PPL={results['awp'][canon_key]:.4f}), skipping.")
            continue

        print(f"\n[awp] sparsity={sparsity*100:.0f}% ...")
        model.load_state_dict(dense_state, strict=False)

        _, _ = prune_awp(model, stats, sparsity, input_device,
                         max_iter=args.max_iter, tol=args.tol)
        ppl = evaluate_ppl(model, tokenizer, input_device, args.seq_len)
        results["awp"][canon_key] = ppl
        print(f"  PPL: {ppl:.4f}")

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'='*50}")
    print(f"Dense baseline: {results.get('dense_ppl', dense_ppl):.2f}")
    print(f"\n{'Sparsity':>10}  {'AWP':>10}")
    for sparsity in SPARSITIES:
        v = results["awp"].get(f"{sparsity:.2f}", float("nan"))
        print(f"{sparsity*100:>9.0f}%  {v:>10.2f}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
