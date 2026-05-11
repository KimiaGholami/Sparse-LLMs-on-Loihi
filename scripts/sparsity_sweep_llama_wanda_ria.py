"""
LLaMA-7B sparsity sweep extension: adds Wanda and RIA at 30–80% sparsity.

Loads the model and collects calibration stats once (full covariance, same as
sparsity_sweep_llama.py). Derives act_rms for Wanda/RIA from the diagonal of
Sigma_X to avoid a second forward pass. Appends results to the existing
sparsity_sweep_llama.json if present.

Usage
-----
python scripts/sparsity_sweep_llama_wanda_ria.py \
    --model_path /path/to/open_llama_7b \
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
    evaluate_ppl,
)
from prune_ria import ria_prune_layer

SPARSITIES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
METHODS    = ["wanda", "ria"]

LLAMA_PATH = ("/mnt/cephfs/share/kimia/hf_cache/hub/models--openlm-research"
              "--open_llama_7b/snapshots/"
              "6fb184ff23774c25bf84b3628e49c8b78372c7be")


def act_rms_from_cov(cov_stats: CovarianceStats) -> torch.Tensor:
    """sqrt(E[x_c^2]) from the diagonal of Sigma_X."""
    return cov_stats.second_moment().diagonal().clamp(min=0).sqrt().float()


@torch.no_grad()
def wanda_prune_layer(W: torch.Tensor, act_rms: torch.Tensor,
                      sparsity: float) -> torch.Tensor:
    out_f, in_f = W.shape
    k = int(in_f * sparsity)
    if k == 0:
        return W.clone()
    scores = W.float().abs() * act_rms.to(W.device).unsqueeze(0)
    _, prune_idx = scores.topk(k, dim=1, largest=False)
    W_out = W.float().clone()
    W_out.scatter_(1, prune_idx, 0.0)
    return W_out


@torch.no_grad()
def prune_and_eval(model, dense_state, stats, sparsity, method,
                   input_device, tokenizer, seq_len):
    model.load_state_dict(dense_state, strict=False)
    label = method.upper()

    for name, module in tqdm(model.named_modules(),
                              desc=f"{label} {sparsity*100:.0f}%"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        W = module.weight.data
        if int(W.shape[1] * sparsity) == 0:
            continue
        act_rms = act_rms_from_cov(stats[name])
        if method == "wanda":
            W_pruned = wanda_prune_layer(W.float(), act_rms, sparsity)
        else:  # ria
            W_pruned = ria_prune_layer(W.float(), act_rms, sparsity, alpha=0.5)
        module.weight.data.copy_(W_pruned.to(W.dtype))
        del act_rms, W_pruned

    return evaluate_ppl(model, tokenizer, input_device, seq_len)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",      type=str, default=LLAMA_PATH)
    p.add_argument("--n_calib_batches", type=int, default=32)
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--seq_len",         type=int, default=512)
    p.add_argument("--output", default="results/sparsity_sweep_llama.json")
    args = p.parse_args()

    print(f"Loading model from {args.model_path} ...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")
    input_device = torch.device(next(iter(model.hf_device_map.values())))

    dense_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("\nCollecting calibration stats ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches,
                                  args.batch_size, args.seq_len)
    stats = collect_covariance_stats(model, batches, input_device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Loaded existing results from {args.output}")
    else:
        results = {"dense_ppl": 8.6409}

    for method in METHODS:
        for sparsity in SPARSITIES:
            key = f"{sparsity:.2f}"
            if key in results.get(method, {}):
                print(f"[{method}] {sparsity*100:.0f}% already done, skipping.")
                continue
            ppl = prune_and_eval(model, dense_state, stats, sparsity, method,
                                 input_device, tokenizer, args.seq_len)
            results.setdefault(method, {})[key] = ppl
            print(f"[{method}] {sparsity*100:.0f}%  PPL: {ppl:.4f}")
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nFull results: {args.output}")
    for method in METHODS:
        print(f"\n{method.upper()}:")
        for sparsity in SPARSITIES:
            key = f"{sparsity:.2f}"
            ppl = results.get(method, {}).get(key, "-")
            print(f"  {sparsity*100:.0f}%  {ppl:.4f}" if isinstance(ppl, float) else f"  {sparsity*100:.0f}%  -")


if __name__ == "__main__":
    main()
