"""
HGRN-1.3B sparsity sweep: SparseGPT vs OBS-cancel-block at 30–80% sparsity.

Usage
-----
python scripts/sparsity_sweep_hgrn.py \
    --model_path exp/hgrn-1.3B-dense-baseline \
    --output results/sparsity_sweep_hgrn.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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

sys.path.insert(0, os.path.dirname(__file__))
from prune_sparsegpt import (
    build_calib_batches, collect_covariance_stats,
    evaluate_ppl, sparsegpt_prune_layer,
)
from prune_obs_cancel import obs_cancel_block_prune_layer

SPARSITIES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
METHODS    = ["sparsegpt", "obs_cancel_block"]


@torch.no_grad()
def prune_and_eval(model, dense_state, stats, sparsity, method,
                   device, tokenizer, seq_len, damp, block_size):
    model.load_state_dict(dense_state)
    fn = obs_cancel_block_prune_layer if method == "obs_cancel_block" else sparsegpt_prune_layer
    label = "OBS-cancel-block" if method == "obs_cancel_block" else "SparseGPT"

    for name, module in tqdm(model.named_modules(),
                              desc=f"{label} {sparsity*100:.0f}%"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        W = module.weight.data
        if int(W.shape[1] * sparsity) == 0:
            continue
        Sigma = stats[name].second_moment().to(device)
        W_corr = fn(W.float(), Sigma, sparsity, damp=damp, block_size=block_size)
        module.weight.data.copy_(W_corr.to(W.dtype))
        del Sigma, W_corr
        torch.cuda.empty_cache()

    return evaluate_ppl(model, tokenizer, device, seq_len)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",      type=str, default="exp/hgrn-1.3B-dense-baseline")
    p.add_argument("--n_calib_batches", type=int, default=64)
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--seq_len",         type=int, default=512)
    p.add_argument("--damp",            type=float, default=0.01)
    p.add_argument("--block_size",      type=int, default=128)
    p.add_argument("--output", default="results/sparsity_sweep_hgrn.json")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    print(f"  {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")

    dense_ppl = evaluate_ppl(model, tokenizer, device, args.seq_len)
    print(f"Dense PPL: {dense_ppl:.4f}")

    dense_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("\nCollecting calibration stats ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches,
                                  args.batch_size, args.seq_len)
    stats = collect_covariance_stats(model, batches, device)

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
                print(f"[{method}] {sparsity*100:.0f}% already done, skipping.")
                continue
            ppl = prune_and_eval(model, dense_state, stats, sparsity, method,
                                 device, tokenizer, args.seq_len,
                                 damp=args.damp, block_size=args.block_size)
            results.setdefault(method, {})[key] = ppl
            print(f"[{method}] {sparsity*100:.0f}%  PPL: {ppl:.4f}")
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n{'Sparsity':>10}  {'SparseGPT':>12}  {'OBS-cancel-block':>18}  {'Improvement':>12}")
    for sparsity in SPARSITIES:
        key = f"{sparsity:.2f}"
        sg  = results["sparsegpt"].get(key, "-")
        oc  = results["obs_cancel_block"].get(key, "-")
        imp = f"{sg/oc:.3f}×" if isinstance(sg, float) and isinstance(oc, float) else "-"
        print(f"{sparsity*100:>9.0f}%  {sg if isinstance(sg,str) else f'{sg:.2f}':>12}  "
              f"{oc if isinstance(oc,str) else f'{oc:.2f}':>18}  {imp:>12}")
    print(f"\nFull results: {args.output}")


if __name__ == "__main__":
    main()
