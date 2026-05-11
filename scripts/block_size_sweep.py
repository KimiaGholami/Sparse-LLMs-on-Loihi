"""
Block size sweep: OBS-cancel-block PPL vs --block_size on the 1B model.

Tests block sizes [64, 128, 256, 512] at 50% sparsity to show how much
cross-block cancellation is captured as the block size grows.  Only PPL
is measured (no model save) so each run is fast.

Usage
-----
python scripts/block_size_sweep.py \\
    --model_path exp/transformer-1B-dense-baseline \\
    --output results/block_size_sweep.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from prune_sparsegpt import (
    build_calib_batches, collect_covariance_stats, evaluate_ppl,
)
from prune_obs_cancel import obs_cancel_block_prune_layer

try:
    import fla  # noqa
    from fla.models.transformer import TransformerConfig, TransformerForCausalLM
    AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
    AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)
except Exception as e:
    print(f"Warning: could not register FLA transformer type: {e}")

BLOCK_SIZES = [64, 128, 256, 512]


@torch.no_grad()
def prune_and_eval(model, dense_state, stats, sparsity, block_size, device,
                   tokenizer, seq_len, damp):
    model.load_state_dict(dense_state)
    for name, module in tqdm(model.named_modules(),
                              desc=f"block_size={block_size}"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        W = module.weight.data
        if int(W.shape[1] * sparsity) == 0:
            continue
        Sigma = stats[name].second_moment().to(device)
        W_corr = obs_cancel_block_prune_layer(
            W.float(), Sigma, sparsity, damp=damp, block_size=block_size,
        )
        module.weight.data.copy_(W_corr.to(W.dtype))
        del Sigma, W_corr
        torch.cuda.empty_cache()
    return evaluate_ppl(model, tokenizer, device, seq_len)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",      type=str,   default="exp/transformer-1B-dense-baseline")
    p.add_argument("--sparsity",        type=float, default=0.5)
    p.add_argument("--n_calib_batches", type=int,   default=64)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--seq_len",         type=int,   default=512)
    p.add_argument("--damp",            type=float, default=0.01)
    p.add_argument("--output", default="results/block_size_sweep.json")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

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
        results = {"dense_ppl": dense_ppl, "sparsity": args.sparsity,
                   "block_size_ppl": {}}

    for bs in BLOCK_SIZES:
        key = str(bs)
        if key in results["block_size_ppl"]:
            print(f"block_size={bs} already done (PPL={results['block_size_ppl'][key]:.4f}), skipping.")
            continue
        print(f"\nblock_size={bs} ...")
        ppl = prune_and_eval(model, dense_state, stats, args.sparsity, bs,
                             device, tokenizer, args.seq_len, args.damp)
        results["block_size_ppl"][key] = ppl
        print(f"  PPL: {ppl:.4f}")
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'Block size':>12}  {'PPL':>10}")
    print(f"{'dense':>12}  {results['dense_ppl']:>10.2f}")
    for bs in BLOCK_SIZES:
        key = str(bs)
        ppl = results["block_size_ppl"].get(key, "-")
        ppl_s = f"{ppl:.2f}" if isinstance(ppl, float) else ppl
        print(f"{bs:>12}  {ppl_s:>10}")
    print(f"\nFull results: {args.output}")


if __name__ == "__main__":
    main()
