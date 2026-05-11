"""
Block size sweep: OBS-cancel-block PPL + task accuracy vs block size on the 1B model.
Tests block sizes [64, 128, 256, 512] at 50% sparsity.

Usage
-----
python scripts/block_size_sweep_acc.py \
    --model_path exp/transformer-1B-dense-baseline \
    --output results/block_size_sweep_acc.json \
    --device cuda
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from prune_sparsegpt import build_calib_batches, collect_covariance_stats, evaluate_ppl
from prune_obs_cancel import obs_cancel_block_prune_layer

try:
    import fla  # noqa
    from fla.models.transformer import TransformerConfig, TransformerForCausalLM
    AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
    AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)
except Exception as e:
    print(f"Warning: could not register FLA transformer: {e}")

import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

BLOCK_SIZES = [64, 128, 256, 512]
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]
ACC_KEYS = {
    "arc_easy":      "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag":     "acc_norm,none",
    "piqa":          "acc_norm,none",
    "winogrande":    "acc,none",
}


def run_lm_eval(model, tokenizer, device, batch_size=16):
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        dtype="bfloat16",
        trust_remote_code=True,
        batch_size=batch_size,
        device=str(device),
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=TASKS,
        num_fewshot=0,
    )
    task_results = results["results"]
    accs = {}
    for task in TASKS:
        key = ACC_KEYS[task]
        accs[task] = task_results[task][key]
    avg = sum(accs.values()) / len(accs)
    accs["avg"] = avg
    return accs


@torch.no_grad()
def prune_and_eval(model, dense_state, stats, sparsity, block_size,
                   device, tokenizer, seq_len, damp, batch_size):
    model.load_state_dict(dense_state)
    for name, module in tqdm(model.named_modules(), desc=f"Pruning b={block_size}"):
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

    ppl = evaluate_ppl(model, tokenizer, device, seq_len)
    print(f"  PPL: {ppl:.4f}")

    print(f"  Running lm-eval ...")
    accs = run_lm_eval(model, tokenizer, device, batch_size)
    for task, val in accs.items():
        print(f"    {task}: {val:.4f}")
    return ppl, accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",      type=str,   default="exp/transformer-1B-dense-baseline")
    p.add_argument("--sparsity",        type=float, default=0.5)
    p.add_argument("--n_calib_batches", type=int,   default=64)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--eval_batch_size", type=int,   default=16)
    p.add_argument("--seq_len",         type=int,   default=512)
    p.add_argument("--damp",            type=float, default=0.01)
    p.add_argument("--output",          type=str,   default="results/block_size_sweep_acc.json")
    p.add_argument("--device",          type=str,
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
    dense_accs = run_lm_eval(model, tokenizer, device, args.eval_batch_size)
    print(f"Dense accs: {dense_accs}")

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
        results = {
            "dense_ppl": dense_ppl,
            "dense_accs": dense_accs,
            "sparsity": args.sparsity,
            "block_sizes": {},
        }

    for bs in BLOCK_SIZES:
        key = str(bs)
        if key in results["block_sizes"]:
            print(f"block_size={bs} already done, skipping.")
            continue
        print(f"\n--- block_size={bs} ---")
        ppl, accs = prune_and_eval(
            model, dense_state, stats, args.sparsity, bs,
            device, tokenizer, args.seq_len, args.damp, args.eval_batch_size,
        )
        results["block_sizes"][key] = {"ppl": ppl, "accs": accs}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {args.output}")

    print(f"\n{'Block':>8}  {'PPL':>8}  {'Avg acc':>8}")
    print(f"{'dense':>8}  {results['dense_ppl']:>8.2f}  {results['dense_accs']['avg']:>8.4f}")
    for bs in BLOCK_SIZES:
        key = str(bs)
        if key in results["block_sizes"]:
            r = results["block_sizes"][key]
            print(f"{bs:>8}  {r['ppl']:>8.2f}  {r['accs']['avg']:>8.4f}")


if __name__ == "__main__":
    main()
