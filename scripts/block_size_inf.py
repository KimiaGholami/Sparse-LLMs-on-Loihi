"""
OBS-cancel-block with block_size=inf (full matrix per layer) at 50% sparsity.

Runs the entire weight matrix as a single block — the theoretical upper bound
of the block size sweep. Uses 2048-token non-overlapping PPL and lm-eval.

Usage
-----
python scripts/block_size_inf.py \
    --model_path ikimyaii/transformer-1B-dense-baseline-continued \
    --output results/benchmark_1b_obs_cancel_block_inf_50pct.json \
    --device cuda:0
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
from prune_sparsegpt import build_calib_batches, collect_covariance_stats
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
from datasets import load_dataset

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]
ACC_KEYS = {
    "arc_easy":      "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag":     "acc_norm,none",
    "piqa":          "acc_norm,none",
    "winogrande":    "acc,none",
}


@torch.no_grad()
def evaluate_ppl_2048(model, tokenizer, device, seq_len=2048):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    n = (len(enc) // seq_len) * seq_len
    enc = enc[:n].view(-1, seq_len).to(device)
    model.eval()
    total_nll = 0.0
    for chunk in tqdm(enc, desc="PPL eval"):
        out = model(chunk.unsqueeze(0), labels=chunk.unsqueeze(0))
        total_nll += out.loss.item()
    ppl = (total_nll / len(enc))
    return float(torch.exp(torch.tensor(ppl)).item())


def run_lm_eval(model, tokenizer, device, batch_size=16):
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        dtype="bfloat16",
        trust_remote_code=True,
        batch_size=batch_size,
        device=str(device),
    )
    results = evaluator.simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0)
    task_results = results["results"]
    accs = {}
    for task in TASKS:
        accs[task] = task_results[task][ACC_KEYS[task]]
    accs["avg"] = sum(accs.values()) / len(accs)
    return accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",      type=str,   default="ikimyaii/transformer-1B-dense-baseline-continued")
    p.add_argument("--sparsity",        type=float, default=0.5)
    p.add_argument("--n_calib_batches", type=int,   default=64)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--seq_len",         type=int,   default=512)
    p.add_argument("--eval_seq_len",    type=int,   default=2048)
    p.add_argument("--damp",            type=float, default=0.01)
    p.add_argument("--eval_batch_size", type=int,   default=16)
    p.add_argument("--output",          type=str,   default="results/benchmark_1b_obs_cancel_block_inf_50pct.json")
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f}B parameters")

    print("Collecting calibration stats ...")
    batches = build_calib_batches(tokenizer, args.n_calib_batches,
                                  args.batch_size, args.seq_len)
    stats = collect_covariance_stats(model, batches, device)

    print("Pruning with block_size=inf (full matrix) ...")
    for name, module in tqdm(model.named_modules(), desc="Pruning"):
        if not isinstance(module, nn.Linear) or name not in stats:
            continue
        W = module.weight.data
        in_f = W.shape[1]
        if int(in_f * args.sparsity) == 0:
            continue
        Sigma = stats[name].second_moment().to(device)
        # Use full matrix width as block_size (∞ block)
        W_corr = obs_cancel_block_prune_layer(
            W.float(), Sigma, args.sparsity, damp=args.damp, block_size=in_f,
        )
        module.weight.data.copy_(W_corr.to(W.dtype))
        del Sigma, W_corr
        torch.cuda.empty_cache()

    print("Evaluating 2048-token PPL ...")
    ppl = evaluate_ppl_2048(model, tokenizer, device, args.eval_seq_len)
    print(f"  PPL: {ppl:.4f}")

    print("Running lm-eval ...")
    accs = run_lm_eval(model, tokenizer, device, args.eval_batch_size)
    for task, val in accs.items():
        print(f"  {task}: {val:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out = {"block_size": "inf", "sparsity": args.sparsity, "ppl": ppl, "accs": accs}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
