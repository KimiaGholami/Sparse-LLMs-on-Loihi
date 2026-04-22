"""
AWP prune-and-benchmark: prunes a model with AWP at a given sparsity,
then immediately runs lm-evaluation-harness on the pruned model in memory.
No intermediate model checkpoint is required.

Usage
-----
# 1B transformer at 50%
CUDA_VISIBLE_DEVICES=0 conda run -n env python scripts/awp_prune_and_bench.py \
    --model_path exp/transformer-1B-dense-baseline \
    --sparsity 0.5 \
    --model_type transformer \
    --bench_output results/benchmark_awp_1b_50pct.json

# LLaMA-7B at 80%
CUDA_VISIBLE_DEVICES=4 conda run -n env python scripts/awp_prune_and_bench.py \
    --model_path /path/to/open_llama_7b \
    --sparsity 0.8 \
    --model_type llama \
    --bench_output results/benchmark_awp_llama_80pct.json

# HGRN-1.3B at 50%
CUDA_VISIBLE_DEVICES=2 conda run -n env python scripts/awp_prune_and_bench.py \
    --model_path exp/hgrn-1.3B-dense-baseline \
    --sparsity 0.5 \
    --model_type hgrn \
    --bench_output results/benchmark_awp_hgrn_50pct.json
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
from prune_awp import build_calib_batches, collect_stats, prune_awp

import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

TASKS_TRANSFORMER = [
    "hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa", "lambada_openai"
]
TASKS_HGRN = [
    "hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa", "lambada_openai"
]
TASKS_LLAMA = [
    "arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"
]

LLAMA_PATH = (
    "/mnt/cephfs/share/kimia/hf_cache/hub/"
    "models--openlm-research--open_llama_7b/snapshots/"
    "6fb184ff23774c25bf84b3628e49c8b78372c7be"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=None,
                   help="Path to dense model. Uses known default if omitted.")
    p.add_argument("--sparsity", type=float, required=True)
    p.add_argument("--model_type", choices=["transformer", "llama", "hgrn"], required=True)
    p.add_argument("--bench_output", required=True,
                   help="Path to write benchmark JSON results.")
    p.add_argument("--n_calib_batches", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--bench_batch_size", type=int, default=8)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve model path
    if args.model_path is None:
        defaults = {
            "transformer": "exp/transformer-1B-dense-baseline",
            "llama": LLAMA_PATH,
            "hgrn": "exp/hgrn-1.3B-dense-baseline",
        }
        args.model_path = defaults[args.model_type]

    # Skip if output already exists
    if os.path.exists(args.bench_output):
        print(f"Output already exists: {args.bench_output}  Skipping.")
        return

    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"Model type : {args.model_type}")
    print(f"Model path : {args.model_path}")
    print(f"Sparsity   : {args.sparsity*100:.0f}%")
    print(f"Output     : {args.bench_output}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.model_type == "llama":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        calib_device = torch.device("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        calib_device = device

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f}B parameters")

    # Collect calibration stats
    print("\nCollecting calibration activations ...")
    batches = build_calib_batches(
        tokenizer, args.n_calib_batches, args.batch_size, args.seq_len,
        calib_dataset="wikitext2",
    )
    stats = collect_stats(model, batches, calib_device)

    # Prune with AWP
    print(f"\nPruning at {args.sparsity*100:.0f}% sparsity (AWP) ...")
    actual_sparsity, layer_info = prune_awp(
        model, stats, args.sparsity, calib_device,
        max_iter=200, tol=1e-4,
    )
    print(f"Actual sparsity: {actual_sparsity*100:.2f}%")

    # Free calibration stats from memory
    del stats, batches
    torch.cuda.empty_cache()

    # Run lm-eval directly on the in-memory pruned model
    tasks = {
        "transformer": TASKS_TRANSFORMER,
        "hgrn": TASKS_HGRN,
        "llama": TASKS_LLAMA,
    }[args.model_type]

    print(f"\nRunning lm-eval on: {tasks}")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        dtype="bfloat16",
        trust_remote_code=True,
        batch_size=args.bench_batch_size,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=args.bench_batch_size,
    )

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.bench_output)), exist_ok=True)
    with open(args.bench_output, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"\nBenchmark saved to {args.bench_output}")

    # Print summary
    print("\nTask accuracies:")
    for task, metrics in results["results"].items():
        acc = metrics.get("acc,none") or metrics.get("acc_norm,none")
        if acc is not None:
            print(f"  {task:<30s}: {acc:.4f}")


if __name__ == "__main__":
    main()
