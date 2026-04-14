"""
lm-evaluation-harness wrapper for LLaMA-family models.

Evaluates a pruned (or dense) LLaMA model on standard zero-shot tasks using
device_map="auto" for multi-GPU inference.

Usage
-----
python scripts/benchmark_llama.py \\
    --model_path /path/to/pruned_llama \\
    --output results/benchmark_llama_sparsegpt_50pct.json

Tasks evaluated: arc_easy, arc_challenge, hellaswag, piqa, winogrande
"""

import argparse
import json
import os

import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--tasks",      default=",".join(TASKS))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_fewshot", type=int, default=0)
    return p.parse_args()


def main():
    args  = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]

    print(f"Model : {args.model_path}")
    print(f"Tasks : {tasks}")

    lm = HFLM(
        pretrained=args.model_path,
        dtype="bfloat16",
        batch_size=args.batch_size,
        # device_map="auto" is passed via parallelize for multi-GPU
        parallelize=True,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results["results"], f, indent=2)

    print(f"\n=== Results ===")
    for task, metrics in results["results"].items():
        acc = metrics.get("acc,none", metrics.get("acc_norm,none", "?"))
        print(f"  {task:<25} acc={acc:.4f}" if isinstance(acc, float) else f"  {task}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
