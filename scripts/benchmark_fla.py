"""
lm-evaluation-harness wrapper that registers FLA model types before evaluation.

Usage:
    python scripts/benchmark_fla.py \
        --model_path exp/transformer-1B-dense-baseline \
        --output /home/ubuntu/results_baseline.json \
        --tasks hellaswag,arc_easy,arc_challenge,winogrande,piqa,lambada_openai
"""
import argparse
import sys

# Register FLA types before lm_eval touches anything
import fla  # noqa
from fla.models.transformer import TransformerConfig, TransformerForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("transformer", TransformerConfig, exist_ok=True)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)

import lm_eval
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import json, os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output", default="/home/ubuntu/benchmark_results.json")
    p.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge,winogrande,piqa,lambada_openai")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_fewshot", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    task_list = [t.strip() for t in args.tasks.split(",")]

    print(f"Model : {args.model_path}")
    print(f"Tasks : {task_list}")

    lm = HFLM(
        pretrained=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        batch_size=args.batch_size,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results["results"], f, indent=2)

    print(f"\n=== Results ===")
    for task, metrics in results["results"].items():
        acc = metrics.get("acc,none") or metrics.get("acc_norm,none") or metrics.get("perplexity,none")
        print(f"  {task:<25s}: {acc:.4f}" if acc is not None else f"  {task}: {metrics}")
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
