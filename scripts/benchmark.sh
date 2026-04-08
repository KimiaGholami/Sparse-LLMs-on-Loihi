#!/usr/bin/env bash
# Benchmark a model using lm-evaluation-harness.
# Usage: bash scripts/benchmark.sh <model_path> <output_json>
#
# Evaluates: hellaswag, arc_easy, arc_challenge, winogrande, piqa, lambada_openai
# Results written to <output_json>.

set -euo pipefail

MODEL_PATH="${1:-exp/transformer-1B-dense-baseline}"
OUTPUT="${2:-/home/ubuntu/benchmark_results.json}"
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,lambada_openai"
BATCH_SIZE=16
NUM_FEW_SHOT=0

echo "============================================"
echo "Model : $MODEL_PATH"
echo "Tasks : $TASKS"
echo "Output: $OUTPUT"
echo "============================================"

conda run -n env python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True" \
    --tasks "${TASKS}" \
    --num_fewshot "${NUM_FEW_SHOT}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT}" \
    --log_samples \
    2>&1

echo "Done. Results at ${OUTPUT}"
