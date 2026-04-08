#!/usr/bin/bash

params=""
if [ $# -ne 0 ]; then
    params="$*"
fi

# use envs as local params for convenience
# e.g.
# NNODE=1 NGPU=8 LOG_RANK=0 ./train.sh
NNODE=${NNODE:-"1"}
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}

if [[ -z "${MASTER_ADDR}" ]]; then
  export MASTER_ADDR="localhost"
fi
if [[ -z "${MASTER_PORT}" ]]; then
  export MASTER_PORT="0"
fi

: '
Usage:

bash train.sh -h

Training a 340M model:

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODE=1 NGPU=4 LOG_RANK=0 \
WANDB_PROJECT="sparsity_baseline" \
bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/transformer-1B-dense-baseline \
  --model.config configs/transformer_1B.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 1000 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 8 \
  --training.seq_len 65536 \
  --training.context_len 4096 \
  --training.gradient_accumulation_steps 4 \
  --training.steps 20480 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 8 \
  --training.prefetch_factor 4 \
  --training.seed 42 \
  --training.compile \
  --training.tensor_parallel_degree 1 \
  --training.disable_loss_parallel \
  --checkpoint.interval 500 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 10 \
  --metrics.enable_wandb
'

echo "Launching training..."

set -x
path=$(grep -oP '(?<=--job.dump_folder )[^ ]+' <<< "$params")
steps=$(grep -oP '(?<=--training.steps )[^ ]+' <<< "$params")
config=$(grep -oP '(?<=--model.config )[^ ]+' <<< "$params")
tokenizer=$(grep -oP '(?<=--model.tokenizer_path )[^ ]+' <<< "$params")
model=$(
  python -c "import fla, sys; from transformers import AutoConfig; print(AutoConfig.from_pretrained(sys.argv[1]).to_json_string())" "$config" | jq -r '.model_type'
)

mkdir -p $path
cp * $path
cp -r configs $path
cp -r flame   $path
cp -r 3rdparty/flash-linear-attention/fla $path
cp -r 3rdparty/torchtitan/torchtitan $path

# for offline systems
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
if [ "$date" == "" ]; then
  date=$(date +%Y%m%d%H%M)
fi
RUN_NAME="$model-$(basename $path)"
RUN_ID="$RUN_NAME-$date"

export WANDB_RESUME=allow
if [[ -z "${WANDB_PROJECT}" ]]; then
  export WANDB_PROJECT="sparsity_baseline"
fi
if [[ -z "${WANDB_NAME}" ]]; then
  export WANDB_NAME="$RUN_NAME"
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  export WANDB_RUN_ID="$RUN_ID"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nnodes=${NNODE} \
  --nproc_per_node=${NGPU} \
  --rdzv_backend c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  --local-ranks-filter ${LOG_RANK} \
  --role rank \
  --tee 3 \
  --log-dir $path/logs \
  -m flame.train \
  $params

status=$?
if [[ $status -ne 0 ]]; then
  echo "TRAINING FAILED (exit code $status). Skipping conversion."
  exit $status
fi

echo "TRAINING DONE!"
echo "Converting the DCP checkpoints to HF format..."

python -m flame.utils.convert_dcp_to_hf \
  --path $path \
  --step $steps \
  --config $config \
  --tokenizer $tokenizer

echo "RUNNING DONE!"
