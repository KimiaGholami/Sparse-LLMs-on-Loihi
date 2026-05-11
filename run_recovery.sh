#!/bin/bash
cd /mnt/cephfs/share/kimia/flame
PYTHON=/home/ubuntu/miniconda3/envs/fla-py311/bin/python
TORCHRUN=/home/ubuntu/miniconda3/envs/fla-py311/bin/torchrun

for method in wanda ria sparsegpt awp; do
  case $method in
    wanda)     ft_gpu=0; dist_gpu=4; port=29500 ;;
    ria)       ft_gpu=1; dist_gpu=5; port=29501 ;;
    sparsegpt) ft_gpu=2; dist_gpu=6; port=29502 ;;
    awp)       ft_gpu=3; dist_gpu=7; port=29503 ;;
  esac

  CUDA_VISIBLE_DEVICES=$ft_gpu $PYTHON scripts/finetune_sparse.py \
    --model_path exp/hgrn-1.3B-${method}-80pct \
    --model_type hgrn \
    --output_path exp/hgrn-1.3B-${method}-80pct-ft \
    --total_steps 20000 --lr 2e-5 --calib_dataset c4 \
    > logs/ft_${method}_80pct.log 2>&1 &
  echo "ft  $method -> GPU $ft_gpu (PID $!)"

  CUDA_VISIBLE_DEVICES=$dist_gpu $TORCHRUN \
    --nproc_per_node=1 --master_port=$port scripts/distill_ddp.py \
    --teacher_path fla-hub/hgrn-1.3B-100B \
    --student_path exp/hgrn-1.3B-${method}-80pct \
    --model_type hgrn \
    --output_path exp/hgrn-1.3B-${method}-80pct-distill \
    --total_steps 20000 --lr 2e-5 \
    > logs/distill_${method}_80pct.log 2>&1 &
  echo "distill $method -> GPU $dist_gpu (PID $!)"
done

echo "All 8 jobs launched. PIDs above."
