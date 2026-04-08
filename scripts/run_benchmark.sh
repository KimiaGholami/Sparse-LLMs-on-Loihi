#!/bin/bash
set -e

BASE=/mnt/cephfs/share/kimia
DATA=$BASE/benchmark_data
TOKENIZER=fla-hub/transformer-1.3B-100B
cd $BASE/flame

run_job() {
    local name=$1 dump=$2 config=$3 data_file=$4 steps=$5
    local seq_len=$6 accum=$7 warmup=$8 lr=$9
    local ckpt_interval=${10} log_freq=${11}
    latest=$(ls $dump/checkpoint/ 2>/dev/null | grep "step-" | sort -t- -k2 -n | tail -1)
    if [ "$latest" = "step-${steps}" ]; then
        echo ">>> SKIP (already done): $name"
        return 0
    fi
    echo ">>> STARTING: $name"
    rm -rf $dump/checkpoint/step-1 2>/dev/null || true
    NGPU=8 bash train.sh --job.config_file flame/models/fla.toml --job.dump_folder $dump --model.config $config --model.tokenizer_path $TOKENIZER --optimizer.name AdamW --optimizer.lr $lr --optimizer.eps 1e-6 --lr_scheduler.warmup_steps $warmup --lr_scheduler.decay_type cosine --lr_scheduler.lr_min 0.1 --training.batch_size 1 --training.seq_len $seq_len --training.context_len $seq_len --training.gradient_accumulation_steps $accum --training.steps $steps --training.max_norm 1.0 --training.skip_nan_inf --training.dataset json --training.dataset_split train --training.data_files $data_file --training.num_workers 4 --training.seed 42 --checkpoint.interval $ckpt_interval --checkpoint.load_step -1 --checkpoint.keep_latest_k 2 --metrics.log_freq $log_freq
    echo ">>> DONE: $name"
}

D=/mnt/cephfs/share/kimia/benchmark_data

run_job "capo/gated_deltanet/N20000"      exp/capo/gated_deltanet/N20000      configs/gated_deltanet_capo.json      $D/capo/N20000/train.jsonl      10000  512  24   500  1e-3  2000  100
run_job "capo/gated_deltanet/N50000"      exp/capo/gated_deltanet/N50000      configs/gated_deltanet_capo.json      $D/capo/N50000/train.jsonl      10000  512  24   500  1e-3  2000  100
run_job "capo/gated_deltanet/N100000"     exp/capo/gated_deltanet/N100000     configs/gated_deltanet_capo.json      $D/capo/N100000/train.jsonl     10000  512  24   500  1e-3  2000  100
run_job "capo/gated_deltanet/N200000"     exp/capo/gated_deltanet/N200000     configs/gated_deltanet_capo.json      $D/capo/N200000/train.jsonl     10000  512  24   500  1e-3  2000  100
run_job "capo/gated_deltanet/N500000"     exp/capo/gated_deltanet/N500000     configs/gated_deltanet_capo.json      $D/capo/N500000/train.jsonl     10000  512  24   500  1e-3  2000  100
run_job "capo/gated_deltaproduct/N20000"  exp/capo/gated_deltaproduct/N20000  configs/gated_deltaproduct_capo.json  $D/capo/N20000/train.jsonl      10000  512  24   500  1e-3  2000  100
run_job "capo/gated_deltaproduct/N50000"  exp/capo/gated_deltaproduct/N50000  configs/gated_deltaproduct_capo.json  $D/capo/N50000/train.jsonl      10000  512  24   500  1e-3  2000  100
run_job "capo/gated_deltaproduct/N100000" exp/capo/gated_deltaproduct/N100000 configs/gated_deltaproduct_capo.json  $D/capo/N100000/train.jsonl     10000  512  24   500  1e-3  2000  100
run_job "capo/gated_deltaproduct/N200000" exp/capo/gated_deltaproduct/N200000 configs/gated_deltaproduct_capo.json  $D/capo/N200000/train.jsonl     10000  512  24   500  1e-3  2000  100
run_job "capo/gated_deltaproduct/N500000" exp/capo/gated_deltaproduct/N500000 configs/gated_deltaproduct_capo.json  $D/capo/N500000/train.jsonl     10000  512  24   500  1e-3  2000  100
run_job "mano/gated_deltanet/L10"         exp/mano/gated_deltanet/L10         configs/gated_deltanet_mano.json      $D/mano/L10/train.jsonl          80000  1024  16  1000  2e-4  5000  200
run_job "mano/gated_deltaproduct/L10"     exp/mano/gated_deltaproduct/L10     configs/gated_deltaproduct_mano.json  $D/mano/L10/train.jsonl          80000  1024  16  1000  2e-4  5000  200
run_job "mano/gated_deltanet/L16"         exp/mano/gated_deltanet/L16         configs/gated_deltanet_mano.json      $D/mano/L16/train.jsonl         110000  1024  16  1000  2e-4  5000  200
run_job "mano/gated_deltaproduct/L16"     exp/mano/gated_deltaproduct/L16     configs/gated_deltaproduct_mano.json  $D/mano/L16/train.jsonl         110000  1024  16  1000  2e-4  5000  200
run_job "mano/gated_deltanet/L24"         exp/mano/gated_deltanet/L24         configs/gated_deltanet_mano.json      $D/mano/L24/train.jsonl         200000  1024  16  1000  2e-4  5000  200
run_job "mano/gated_deltaproduct/L24"     exp/mano/gated_deltaproduct/L24     configs/gated_deltaproduct_mano.json  $D/mano/L24/train.jsonl         200000  1024  16  1000  2e-4  5000  200
run_job "multihop/gated_deltanet/N5000"      exp/multihop/gated_deltanet/N5000      configs/gated_deltanet_multihop.json      $D/multihop/train_N5000/train_text.jsonl    20000  1024  256  1000  5e-4  2000  100
run_job "multihop/gated_deltaproduct/N5000"  exp/multihop/gated_deltaproduct/N5000  configs/gated_deltaproduct_multihop.json  $D/multihop/train_N5000/train_text.jsonl    20000  1024  256  1000  5e-4  2000  100
run_job "multihop/gated_deltanet/N10000"     exp/multihop/gated_deltanet/N10000     configs/gated_deltanet_multihop.json      $D/multihop/train_N10000/train_text.jsonl   20000  1024  256  1000  5e-4  2000  100
run_job "multihop/gated_deltaproduct/N10000" exp/multihop/gated_deltaproduct/N10000 configs/gated_deltaproduct_multihop.json  $D/multihop/train_N10000/train_text.jsonl   20000  1024  256  1000  5e-4  2000  100
run_job "multihop/gated_deltanet/N15000"     exp/multihop/gated_deltanet/N15000     configs/gated_deltanet_multihop.json      $D/multihop/train_N15000/train_text.jsonl   20000  1024  256  1000  5e-4  2000  100
run_job "multihop/gated_deltaproduct/N15000" exp/multihop/gated_deltaproduct/N15000 configs/gated_deltaproduct_multihop.json  $D/multihop/train_N15000/train_text.jsonl   20000  1024  256  1000  5e-4  2000  100
run_job "multihop/gated_deltanet/N20000"     exp/multihop/gated_deltanet/N20000     configs/gated_deltanet_multihop.json      $D/multihop/train_N20000/train_text.jsonl   20000  1024  256  1000  5e-4  2000  100
run_job "multihop/gated_deltaproduct/N20000" exp/multihop/gated_deltaproduct/N20000 configs/gated_deltaproduct_multihop.json  $D/multihop/train_N20000/train_text.jsonl   20000  1024  256  1000  5e-4  2000  100

echo "ALL BENCHMARK RUNS COMPLETE"
