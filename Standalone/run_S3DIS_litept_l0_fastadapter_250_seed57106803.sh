#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the S3DIS dataset root}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/KPConvX/results}"
LOG_PATH="${LOG_PATH:-$RESULT_ROOT/s3dis_litept_l0_fastadapter_b12a2_250_seed57106803}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[[ -d "$DATASET_PATH/Area_5" ]] || { echo "S3DIS Area_5 not found: $DATASET_PATH" >&2; exit 1; }
[[ ! -e "$LOG_PATH" ]] || { echo "Refusing to overwrite: $LOG_PATH" >&2; exit 1; }
export DATASET_PATH LOG_PATH PYTHON_BIN OMP_NUM_THREADS CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF
export SEED=57106803 FA_ENABLED=1
cd "$SCRIPT_DIR"
exec ./train_S3DIS_litept.sh \
  --batch_size 12 \
  --accum_batch 2 \
  --layer_blocks 3 3 9 12 3 \
  --litept_light_decoder 0 \
  --decoder_layer 1 \
  --max_epoch 250 \
  --cyc_decrease10 62 \
  --checkpoint_start 100 \
  --checkpoint_gap 10 \
  --monitor_enabled 1 \
  --monitor_interval 50
