#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the S3DIS dataset root}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/KPConvX/results}"
LOG_PATH="${LOG_PATH:-$RESULT_ROOT/s3dis_litept_l2_no_rope_250_seed57106803}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
[[ -d "$DATASET_PATH/Area_5" ]] || { echo "S3DIS Area_5 not found: $DATASET_PATH" >&2; exit 1; }
[[ ! -e "$LOG_PATH" ]] || { echo "Refusing to overwrite: $LOG_PATH" >&2; exit 1; }
export DATASET_PATH LOG_PATH PYTHON_BIN SEED=57106803 FA_ENABLED=0
cd "$SCRIPT_DIR"
exec ./train_S3DIS_litept.sh \
  --batch_size 24 \
  --accum_batch 1 \
  --layer_blocks 2 2 2 6 2 \
  --litept_light_decoder 1 \
  --decoder_layer 0 \
  --litept_rope_enabled 0 \
  --max_epoch 250 \
  --cyc_decrease10 62 \
  --checkpoint_start 200 \
  --checkpoint_gap 10 \
  --monitor_enabled 1 \
  --monitor_interval 50
