#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/KPConvX"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/data/ScanObjectNN/main_split}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-57106803}"
RESUME_PATH="${RESUME_PATH:-}"
LOG_ARGS=()
if [[ -n "${LOG_PATH:-}" ]]; then
  LOG_ARGS=(--log_path "$LOG_PATH")
fi

if [[ -n "$RESUME_PATH" ]]; then
  exec "$PYTHON_BIN" experiments/ScanObjectNN/train_ScanObj.py \
    --dataset_path "$DATASET_PATH" \
    "${LOG_ARGS[@]}" \
    --resume_path "$RESUME_PATH" \
    "$@"
fi

exec "$PYTHON_BIN" experiments/ScanObjectNN/train_ScanObj.py \
  --dataset_path "$DATASET_PATH" \
  "${LOG_ARGS[@]}" \
  --seed "$SEED" \
  --kp_mode kpconvx \
  --fa_enabled 1 \
  --fa_train_mode joint \
  --fa_num_anchors 64 \
  --fa_anchor_mode fps \
  --fa_geometry_dim 16 \
  --fa_attention_dim 64 \
  --fa_attention_heads 4 \
  --fa_chunk_size 4096 \
  --fa_cross_layer 1 \
  --fa_spatial 1 \
  "$@"
