#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/KPConvX"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/data/s3dis}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-57106803}"
AMP_ENABLED="${AMP_ENABLED:-}"
AMP_DTYPE="${AMP_DTYPE:-}"
AMP_ARGS=()
if [[ -n "$AMP_ENABLED" ]]; then
  AMP_ARGS+=(--amp_enabled "$AMP_ENABLED")
fi
if [[ -n "$AMP_DTYPE" ]]; then
  AMP_ARGS+=(--amp_dtype "$AMP_DTYPE")
fi
LOG_ARGS=()
if [[ -n "${LOG_PATH:-}" ]]; then
  LOG_ARGS=(--log_path "$LOG_PATH")
fi

TRAIN_COMMAND=(
  "$PYTHON_BIN" experiments/S3DIS/train_S3DIS.py
  --dataset_path "$DATASET_PATH"
  "${LOG_ARGS[@]}"
  --seed "$SEED"
  --kp_mode kpconvx
  --fa_enabled 1
  --fa_train_mode joint
  --fa_num_anchors 100
  --fa_anchor_mode fps
  --fa_geometry_dim 16
  --fa_attention_dim 64
  --fa_attention_heads 4
  --fa_chunk_size 16384
  --fa_cross_layer 1
  --fa_spatial 1
  "${AMP_ARGS[@]}"
  "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Command:'
  printf ' %q' "${TRAIN_COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${TRAIN_COMMAND[@]}"
