#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/KPConvX"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/data/ScanObjectNN/main_split}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-57106803}"
PATCH_SIZE="${PATCH_SIZE:-64}"
HANDOVER_STAGE="${HANDOVER_STAGE:-0}"
if [[ -n "${CONV_STAGES:-}" ]]; then
  CONV_STAGES="$CONV_STAGES"
elif (( HANDOVER_STAGE > 0 )); then
  CONV_STAGES="$((HANDOVER_STAGE - 1))"
else
  CONV_STAGES=3
fi
FA_ENABLED="${FA_ENABLED:-0}"
AMP_ENABLED="${AMP_ENABLED:-}"
AMP_DTYPE="${AMP_DTYPE:-}"
AMP_ARGS=()
if [[ -n "$AMP_ENABLED" ]]; then
  AMP_ARGS+=(--amp_enabled "$AMP_ENABLED")
fi
if [[ -n "$AMP_DTYPE" ]]; then
  AMP_ARGS+=(--amp_dtype "$AMP_DTYPE")
fi
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
  --layer_blocks 2 2 2 6 2 \
  --litept_enabled 1 \
  --litept_conv_stages "$CONV_STAGES" \
  --litept_handover_stage "$HANDOVER_STAGE" \
  --litept_patch_size "$PATCH_SIZE" \
  --litept_num_heads 8 \
  --litept_attention_ratio 1.0 \
  --litept_mlp_ratio 4.0 \
  --litept_rope_base 100.0 \
  --litept_rope_enabled 1 \
  --litept_orders z,z-trans \
  --litept_light_decoder 0 \
  --fa_enabled "$FA_ENABLED" \
  --fa_train_mode joint \
  --fa_num_anchors 64 \
  --fa_anchor_mode fps \
  --fa_geometry_dim 16 \
  --fa_attention_dim 64 \
  --fa_attention_heads 4 \
  --fa_chunk_size 4096 \
  --fa_cross_layer 1 \
  --fa_spatial 1 \
  "${AMP_ARGS[@]}" \
  "$@"
