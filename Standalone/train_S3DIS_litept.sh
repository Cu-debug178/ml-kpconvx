#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/KPConvX"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/data/s3dis}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-57106803}"
PATCH_SIZE="${PATCH_SIZE:-128}"
HANDOVER_STAGE="${HANDOVER_STAGE:-0}"
if [[ -n "${CONV_STAGES:-}" ]]; then
  CONV_STAGES="$CONV_STAGES"
elif (( HANDOVER_STAGE > 0 )); then
  CONV_STAGES="$((HANDOVER_STAGE - 1))"
else
  CONV_STAGES=3
fi
FA_ENABLED="${FA_ENABLED:-0}"
RESUME_PATH="${RESUME_PATH:-}"
LOG_ARGS=()
if [[ -n "${LOG_PATH:-}" ]]; then
  LOG_ARGS=(--log_path "$LOG_PATH")
fi

# A resumed run must use the configuration stored beside its checkpoint.  The
# training entry rejects architecture overrides on resume, so do not repeat the
# LitePT defaults in this branch.
if [[ -n "$RESUME_PATH" ]]; then
  TRAIN_COMMAND=(
    "$PYTHON_BIN" experiments/S3DIS/train_S3DIS.py
    --dataset_path "$DATASET_PATH"
    "${LOG_ARGS[@]}"
    --resume_path "$RESUME_PATH"
    "$@"
  )
else
  # The default (2,2,2,6,2) follows LitePT-S depth allocation while preserving
  # KPConvX's stem, point pyramid and channel schedule. Pass --layer_blocks after
  # this script to override it, for example with 3 3 9 12 3 for a depth-matched run.
  TRAIN_COMMAND=(
    "$PYTHON_BIN" experiments/S3DIS/train_S3DIS.py
    --dataset_path "$DATASET_PATH"
    "${LOG_ARGS[@]}"
    --seed "$SEED"
    --kp_mode kpconvx
    --layer_blocks 2 2 2 6 2
    --litept_enabled 1
    --litept_conv_stages "$CONV_STAGES"
    --litept_handover_stage "$HANDOVER_STAGE"
    --litept_patch_size "$PATCH_SIZE"
    --litept_num_heads 8
    --litept_attention_ratio 1.0
    --litept_mlp_ratio 4.0
    --litept_rope_base 100.0
    --litept_rope_enabled 1
    --litept_orders z,z-trans
    --litept_light_decoder 1
    --decoder_layer 0
    --fa_enabled "$FA_ENABLED"
    --fa_train_mode joint
    --fa_num_anchors 100
    --fa_anchor_mode fps
    --fa_geometry_dim 16
    --fa_attention_dim 64
    --fa_attention_heads 4
    --fa_chunk_size 16384
    --fa_cross_layer 1
    --fa_spatial 1
    "$@"
  )
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'SEED=%q FA_ENABLED=%q OMP_NUM_THREADS=%q CUDA_VISIBLE_DEVICES=%q PYTORCH_CUDA_ALLOC_CONF=%q LITEPT_PROFILE_SERIALIZATION=%q LITEPT_SMOKE_METRICS=%q\n' \
    "$SEED" "$FA_ENABLED" "${OMP_NUM_THREADS:-}" "${CUDA_VISIBLE_DEVICES:-}" \
    "${PYTORCH_CUDA_ALLOC_CONF:-}" "${LITEPT_PROFILE_SERIALIZATION:-0}" \
    "${LITEPT_SMOKE_METRICS:-0}"
  printf 'Command:'
  printf ' %q' "${TRAIN_COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${TRAIN_COMMAND[@]}"
