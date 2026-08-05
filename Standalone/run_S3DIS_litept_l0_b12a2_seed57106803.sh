#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the S3DIS dataset root}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/KPConvX/results}"
LOG_PATH="${LOG_PATH:-$RESULT_ROOT/s3dis_litept_l0_b12a2_seed57106803}"
SMOKE_LOG_PATH="${SMOKE_LOG_PATH:-$RESULT_ROOT/s3dis_litept_l1_smoke}"
STABILITY_LOG_PATH="${STABILITY_LOG_PATH:-$RESULT_ROOT/stability_batch12_accum2_300step_seed57106803}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! PYTHON_BIN="$(command -v "$PYTHON_BIN")"; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$DATASET_PATH/Area_5" ]]; then
    echo "S3DIS Area_5 not found under: $DATASET_PATH" >&2
    exit 1
fi
if [[ -e "$LOG_PATH" ]]; then
    echo "L0 b12a2 output already exists; refusing to overwrite: $LOG_PATH" >&2
    exit 1
fi

export DATASET_PATH LOG_PATH PYTHON_BIN
export SEED=57106803
export FA_ENABLED=0
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset LITEPT_PROFILE_SERIALIZATION LITEPT_SMOKE_METRICS

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    cd "$SCRIPT_DIR"
    ./train_S3DIS_litept.sh \
        --batch_size 12 \
        --accum_batch 2 \
        --layer_blocks 3 3 9 12 3 \
        --litept_light_decoder 0 \
        --decoder_layer 1
    exit 0
fi

if [[ ! -f "$SMOKE_LOG_PATH/automatic_checks_passed" ]]; then
    echo "The L1 smoke run has not passed its automatic checks: $SMOKE_LOG_PATH" >&2
    exit 1
fi
if [[ ! -f "$STABILITY_LOG_PATH/checkpoints/current_chkp.tar" ]] || \
   [[ "$(wc -l < "$STABILITY_LOG_PATH/training.txt")" -ne 301 ]] || \
   [[ ! -s "$STABILITY_LOG_PATH/val_IoUs.txt" ]]; then
    echo "The 300-step b12a2 stability run is incomplete: $STABILITY_LOG_PATH" >&2
    exit 1
fi
if [[ "${SMOKE_APPROVED:-0}" != "1" ]]; then
    echo "Review the L1 smoke and b12a2 stability results first." >&2
    echo "Then rerun with SMOKE_APPROVED=1." >&2
    exit 1
fi
if ! "$PYTHON_BIN" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("CUDA is not available in the selected Python environment", file=sys.stderr)
    raise SystemExit(1)
print(torch.cuda.get_device_name(0))
PY
then
    echo "Enable a GPU runtime before starting the L0 b12a2 experiment." >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT"
SOURCE_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
TRACKED_DIFF_SHA256="$(git -C "$REPO_DIR" diff --binary | sha256sum | awk '{print $1}')"
LAUNCHER_SHA256="$(sha256sum "$SCRIPT_DIR/run_S3DIS_litept_l0_b12a2_seed57106803.sh" | awk '{print $1}')"
printf 'Source state | commit=%s tracked_diff_sha256=%s launcher_sha256=%s\n' \
    "$SOURCE_COMMIT" "$TRACKED_DIFF_SHA256" "$LAUNCHER_SHA256"
printf 'Protocol variant | batch_size=12 accum_batch=2 effective_batch=24\n'
cd "$SCRIPT_DIR"
exec ./train_S3DIS_litept.sh \
    --batch_size 12 \
    --accum_batch 2 \
    --layer_blocks 3 3 9 12 3 \
    --litept_light_decoder 0 \
    --decoder_layer 1
