#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the S3DIS dataset root}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/KPConvX/results}"
LOG_PATH="${LOG_PATH:-$RESULT_ROOT/s3dis_litept_l1_smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONSOLE_LOG="${CONSOLE_LOG:-$RESULT_ROOT/s3dis_litept_l1_smoke.console.log}"

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
if [[ -e "$LOG_PATH" || -e "$CONSOLE_LOG" ]]; then
    echo "Smoke output already exists; refusing to overwrite: $LOG_PATH" >&2
    exit 1
fi

export DATASET_PATH LOG_PATH PYTHON_BIN
export SEED=57106803
export FA_ENABLED=0
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LITEPT_PROFILE_SERIALIZATION=1
export LITEPT_SMOKE_METRICS=1

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    cd "$SCRIPT_DIR"
    ./train_S3DIS_litept.sh --max_epoch 1
    exit 0
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
    echo "Enable a GPU runtime before starting the smoke experiment." >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT"
cd "$SCRIPT_DIR"
./train_S3DIS_litept.sh --max_epoch 1 2>&1 | tee "$CONSOLE_LOG"

"$PYTHON_BIN" - "$LOG_PATH/training.txt" <<'PY'
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
rows = [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]
if not rows:
    raise SystemExit("Smoke check failed: training.txt has no metric rows")
losses = [float(row[2]) for row in rows]
if not all(math.isfinite(loss) for loss in losses):
    raise SystemExit("Smoke check failed: a non-finite loss was recorded")
print(f"Automatic smoke checks passed: {len(rows)} finite loss rows")
PY

if grep -Eiq 'CUDA error|CUDA out of memory|device-side assert|non-finite loss' "$CONSOLE_LOG"; then
    echo "Smoke check failed: CUDA/non-finite error found in console log" >&2
    exit 1
fi

touch "$LOG_PATH/automatic_checks_passed"
echo "Review the 'Smoke metrics' lines before approving the formal L0 run."
