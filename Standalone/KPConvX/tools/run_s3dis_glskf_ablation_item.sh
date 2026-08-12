#!/usr/bin/env bash

set -Eeuo pipefail

if (($# != 1)); then
    printf 'usage: %s MODE\n' "$0" >&2
    exit 2
fi
MODE="$1"
case "${MODE}" in
    true|shuffled|room_mean|zero_context|neutral_gate|branch_off) ;;
    *) printf 'unsupported mode: %s\n' "${MODE}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${KP_CONVX_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
DATASET_PATH="${S3DIS_DATASET_PATH:-/root/autodl-tmp/data/s3dis}"
SOURCE_RUN="${GLSKF_SOURCE_RUN:?GLSKF_SOURCE_RUN must name a completed GLSKF training directory}"
CHECKPOINT="${GLSKF_CHECKPOINT:-${SOURCE_RUN}/checkpoints/best_val_chkp.tar}"
OUTPUT_ROOT="${GLSKF_ABLATION_ROOT:-${SCRIPT_DIR}/results/s3dis_glskf_same_checkpoint_ablations_20260812}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/tools/evaluate_s3dis_glskf_ablation.py" \
    --source-log "${SOURCE_RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-path "${DATASET_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --mode "${MODE}" \
    --seed 57106803 \
    --gpu 0

"${PYTHON_BIN}" - "${OUTPUT_DIR}/result.json" "${MODE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2]
with path.open(encoding="utf-8") as stream:
    result = json.load(stream)
if result.get("status") != "completed" or result.get("mode") != mode:
    raise SystemExit("result artifact is incomplete or belongs to another mode")
PY
