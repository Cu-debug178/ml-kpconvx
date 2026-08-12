#!/usr/bin/env bash

set -Eeuo pipefail

if (($# != 1)); then
    printf 'usage: %s MODE\n' "$0" >&2
    exit 2
fi

MODE="$1"
case "${MODE}" in
    true|shuffled|zero|room_mean|branch_off) ;;
    *) printf 'unsupported mode: %s\n' "${MODE}" >&2; exit 2 ;;
esac

: "${S3DIS_DATASET_PATH:?S3DIS_DATASET_PATH must name the S3DIS dataset root}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN="${SCRIPT_DIR}/results/s3dis_ktha_m1_concat_best6_joint20_bf16_b12a2_lr1e4_wd001_identity_seed57106803"
CHECKPOINT="${SOURCE_RUN}/checkpoints/best_val_chkp.tar"
OUTPUT_ROOT="${SCRIPT_DIR}/results/s3dis_ktha_m1_jointbest_inference_ablations_20260812"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODE}"
PYTHON_BIN="${KP_CONVX_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/tools/evaluate_s3dis_ktha_ablation.py" \
    --source-log "${SOURCE_RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-path "${S3DIS_DATASET_PATH}" \
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
