#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${KP_CONVX_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
DATASET_PATH="${S3DIS_DATASET_PATH:-/root/autodl-tmp/data/s3dis}"
SOURCE_RUN="${GLSKF_SOURCE_RUN:?GLSKF_SOURCE_RUN must name the GLSKF run whose architecture/configuration is reused}"
CHECKPOINT="${GLSKF_L0_CHECKPOINT:-${SOURCE_RUN}/checkpoints/chkp_0210.tar}"
OUTPUT_ROOT="${GLSKF_ABLATION_ROOT:-${SCRIPT_DIR}/results/s3dis_glskf_same_checkpoint_ablations_20260812}"
OUTPUT_DIR="${OUTPUT_ROOT}/baseline"

"${PYTHON_BIN}" "${SCRIPT_DIR}/tools/evaluate_s3dis_glskf_ablation.py" \
    --source-log "${SOURCE_RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-path "${DATASET_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --mode baseline \
    --seed 57106803 \
    --gpu 0
