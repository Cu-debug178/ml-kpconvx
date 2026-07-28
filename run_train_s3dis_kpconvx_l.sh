#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/pointcept/bin/python}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data/s3dis}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/Standalone/KPConvX/results/S3DIS_KPConvX-L-4090D-24G}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="${ROOT_DIR}/Standalone/KPConvX${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${DATA_DIR}/Area_5" ]]; then
    echo "S3DIS Area_5 not found under: ${DATA_DIR}" >&2
    exit 1
fi
if [[ -e "${LOG_DIR}" ]]; then
    echo "Log directory already exists, refusing to overwrite: ${LOG_DIR}" >&2
    exit 1
fi

cd "${ROOT_DIR}/Standalone/KPConvX"
exec "${PYTHON_BIN}" experiments/S3DIS/train_S3DIS.py \
    --dataset_path "${DATA_DIR}" \
    --log_path "${LOG_DIR}"
