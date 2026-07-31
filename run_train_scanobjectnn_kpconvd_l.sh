#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/pointcept/bin/python}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data/ScanObjectNN/main_split}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/Standalone/KPConvX/results/ScanObjectNN_KPConvD-L-official-4090D-24G}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}/Standalone/KPConvX${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
for split_file in \
    training_objectdataset_augmentedrot_scale75.h5 \
    test_objectdataset_augmentedrot_scale75.h5; do
    if [[ ! -f "${DATA_DIR}/${split_file}" ]]; then
        echo "ScanObjectNN split not found: ${DATA_DIR}/${split_file}" >&2
        exit 1
    fi
done
if [[ -e "${LOG_DIR}" ]]; then
    echo "Log directory already exists, refusing to overwrite: ${LOG_DIR}" >&2
    exit 1
fi

cd "${ROOT_DIR}/Standalone/KPConvX"
exec "${PYTHON_BIN}" experiments/ScanObjectNN/train_ScanObj.py \
    --dataset_path "${DATA_DIR}" \
    --log_path "${LOG_DIR}"
