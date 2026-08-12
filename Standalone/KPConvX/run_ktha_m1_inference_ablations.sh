#!/usr/bin/env bash

set -Eeuo pipefail

if (($# != 1)); then
    printf 'usage: %s DATASET_PATH\n' "$0" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export KP_CONVX_PYTHON="${KP_CONVX_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
export S3DIS_DATASET_PATH="$1"
[[ -d "${S3DIS_DATASET_PATH}" ]] || {
    printf 'dataset directory not found: %s\n' "${S3DIS_DATASET_PATH}" >&2
    exit 66
}
[[ -f "${SCRIPT_DIR}/results/s3dis_ktha_m1_concat_best6_joint20_bf16_b12a2_lr1e4_wd001_identity_seed57106803/checkpoints/best_val_chkp.tar" ]] || {
    printf 'source checkpoint is missing\n' >&2
    exit 66
}
SOURCE_RUN="${SCRIPT_DIR}/results/s3dis_ktha_m1_concat_best6_joint20_bf16_b12a2_lr1e4_wd001_identity_seed57106803"
"${KP_CONVX_PYTHON}" "${SCRIPT_DIR}/tools/preflight_s3dis_ktha_ablations.py" \
    --source-log "${SOURCE_RUN}" \
    --checkpoint "${SOURCE_RUN}/checkpoints/best_val_chkp.tar" \
    --dataset-path "${S3DIS_DATASET_PATH}"
"${KP_CONVX_PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; switch the instance to GPU mode before launching")
PY

exec "${SCRIPT_DIR}/tools/run_resilient_queue.sh" \
    --jobs-dir "${SCRIPT_DIR}/queue_jobs/ktha_m1_inference_ablations" \
    --state-dir "${SCRIPT_DIR}/results/ktha_m1_inference_ablations_queue_20260812" \
    --max-consecutive-failures 2 \
    --retry 0
