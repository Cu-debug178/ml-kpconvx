#!/usr/bin/env bash
set -u

# One-shot L0D protocol: train with the canonical runner, then evaluate the
# six requested late checkpoints before the finalizer is allowed to shut down.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/KPConvX"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the S3DIS dataset root}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/s3dis_litept_l0d_250_seed57106803}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONSOLE_LOG="${CONSOLE_LOG:-${LOG_PATH}.console.log}"
EVAL_DIR="${LOG_PATH}/eval_200_250"
MANIFEST="${EVAL_DIR}/manifest.csv"

set +e
DATASET_PATH="${DATASET_PATH}" \
RESULT_ROOT="${RESULT_ROOT}" \
LOG_PATH="${LOG_PATH}" \
PYTHON_BIN="${PYTHON_BIN}" \
"${SCRIPT_DIR}/run_S3DIS_litept_l0d_250_seed57106803.sh" 2>&1 | tee "${CONSOLE_LOG}"
training_rc="${PIPESTATUS[0]}"
set -e
if [[ "${training_rc}" -ne 0 ]]; then
    printf 'training_exit_code=%s\n' "${training_rc}"
    exit "${training_rc}"
fi

mkdir -p "${EVAL_DIR}/logs"
printf 'checkpoint,internal_epoch,return_code,test_dir,report_path\n' > "${MANIFEST}"
evaluation_rc=0

for epoch in 0200 0210 0220 0230 0240 0250; do
    weight="${LOG_PATH}/checkpoints/chkp_${epoch}.tar"
    log_file="${EVAL_DIR}/logs/chkp_${epoch}.log"
    if [[ ! -f "${weight}" ]]; then
        printf 'chkp_%s,missing,missing,missing,missing\n' "${epoch}" >> "${MANIFEST}"
        continue
    fi

    internal_epoch="$("${PYTHON_BIN}" -c \
        'import sys, torch; c=torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(c.get("epoch", "unknown"))' \
        "${weight}")"
    before="$(find "${LOG_PATH}/test" -maxdepth 1 -type d -name 'test_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
    set +e
    (
        cd "${PROJECT_DIR}" || exit 1
        export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
        "${PYTHON_BIN}" experiments/S3DIS/test_S3DIS.py \
            --dataset_path "${DATASET_PATH}" \
            --log_path "${LOG_PATH}" \
            --weight_path "${weight}"
    ) > "${log_file}" 2>&1
    rc=$?
    set -e
    [[ "${rc}" -eq 0 ]] || evaluation_rc=1
    after="$(find "${LOG_PATH}/test" -maxdepth 1 -type d -name 'test_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
    [[ "${after}" == "${before}" ]] && after=""
    report="${LOG_PATH}/test/${after}/report.txt"
    [[ -f "${report}" ]] || report=""
    printf 'chkp_%s,%s,%s,%s,%s\n' "${epoch}" "${internal_epoch}" "${rc}" "${after}" "${report}" >> "${MANIFEST}"
done

touch "${EVAL_DIR}/evaluation_complete"
printf 'evaluation_complete=%s\n' "${EVAL_DIR}/evaluation_complete"
exit "${evaluation_rc}"
