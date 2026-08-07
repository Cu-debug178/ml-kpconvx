#!/usr/bin/env bash
set -u

# L1 training wrapper. The caller arms the existing finalizer against this
# process so console logging and shutdown monitoring are established before
# training starts; this file is not edited during a run.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/KPConvX/results}"
LOG_PATH="${LOG_PATH:-${RESULT_ROOT}/s3dis_litept_l1_250_seed57106803}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the S3DIS dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONSOLE_LOG="${CONSOLE_LOG:-${LOG_PATH}.console.log}"

set +e
DATASET_PATH="${DATASET_PATH}" \
RESULT_ROOT="${RESULT_ROOT}" \
LOG_PATH="${LOG_PATH}" \
PYTHON_BIN="${PYTHON_BIN}" \
"${SCRIPT_DIR}/run_S3DIS_litept_l1_250_seed57106803.sh" 2>&1 | tee "${CONSOLE_LOG}"
training_rc="${PIPESTATUS[0]}"
set -e
printf 'training_exit_code=%s\n' "${training_rc}"
exit "${training_rc}"
