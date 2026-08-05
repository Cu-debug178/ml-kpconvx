#!/usr/bin/env bash
set -u

# Run after training: test the monitored best, final current checkpoint, and
# every retained checkpoint in a stable order.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_DIR="${WORKTREE_DIR:-${ROOT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the S3DIS dataset root}"
LOG_DIR="${LOG_DIR:?Set LOG_DIR to the training result directory}"
TRAIN_PID="${TRAIN_PID:?Set TRAIN_PID to the training process PID}"
WAIT_INTERVAL="${WAIT_INTERVAL:-60}"
MONITOR_DRAIN_TIMEOUT="${MONITOR_DRAIN_TIMEOUT:-180}"
TEST_ROOT="${TEST_ROOT:-${LOG_DIR}/all_checkpoint_tests}"
RUN_LOG_DIR="${TEST_ROOT}/logs"
CHECKPOINT_DIR="${LOG_DIR}/checkpoints"
TEST_SCRIPT="${WORKTREE_DIR}/Standalone/KPConvX/experiments/S3DIS/test_S3DIS.py"
RESULTS_CSV="${TEST_ROOT}/results.csv"
RUNNER_LOG="${TEST_ROOT}/runner.log"

if ! PYTHON_BIN="$(command -v "$PYTHON_BIN")"; then
    printf '%s\n' "ERROR: Python executable not found" >&2
    exit 1
fi

mkdir -p "${TEST_ROOT}" "${RUN_LOG_DIR}"

log() {
    printf '%s %s\n' "$(date -Is)" "$1" | tee -a "${RUNNER_LOG}"
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    log "ERROR: Python executable not found: ${PYTHON_BIN}"
    exit 1
fi
if [[ ! -f "${TEST_SCRIPT}" || ! -d "${LOG_DIR}" ]]; then
    log "ERROR: test script or log directory is missing"
    exit 1
fi

if [[ ! -s "${RESULTS_CSV}" ]]; then
    printf 'started_utc,status,checkpoint_epoch,weight_path,test_dir,log_path,report_path,full_miou\n' > "${RESULTS_CSV}"
fi

log "Waiting for training PID ${TRAIN_PID} to finish before testing"
while [[ -d "/proc/${TRAIN_PID}" ]]; do
    sleep "${WAIT_INTERVAL}"
done

# Let the checkpoint monitor finish its final copy after the trainer exits.
drain_deadline=$((SECONDS + MONITOR_DRAIN_TIMEOUT))
while (( SECONDS < drain_deadline )); do
    if ! screen -ls 2>/dev/null | grep -q 's3dis_checkpoint_monitor'; then
        break
    fi
    sleep "${WAIT_INTERVAL}"
done

if [[ ! -f "${CHECKPOINT_DIR}/current_chkp.tar" ]]; then
    log "ERROR: final current checkpoint is missing"
    exit 1
fi

shopt -s nullglob
weights=()
add_weight() {
    local candidate="$1"
    [[ -f "${candidate}" ]] || return 0
    local existing
    for existing in "${weights[@]}"; do
        [[ "${existing}" == "${candidate}" ]] && return 0
    done
    weights+=("${candidate}")
}

# Priority requested by the user.
add_weight "${CHECKPOINT_DIR}/best_mIoU_chkp.tar"
add_weight "${CHECKPOINT_DIR}/current_chkp.tar"

for weight in "${CHECKPOINT_DIR}"/snapshots/chkp_*.tar; do
    add_weight "${weight}"
done
for weight in "${CHECKPOINT_DIR}"/chkp_*.tar; do
    add_weight "${weight}"
done

if (( ${#weights[@]} == 0 )); then
    log "ERROR: no checkpoints found"
    exit 1
fi

log "Testing ${#weights[@]} retained checkpoint files"

for weight in "${weights[@]}"; do
    if awk -F',' -v target="${weight}" '$2 == "ok" && $4 == target { found = 1 } END { exit !found }' "${RESULTS_CSV}"; then
        log "Skipping already successful checkpoint: ${weight}"
        continue
    fi

    base_name="$(basename "${weight}" .tar)"
    parent_name="$(basename "$(dirname "${weight}")")"
    label="${parent_name}_${base_name}"
    output_log="${RUN_LOG_DIR}/${label}.log"

    # test_model creates the next test_XXX folder under LOG_DIR/test.
    test_index=1
    while [[ -d "${LOG_DIR}/test/test_$(printf '%03d' "${test_index}")" ]]; do
        test_index=$((test_index + 1))
    done
    test_dir="${LOG_DIR}/test/test_$(printf '%03d' "${test_index}")"

    checkpoint_epoch="$(${PYTHON_BIN} -c "import torch; print(torch.load('${weight}', map_location='cpu')['epoch'])" 2>/dev/null || printf 'unknown')"
    log "Starting ${label} (checkpoint epoch ${checkpoint_epoch})"
    started="$(date -Is)"

    env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
        PYTHONPATH="${WORKTREE_DIR}/Standalone/KPConvX${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" "${TEST_SCRIPT}" \
        --dataset_path "${DATA_DIR}" \
        --log_path "${LOG_DIR}" \
        --weight_path "${weight}" > "${output_log}" 2>&1
    status=$?

    report_path="${test_dir}/report.txt"
    full_miou="NA"
    if [[ -f "${report_path}" ]]; then
        full_miou="$(awk -F'|' '{v=$2; gsub(/[[:space:]]/, "", v); if (v ~ /^[0-9]+([.][0-9]+)?$/) last=v} END{print (last == "" ? "NA" : last)}' "${report_path}")"
    fi

    if (( status == 0 )); then
        result_status="ok"
        log "Finished ${label}: full_mIoU=${full_miou}"
    else
        result_status="failed_${status}"
        log "FAILED ${label}: exit=${status}"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${started}" "${result_status}" "${checkpoint_epoch}" "${weight}" \
        "${test_dir}" "${output_log}" "${report_path}" "${full_miou}" >> "${RESULTS_CSV}"
done

log "All checkpoint tests completed"
