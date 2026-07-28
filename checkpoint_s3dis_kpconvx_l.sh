#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/ml-kpconvx"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/Standalone/KPConvX/results/S3DIS_KPConvX-L-4090D-24G}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/pointcept/bin/python}"
INTERVAL="${INTERVAL:-15}"
SNAPSHOT_GAP="${SNAPSHOT_GAP:-10}"
VAL_LOG="${RESULT_DIR}/val_IoUs.txt"
CHECKPOINT_DIR="${RESULT_DIR}/checkpoints"
CURRENT_CHECKPOINT="${CHECKPOINT_DIR}/current_chkp.tar"
BEST_CHECKPOINT="${CHECKPOINT_DIR}/best_mIoU_chkp.tar"
SNAPSHOT_DIR="${CHECKPOINT_DIR}/snapshots"
SELECTION_LOG="${RESULT_DIR}/checkpoint_selection.csv"
MONITOR_LOG="${RESULT_DIR}/checkpoint_monitor.log"

find_train_pid() {
    ps -eo pid=,args= | awk '
        /experiments\/S3DIS\/train_S3DIS.py/ &&
        /S3DIS_KPConvX-L-4090D-24G/ &&
        ! /awk/ {
            print $1
            exit
        }
    '
}

checkpoint_epoch() {
    "${PYTHON_BIN}" -c "import torch; print(torch.load('${CURRENT_CHECKPOINT}', map_location='cpu')['epoch'])" 2>/dev/null
}

latest_miou() {
    awk 'NF { sum = 0; for (i = 1; i <= NF; i++) sum += $i; printf "%.9f\n", sum / NF }' "${VAL_LOG}" | tail -1
}

copy_checkpoint() {
    local destination="$1"
    local temp_path="${destination}.tmp.$$"

    cp "${CURRENT_CHECKPOINT}" "${temp_path}"
    mv -f "${temp_path}" "${destination}"
}

record_event() {
    local event="$1"
    local epoch="$2"
    local miou="$3"
    local checkpoint="$4"
    printf '%s,%s,%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${event}" "${epoch}" "${miou}" "${checkpoint}" >> "${SELECTION_LOG}"
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${CURRENT_CHECKPOINT}" || ! -f "${VAL_LOG}" ]]; then
    echo "Expected checkpoint or validation log not found under: ${RESULT_DIR}" >&2
    exit 1
fi
if [[ ! "${SNAPSHOT_GAP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SNAPSHOT_GAP must be a positive integer." >&2
    exit 1
fi

TRAIN_PID="${TRAIN_PID:-$(find_train_pid)}"
if [[ -z "${TRAIN_PID}" || ! -d "/proc/${TRAIN_PID}" ]]; then
    echo "No active S3DIS training process found." >&2
    exit 1
fi

mkdir -p "${SNAPSHOT_DIR}"
if [[ ! -s "${SELECTION_LOG}" ]]; then
    printf 'timestamp_utc,event,epoch,mean_iou,checkpoint\n' > "${SELECTION_LOG}"
fi

best_miou="$(awk -F, '$2 == "best" { value = $4 } END { print value }' "${SELECTION_LOG}")"
if [[ -z "${best_miou}" ]]; then
    best_miou="$(latest_miou)"
    epoch="$(checkpoint_epoch)"
    copy_checkpoint "${BEST_CHECKPOINT}"
    record_event "best" "${epoch}" "${best_miou}" "${BEST_CHECKPOINT}"
    printf '%s initialized best checkpoint: epoch=%s mean_iou=%s\n' "$(date -Is)" "${epoch}" "${best_miou}" | tee -a "${MONITOR_LOG}"
fi

last_val_lines="$(wc -l < "${VAL_LOG}")"
printf '%s monitoring train_pid=%s snapshot_gap=%s best_miou=%s\n' "$(date -Is)" "${TRAIN_PID}" "${SNAPSHOT_GAP}" "${best_miou}" | tee -a "${MONITOR_LOG}"

while [[ -d "/proc/${TRAIN_PID}" ]]; do
    current_val_lines="$(wc -l < "${VAL_LOG}")"
    if (( current_val_lines > last_val_lines )); then
        # validation_epoch writes val_IoUs after current_chkp.tar has been saved.
        sleep 2
        epoch="$(checkpoint_epoch)"
        miou="$(latest_miou)"
        snapshot_path="${SNAPSHOT_DIR}/chkp_$(printf '%04d' "${epoch}").tar"

        if (( epoch % SNAPSHOT_GAP == 0 )) && [[ ! -e "${snapshot_path}" ]]; then
            copy_checkpoint "${snapshot_path}"
            record_event "snapshot" "${epoch}" "${miou}" "${snapshot_path}"
            printf '%s saved snapshot: epoch=%s mean_iou=%s\n' "$(date -Is)" "${epoch}" "${miou}" | tee -a "${MONITOR_LOG}"
        fi

        if awk -v value="${miou}" -v best="${best_miou}" 'BEGIN { exit !(value > best) }'; then
            copy_checkpoint "${BEST_CHECKPOINT}"
            best_miou="${miou}"
            record_event "best" "${epoch}" "${miou}" "${BEST_CHECKPOINT}"
            printf '%s updated best checkpoint: epoch=%s mean_iou=%s\n' "$(date -Is)" "${epoch}" "${miou}" | tee -a "${MONITOR_LOG}"
        fi

        last_val_lines="${current_val_lines}"
    fi
    sleep "${INTERVAL}"
done

printf '%s training process %s ended; checkpoint monitor stopped\n' "$(date -Is)" "${TRAIN_PID}" | tee -a "${MONITOR_LOG}"
