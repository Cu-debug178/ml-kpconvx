#!/usr/bin/env bash
set -u

# Wait for the remaining L0D checkpoint suite, then start L1 exactly once and
# arm the existing finalizer before training can finish.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the S3DIS dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
L0D_RESULT="${ROOT_DIR}/Standalone/KPConvX/results/s3dis_litept_l0d_250_seed57106803"
SIX_EVAL="${L0D_RESULT}/eval_200_250_20260806"
L0D_EVAL="${L0D_RESULT}/eval_remaining_20260806"
L1_RESULT="${L1_RESULT:-${ROOT_DIR}/Standalone/KPConvX/results/s3dis_litept_l1_250_seed57106803}"
L1_CONSOLE="${L1_RESULT}.console.log"
POLL_SECONDS="${POLL_SECONDS:-30}"

while [[ ! -f "${L0D_EVAL}/evaluation_complete" ]]; do
    sleep "${POLL_SECONDS}"
done

manifest="${L0D_EVAL}/manifest.csv"
[[ -f "${manifest}" ]] || { echo "Missing L0D manifest: ${manifest}" >&2; exit 1; }
six_manifest="${SIX_EVAL}/manifest.csv"
[[ -f "${six_manifest}" ]] || { echo "Missing six-checkpoint manifest: ${six_manifest}" >&2; exit 1; }
six_rows="$(awk 'NR > 1 {n++} END {print n + 0}' "${six_manifest}")"
[[ "${six_rows}" -eq 6 ]] || { echo "Expected 6 initial L0D rows, got ${six_rows}" >&2; exit 1; }
if ! awk -F, 'NR > 1 && $4 != 0 {bad=1} END {exit bad ? 1 : 0}' "${six_manifest}"; then
    echo "Initial six-checkpoint evaluation contains failures; L1 was not started." >&2
    exit 1
fi
rows="$(awk 'NR > 1 {n++} END {print n + 0}' "${manifest}")"
[[ "${rows}" -eq 25 ]] || { echo "Expected 25 remaining L0D rows, got ${rows}" >&2; exit 1; }
if ! awk -F, 'NR > 1 && $4 != 0 {bad=1} END {exit bad ? 1 : 0}' "${manifest}"; then
    echo "L0D remaining evaluation contains failures; L1 was not started." >&2
    exit 1
fi
if [[ -e "${L1_RESULT}" || -e "${L1_CONSOLE}" ]]; then
    echo "L1 output already exists; refusing to overwrite: ${L1_RESULT}" >&2
    exit 1
fi

screen -dmS s3dis_l1_250 bash -lc "cd '${ROOT_DIR}' && exec env DATASET_PATH='${DATASET_PATH}' RESULT_ROOT='${ROOT_DIR}/Standalone/KPConvX/results' LOG_PATH='${L1_RESULT}' PYTHON_BIN='${PYTHON_BIN}' SEED=57106803 FA_ENABLED=0 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash '${ROOT_DIR}/Standalone/run_S3DIS_litept_l1_250_shutdown.sh'"

screen_pid=""
l1_pid=""
for _ in $(seq 1 30); do
    screen_pid="$(ps -eo pid=,args= | awk '$0 ~ /SCREEN -DmS s3dis_l1_250 bash -lc/ {print $1; exit}')"
    if [[ -n "${screen_pid}" ]]; then
        l1_pid="$(ps -eo pid=,ppid=,args= | awk -v parent="${screen_pid}" '$2 == parent && $0 ~ /run_S3DIS_litept_l1_250_shutdown.sh/ {print $1; exit}')"
    fi
    [[ -n "${l1_pid}" && -r "/proc/${l1_pid}/stat" ]] && break
    sleep 1
done
[[ -n "${l1_pid}" && -r "/proc/${l1_pid}/stat" ]] || { echo "Could not identify L1 process" >&2; exit 1; }
start_ticks="$(awk '{print $22}' "/proc/${l1_pid}/stat")"
state_dir="${L1_RESULT}/auto_shutdown"
milestones="100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250"
screen -dmS s3dis_l1_250_finalize bash -lc "exec '${PYTHON_BIN}' '${ROOT_DIR}/watch_scanobjectnn_finalize_shutdown.py' --pid '${l1_pid}' --start-ticks '${start_ticks}' --result-dir '${L1_RESULT}' --console-log '${L1_CONSOLE}' --state-dir '${state_dir}' --checkpoint-python '${PYTHON_BIN}' --expect-command run_S3DIS_litept_l1_250_shutdown.sh --milestones '${milestones}' --final-epoch 250 --interval-seconds 30 --dead-confirmations 3 --shutdown-delay-seconds 60 --shutdown-on-finish --shutdown-on-interruption"
printf 'L1 queued and started: pid=%s result=%s\n' "${l1_pid}" "${L1_RESULT}"
