#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_FOR_PID="${WAIT_FOR_PID:-134977}"
BASELINE_RESULT="${BASELINE_RESULT:-${ROOT_DIR}/Standalone/KPConvX/results/ScanObjectNN_KPConvD-L-official-4090D-24G}"
BASELINE_CONSOLE="${BASELINE_CONSOLE:-${BASELINE_RESULT}.console.log}"
TARGET_RESULT="${TARGET_RESULT:-${ROOT_DIR}/Standalone/KPConvX/results/ScanObjectNN_KPConvX-L-FastAdapter-seed57106803-4090D-24G}"
TARGET_CONSOLE="${TARGET_CONSOLE:-${TARGET_RESULT}.console.log}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/pointcept/bin/python}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/data/ScanObjectNN/main_split}"
SEED="${SEED:-57106803}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-300}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-3}"

current_run_is_alive() {
  [[ -r "/proc/${WAIT_FOR_PID}/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' < "/proc/${WAIT_FOR_PID}/cmdline")"
  [[ "${command_line}" == *"experiments/ScanObjectNN/train_ScanObj.py"* &&
     "${command_line}" == *"${BASELINE_RESULT}"* ]]
}

printf '%s waiting for ScanObjectNN baseline pid=%s\n' "$(date -Is)" "${WAIT_FOR_PID}"
last_heartbeat=0
while current_run_is_alive; do
  now="$(date +%s)"
  if (( now - last_heartbeat >= HEARTBEAT_INTERVAL )); then
    epoch_step="$(tail -1 "${BASELINE_RESULT}/training.txt" 2>/dev/null \
      | awk '{print $1 ":" $2}' || true)"
    disk_available_gib="$(df --output=avail -B1 "${ROOT_DIR}" \
      | tail -1 | awk '{printf "%.2f", $1 / 1073741824}')"
    printf '%s baseline alive epoch_step=%s disk_available_gib=%s\n' \
      "$(date -Is)" "${epoch_step:-unknown}" "${disk_available_gib:-unknown}"
    last_heartbeat="${now}"
  fi
  sleep "${POLL_INTERVAL}"
done

# Give buffered console output a moment to flush after the process exits.
sleep 5
if ! grep -q 'Finished Training' "${BASELINE_CONSOLE}" 2>/dev/null; then
  printf '%s baseline ended without a normal completion marker; FastAdapter was not started\n' \
    "$(date -Is)" >&2
  exit 1
fi

if [[ -e "${TARGET_RESULT}" || -e "${TARGET_CONSOLE}" ]]; then
  printf 'Refusing to overwrite an existing FastAdapter result: %s\n' "${TARGET_RESULT}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi
for split_file in \
  training_objectdataset_augmentedrot_scale75.h5 \
  test_objectdataset_augmentedrot_scale75.h5; do
  if [[ ! -f "${DATASET_PATH}/${split_file}" ]]; then
    printf 'ScanObjectNN split not found: %s\n' "${DATASET_PATH}/${split_file}" >&2
    exit 1
  fi
done

disk_available_bytes="$(df --output=avail -B1 "${ROOT_DIR}" | tail -1 | tr -d ' ')"
minimum_disk_bytes="$((MIN_FREE_DISK_GIB * 1024 * 1024 * 1024))"
if (( disk_available_bytes < minimum_disk_bytes )); then
  printf 'Only %.2f GiB is available; at least %s GiB is required\n' \
    "$(awk -v bytes="${disk_available_bytes}" 'BEGIN {print bytes / 1073741824}')" \
    "${MIN_FREE_DISK_GIB}" >&2
  exit 1
fi

# Do not overlap another compute job that may have claimed the GPU meanwhile.
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '^[[:space:]]*[0-9]+'; do
  printf '%s waiting for GPU compute processes to exit\n' "$(date -Is)"
  sleep "${POLL_INTERVAL}"
done

mkdir -p "${TARGET_RESULT}"
printf '%s launching ScanObjectNN KPConvX FastAdapter with seed=%s\n' \
  "$(date -Is)" "${SEED}"

if command -v screen >/dev/null 2>&1; then
  screen -dmS scanobjectnn_fastadapter_monitor env \
    TRAIN_PID="$$" \
    PROJECT_DIR="${ROOT_DIR}" \
    RESULT_DIR="${TARGET_RESULT}" \
    CONSOLE_LOG="${TARGET_CONSOLE}" \
    INTERVAL=15 \
    "${ROOT_DIR}/monitor_scanobjectnn_kpconvd_l.sh"
fi

export PYTHON_BIN DATASET_PATH SEED
export LOG_PATH="${TARGET_RESULT}"
exec "${ROOT_DIR}/Standalone/train_ScanObjectNN_fastadapter.sh" \
  > "${TARGET_CONSOLE}" 2>&1
