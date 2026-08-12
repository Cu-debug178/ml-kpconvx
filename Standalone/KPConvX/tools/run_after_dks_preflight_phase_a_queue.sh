#!/usr/bin/env bash

# 逻辑：核验并等待 DKS 预检完成；预检产物完整后，串行启动 3 seed x 6 arm Phase-A 队列。
# 使用：run_after_dks_preflight_phase_a_queue.sh PREFLIGHT_WAITER_PID START_TICKS
# 范围：仅用于本项目 DKS S3DIS Phase-A；不绕过预检，也不负责关机。

set -Eeuo pipefail

if (($# != 2)); then
    printf 'usage: %s PREFLIGHT_WAITER_PID PREFLIGHT_WAITER_START_TICKS\n' "$0" >&2
    exit 2
fi
[[ "$1" =~ ^[0-9]+$ ]] || { printf 'PREFLIGHT_WAITER_PID must be an integer\n' >&2; exit 2; }
[[ "$2" =~ ^[0-9]+$ ]] || { printf 'PREFLIGHT_WAITER_START_TICKS must be an integer\n' >&2; exit 2; }

WAIT_PID="$1"
WAIT_START_TICKS="$2"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
KPCONVX_DIR="${ROOT_DIR}/Standalone/KPConvX"
RESULTS_DIR="${KPCONVX_DIR}/results"
STATE_DIR="${RESULTS_DIR}/dks_phase_a_queue_waiter_20260812"
LOG_FILE="${STATE_DIR}/waiter.log"
PREFLIGHT_ROOT="${RESULTS_DIR}/dks_phase_a_preflight_seed57106803"
QUEUE_STATE="${RESULTS_DIR}/dks_stage2_queue"

mkdir -p "${STATE_DIR}"
exec >>"${LOG_FILE}" 2>&1

record_failure() {
    local code="$?"
    trap - EXIT
    if ((code != 0)) && [[ ! -f "${STATE_DIR}/success.tsv" ]]; then
        printf 'phase=dks_phase_a_queue\nexit_code=%s\nfailed_at=%s\nlog_file=%s\nqueue_state=%s\n' \
            "${code}" "$(date --iso-8601=seconds)" "${LOG_FILE}" "${QUEUE_STATE}" > "${STATE_DIR}/failure.tsv"
    fi
    exit "${code}"
}
trap record_failure EXIT

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

proc_start_ticks() {
    local pid="$1" stat_line rest
    [[ -r "/proc/${pid}/stat" ]] || return 1
    stat_line="$(<"/proc/${pid}/stat")"
    rest="${stat_line#*) }"
    [[ "${rest}" != "${stat_line}" ]] || return 1
    set -- ${rest}
    printf '%s\n' "${20}"
}

proc_matches_waiter() {
    local cmdline
    [[ -r "/proc/${WAIT_PID}/cmdline" ]] || return 1
    cmdline="$(tr '\0' ' ' < "/proc/${WAIT_PID}/cmdline" 2>/dev/null || true)"
    [[ "${cmdline}" == *"run_after_glskf_b1_postrun_dks_preflight.sh"* ]]
}

preflight_complete() {
    [[ -f "${PREFLIGHT_ROOT}/success.tsv" ]] || return 1
    [[ -f "${PREFLIGHT_ROOT}/summary.json" ]] || return 1
    python3 - "${PREFLIGHT_ROOT}/summary.json" <<'PY' >/dev/null
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
if data.get("status") != "completed":
    raise SystemExit(1)
PY
}

dks_queue_complete() {
    local state_dir="$1" manifest_path="$1/queue_manifest.tsv"
    [[ -f "${state_dir}/status.tsv" ]] || return 1
    [[ -f "${manifest_path}" ]] || return 1
    [[ "$(find "${state_dir}/markers" -maxdepth 1 -type f -name '*.success' -printf '.' 2>/dev/null | wc -c)" -eq 18 ]] || return 1
    python3 - "${manifest_path}" "${state_dir}" "${RESULTS_DIR}" <<'PY'
import csv
import os
import sys

manifest_path, state_dir, results_dir = sys.argv[1:]
with open(manifest_path, newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
if len(rows) != 18:
    raise SystemExit(f"expected 18 manifest rows, got {len(rows)}")
for row in rows:
    job = row["job"]
    run_dir = os.path.join(results_dir, row["run_name"])
    marker = os.path.join(state_dir, "markers", job + ".success")
    required = [
        marker,
        os.path.join(run_dir, "val_IoUs.txt"),
        os.path.join(run_dir, "checkpoints", "current_chkp.tar"),
        os.path.join(run_dir, "checkpoints", "best_val_chkp.tar"),
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise SystemExit(f"missing artifacts for {job}: {missing!r}")
    with open(required[1], encoding="utf-8", errors="replace") as stream:
        if sum(1 for line in stream if line.strip()) < int(row["epochs"]):
            raise SystemExit(f"validation rows are incomplete for {job}")
    if row["arm"] == "learned":
        stats_path = os.path.join(run_dir, "dks_alpha_stats.csv")
        if not os.path.isfile(stats_path):
            raise SystemExit(f"learned alpha diagnostics are missing for {job}")
        with open(stats_path, newline="", encoding="utf-8") as stream:
            stats = list(csv.DictReader(stream))
        if not stats:
            raise SystemExit(f"learned alpha diagnostics are empty for {job}")
        last = stats[-1]
        try:
            gate = float(last["gate"])
            std = float(last["std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid learned alpha diagnostics for {job}: {exc}")
        diagnostic_path = os.path.join(state_dir, "learned_diagnostics.tsv")
        with open(diagnostic_path, "a", encoding="utf-8") as stream:
            stream.write(
                f"{job}\t{row['seed']}\t{gate:.9g}\t{std:.9g}\t"
                f"{'collapsed' if abs(gate) <= 1e-8 or std <= 0.02 else 'noncollapsed'}\n"
            )
PY
}

main() {
    local current_ticks recheck_ticks queue_code
    current_ticks="$(proc_start_ticks "${WAIT_PID}" 2>/dev/null || true)"
    [[ "${current_ticks}" == "${WAIT_START_TICKS}" ]] ||
        die "preflight waiter identity changed before waiting began"
    proc_matches_waiter || die "preflight waiter command identity does not match"

    printf 'waiter_pid=%s\nwaiter_start_ticks=%s\npreflight_waiter_pid=%s\npreflight_waiter_start_ticks=%s\nqueued_at=%s\n' \
        "$$" "$(proc_start_ticks "$$")" "${WAIT_PID}" "${WAIT_START_TICKS}" \
        "$(date --iso-8601=seconds)" > "${STATE_DIR}/queued.tsv"
    log "Waiting for verified DKS preflight pid=${WAIT_PID}, start_ticks=${WAIT_START_TICKS}."

    while :; do
        current_ticks="$(proc_start_ticks "${WAIT_PID}" 2>/dev/null || true)"
        [[ -n "${current_ticks}" ]] || break
        [[ "${current_ticks}" == "${WAIT_START_TICKS}" ]] || die "preflight waiter PID was reused"
        if ! proc_matches_waiter; then
            recheck_ticks="$(proc_start_ticks "${WAIT_PID}" 2>/dev/null || true)"
            [[ -z "${recheck_ticks}" ]] && break
            [[ "${recheck_ticks}" == "${WAIT_START_TICKS}" ]] || die "preflight waiter PID was reused during identity re-check"
            die "preflight waiter command identity changed while process still exists"
        fi
        sleep 30
    done

    log "Preflight waiter exited; checking preflight artifacts rather than inferring status from PID disappearance."
    preflight_complete || die "DKS preflight is incomplete; Phase-A queue will not use stale outputs"
    log "Starting the independent 3-seed x 6-arm DKS Phase-A queue."
    set +e
    KP_CONVX_PYTHON="${KP_CONVX_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}" \
    S3DIS_DATASET_PATH="${S3DIS_DATASET_PATH:-/root/autodl-tmp/data/s3dis}" \
    DKS_L0_CHECKPOINT="${DKS_L0_CHECKPOINT:-${RESULTS_DIR}/s3dis_litept_l0_b12a2_seed57106803/checkpoints/chkp_0210.tar}" \
    DKS_QUEUE_STATE_DIR="${DKS_QUEUE_STATE_DIR:-${QUEUE_STATE}}" \
        bash "${KPCONVX_DIR}/tools/run_stage2_dks_queue.sh"
    queue_code=$?
    set -e
    printf 'queue_exit_code=%s\ncompleted_at=%s\n' "${queue_code}" "$(date --iso-8601=seconds)" > "${STATE_DIR}/queue_exit.tsv"
    ((queue_code == 0)) || die "DKS Phase-A queue exited with code ${queue_code}; inspect ${QUEUE_STATE}/status.tsv and diagnostics"
    dks_queue_complete "${QUEUE_STATE}" || die "DKS Phase-A queue lacks 18 successful jobs with complete artifacts"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" > "${STATE_DIR}/success.tsv"
    log "DKS Phase-A queue completed successfully."
}

main "$@"
