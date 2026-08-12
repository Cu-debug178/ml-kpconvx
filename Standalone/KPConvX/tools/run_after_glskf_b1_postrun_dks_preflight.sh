#!/usr/bin/env bash

# Wait for the complete GLSKF B1 post-run pipeline, then run the DKS Phase-A
# preflight. A vanished PID is neutral: only upstream success artifacts decide
# completion. PID start ticks and command identity prevent PID-reuse mistakes.

set -Eeuo pipefail

if (($# != 2)); then
    printf 'usage: %s GLSKF_POSTRUN_PID GLSKF_POSTRUN_START_TICKS\n' "$0" >&2
    exit 2
fi
[[ "$1" =~ ^[0-9]+$ ]] || { printf 'PID must be an integer\n' >&2; exit 2; }
[[ "$2" =~ ^[0-9]+$ ]] || { printf 'START_TICKS must be an integer\n' >&2; exit 2; }

UPSTREAM_PID="$1"
UPSTREAM_START_TICKS="$2"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
KPCONVX_DIR="${ROOT_DIR}/Standalone/KPConvX"
RESULTS_DIR="${KPCONVX_DIR}/results"
UPSTREAM_STATE="${RESULTS_DIR}/glskf_b1_postrun_20260812"
UPSTREAM_SUMMARY="${RESULTS_DIR}/s3dis_glskf_b1_e180_same_checkpoint_ablations_seed57106803/summary.json"
STATE_DIR="${RESULTS_DIR}/dks_preflight_waiter_20260812"
LOG_FILE="${STATE_DIR}/waiter.log"
PREFLIGHT_ROOT="${RESULTS_DIR}/dks_phase_a_preflight_seed57106803"

mkdir -p "${STATE_DIR}"
exec >>"${LOG_FILE}" 2>&1

record_failure() {
    local code="$?"
    trap - EXIT
    if ((code != 0)) && [[ ! -f "${STATE_DIR}/success.tsv" ]]; then
        printf 'phase=dks_preflight\nexit_code=%s\nfailed_at=%s\nlog_file=%s\n' \
            "${code}" "$(date --iso-8601=seconds)" "${LOG_FILE}" > "${STATE_DIR}/failure.tsv"
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

proc_matches_upstream() {
    local cmdline
    [[ -r "/proc/${UPSTREAM_PID}/cmdline" ]] || return 1
    cmdline="$(tr '\0' ' ' < "/proc/${UPSTREAM_PID}/cmdline" 2>/dev/null || true)"
    [[ "${cmdline}" == *"run_after_glskf_b1_ablations.sh"* ]]
}

upstream_artifacts_complete() {
    [[ -f "${UPSTREAM_STATE}/success.tsv" ]] || return 1
    [[ -f "${UPSTREAM_SUMMARY}" ]] || return 1
    python3 - "${UPSTREAM_SUMMARY}" <<'PY' >/dev/null
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(data, dict):
    raise SystemExit(1)
PY
}

main() {
    local current_ticks recheck_ticks
    current_ticks="$(proc_start_ticks "${UPSTREAM_PID}" 2>/dev/null || true)"
    [[ "${current_ticks}" == "${UPSTREAM_START_TICKS}" ]] ||
        die "upstream PID identity changed before waiting began"
    proc_matches_upstream || die "upstream command identity does not match"

    printf 'waiter_pid=%s\nwaiter_start_ticks=%s\nupstream_pid=%s\nupstream_start_ticks=%s\nqueued_at=%s\n' \
        "$$" "$(proc_start_ticks "$$")" "${UPSTREAM_PID}" "${UPSTREAM_START_TICKS}" \
        "$(date --iso-8601=seconds)" > "${STATE_DIR}/queued.tsv"
    log "Waiting for complete GLSKF post-run pipeline pid=${UPSTREAM_PID}, start_ticks=${UPSTREAM_START_TICKS}."

    while :; do
        current_ticks="$(proc_start_ticks "${UPSTREAM_PID}" 2>/dev/null || true)"
        [[ -n "${current_ticks}" ]] || break
        [[ "${current_ticks}" == "${UPSTREAM_START_TICKS}" ]] ||
            die "upstream numeric PID was reused"
        if ! proc_matches_upstream; then
            # /proc can disappear between stat and cmdline reads. Re-check: an
            # absent process is not failure; artifacts below decide completion.
            recheck_ticks="$(proc_start_ticks "${UPSTREAM_PID}" 2>/dev/null || true)"
            [[ -z "${recheck_ticks}" ]] && break
            [[ "${recheck_ticks}" == "${UPSTREAM_START_TICKS}" ]] ||
                die "upstream numeric PID was reused during identity re-check"
            die "upstream command identity changed while the process still exists"
        fi
        sleep 30
    done

    log "Upstream process exited; checking completion artifacts rather than inferring status from PID disappearance."
    upstream_artifacts_complete ||
        die "GLSKF post-run completion artifacts are incomplete; DKS preflight will not use stale outputs"

    KP_CONVX_PYTHON="${KP_CONVX_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}" \
    S3DIS_DATASET_PATH="${S3DIS_DATASET_PATH:-/root/autodl-tmp/data/s3dis}" \
    DKS_PREFLIGHT_ROOT="${PREFLIGHT_ROOT}" \
        bash "${KPCONVX_DIR}/tools/run_dks_phase_a_preflight.sh"

    [[ -f "${PREFLIGHT_ROOT}/success.tsv" ]] || die "DKS preflight success artifact is missing"
    [[ -f "${PREFLIGHT_ROOT}/summary.json" ]] || die "DKS preflight summary is missing"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" > "${STATE_DIR}/success.tsv"
    log "DKS preflight completed after verified GLSKF post-run completion."
}

main "$@"
