#!/usr/bin/env bash

# 逻辑：核验既有 GLSKF Stage-2 队列身份并等待其退出；五路任务与汇总全部成功且
# 产物完整后，以前台子进程启动 B1 scratch 180 epoch。上游失败时不启动长训。
# 使用：run_after_glskf_stage2_b1_scratch_e180.sh QUEUE_PID QUEUE_START_TICKS
# 范围：本项目 GLSKF Stage-2 -> B1 scratch 长训的依赖流水线。

set -Eeuo pipefail

if (($# != 2)); then
    printf 'usage: %s QUEUE_PID QUEUE_START_TICKS\n' "$0" >&2
    exit 2
fi
[[ "$1" =~ ^[0-9]+$ ]] || { printf 'QUEUE_PID must be an integer\n' >&2; exit 2; }
[[ "$2" =~ ^[0-9]+$ ]] || { printf 'QUEUE_START_TICKS must be an integer\n' >&2; exit 2; }

QUEUE_PID="$1"
QUEUE_START_TICKS="$2"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
KPCONVX_DIR="${ROOT_DIR}/Standalone/KPConvX"
STATE_DIR="${KPCONVX_DIR}/results/glskf_stage2_queue_20260812"
PIPELINE_STATE="${KPCONVX_DIR}/results/glskf_b1_after_stage2_20260812"
PIPELINE_LOG="${PIPELINE_STATE}/pipeline.log"

mkdir -p "${PIPELINE_STATE}"
exec >>"${PIPELINE_LOG}" 2>&1

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

proc_matches_queue() {
    local cmdline
    [[ -r "/proc/${QUEUE_PID}/cmdline" ]] || return 1
    cmdline="$(tr '\0' ' ' < "/proc/${QUEUE_PID}/cmdline" 2>/dev/null || true)"
    [[ "${cmdline}" == *"run_resilient_queue.sh"* ]] &&
        [[ "${cmdline}" == *"stage2_glskf_screen"* ]]
}

status_success() {
    local status_file="${STATE_DIR}/status.tsv" job
    [[ -f "${status_file}" ]] || return 1
    for job in 10_l0_head 20_true 30_shuffled 40_room_mean 50_matched_mlp 90_summarize; do
        awk -F '\t' -v wanted="${job}" \
            '$1 == wanted && $2 == "success" {found=1} END {exit !found}' \
            "${status_file}" || return 1
    done
}

artifacts_complete() {
    local results_dir="${KPCONVX_DIR}/results" run
    for run in \
        s3dis_glskf_l0_head_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803 \
        s3dis_glskf_kernel_gate_true_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803 \
        s3dis_glskf_kernel_gate_shuffled_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803 \
        s3dis_glskf_kernel_gate_room_mean_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803 \
        s3dis_glskf_matched_mlp_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803; do
        [[ -f "${results_dir}/${run}/checkpoints/current_chkp.tar" ]] || return 1
        [[ -f "${results_dir}/${run}/checkpoints/best_val_chkp.tar" ]] || return 1
        [[ "$(wc -l < "${results_dir}/${run}/val_IoUs.txt" 2>/dev/null || printf 0)" -ge 10 ]] || return 1
    done
    [[ -f "${results_dir}/glskf_stage2_screen_20260812/warm10_summary.csv" ]] &&
        [[ -f "${results_dir}/glskf_stage2_screen_20260812/warm10_summary.json" ]]
}

main() {
    local current_ticks
    current_ticks="$(proc_start_ticks "${QUEUE_PID}" 2>/dev/null || true)"
    [[ "${current_ticks}" == "${QUEUE_START_TICKS}" ]] ||
        die "GLSKF queue PID identity changed before waiting began"
    proc_matches_queue || die "GLSKF queue PID command identity does not match"

    printf 'waiter_pid=%s\nwaiter_start_ticks=%s\nqueue_pid=%s\nqueue_start_ticks=%s\nqueued_at=%s\n' \
        "$$" "$(proc_start_ticks "$$")" "${QUEUE_PID}" "${QUEUE_START_TICKS}" \
        "$(date --iso-8601=seconds)" \
        > "${PIPELINE_STATE}/queued.tsv"
    log "Waiting for GLSKF Stage-2 queue pid=${QUEUE_PID}, start_ticks=${QUEUE_START_TICKS}."

    while :; do
        current_ticks="$(proc_start_ticks "${QUEUE_PID}" 2>/dev/null || true)"
        [[ -n "${current_ticks}" ]] || break
        [[ "${current_ticks}" == "${QUEUE_START_TICKS}" ]] || die "GLSKF queue PID was reused"
        proc_matches_queue || die "GLSKF queue command identity changed"
        sleep 30
    done

    status_success || die "GLSKF queue exited without six successful job records"
    artifacts_complete || die "GLSKF queue artifacts are incomplete"
    log "GLSKF Stage-2 completion evidence is valid; starting B1 scratch as foreground child."

    bash "${KPCONVX_DIR}/tools/run_stage3_glskf_b1_scratch_e180.sh"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" > "${PIPELINE_STATE}/success.tsv"
    log "B1 scratch training and 10-vote evaluation completed."
}

main "$@"
