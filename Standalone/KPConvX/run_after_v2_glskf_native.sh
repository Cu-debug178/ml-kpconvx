#!/usr/bin/env bash

# 逻辑：等待既有 Stage-2 KTHA V2 队列完成；完成证据必须同时包括队列进程
# 已退出、status.tsv 中四个任务均 success、以及 true/shuffled/matched 的
# 完整 checkpoint 与 10 行 full_identity 指标。随后 SSH fetch + fast-forward
# 主工作区，再以前台子进程启动 GLSKF Stage-2 队列。
# 使用：在 GPU 训练启动后执行：
#   run_after_v2_glskf_native.sh QUEUE_PID QUEUE_START_TICKS
# 其中 QUEUE_START_TICKS 是 `/proc/QUEUE_PID/stat` 第 22 项；可用 `awk
# '{print $22}' /proc/PID/stat` 读取。脚本只用于依赖流水线，不替代训练监控。
# 范围：本项目 S3DIS KTHA V2 -> GLSKF Stage-2；不用于独立、无依赖任务。

set -Eeuo pipefail

if (($# != 2)); then
    printf 'usage: %s V2_QUEUE_PID V2_QUEUE_START_TICKS\n' "$0" >&2
    exit 2
fi

[[ "$1" =~ ^[0-9]+$ ]] || { printf 'V2_QUEUE_PID must be an integer\n' >&2; exit 2; }
[[ "$2" =~ ^[0-9]+$ ]] || { printf 'V2_QUEUE_START_TICKS must be an integer\n' >&2; exit 2; }

V2_QUEUE_PID="$1"
V2_QUEUE_START_TICKS="$2"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN_DIR="${V2_MAIN_DIR:-${ROOT_DIR}}"
V2_STATE_DIR="${V2_STATE_DIR:-${MAIN_DIR}/Standalone/KPConvX/results/queue_state/stage2_ktha_v2_screen}"
GLSKF_LOG="${GLSKF_PIPELINE_LOG:-${MAIN_DIR}/Standalone/KPConvX/results/glskf_after_v2_native_20260812.log}"
REMOTE="${V2_GIT_REMOTE:-fork}"
BRANCH="${V2_GIT_BRANCH:-codex/s3dis-resume-throughput}"

mkdir -p "$(dirname -- "${GLSKF_LOG}")"
exec >>"${GLSKF_LOG}" 2>&1

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

[[ -d "${MAIN_DIR}/.git" ]] || die "V2_MAIN_DIR is not a git worktree: ${MAIN_DIR}"

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
    [[ -r "/proc/${V2_QUEUE_PID}/cmdline" ]] || return 1
    cmdline="$(tr '\0' ' ' < "/proc/${V2_QUEUE_PID}/cmdline" 2>/dev/null || true)"
    [[ "${cmdline}" == *"run_resilient_queue.sh"* ]] &&
        [[ "${cmdline}" == *"stage2_ktha_v2_screen"* ]]
}

status_success() {
    local status_file="${V2_STATE_DIR}/status.tsv"
    [[ -f "${status_file}" ]] || return 1
    for job in 10_v2_true 20_v2_shuffled 30_v2_matched_mlp 90_summarize; do
        awk -F '\t' -v wanted="${job}" '$1 == wanted && $2 == "success" {found=1} END {exit !found}' "${status_file}" || return 1
    done
}

artifacts_complete() {
    local results_dir="${MAIN_DIR}/Standalone/KPConvX/results"
    for run in \
        s3dis_ktha_v2_pairwise_true_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803 \
        s3dis_ktha_v2_pairwise_shuffled_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803 \
        s3dis_ktha_v2_matched_mlp_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803; do
        [[ -f "${results_dir}/${run}/checkpoints/current_chkp.tar" ]] || return 1
        [[ "$(wc -l < "${results_dir}/${run}/val_IoUs.txt" 2>/dev/null || printf 0)" -ge 10 ]] || return 1
    done
    [[ -f "${results_dir}/stage2_ktha_v2_warm10_screen.csv" ]]
}

wait_for_v2() {
    local current_ticks
    current_ticks="$(proc_start_ticks "${V2_QUEUE_PID}" 2>/dev/null || true)"
    [[ "${current_ticks}" == "${V2_QUEUE_START_TICKS}" ]] ||
        die "V2 queue PID identity changed before waiting began"
    log "Waiting for V2 queue pid=${V2_QUEUE_PID}, start_ticks=${V2_QUEUE_START_TICKS}."
    while :; do
        current_ticks="$(proc_start_ticks "${V2_QUEUE_PID}" 2>/dev/null || true)"
        if [[ -z "${current_ticks}" ]]; then
            break
        fi
        [[ "${current_ticks}" == "${V2_QUEUE_START_TICKS}" ]] || die "V2 queue PID was reused"
        proc_matches_queue || die "V2 queue PID command identity changed"
        sleep 30
    done
    status_success || die "V2 queue exited without four successful job records"
    artifacts_complete || die "V2 queue status is incomplete or artifacts are missing"
    log "V2 completion evidence is valid."
}

update_main() {
    cd "${MAIN_DIR}"
    git fetch "${REMOTE}" "${BRANCH}"
    git merge --ff-only "${REMOTE}/${BRANCH}"
    [[ "$(git rev-parse HEAD)" == "$(git rev-parse "${REMOTE}/${BRANCH}")" ]] ||
        die "main worktree did not fast-forward to ${REMOTE}/${BRANCH}"
}

main() {
    [[ "$(git -C "${MAIN_DIR}" rev-parse --show-toplevel 2>/dev/null)" == "${MAIN_DIR}" ]] ||
        die "V2_MAIN_DIR is not the expected main worktree: ${MAIN_DIR}"
    wait_for_v2
    update_main
    log "Starting GLSKF Stage-2 queue as a foreground child."
    bash "${MAIN_DIR}/Standalone/KPConvX/run_stage2_glskf_queue.sh"
    log "GLSKF Stage-2 queue completed."
}

main "$@"
