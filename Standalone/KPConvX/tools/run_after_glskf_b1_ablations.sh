#!/usr/bin/env bash

# 逻辑：核验并等待正在运行的 B1 主脚本；180 轮训练和自动 10-vote 产物完整后，
# 顺序执行外部 L0 参考与 B1 最佳权重的六路同 checkpoint 推理干预。
# 使用：run_after_glskf_b1_ablations.sh B1_SCRIPT_PID B1_SCRIPT_START_TICKS
# 范围：仅用于本项目 S3DIS GLSKF B1 scratch e180 后处理，不负责启动或恢复训练。

set -Eeuo pipefail

if (($# != 2)); then
    printf 'usage: %s B1_SCRIPT_PID B1_SCRIPT_START_TICKS\n' "$0" >&2
    exit 2
fi
[[ "$1" =~ ^[0-9]+$ ]] || { printf 'B1_SCRIPT_PID must be an integer\n' >&2; exit 2; }
[[ "$2" =~ ^[0-9]+$ ]] || { printf 'B1_SCRIPT_START_TICKS must be an integer\n' >&2; exit 2; }

B1_PID="$1"
B1_START_TICKS="$2"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
KPCONVX_DIR="${ROOT_DIR}/Standalone/KPConvX"
RESULTS_DIR="${KPCONVX_DIR}/results"
B1_RUN="${RESULTS_DIR}/s3dis_glskf_b1_true_scratch_e180_bf16_b12a2_lr5e3_wd001_seed57106803"
PIPELINE_STATE="${RESULTS_DIR}/glskf_b1_postrun_20260812"
PIPELINE_LOG="${PIPELINE_STATE}/pipeline.log"
ABLATION_ROOT="${RESULTS_DIR}/s3dis_glskf_b1_e180_same_checkpoint_ablations_seed57106803"
QUEUE_STATE="${RESULTS_DIR}/glskf_b1_e180_ablation_queue_20260812"
L0_CHECKPOINT="${RESULTS_DIR}/s3dis_litept_l0_b12a2_seed57106803/checkpoints/chkp_0210.tar"

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

proc_matches_b1() {
    local cmdline
    [[ -r "/proc/${B1_PID}/cmdline" ]] || return 1
    cmdline="$(tr '\0' ' ' < "/proc/${B1_PID}/cmdline" 2>/dev/null || true)"
    [[ "${cmdline}" == *"run_stage3_glskf_b1_scratch_e180.sh"* ]]
}

b1_artifacts_complete() {
    local val_lines=0
    [[ -f "${B1_RUN}/val_IoUs.txt" ]] || return 1
    val_lines="$(wc -l < "${B1_RUN}/val_IoUs.txt")"
    ((val_lines >= 180)) &&
        [[ -f "${B1_RUN}/checkpoints/current_chkp.tar" ]] &&
        [[ -f "${B1_RUN}/checkpoints/best_val_chkp.tar" ]] &&
        [[ -f "${B1_RUN}/test/test_001/report.txt" ]]
}

main() {
    local current_ticks
    current_ticks="$(proc_start_ticks "${B1_PID}" 2>/dev/null || true)"
    [[ "${current_ticks}" == "${B1_START_TICKS}" ]] ||
        die "B1 script PID identity changed before waiting began"
    proc_matches_b1 || die "B1 script command identity does not match"

    printf 'waiter_pid=%s\nwaiter_start_ticks=%s\nb1_pid=%s\nb1_start_ticks=%s\nqueued_at=%s\n' \
        "$$" "$(proc_start_ticks "$$")" "${B1_PID}" "${B1_START_TICKS}" \
        "$(date --iso-8601=seconds)" > "${PIPELINE_STATE}/queued.tsv"
    log "Waiting for B1 script pid=${B1_PID}, start_ticks=${B1_START_TICKS}."

    while :; do
        current_ticks="$(proc_start_ticks "${B1_PID}" 2>/dev/null || true)"
        [[ -n "${current_ticks}" ]] || break
        [[ "${current_ticks}" == "${B1_START_TICKS}" ]] || die "B1 script PID was reused"
        proc_matches_b1 || die "B1 script command identity changed"
        sleep 30
    done

    b1_artifacts_complete || die "B1 exited without complete 180-epoch and 10-vote artifacts"
    [[ -f "${L0_CHECKPOINT}" ]] || die "external L0 reference checkpoint is unavailable"
    log "B1 completion evidence is valid; starting external reference and same-checkpoint interventions."

    GLSKF_SOURCE_RUN="${B1_RUN}" \
    GLSKF_CHECKPOINT="${B1_RUN}/checkpoints/best_val_chkp.tar" \
    GLSKF_L0_CHECKPOINT="${L0_CHECKPOINT}" \
    GLSKF_ABLATION_ROOT="${ABLATION_ROOT}" \
    GLSKF_ABLATION_QUEUE_STATE="${QUEUE_STATE}" \
        bash "${KPCONVX_DIR}/run_glskf_inference_ablations.sh"

    [[ -f "${ABLATION_ROOT}/summary.json" ]] || die "GLSKF ablation summary is missing"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" > "${PIPELINE_STATE}/success.tsv"
    log "B1 post-run intervention queue completed."
}

main "$@"
