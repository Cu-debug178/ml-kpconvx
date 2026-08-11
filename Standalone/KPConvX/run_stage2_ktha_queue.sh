#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/root/autodl-tmp/envs/pointcept/bin/python"
DATASET_PATH="/root/autodl-tmp/data/s3dis"
RESULTS_DIR="${SCRIPT_DIR}/results"
FINETUNE_PATH="${RESULTS_DIR}/s3dis_litept_l0_b12a2_seed57106803/checkpoints/chkp_0210.tar"
M2_DIR="${RESULTS_DIR}/s3dis_ktha_m2_qk_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803"
M2_PID="31641"
QUEUE_LOG="${RESULTS_DIR}/stage2_ktha_queue_20260811.log"

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${QUEUE_LOG}"
}

die() {
    log "ERROR: $*"
    exit 1
}

completed_run() {
    local run_dir="$1"
    local expected_epochs="${2:-10}"
    local val_lines=0

    if [[ -f "${run_dir}/val_IoUs.txt" ]]; then
        val_lines="$(wc -l < "${run_dir}/val_IoUs.txt")"
    fi

    [[ -f "${run_dir}/checkpoints/current_chkp.tar" ]] &&
        (( val_lines >= expected_epochs ))
}

wait_for_m2() {
    local cmdline=""

    log "Waiting for the active M2 process (PID ${M2_PID})."
    while [[ -r "/proc/${M2_PID}/cmdline" ]]; do
        # /proc can disappear after the readability test if M2 exits here.
        # Capture cmdline separately so that case is handled as normal process
        # termination instead of being mislabeled as PID reuse.
        if ! cmdline="$(tr '\0' ' ' < "/proc/${M2_PID}/cmdline" 2>/dev/null)"; then
            break
        fi

        if [[ "${cmdline}" != *"${M2_DIR}"* ]]; then
            # The process exited or the numeric PID now belongs to another
            # process. Artifacts, rather than PID identity, decide whether M2
            # completed successfully.
            if completed_run "${M2_DIR}" 10; then
                log "M2 process ended after all expected artifacts were saved."
                return 0
            fi
            die "PID ${M2_PID} no longer identifies M2 before its expected artifacts were complete."
        fi
        sleep 20
    done

    completed_run "${M2_DIR}" 10 ||
        die "M2 exited without 10 validation rows and a final checkpoint."
    log "M2 completed successfully."
}

run_experiment() {
    local label="$1"
    local mode="$2"
    local shuffle_geometry="$3"
    local run_dir="$4"

    if completed_run "${run_dir}" 10; then
        log "${label} is already complete; skipping it."
        return 0
    fi
    [[ ! -e "${run_dir}" ]] ||
        die "${label} output directory already exists but is incomplete: ${run_dir}"

    log "Starting ${label}: mode=${mode}, shuffle_geometry=${shuffle_geometry}."
    "${PYTHON_BIN}" experiments/S3DIS/train_S3DIS.py \
        --dataset_path "${DATASET_PATH}" \
        --log_path "${run_dir}" \
        --finetune_path "${FINETUNE_PATH}" \
        --seed 57106803 \
        --kp_mode kpconvx \
        --layer_blocks 3 3 9 12 3 \
        --neighbor_limits 12 16 20 20 20 \
        --litept_enabled 1 \
        --litept_conv_stages 3 \
        --litept_handover_stage 0 \
        --litept_patch_size 128 \
        --litept_num_heads 8 \
        --litept_attention_ratio 1.0 \
        --litept_mlp_ratio 4.0 \
        --litept_rope_base 100.0 \
        --litept_rope_enabled 1 \
        --litept_orders z,z-trans \
        --litept_light_decoder 0 \
        --litept_legacy_kpconvd_encoder 1 \
        --decoder_layer 1 \
        --ktha_mode "${mode}" \
        --ktha_source_stage 3 \
        --ktha_target_stages 4 \
        --ktha_relation_dim 8 \
        --ktha_shuffle_geometry "${shuffle_geometry}" \
        --ktha_train_mode module_head \
        --validation_mode full_identity \
        --max_epoch 10 \
        --batch_size 24 \
        --accum_batch 1 \
        --amp_enabled 1 \
        --amp_dtype bfloat16 \
        --weight_decay 0.01 \
        --auto_test_vote10 0 \
        --options \
            train.steps_per_epoch=300 \
            train.num_workers=10 \
            train.monitor_enabled=True \
            train.monitor_interval=25 \
            train.save_best_val=True \
            train.save_latest_val=True \
            train.save_best_val_cycle=False \
            train.checkpoint_gap=2 \
            train.cyc_lr0=0.005 \
            train.cyc_lr1=0.005 \
            train.cyc_raise_n=1 \
            train.cyc_plateau=0 \
            train.cyc_decrease10=60 \
        >> "${QUEUE_LOG}" 2>&1

    completed_run "${run_dir}" 10 ||
        die "${label} returned success but its expected artifacts are incomplete."
    log "${label} completed successfully."
}

main() {
    [[ -x "${PYTHON_BIN}" ]] || die "Python executable not found: ${PYTHON_BIN}"
    [[ -f "${FINETUNE_PATH}" ]] || die "L0 epoch-210 checkpoint not found: ${FINETUNE_PATH}"
    cd "${SCRIPT_DIR}"

    log "Stage-2 queue started. Runs are serialized and stop on the first failure."
    wait_for_m2

    run_experiment \
        "M3 relation bias" \
        "relation_bias" \
        0 \
        "${RESULTS_DIR}/s3dis_ktha_m3_relation_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803"

    run_experiment \
        "M3 shuffled geometry" \
        "relation_bias" \
        1 \
        "${RESULTS_DIR}/s3dis_ktha_m3_relation_shuffled_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803"

    run_experiment \
        "Matched-MLP parameter control" \
        "matched_mlp" \
        0 \
        "${RESULTS_DIR}/s3dis_ktha_matched_mlp_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803"

    log "Stage-2 queue completed successfully."
}

main "$@"
