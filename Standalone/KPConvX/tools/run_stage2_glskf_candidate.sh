#!/usr/bin/env bash

set -Eeuo pipefail

if (($# != 5)); then
    printf 'usage: %s GLSKF_MODE CONTEXT_CONTROL TRAIN_MODE RUN_NAME EXPECTED_EPOCHS\n' "$0" >&2
    exit 2
fi

GLSKF_MODE="$1"
CONTEXT_CONTROL="$2"
TRAIN_MODE="$3"
RUN_NAME="$4"
EXPECTED_EPOCHS="$5"

case "${GLSKF_MODE}" in
    none|kernel_gate|matched_mlp) ;;
    *) printf 'unsupported GLSKF mode: %s\n' "${GLSKF_MODE}" >&2; exit 2 ;;
esac
case "${CONTEXT_CONTROL}" in
    none|shuffle|room_mean) ;;
    *) printf 'unsupported context control: %s\n' "${CONTEXT_CONTROL}" >&2; exit 2 ;;
esac
case "${TRAIN_MODE}" in
    module_head|head_only) ;;
    *) printf 'unsupported training mode: %s\n' "${TRAIN_MODE}" >&2; exit 2 ;;
esac
if [[ "${GLSKF_MODE}" == "none" && "${TRAIN_MODE}" != "head_only" ]]; then
    printf 'glskf_mode=none is only valid for the head_only L0 control\n' >&2
    exit 2
fi
if [[ "${GLSKF_MODE}" == "matched_mlp" && "${CONTEXT_CONTROL}" != "none" ]]; then
    printf 'matched_mlp does not accept a deep-context control\n' >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
RUN_DIR="${RESULTS_DIR}/${RUN_NAME}"
PYTHON_BIN="${KP_CONVX_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
DATASET_PATH="${S3DIS_DATASET_PATH:-/root/autodl-tmp/data/s3dis}"
FINETUNE_PATH="${GLSKF_L0_CHECKPOINT:-${RESULTS_DIR}/s3dis_litept_l0_b12a2_seed57106803/checkpoints/chkp_0210.tar}"
[[ "${EXPECTED_EPOCHS}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'EXPECTED_EPOCHS must be a positive integer\n' >&2
    exit 2
}

completed_run() {
    local val_lines=0
    [[ -f "${RUN_DIR}/val_IoUs.txt" ]] && val_lines="$(wc -l < "${RUN_DIR}/val_IoUs.txt")"
    [[ -f "${RUN_DIR}/checkpoints/current_chkp.tar" ]] &&
        [[ -f "${RUN_DIR}/checkpoints/best_val_chkp.tar" ]] &&
        ((val_lines >= EXPECTED_EPOCHS))
}

if completed_run; then
    printf 'Result is already complete; skipping: %s\n' "${RUN_DIR}"
    exit 0
fi
[[ ! -e "${RUN_DIR}" ]] || {
    printf 'Incomplete result directory requires manual diagnosis; preserving it: %s\n' "${RUN_DIR}" >&2
    exit 75
}
[[ -x "${PYTHON_BIN}" ]] || {
    printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
    exit 66
}
[[ -d "${DATASET_PATH}/Area_5" ]] || {
    printf 'S3DIS Area_5 directory not found: %s\n' "${DATASET_PATH}/Area_5" >&2
    exit 66
}
[[ -f "${FINETUNE_PATH}" ]] || {
    printf 'L0 epoch-210 checkpoint not found: %s\n' "${FINETUNE_PATH}" >&2
    exit 66
}

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" experiments/S3DIS/train_S3DIS.py \
    --dataset_path "${DATASET_PATH}" \
    --log_path "${RUN_DIR}" \
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
    --glskf_mode "${GLSKF_MODE}" \
    --glskf_refine_stage 3 \
    --glskf_context_stages 4,5 \
    --glskf_groups 8 \
    --glskf_hidden_dim 64 \
    --glskf_context_control "${CONTEXT_CONTROL}" \
    --glskf_train_mode "${TRAIN_MODE}" \
    --validation_mode full_identity \
    --max_epoch "${EXPECTED_EPOCHS}" \
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
        train.cyc_decrease10=60

completed_run || {
    printf 'Training returned success but expected artifacts are incomplete: %s\n' "${RUN_DIR}" >&2
    exit 76
}
