#!/usr/bin/env bash

set -Eeuo pipefail

if (($# != 2)); then
    printf 'usage: %s CHECKPOINT RUN_NAME\n' "$0" >&2
    exit 2
fi

CHECKPOINT="$1"
RUN_NAME="$2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${SCRIPT_DIR}/results/${RUN_NAME}"
PYTHON_BIN="/root/autodl-tmp/envs/pointcept/bin/python"

completed_run() {
    local val_lines=0
    [[ -f "${RUN_DIR}/val_IoUs.txt" ]] && val_lines="$(wc -l < "${RUN_DIR}/val_IoUs.txt")"
    [[ -f "${RUN_DIR}/checkpoints/current_chkp.tar" ]] && ((val_lines >= 180))
}

if completed_run; then
    printf 'Result is already complete; skipping: %s\n' "${RUN_DIR}"
    exit 0
fi
[[ ! -e "${RUN_DIR}" ]] || {
    printf 'Incomplete result directory requires manual diagnosis: %s\n' "${RUN_DIR}" >&2
    exit 75
}
[[ -f "${CHECKPOINT}" ]] || {
    printf 'Checkpoint not found: %s\n' "${CHECKPOINT}" >&2
    exit 66
}

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" experiments/S3DIS/train_S3DIS.py \
    --dataset_path /root/autodl-tmp/data/s3dis \
    --log_path "${RUN_DIR}" \
    --finetune_path "${CHECKPOINT}" \
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
    --ktha_mode concat \
    --ktha_source_stage 3 \
    --ktha_target_stages 4 \
    --ktha_relation_dim 8 \
    --ktha_shuffle_geometry 0 \
    --ktha_train_mode joint \
    --validation_mode full_identity \
    --max_epoch 180 \
    --batch_size 12 \
    --accum_batch 2 \
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
        train.checkpoint_gap=36 \
        train.cyc_lr0=0.0001 \
        train.cyc_lr1=0.0001 \
        train.cyc_raise_n=1 \
        train.cyc_plateau=0 \
        train.cyc_decrease10=60

completed_run || {
    printf 'Training returned success but expected artifacts are incomplete: %s\n' "${RUN_DIR}" >&2
    exit 76
}
