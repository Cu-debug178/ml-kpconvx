#!/usr/bin/env bash

# 逻辑：从随机初始化联合训练 L0 + true-context GLSKF 180 epoch，完成后执行 10-vote。
# 使用：GPU 空闲后直接运行；可用 S3DIS_DATASET_PATH、KP_CONVX_PYTHON 覆盖路径。
# 范围：S3DIS Area_5 的 B1 探索实验；不加载已有 checkpoint，不是严格配对 B0。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
RUN_NAME="s3dis_glskf_b1_true_scratch_e180_bf16_b12a2_lr5e3_wd001_seed57106803"
RUN_DIR="${RESULTS_DIR}/${RUN_NAME}"
PYTHON_BIN="${KP_CONVX_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
DATASET_PATH="${S3DIS_DATASET_PATH:-/root/autodl-tmp/data/s3dis}"

completed_run() {
    local val_lines=0
    [[ -f "${RUN_DIR}/val_IoUs.txt" ]] &&
        val_lines="$(wc -l < "${RUN_DIR}/val_IoUs.txt")"
    [[ -f "${RUN_DIR}/checkpoints/current_chkp.tar" ]] &&
        [[ -f "${RUN_DIR}/checkpoints/best_val_chkp.tar" ]] &&
        ((val_lines >= 180)) &&
        [[ -f "${RUN_DIR}/test/test_001/report.txt" ]]
}

if completed_run; then
    printf 'B1 scratch result is already complete; skipping: %s\n' "${RUN_DIR}"
    exit 0
fi
[[ ! -e "${RUN_DIR}" ]] || {
    printf 'Incomplete B1 result requires manual diagnosis: %s\n' "${RUN_DIR}" >&2
    exit 75
}
[[ -x "${PYTHON_BIN}" ]] || {
    printf 'Python executable is unavailable: %s\n' "${PYTHON_BIN}" >&2
    exit 66
}
[[ -d "${DATASET_PATH}/Area_5" ]] || {
    printf 'S3DIS Area_5 directory not found: %s\n' "${DATASET_PATH}/Area_5" >&2
    exit 66
}

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" experiments/S3DIS/train_S3DIS.py \
    --dataset_path "${DATASET_PATH}" \
    --log_path "${RUN_DIR}" \
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
    --ktha_mode none \
    --glskf_mode kernel_gate \
    --glskf_refine_stage 3 \
    --glskf_context_stages 4,5 \
    --glskf_groups 8 \
    --glskf_hidden_dim 64 \
    --glskf_context_control none \
    --glskf_train_mode joint \
    --validation_mode full_identity \
    --max_epoch 180 \
    --batch_size 12 \
    --accum_batch 2 \
    --amp_enabled 1 \
    --amp_dtype bfloat16 \
    --weight_decay 0.01 \
    --auto_test_vote10 1 \
    --options \
        train.steps_per_epoch=300 \
        train.num_workers=10 \
        train.monitor_enabled=True \
        train.monitor_interval=25 \
        train.save_best_val=True \
        train.save_latest_val=True \
        train.save_best_val_cycle=False \
        train.checkpoint_start=90 \
        train.checkpoint_gap=10 \
        train.cyc_lr0=0.005 \
        train.cyc_lr1=0.005 \
        train.cyc_raise_n=1 \
        train.cyc_plateau=0 \
        train.cyc_decrease10=60

completed_run || {
    printf 'B1 command returned success but required artifacts are incomplete: %s\n' "${RUN_DIR}" >&2
    exit 76
}
