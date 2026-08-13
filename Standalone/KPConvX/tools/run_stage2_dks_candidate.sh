#!/usr/bin/env bash

# Run one independent DKS warm-start arm. Existing completed runs are skipped;
# incomplete result directories are preserved for diagnosis and never overwritten.

set -Eeuo pipefail

if (($# != 4)); then
    printf 'usage: %s ARM SEED RUN_NAME EXPECTED_EPOCHS\n' "$0" >&2
    printf 'ARM: l0_head | fixed_1.0 | fixed_0.8 | fixed_1.15 | random | learned\n' >&2
    exit 2
fi

ARM="$1"
SEED="$2"
RUN_NAME="$3"
EXPECTED_EPOCHS="$4"
STEPS_PER_EPOCH="${DKS_STEPS_PER_EPOCH:-300}"
BATCH_SIZE="${DKS_BATCH_SIZE:-24}"
ACCUM_BATCH="${DKS_ACCUM_BATCH:-1}"
MONITOR_INTERVAL="${DKS_MONITOR_INTERVAL:-50}"

case "${ARM}" in
    l0_head) MODE=learned; TRAIN_MODE=head_only; FIXED_ALPHA=1.0 ;;
    fixed_1.0) MODE=fixed; TRAIN_MODE=head_only; FIXED_ALPHA=1.0 ;;
    fixed_0.8) MODE=fixed; TRAIN_MODE=head_only; FIXED_ALPHA=0.8 ;;
    fixed_1.15) MODE=fixed; TRAIN_MODE=head_only; FIXED_ALPHA=1.15 ;;
    random) MODE=random; TRAIN_MODE=head_only; FIXED_ALPHA=1.0 ;;
    learned) MODE=learned; TRAIN_MODE=module_head; FIXED_ALPHA=1.0 ;;
    *) printf 'unsupported DKS arm: %s\n' "${ARM}" >&2; exit 2 ;;
esac
[[ "${SEED}" =~ ^[0-9]+$ ]] || { printf 'SEED must be an integer\n' >&2; exit 2; }
[[ "${EXPECTED_EPOCHS}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'EXPECTED_EPOCHS must be a positive integer\n' >&2
    exit 2
}
for pair in \
    "DKS_STEPS_PER_EPOCH:${STEPS_PER_EPOCH}" \
    "DKS_BATCH_SIZE:${BATCH_SIZE}" \
    "DKS_ACCUM_BATCH:${ACCUM_BATCH}" \
    "DKS_MONITOR_INTERVAL:${MONITOR_INTERVAL}"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        printf '%s must be a positive integer\n' "${name}" >&2
        exit 2
    }
done

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
RUN_DIR="${RESULTS_DIR}/${RUN_NAME}"
# Prefer the environment used by the completed DKS/L0 runs.  Set
# KP_CONVX_PYTHON explicitly when reproducing on another machine.
if [[ -n "${KP_CONVX_PYTHON:-}" ]]; then
    PYTHON_BIN="${KP_CONVX_PYTHON}"
elif [[ -x "/root/autodl-tmp/envs/pointcept/bin/python" ]]; then
    PYTHON_BIN="/root/autodl-tmp/envs/pointcept/bin/python"
else
    PYTHON_BIN="python3"
fi
DATASET_PATH="${S3DIS_DATASET_PATH:-<DATASET_PATH>}"
FINETUNE_PATH="${DKS_L0_CHECKPOINT:-<L0_EPOCH210_CHECKPOINT>}"

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
    printf 'Incomplete result directory requires manual diagnosis: %s\n' "${RUN_DIR}" >&2
    exit 75
}
[[ -x "${PYTHON_BIN}" ]] || { printf 'Python not found: %s\n' "${PYTHON_BIN}" >&2; exit 66; }
"${PYTHON_BIN}" -c 'import torch, easydict' >/dev/null 2>&1 || {
    printf 'Python environment lacks required torch/easydict imports: %s\n' "${PYTHON_BIN}" >&2
    exit 66
}
[[ -d "${DATASET_PATH}/Area_5" ]] || { printf 'S3DIS Area_5 not found\n' >&2; exit 66; }
[[ -f "${FINETUNE_PATH}" ]] || { printf 'L0 checkpoint not found: %s\n' "${FINETUNE_PATH}" >&2; exit 66; }

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" experiments/S3DIS/train_S3DIS.py \
    --dataset_path "${DATASET_PATH}" \
    --log_path "${RUN_DIR}" \
    --finetune_path "${FINETUNE_PATH}" \
    --seed "${SEED}" \
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
    --dks_mode "${MODE}" \
    --dks_stages 3 \
    --dks_hidden_dim 32 \
    --dks_alpha_min 0.5 \
    --dks_alpha_max 1.2 \
    --dks_fixed_alpha "${FIXED_ALPHA}" \
    --dks_train_mode "${TRAIN_MODE}" \
    --dks_log_stats 1 \
    --validation_mode full_identity \
    --max_epoch "${EXPECTED_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --accum_batch "${ACCUM_BATCH}" \
    --amp_enabled 1 \
    --amp_dtype bfloat16 \
    --weight_decay 0.01 \
    --auto_test_vote10 0 \
    --options \
        train.steps_per_epoch="${STEPS_PER_EPOCH}" \
        train.num_workers=10 \
        train.monitor_enabled=True \
        train.monitor_interval="${MONITOR_INTERVAL}" \
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
    printf 'Training returned success but artifacts are incomplete: %s\n' "${RUN_DIR}" >&2
    exit 76
}
