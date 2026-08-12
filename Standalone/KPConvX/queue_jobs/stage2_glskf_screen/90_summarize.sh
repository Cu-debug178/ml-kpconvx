#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
ROOT="${RESULTS_DIR}/glskf_stage2_screen_20260812"
"${SCRIPT_DIR}/tools/summarize_stage2_glskf_screen.py" \
    --root "${ROOT}" \
    --run l0_head "${RESULTS_DIR}/s3dis_glskf_l0_head_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803" \
    --run true "${RESULTS_DIR}/s3dis_glskf_kernel_gate_true_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803" \
    --run shuffled "${RESULTS_DIR}/s3dis_glskf_kernel_gate_shuffled_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803" \
    --run room_mean "${RESULTS_DIR}/s3dis_glskf_kernel_gate_room_mean_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803" \
    --run matched_mlp "${RESULTS_DIR}/s3dis_glskf_matched_mlp_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803"
