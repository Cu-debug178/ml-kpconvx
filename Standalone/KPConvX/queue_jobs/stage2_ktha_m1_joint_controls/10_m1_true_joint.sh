#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
"${SCRIPT_DIR}/tools/run_stage2_ktha_joint_candidate.sh" \
    "${SCRIPT_DIR}/results/s3dis_ktha_m1_concat_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803/checkpoints/best_val_chkp.tar" \
    concat 0 \
    s3dis_ktha_m1_concat_best6_joint20_bf16_b12a2_lr1e4_wd001_identity_seed57106803
