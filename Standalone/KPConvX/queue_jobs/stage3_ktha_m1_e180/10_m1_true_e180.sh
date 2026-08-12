#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
"${SCRIPT_DIR}/tools/run_stage3_ktha_m1_e180.sh" \
    "${SCRIPT_DIR}/results/s3dis_ktha_m1_concat_best6_joint20_bf16_b12a2_lr1e4_wd001_identity_seed57106803/checkpoints/best_val_chkp.tar" \
    s3dis_ktha_m1_concat_from_jointbest_e180_bf16_b12a2_lr1e4_wd001_identity_seed57106803
