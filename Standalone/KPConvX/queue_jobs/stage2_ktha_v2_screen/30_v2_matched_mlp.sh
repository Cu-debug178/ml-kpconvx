#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
"${SCRIPT_DIR}/tools/run_stage2_ktha_candidate.sh" \
    matched_mlp_v2 0 \
    s3dis_ktha_v2_matched_mlp_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803
