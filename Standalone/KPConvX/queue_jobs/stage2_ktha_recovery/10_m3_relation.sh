#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
"${SCRIPT_DIR}/tools/run_stage2_ktha_candidate.sh" \
    relation_bias 0 \
    s3dis_ktha_m3_relation_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803
