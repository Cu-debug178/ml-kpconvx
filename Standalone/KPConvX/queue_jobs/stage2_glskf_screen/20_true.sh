#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
"${SCRIPT_DIR}/tools/run_stage2_glskf_candidate.sh" \
    kernel_gate none module_head \
    s3dis_glskf_kernel_gate_true_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803 10
