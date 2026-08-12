#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${GLSKF_ABLATION_ROOT:-${SCRIPT_DIR}/results/s3dis_glskf_same_checkpoint_ablations_20260812}"
PYTHON_BIN="${KP_CONVX_PYTHON:-/root/autodl-tmp/envs/pointcept/bin/python}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/tools/summarize_s3dis_glskf_ablations.py" --root "${ROOT}"
