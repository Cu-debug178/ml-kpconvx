#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${SCRIPT_DIR}/tools/run_s3dis_glskf_baseline_item.sh"
