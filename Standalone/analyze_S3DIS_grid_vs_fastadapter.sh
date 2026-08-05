#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the S3DIS root}"
BASELINE_LOG="${BASELINE_LOG:?Set BASELINE_LOG to the grid-only experiment directory}"
FASTADAPTER_LOG="${FASTADAPTER_LOG:?Set FASTADAPTER_LOG to the grid+FastAdapter experiment directory}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/KPConvX/results/stage_diagnostics_grid_vs_fastadapter}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INCLUDE_ADAPTER_BYPASS="${INCLUDE_ADAPTER_BYPASS:-1}"

cd "$SCRIPT_DIR/KPConvX"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  tools/analyze_stage_representations.py
  --dataset_path "$DATASET_PATH"
  --baseline_log "$BASELINE_LOG"
  --comparison_log "$FASTADAPTER_LOG"
  --baseline_name "Grid"
  --comparison_name "Grid+FastAdapter"
  --output_dir "$OUTPUT_DIR"
)
if [[ "$INCLUDE_ADAPTER_BYPASS" == "1" ]]; then
  ARGS+=(--include_adapter_bypass)
fi
if [[ -n "${BASELINE_CHECKPOINT:-}" ]]; then
  ARGS+=(--baseline_checkpoint "$BASELINE_CHECKPOINT")
fi
if [[ -n "${FASTADAPTER_CHECKPOINT:-}" ]]; then
  ARGS+=(--comparison_checkpoint "$FASTADAPTER_CHECKPOINT")
fi

exec "$PYTHON_BIN" "${ARGS[@]}" "$@"
