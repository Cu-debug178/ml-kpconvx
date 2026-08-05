#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the S3DIS root}"
BASELINE_LOG="${BASELINE_LOG:?Set BASELINE_LOG to the grid-only experiment directory}"
FASTADAPTER_LOG="${FASTADAPTER_LOG:?Set FASTADAPTER_LOG to the grid+FastAdapter experiment directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/KPConvX/results/stage_diagnostics_grid_vs_fastadapter_multi}"
NUM_RUNS="${NUM_RUNS:-10}"
BASE_SEED="${BASE_SEED:-57106803}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCENE_INDICES="${SCENE_INDICES:-}"
mkdir -p "$OUTPUT_ROOT"
if [[ -n "$SCENE_INDICES" ]]; then
  read -r -a scene_indices <<< "$SCENE_INDICES"
  NUM_RUNS="${#scene_indices[@]}"
else
  scene_indices=()
  for ((i=0; i<NUM_RUNS; i++)); do
    scene_indices+=("$i")
  done
fi
for ((i=0; i<NUM_RUNS; i++)); do
  seed=$((BASE_SEED + i))
  scene_index="${scene_indices[$i]}"
  output="$OUTPUT_ROOT/run_$(printf '%02d' "$i")_scene_${scene_index}_seed_${seed}"
  DATASET_PATH="$DATASET_PATH" \
  BASELINE_LOG="$BASELINE_LOG" \
  FASTADAPTER_LOG="$FASTADAPTER_LOG" \
  OUTPUT_DIR="$output" \
  PYTHON_BIN="$PYTHON_BIN" \
  "$SCRIPT_DIR/analyze_S3DIS_grid_vs_fastadapter.sh" \
    --seed "$seed" \
    --scene_index "$scene_index"
done
cd "$SCRIPT_DIR/KPConvX"
exec "$PYTHON_BIN" tools/aggregate_stage_diagnostics.py \
  --input_root "$OUTPUT_ROOT" \
  --output_dir "$OUTPUT_ROOT/summary"
