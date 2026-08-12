#!/usr/bin/env bash

set -Eeuo pipefail

# 逻辑：先评估原始 L0 checkpoint，再固定一个已训练 GLSKF checkpoint，顺序执行 true、shuffle、room_mean、
# zero_context、neutral_gate、branch_off 六种推理干预，并写入独立结果目录。
# 使用：设置 GLSKF_SOURCE_RUN 后，在 GPU 模式执行；可用 GLSKF_CHECKPOINT、
# S3DIS_DATASET_PATH、GLSKF_ABLATION_ROOT 覆盖默认路径。
# 范围：仅用于 S3DIS Area_5 单视角 full_identity 因果检查，不用于训练。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JOBS_DIR="${SCRIPT_DIR}/queue_jobs/glskf_inference_ablations"
STATE_DIR="${GLSKF_ABLATION_QUEUE_STATE:-${SCRIPT_DIR}/results/glskf_inference_ablations_queue_20260812}"

exec "${SCRIPT_DIR}/tools/run_resilient_queue.sh" \
    --jobs-dir "${JOBS_DIR}" \
    --state-dir "${STATE_DIR}" \
    --max-consecutive-failures 2 \
    --retry 0
