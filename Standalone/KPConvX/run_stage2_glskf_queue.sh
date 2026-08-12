#!/usr/bin/env bash

set -Eeuo pipefail

# 逻辑：从 L0 epoch-210 顺序训练 GLSKF 真语义、打乱语义、房间均值语义、
# 等参数 MLP，以及只更新 L0 head 的控制；独立任务失败会留存诊断并继续。
# 使用：有 GPU 且数据/权重路径可用时执行本脚本；可用环境变量覆盖默认路径。
# 范围：仅用于 S3DIS Area_5 的 GLSKF Stage-2 短程筛选，不替代长训。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${SCRIPT_DIR}/results/glskf_stage2_queue_20260812"
JOBS_DIR="${SCRIPT_DIR}/queue_jobs/stage2_glskf_screen"

exec "${SCRIPT_DIR}/tools/run_resilient_queue.sh" \
    --jobs-dir "${JOBS_DIR}" \
    --state-dir "${STATE_DIR}" \
    --max-consecutive-failures 2 \
    --retry 0
