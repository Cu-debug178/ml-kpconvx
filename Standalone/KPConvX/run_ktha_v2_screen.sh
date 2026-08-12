#!/usr/bin/env bash

# 运行 V2 的 true、shuffled 和同参数量 MLP 三路短程筛选。
# 每路独立记录；连续两路失败时停止。不会启动长训或关机。

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${SCRIPT_DIR}/results/queue_state/stage2_ktha_v2_screen"
export OMP_NUM_THREADS=1

exec "${SCRIPT_DIR}/tools/run_resilient_queue.sh" \
    --jobs-dir "${SCRIPT_DIR}/queue_jobs/stage2_ktha_v2_screen" \
    --state-dir "${STATE_DIR}" \
    --max-consecutive-failures 2
