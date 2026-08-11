#!/usr/bin/env bash

# 逻辑：按文件名顺序执行 jobs-dir 下的 *.sh。单项失败时记录日志并继续；
# 连续失败达到阈值时停止。复用 state-dir 可自动跳过已成功任务。
#
# 用法：
#   tools/run_resilient_queue.sh \
#       --jobs-dir /path/to/jobs \
#       --state-dir /path/to/runtime-state
#
# 适用：相互独立、可在部分失败后继续的任务队列。
# 有依赖关系的流水线使用 --fail-fast，或拆分为独立队列。
#
# 选项：
#   --fail-fast                    首次失败即停止。
#   --max-consecutive-failures N   连续失败 N 次后停止，默认 2；0 表示不限制。
#   --retry N                      每项失败后重试 N 次，默认 0。

set -uo pipefail

JOBS_DIR=""
STATE_DIR=""
FAIL_FAST=0
MAX_CONSECUTIVE_FAILURES=2
RETRY_COUNT=0
QUEUE_INTERRUPTED=0

usage() {
    sed -n '3,18p' "$0" >&2
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

while (($# > 0)); do
    case "$1" in
        --jobs-dir)
            (($# >= 2)) || die "--jobs-dir requires a value"
            JOBS_DIR="$2"
            shift 2
            ;;
        --state-dir)
            (($# >= 2)) || die "--state-dir requires a value"
            STATE_DIR="$2"
            shift 2
            ;;
        --fail-fast)
            FAIL_FAST=1
            shift
            ;;
        --max-consecutive-failures)
            (($# >= 2)) || die "--max-consecutive-failures requires a value"
            MAX_CONSECUTIVE_FAILURES="$2"
            shift 2
            ;;
        --retry)
            (($# >= 2)) || die "--retry requires a value"
            RETRY_COUNT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "${JOBS_DIR}" ]] || die "--jobs-dir is required"
[[ -n "${STATE_DIR}" ]] || die "--state-dir is required"
[[ -d "${JOBS_DIR}" ]] || die "jobs directory does not exist: ${JOBS_DIR}"
is_nonnegative_integer "${MAX_CONSECUTIVE_FAILURES}" ||
    die "--max-consecutive-failures must be a non-negative integer"
is_nonnegative_integer "${RETRY_COUNT}" ||
    die "--retry must be a non-negative integer"

mkdir -p "${STATE_DIR}/logs" "${STATE_DIR}/diagnostics" "${STATE_DIR}/markers"

QUEUE_LOG="${STATE_DIR}/queue.log"
STATUS_TSV="${STATE_DIR}/status.tsv"

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${QUEUE_LOG}"
}

append_status() {
    local job_name="$1"
    local status="$2"
    local exit_code="$3"
    local attempt="$4"
    local start_time="$5"
    local end_time="$6"
    local duration_seconds="$7"

    if [[ ! -f "${STATUS_TSV}" ]]; then
        printf 'job\tstatus\texit_code\tattempt\tstarted_at\tended_at\tduration_seconds\n' > "${STATUS_TSV}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${job_name}" "${status}" "${exit_code}" "${attempt}" \
        "${start_time}" "${end_time}" "${duration_seconds}" >> "${STATUS_TSV}"
}

capture_failure_diagnostics() {
    local job_name="$1"
    local exit_code="$2"
    local job_path="$3"
    local job_log="$4"
    local diagnostic_path="$5"

    {
        printf 'captured_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'job=%s\n' "${job_name}"
        printf 'job_path=%s\n' "${job_path}"
        printf 'exit_code=%s\n' "${exit_code}"
        printf '\n[last 200 job-log lines]\n'
        tail -n 200 "${job_log}" 2>&1 || true
        printf '\n[nvidia-smi summary]\n'
        nvidia-smi 2>&1 || true
        printf '\n[GPU compute processes]\n'
        nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>&1 || true
        printf '\n[memory]\n'
        free -h 2>&1 || true
        printf '\n[disk usage]\n'
        df -h "${JOBS_DIR}" "${STATE_DIR}" 2>&1 || true
    } > "${diagnostic_path}"
}

on_signal() {
    QUEUE_INTERRUPTED=1
    log "Queue received a termination signal; the active child will be allowed to report its exit status."
}

trap on_signal INT TERM HUP

mapfile -d '' JOB_PATHS < <(
    find "${JOBS_DIR}" -maxdepth 1 -type f -name '*.sh' -print0 | sort -z
)
((${#JOB_PATHS[@]} > 0)) || die "no *.sh jobs found in ${JOBS_DIR}"

queue_failures=0
consecutive_failures=0
circuit_open=0

log "Queue started: jobs=${#JOB_PATHS[@]}, fail_fast=${FAIL_FAST}, max_consecutive_failures=${MAX_CONSECUTIVE_FAILURES}, retry=${RETRY_COUNT}."

for job_path in "${JOB_PATHS[@]}"; do
    job_file="$(basename -- "${job_path}")"
    job_name="${job_file%.sh}"
    success_marker="${STATE_DIR}/markers/${job_name}.success"
    failure_marker="${STATE_DIR}/markers/${job_name}.failed"
    job_log="${STATE_DIR}/logs/${job_name}.log"

    if [[ -f "${success_marker}" ]]; then
        log "SKIP ${job_name}: success marker already exists."
        continue
    fi

    [[ -x "${job_path}" ]] || {
        log "FAIL ${job_name}: job file is not executable."
        printf 'exit_code=126\nfailed_at=%s\n' "$(date --iso-8601=seconds)" > "${failure_marker}"
        append_status "${job_name}" failed 126 0 "$(date --iso-8601=seconds)" "$(date --iso-8601=seconds)" 0
        ((queue_failures++))
        ((consecutive_failures++))
        if ((FAIL_FAST == 1)) ||
            ((MAX_CONSECUTIVE_FAILURES > 0 && consecutive_failures >= MAX_CONSECUTIVE_FAILURES)); then
            circuit_open=1
            break
        fi
        continue
    }

    job_succeeded=0
    for ((attempt = 1; attempt <= RETRY_COUNT + 1; attempt++)); do
        start_time="$(date --iso-8601=seconds)"
        start_seconds="$(date +%s)"
        log "START ${job_name}: attempt ${attempt}/$((RETRY_COUNT + 1))."

        bash "${job_path}" >> "${job_log}" 2>&1
        exit_code=$?

        end_seconds="$(date +%s)"
        end_time="$(date --iso-8601=seconds)"
        duration_seconds=$((end_seconds - start_seconds))

        if ((exit_code == 0)); then
            append_status "${job_name}" success 0 "${attempt}" \
                "${start_time}" "${end_time}" "${duration_seconds}"
            printf 'completed_at=%s\nattempt=%s\nduration_seconds=%s\n' \
                "${end_time}" "${attempt}" "${duration_seconds}" > "${success_marker}"
            rm -f -- "${failure_marker}"
            log "SUCCESS ${job_name}: ${duration_seconds}s."
            job_succeeded=1
            consecutive_failures=0
            break
        fi

        append_status "${job_name}" failed "${exit_code}" "${attempt}" \
            "${start_time}" "${end_time}" "${duration_seconds}"
        diagnostic_path="${STATE_DIR}/diagnostics/${job_name}.attempt-${attempt}.txt"
        capture_failure_diagnostics \
            "${job_name}" "${exit_code}" "${job_path}" "${job_log}" "${diagnostic_path}"
        printf 'exit_code=%s\nfailed_at=%s\nattempt=%s\ndiagnostic=%s\n' \
            "${exit_code}" "${end_time}" "${attempt}" "${diagnostic_path}" > "${failure_marker}"
        log "FAIL ${job_name}: exit=${exit_code}, diagnostics=${diagnostic_path}."
    done

    if ((job_succeeded == 0)); then
        ((queue_failures++))
        ((consecutive_failures++))
        if ((FAIL_FAST == 1)); then
            log "Stopping because --fail-fast is enabled."
            circuit_open=1
            break
        fi
        if ((MAX_CONSECUTIVE_FAILURES > 0 && consecutive_failures >= MAX_CONSECUTIVE_FAILURES)); then
            log "Circuit breaker opened after ${consecutive_failures} consecutive failures."
            circuit_open=1
            break
        fi
        log "Continuing to the next independent job after ${job_name} failed."
    fi

    if ((QUEUE_INTERRUPTED == 1)); then
        log "Stopping queue after signal handling."
        circuit_open=1
        break
    fi
done

if ((circuit_open == 1)); then
    log "Queue stopped early: failures=${queue_failures}, consecutive_failures=${consecutive_failures}."
    exit 1
fi

if ((queue_failures > 0)); then
    log "Queue finished all runnable jobs with ${queue_failures} failed job(s)."
    exit 1
fi

log "Queue completed successfully."
