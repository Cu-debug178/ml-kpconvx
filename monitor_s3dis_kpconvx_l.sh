#!/usr/bin/env bash

set -u

PROJECT_DIR="/root/autodl-tmp/ml-kpconvx"
RESULT_DIR="${PROJECT_DIR}/Standalone/KPConvX/results/S3DIS_KPConvX-L-4090D-24G"
TRAIN_LOG="${RESULT_DIR}/training.txt"
SCREEN_LOG="${SCREEN_LOG:-${RESULT_DIR}/monitor.log}"
GPU_LOG="${RESULT_DIR}/gpu_usage.csv"
STATS_LOG="${RESULT_DIR}/gpu_usage_summary.csv"
MONITOR_LOG="${RESULT_DIR}/monitor.log"
ALERT_LOG="${RESULT_DIR}/monitor_alerts.log"
INTERVAL="${INTERVAL:-30}"

find_train_pid() {
    ps -eo pid=,ppid=,args= | awk '
        /experiments\/S3DIS\/train_S3DIS.py/ &&
        /S3DIS_KPConvX-L-4090D-24G/ &&
        ! /awk/ && ! /monitor_s3dis/ {
            if ($2 == 1 || parent == "") {
                parent = $1
            }
        }
        END { if (parent != "") print parent }
    '
}

main_pid="${TRAIN_PID:-}"
if [[ -z "${main_pid}" ]]; then
    main_pid="$(find_train_pid)"
fi

if [[ -z "${main_pid}" || ! -d "/proc/${main_pid}" ]]; then
    printf '%s no active S3DIS training process found\n' "$(date -Is)" | tee -a "${MONITOR_LOG}"
    exit 1
fi

if [[ ! -s "${GPU_LOG}" ]]; then
    printf 'timestamp_utc,train_pid,train_state,phase,epoch_step, gpu_index,gpu_name,gpu_mem_used_mib,gpu_mem_total_mib,gpu_util_pct,gpu_temp_c,gpu_power_w,train_process_gpu_mem_mib\n' > "${GPU_LOG}"
fi

if [[ ! -s "${STATS_LOG}" ]]; then
    printf 'updated_utc,samples,gpu_mem_avg_mib,gpu_mem_peak_mib,gpu_util_avg_pct,gpu_util_peak_pct,gpu_temp_avg_c,gpu_temp_peak_c,gpu_power_avg_w,gpu_power_peak_w,train_process_gpu_mem_avg_mib,train_process_gpu_mem_peak_mib\n' > "${STATS_LOG}"
fi

update_stats() {
    awk -F',' -v now="$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
        NR > 1 && $8 ~ /^[0-9.]+$/ {
            n++
            mem += $8; util += $10; temp += $11; power += $12; proc += $13
            if ($8 > mem_peak) mem_peak = $8
            if ($10 > util_peak) util_peak = $10
            if ($11 > temp_peak) temp_peak = $11
            if ($12 > power_peak) power_peak = $12
            if ($13 > proc_peak) proc_peak = $13
        }
        END {
            if (n > 0) {
                printf "%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                    now, n, mem / n, mem_peak, util / n, util_peak,
                    temp / n, temp_peak, power / n, power_peak,
                    proc / n, (proc_peak > 0 ? proc_peak : 0)
            }
        }
    ' "${GPU_LOG}" > "${STATS_LOG}.tmp"
    if [[ -s "${STATS_LOG}.tmp" ]]; then
        { head -1 "${STATS_LOG}"; cat "${STATS_LOG}.tmp"; } > "${STATS_LOG}.new"
        mv "${STATS_LOG}.new" "${STATS_LOG}"
    fi
    rm -f "${STATS_LOG}.tmp"
}

check_alerts() {
    local alerts
    alerts="$(tail -n 120 "${SCREEN_LOG}" 2>/dev/null | grep -Ei 'Traceback|CUDA out of memory|The network is too big|Killed|RuntimeError:' | tail -3 || true)"
    if [[ -n "${alerts}" ]]; then
        {
            printf '%s detected training error text:\n' "$(date -Is)"
            printf '%s\n' "${alerts}"
        } >> "${ALERT_LOG}"
    fi
}

printf '%s monitoring train_pid=%s interval=%ss\n' "$(date -Is)" "${main_pid}" "${INTERVAL}" | tee -a "${MONITOR_LOG}"

while [[ -d "/proc/${main_pid}" ]]; do
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    train_state="$(ps -p "${main_pid}" -o stat= 2>/dev/null | tr -d ' ' || true)"
    phase="$(grep -Eo 'Training epoch [0-9]+|Validation epoch [0-9]+' "${SCREEN_LOG}" 2>/dev/null | tail -1 || true)"
    if [[ -z "${phase}" ]]; then
        phase="$(grep -Eo 'Training epoch [0-9]+|Validation epoch [0-9]+' "${SCREEN_LOG}" 2>/dev/null | tail -1 || true)"
    fi
    epoch_step="$(tail -1 "${TRAIN_LOG}" 2>/dev/null | awk '{print $1 ":" $2}' || true)"
    gpu_line="$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
    process_gpu_mem="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v pid="${main_pid}" '$1 == pid {print $2; found=1} END {if (!found) print 0}' || true)"

    if [[ -n "${gpu_line}" ]]; then
        printf '%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${timestamp}" "${main_pid}" "${train_state}" "${phase}" "${epoch_step}" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $1}')" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $2}')" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $3}')" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $4}')" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $5}')" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $6}')" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $7}')" \
            "${process_gpu_mem}" >> "${GPU_LOG}"
        update_stats
        check_alerts
        printf '%s sample phase=%s epoch_step=%s\n' "$(date -Is)" "${phase:-unknown}" "${epoch_step:-unknown}" >> "${MONITOR_LOG}"
    fi

    sleep "${INTERVAL}"
done

printf '%s training process %s ended; monitor stopped\n' "$(date -Is)" "${main_pid}" | tee -a "${MONITOR_LOG}"
