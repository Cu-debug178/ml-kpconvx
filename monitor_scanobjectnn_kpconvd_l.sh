#!/usr/bin/env bash

set -u

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/ml-kpconvx}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/Standalone/KPConvX/results/ScanObjectNN_KPConvD-L-official-4090D-24G}"
TRAIN_LOG="${RESULT_DIR}/training.txt"
CONSOLE_LOG="${CONSOLE_LOG:-${RESULT_DIR}.console.log}"
GPU_LOG="${RESULT_DIR}/resource_usage.csv"
STATS_LOG="${RESULT_DIR}/resource_usage_summary.csv"
MONITOR_LOG="${RESULT_DIR}/monitor.log"
ALERT_LOG="${RESULT_DIR}/monitor_alerts.log"
INTERVAL="${INTERVAL:-30}"

find_train_pid() {
    ps -eo pid=,args= | awk -v result_dir="${RESULT_DIR}" '
        /experiments\/ScanObjectNN\/train_ScanObj.py/ &&
        ! /awk/ && ! /monitor_scanobjectnn/ {
            if (index($0, result_dir) > 0) {
                print $1
                exit
            }
        }
    '
}

main_pid="${TRAIN_PID:-$(find_train_pid)}"
if [[ -z "${main_pid}" || ! -d "/proc/${main_pid}" ]]; then
    printf '%s no active ScanObjectNN training process found\n' "$(date -Is)" | tee -a "${MONITOR_LOG}"
    exit 1
fi

if [[ ! -s "${GPU_LOG}" ]]; then
    printf '%s\n' 'timestamp_utc,train_pid,train_state,phase,epoch_step,gpu_mem_used_mib,gpu_mem_total_mib,gpu_util_pct,gpu_temp_c,gpu_power_w,train_process_gpu_mem_mib,train_rss_mib,train_cpu_pct,system_mem_available_mib,disk_available_gib' > "${GPU_LOG}"
fi

if [[ ! -s "${STATS_LOG}" ]]; then
    printf '%s\n' 'updated_utc,samples,gpu_mem_avg_mib,gpu_mem_peak_mib,gpu_util_avg_pct,gpu_util_peak_pct,gpu_temp_avg_c,gpu_temp_peak_c,gpu_power_avg_w,gpu_power_peak_w,train_process_gpu_mem_avg_mib,train_process_gpu_mem_peak_mib,train_rss_avg_mib,train_rss_peak_mib,train_cpu_avg_pct,train_cpu_peak_pct,system_mem_available_min_mib,disk_available_min_gib' > "${STATS_LOG}"
fi

update_stats() {
    awk -F',' -v now="$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
        NR > 1 && $6 ~ /^[0-9.]+$/ {
            n++
            gmem += $6; util += $8; temp += $9; power += $10; pmem += $11
            rss += $12; cpu += $13
            if ($6 > gmem_peak) gmem_peak = $6
            if ($8 > util_peak) util_peak = $8
            if ($9 > temp_peak) temp_peak = $9
            if ($10 > power_peak) power_peak = $10
            if ($11 > pmem_peak) pmem_peak = $11
            if ($12 > rss_peak) rss_peak = $12
            if ($13 > cpu_peak) cpu_peak = $13
            if (n == 1 || $14 < mem_min) mem_min = $14
            if (n == 1 || $15 < disk_min) disk_min = $15
        }
        END {
            if (n > 0) {
                printf "%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                    now, n, gmem / n, gmem_peak, util / n, util_peak,
                    temp / n, temp_peak, power / n, power_peak,
                    pmem / n, pmem_peak, rss / n, rss_peak,
                    cpu / n, cpu_peak, mem_min, disk_min
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
    alerts="$(tail -n 160 "${CONSOLE_LOG}" 2>/dev/null \
        | grep -Ei 'Traceback|CUDA out of memory|The network is too big|Killed|RuntimeError:|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
        | tail -3 || true)"
    if [[ -n "${alerts}" && "${alerts}" != "${last_alerts}" ]]; then
        {
            printf '%s detected training error text:\n' "$(date -Is)"
            printf '%s\n' "${alerts}"
        } >> "${ALERT_LOG}"
        last_alerts="${alerts}"
    fi
}

last_alerts=""
printf '%s monitoring train_pid=%s interval=%ss\n' "$(date -Is)" "${main_pid}" "${INTERVAL}" | tee -a "${MONITOR_LOG}"

while [[ -d "/proc/${main_pid}" ]]; do
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    read -r train_state train_cpu train_rss_kib < <(ps -p "${main_pid}" -o stat=,%cpu=,rss= 2>/dev/null || printf 'ended 0 0\n')
    phase="$(grep -Eo 'Training epoch [0-9]+|Validation epoch [0-9]+' "${CONSOLE_LOG}" 2>/dev/null | tail -1 || true)"
    epoch_step="$(tail -1 "${TRAIN_LOG}" 2>/dev/null | awk '{print $1 ":" $2}' || true)"
    gpu_line="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
    process_gpu_mem="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v pid="${main_pid}" '$1 == pid {print $2; found=1} END {if (!found) print 0}' || true)"
    system_mem_available="$(awk '/MemAvailable:/ {printf "%.2f", $2 / 1024}' /proc/meminfo)"
    disk_available="$(df --output=avail -B1 "${PROJECT_DIR}" | tail -1 | awk '{printf "%.2f", $1 / 1073741824}')"

    if [[ -n "${gpu_line}" ]]; then
        printf '%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,%.2f,%s,%s,%s\n' \
            "${timestamp}" "${main_pid}" "${train_state}" "${phase}" "${epoch_step}" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $1}')" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $2}')" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $3}')" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $4}')" \
            "$(printf '%s' "${gpu_line}" | awk -F', ' '{print $5}')" \
            "${process_gpu_mem}" "$(awk -v rss="${train_rss_kib}" 'BEGIN {print rss / 1024}')" \
            "${train_cpu}" "${system_mem_available}" "${disk_available}" >> "${GPU_LOG}"
        update_stats
        check_alerts
    fi

    sleep "${INTERVAL}"
done

check_alerts
if grep -q 'Finished Training' "${CONSOLE_LOG}" 2>/dev/null; then
    printf '%s training process %s completed normally; monitor stopped\n' "$(date -Is)" "${main_pid}" | tee -a "${MONITOR_LOG}"
else
    printf '%s training process %s ended before normal completion; inspect console and alerts\n' "$(date -Is)" "${main_pid}" | tee -a "${MONITOR_LOG}" "${ALERT_LOG}"
fi
