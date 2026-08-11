#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
QUEUE_PID="${1:?usage: summarize_stage2_ktha_queue.sh QUEUE_PID}"
QUEUE_LOG="${RESULTS_DIR}/stage2_ktha_queue_20260811.log"
SUMMARY_PATH="${RESULTS_DIR}/stage2_ktha_screen_summary_20260811.csv"

while [[ -d "/proc/${QUEUE_PID}" ]]; do
    sleep 20
done

grep -Fq "Stage-2 queue completed successfully." "${QUEUE_LOG}" || exit 1

tmp_summary="${SUMMARY_PATH}.tmp"
printf 'candidate,best_epoch,best_miou_percent,final_epoch,final_miou_percent,validation_rows,result_directory\n' > "${tmp_summary}"

summarize_run() {
    local candidate="$1"
    local run_name="$2"
    local values_path="${RESULTS_DIR}/${run_name}/val_IoUs.txt"

    awk -v candidate="${candidate}" -v run_name="${run_name}" '
        {
            total = 0
            for (i = 1; i <= NF; i++) total += $i
            miou = 100 * total / NF
            if (NR == 1 || miou > best) {
                best = miou
                best_epoch = NR
            }
            final = miou
        }
        END {
            printf "%s,%d,%.6f,%d,%.6f,%d,%s\n", candidate, best_epoch, best, NR, final, NR, run_name
        }
    ' "${values_path}" >> "${tmp_summary}"
}

summarize_run \
    M1_concat \
    s3dis_ktha_m1_concat_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803
summarize_run \
    M2_qk \
    s3dis_ktha_m2_qk_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803
summarize_run \
    M3_relation_bias \
    s3dis_ktha_m3_relation_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803
summarize_run \
    M3_relation_bias_shuffled \
    s3dis_ktha_m3_relation_shuffled_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803
summarize_run \
    matched_mlp \
    s3dis_ktha_matched_mlp_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803

mv "${tmp_summary}" "${SUMMARY_PATH}"
