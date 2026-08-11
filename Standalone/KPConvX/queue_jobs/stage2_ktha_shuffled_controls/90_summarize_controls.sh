#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
SUMMARY_PATH="${RESULTS_DIR}/stage2_ktha_shuffled_controls_20260811.csv"
TMP_PATH="${SUMMARY_PATH}.tmp"

printf 'candidate,true_mean,true_best,true_best_epoch,shuffled_mean,shuffled_best,shuffled_best_epoch,paired_mean_diff\n' > "${TMP_PATH}"

summarize_pair() {
    local candidate="$1"
    local true_run="$2"
    local shuffled_run="$3"
    local true_values="${RESULTS_DIR}/${true_run}/val_IoUs.txt"
    local shuffled_values="${RESULTS_DIR}/${shuffled_run}/val_IoUs.txt"

    [[ "$(wc -l < "${true_values}")" -ge 10 ]]
    [[ "$(wc -l < "${shuffled_values}")" -ge 10 ]]

    paste \
        <(awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print 100*s/NF}' "${true_values}") \
        <(awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print 100*s/NF}' "${shuffled_values}") | \
        awk -v candidate="${candidate}" '
            {
                true_sum += $1
                shuffled_sum += $2
                diff_sum += $1 - $2
                if (NR == 1 || $1 > true_best) {
                    true_best = $1
                    true_epoch = NR
                }
                if (NR == 1 || $2 > shuffled_best) {
                    shuffled_best = $2
                    shuffled_epoch = NR
                }
            }
            END {
                printf "%s,%.6f,%.6f,%d,%.6f,%.6f,%d,%+.6f\n", \
                    candidate, true_sum/NR, true_best, true_epoch, \
                    shuffled_sum/NR, shuffled_best, shuffled_epoch, diff_sum/NR
            }
        ' >> "${TMP_PATH}"
}

summarize_pair \
    M1_concat \
    s3dis_ktha_m1_concat_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803 \
    s3dis_ktha_m1_concat_shuffled_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803
summarize_pair \
    M2_qk \
    s3dis_ktha_m2_qk_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803 \
    s3dis_ktha_m2_qk_shuffled_from210_bf16_b24_paperopt5e3_wd001_warm10_identity_seed57106803

mv "${TMP_PATH}" "${SUMMARY_PATH}"
