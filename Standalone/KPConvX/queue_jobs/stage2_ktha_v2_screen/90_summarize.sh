#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
TRUE_VALUES="${RESULTS_DIR}/s3dis_ktha_v2_pairwise_true_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803/val_IoUs.txt"
SHUFFLED_VALUES="${RESULTS_DIR}/s3dis_ktha_v2_pairwise_shuffled_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803/val_IoUs.txt"
MATCHED_VALUES="${RESULTS_DIR}/s3dis_ktha_v2_matched_mlp_from210_bf16_b24_lr5e3_wd001_warm10_identity_seed57106803/val_IoUs.txt"
SUMMARY="${RESULTS_DIR}/stage2_ktha_v2_warm10_screen.csv"
TMP="${SUMMARY}.tmp"

for values in "${TRUE_VALUES}" "${SHUFFLED_VALUES}" "${MATCHED_VALUES}"; do
    [[ -f "${values}" ]] || {
        printf 'missing validation metrics: %s\n' "${values}" >&2
        exit 66
    }
    [[ "$(wc -l < "${values}")" -ge 10 ]] || {
        printf 'incomplete validation metrics: %s\n' "${values}" >&2
        exit 65
    }
done

printf 'epoch,true_miou,shuffled_miou,matched_mlp_miou,true_minus_shuffled,true_minus_matched\n' > "${TMP}"
paste \
    <(awk '{s=0;for(i=1;i<=NF;i++)s+=$i;print 100*s/NF}' "${TRUE_VALUES}") \
    <(awk '{s=0;for(i=1;i<=NF;i++)s+=$i;print 100*s/NF}' "${SHUFFLED_VALUES}") \
    <(awk '{s=0;for(i=1;i<=NF;i++)s+=$i;print 100*s/NF}' "${MATCHED_VALUES}") | \
    awk '{printf "%d,%.6f,%.6f,%.6f,%+.6f,%+.6f\n",NR,$1,$2,$3,$1-$2,$1-$3}' >> "${TMP}"
mv "${TMP}" "${SUMMARY}"
printf 'summary=%s\n' "${SUMMARY}"
