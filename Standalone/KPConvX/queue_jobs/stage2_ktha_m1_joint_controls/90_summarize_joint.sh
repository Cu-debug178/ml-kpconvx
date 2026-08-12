#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
TRUE_VALUES="${RESULTS_DIR}/s3dis_ktha_m1_concat_best6_joint20_bf16_b12a2_lr1e4_wd001_identity_seed57106803/val_IoUs.txt"
SHUFFLED_VALUES="${RESULTS_DIR}/s3dis_ktha_m1_concat_shuffled_best6_joint20_bf16_b12a2_lr1e4_wd001_identity_seed57106803/val_IoUs.txt"
SUMMARY="${RESULTS_DIR}/stage2_ktha_m1_joint20_control_20260811.csv"
TMP="${SUMMARY}.tmp"

[[ "$(wc -l < "${TRUE_VALUES}")" -ge 20 ]]
[[ "$(wc -l < "${SHUFFLED_VALUES}")" -ge 20 ]]
printf 'epoch,true_miou,shuffled_miou,true_minus_shuffled\n' > "${TMP}"
paste \
    <(awk '{s=0;for(i=1;i<=NF;i++)s+=$i;print 100*s/NF}' "${TRUE_VALUES}") \
    <(awk '{s=0;for(i=1;i<=NF;i++)s+=$i;print 100*s/NF}' "${SHUFFLED_VALUES}") | \
    awk '{printf "%d,%.6f,%.6f,%+.6f\n",NR,$1,$2,$1-$2}' >> "${TMP}"
mv "${TMP}" "${SUMMARY}"
