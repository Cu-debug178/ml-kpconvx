#!/usr/bin/env bash
set -u

# Identity-screen saved checkpoints, then run 10-vote and fixed 13-TTA on the
# Identity-best checkpoint. There are intentionally no power-management calls.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}/Standalone/KPConvX}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/results/s3dis_litept_l0_fastadapter_b12a2_250_seed57106803}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_PID="${TRAIN_PID:?Set TRAIN_PID}"
TRAIN_START_TICKS="${TRAIN_START_TICKS:?Set TRAIN_START_TICKS}"
TRAIN_LOG="${TRAIN_LOG:-${RESULT_DIR}/resume_console.log}"
EVAL_DIR="${EVAL_DIR:-${RESULT_DIR}/posttrain_identity}"

mkdir -p "${EVAL_DIR}/logs"
exec > >(tee -a "${EVAL_DIR}/pipeline.log") 2>&1

record_exit() {
    local status=$?
    printf 'return_code=%s\nfinished_at_utc=%s\n' \
        "${status}" "$(date -u +%FT%TZ)" > "${EVAL_DIR}/pipeline_exit_status.txt"
}
trap record_exit EXIT

proc_start_ticks() {
    local raw tail
    local -a fields
    raw="$(<"/proc/${1}/stat")"
    tail="${raw##*) }"
    read -r -a fields <<< "${tail}"
    printf '%s\n' "${fields[19]}"
}

training_alive() {
    [[ -r "/proc/${TRAIN_PID}/stat" ]] || return 1
    [[ "$(proc_start_ticks "${TRAIN_PID}")" == "${TRAIN_START_TICKS}" ]] || return 1
    tr '\0' ' ' < "/proc/${TRAIN_PID}/cmdline" | grep -q 'experiments/S3DIS/train_S3DIS.py'
}

checkpoint_epoch() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(c.get("epoch", "unknown"))
PY
}

extract_miou() {
    grep -E '^\|[[:space:]]*[0-9]+(\.[0-9]+)?[[:space:]]+\|' "$1" \
        | tail -n 1 \
        | sed -E 's/^\|[[:space:]]*([0-9.]+)[[:space:]]+\|.*/\1/'
}

printf 'name,internal_epoch,return_code,mIoU,test_dir,weight,log\n' > "${EVAL_DIR}/identity_manifest.csv"
declare -A done_sha=()

evaluate_available() {
    shopt -s nullglob
    local weight name digest before after report miou epoch rc log
    for weight in "${RESULT_DIR}/checkpoints"/chkp_*.tar; do
        digest="$(sha256sum "${weight}" | awk '{print $1}')"
        [[ -n "${done_sha[${digest}]:-}" ]] && continue
        name="$(basename "${weight}" .tar)"
        epoch="$(checkpoint_epoch "${weight}")"
        before="$(find "${RESULT_DIR}/test" -maxdepth 1 -type d -name 'test_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
        log="${EVAL_DIR}/logs/${name}.log"
        printf '%s Identity start %s epoch=%s\n' "$(date -u +%FT%TZ)" "${name}" "${epoch}"
        set +e
        (cd "${PROJECT_DIR}" && KP_CONVX_PROJECT_DIR="${PROJECT_DIR}" PYTHONPATH="${PROJECT_DIR}" OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "${PYTHON_BIN}" "${ROOT_DIR}/run_s3dis_l0_identity.py" --log_path "${RESULT_DIR}" --weight_path "${weight}" --dataset_path "${DATASET_PATH}") >"${log}" 2>&1
        rc=$?
        set -e
        after="$(find "${RESULT_DIR}/test" -maxdepth 1 -type d -name 'test_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
        [[ "${after}" == "${before}" ]] && after=""
        report="${RESULT_DIR}/test/${after}/report.txt"
        miou="NA"
        [[ -f "${report}" ]] && miou="$(extract_miou "${report}")"
        printf '%s,%s,%s,%s,%s,%s,%s\n' "${name}" "${epoch}" "${rc}" "${miou}" "${after}" "${weight}" "${log}" >> "${EVAL_DIR}/identity_manifest.csv"
        done_sha["${digest}"]="${weight}"
        [[ ${rc} -eq 0 && -n "${after}" && "${miou}" != "NA" ]] || return 1
    done
}

while training_alive; do
    evaluate_available || exit 1
    sleep 30
done

sleep 20
grep -q 'Finished Training' "${TRAIN_LOG}" || exit 1
[[ -f "${RESULT_DIR}/checkpoints/chkp_0250.tar" ]] || exit 1
evaluate_available || exit 1

best_row="$(awk -F',' 'NR>1 && $3 == 0 && $4 != "NA" { if ($4 > best) { best=$4; row=$0 } } END { print row }' "${EVAL_DIR}/identity_manifest.csv")"
[[ -n "${best_row}" ]] || exit 1
best_weight="$(printf '%s\n' "${best_row}" | awk -F',' '{print $6}')"
best_epoch="$(printf '%s\n' "${best_row}" | awk -F',' '{print $2}')"
printf '%s Identity best epoch=%s weight=%s\n' "$(date -u +%FT%TZ)" "${best_epoch}" "${best_weight}"
printf '%s\n' "${best_weight}" > "${EVAL_DIR}/identity_best_weight.txt"

set +e
(cd "${PROJECT_DIR}" && KP_CONVX_PROJECT_DIR="${PROJECT_DIR}" PYTHONPATH="${PROJECT_DIR}" OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" experiments/S3DIS/test_S3DIS.py --dataset_path "${DATASET_PATH}" --log_path "${RESULT_DIR}" --weight_path "${best_weight}") >"${EVAL_DIR}/best_vote10.log" 2>&1
vote_rc=$?
(cd "${ROOT_DIR}" && KP_CONVX_PROJECT_DIR="${PROJECT_DIR}" DATASET_PATH="${DATASET_PATH}" "${PYTHON_BIN}" "${ROOT_DIR}/run_s3dis_l0_best_pointcept_tta13.py" --log_path "${RESULT_DIR}" --weight_path "${best_weight}") >"${EVAL_DIR}/best_tta13.log" 2>&1
tta_rc=$?
set -e
printf 'vote10_return_code=%s\ntta13_return_code=%s\n' "${vote_rc}" "${tta_rc}" > "${EVAL_DIR}/final_tests_status.txt"
[[ ${vote_rc} -eq 0 && ${tta_rc} -eq 0 ]] || exit 1
printf '%s\n' completed > "${EVAL_DIR}/final_tests_complete"
