#!/usr/bin/env bash
set -u

# Exhaustive, serial evaluation of every distinct retained L0 checkpoint.
# Runtime outputs stay under Standalone/KPConvX/results and are not versioned.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ROOT_DIR}/Standalone/KPConvX"
RESULT_DIR="${PROJECT_DIR}/results/s3dis_litept_l0_b12a2_seed57106803"
CHECKPOINT_DIR="${RESULT_DIR}/checkpoints"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the S3DIS dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DIR="${RUN_DIR:-${RESULT_DIR}/test_runs_all_20260805}"
LOG_DIR="${RUN_DIR}/logs"
MANIFEST="${RUN_DIR}/manifest.csv"

mkdir -p "${LOG_DIR}"
if [[ ! -s "${MANIFEST}" ]]; then
    printf 'name,checkpoint,internal_epoch,sha256,return_code,test_dir,full_miou,log_path\n' > "${MANIFEST}"
fi

extract_miou() {
    local report="$1"
    "${PYTHON_BIN}" - "${report}" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors='replace')
last_vote = text.rsplit("Vote ", 1)[-1]
rows = re.findall(r'^\|\s*([0-9]+(?:\.[0-9]+)?)\s+\|', last_vote, flags=re.M)
print(rows[-1] if rows else "NA")
PY
}

shopt -s nullglob
declare -a weights=()
declare -A seen_sha=()
for checkpoint in "${CHECKPOINT_DIR}"/*.tar; do
    [[ -f "${checkpoint}" ]] || continue
    sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
    if [[ -n "${seen_sha[${sha256}]:-}" ]]; then
        printf '[%s] duplicate checkpoint skipped: %s (same as %s)\n' "$(date -Is)" "${checkpoint}" "${seen_sha[${sha256}]}"
        continue
    fi
    seen_sha["${sha256}"]="${checkpoint}"
    weights+=("${checkpoint}")
done

for checkpoint in "${weights[@]}"; do
    name="$(basename "${checkpoint}" .tar)"
    sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
    if awk -F',' -v target="${checkpoint}" '$2 == target && $5 == 0 { found=1 } END { exit !found }' "${MANIFEST}"; then
        printf '[%s] already passed, skipping: %s\n' "$(date -Is)" "${name}"
        continue
    fi
    internal_epoch="$(${PYTHON_BIN} - "${checkpoint}" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(checkpoint.get('epoch', 'unknown'))
PY
)"
    before_test_dir="$(find "${RESULT_DIR}/test" -maxdepth 1 -type d -name 'test_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
    log_path="${LOG_DIR}/${name}.log"
    printf '[%s] testing %s (internal epoch %s)\n' "$(date -Is)" "${name}" "${internal_epoch}" | tee "${log_path}"
    set +e
    (
        cd "${PROJECT_DIR}" || exit 1
        export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
        export OMP_NUM_THREADS=1
        export CUDA_VISIBLE_DEVICES=0
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
        exec "${PYTHON_BIN}" experiments/S3DIS/test_S3DIS.py \
            --dataset_path "${DATASET_PATH}" \
            --log_path "${RESULT_DIR}" \
            --weight_path "${checkpoint}"
    ) >> "${log_path}" 2>&1
    return_code=$?
    set -e
    after_test_dir="$(find "${RESULT_DIR}/test" -maxdepth 1 -type d -name 'test_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
    [[ "${after_test_dir}" == "${before_test_dir}" ]] && after_test_dir=""
    full_miou="NA"
    report_path="${RESULT_DIR}/test/${after_test_dir}/report.txt"
    if [[ -n "${after_test_dir}" && -f "${report_path}" ]]; then
        full_miou="$(extract_miou "${report_path}")"
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${name}" "${checkpoint}" "${internal_epoch}" "${sha256}" \
        "${return_code}" "${after_test_dir}" "${full_miou}" "${log_path}" >> "${MANIFEST}"
    printf '[%s] finished %s return_code=%s test_dir=%s full_mIoU=%s\n' \
        "$(date -Is)" "${name}" "${return_code}" "${after_test_dir}" "${full_miou}" | tee -a "${log_path}"
done

printf '[%s] exhaustive checkpoint test suite finished (%s distinct files)\n' "$(date -Is)" "${#weights[@]}" | tee "${RUN_DIR}/suite_finished.txt"
