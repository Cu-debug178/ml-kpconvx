#!/usr/bin/env bash
set -u

# Evaluate every distinct checkpoint epoch in the L0D result directory except
# the six checkpoints already covered by eval_200_250_20260806. Files with the
# same internal epoch (regular, exact archive, and current_chkp) are tested
# once only; the regular chkp_XXXX file is preferred by sort order.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/KPConvX"
RESULT_DIR="${RESULT_DIR:-${PROJECT_DIR}/results/s3dis_litept_l0d_250_seed57106803}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the S3DIS dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_DIR="${EVAL_DIR:-${RESULT_DIR}/eval_remaining_20260806}"
LOG_DIR="${EVAL_DIR}/logs"
MANIFEST="${EVAL_DIR}/manifest.csv"

mkdir -p "${LOG_DIR}"
printf 'checkpoint,internal_epoch,sha256,return_code,test_dir,full_miou,report_path,log_path\n' > "${MANIFEST}"

extract_miou() {
    "${PYTHON_BIN}" - "${1}" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
last_vote = text.rsplit("Vote ", 1)[-1]
rows = re.findall(r'^\|\s*([0-9]+(?:\.[0-9]+)?)\s+\|', last_vote, flags=re.M)
print(rows[-1] if rows else "NA")
PY
}

declare -A seen_epochs=()
for weight in "${RESULT_DIR}/checkpoints"/*.tar; do
    [[ -f "${weight}" ]] || continue
    name="$(basename "${weight}" .tar)"
    internal_epoch="$("${PYTHON_BIN}" -c \
        'import sys, torch; c=torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(c.get("epoch", "unknown"))' \
        "${weight}")"

    case ",200,210,220,230,240,250," in
        *,"${internal_epoch}",*)
            printf 'skip %s internal_epoch=%s (already in six-checkpoint evaluation)\n' "${name}" "${internal_epoch}"
            continue
            ;;
    esac
    if [[ -n "${seen_epochs[${internal_epoch}]:-}" ]]; then
        printf 'skip %s internal_epoch=%s (duplicate of %s)\n' \
            "${name}" "${internal_epoch}" "${seen_epochs[${internal_epoch}]}"
        continue
    fi
    seen_epochs["${internal_epoch}"]="${name}"

    log_path="${LOG_DIR}/${name}.log"
    sha256="$(sha256sum "${weight}" | awk '{print $1}')"
    before="$(find "${RESULT_DIR}/test" -maxdepth 1 -type d -name 'test_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
    (
        cd "${PROJECT_DIR}" || exit 1
        export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
        export OMP_NUM_THREADS=1
        export CUDA_VISIBLE_DEVICES=0
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
        exec "${PYTHON_BIN}" experiments/S3DIS/test_S3DIS.py \
            --dataset_path "${DATASET_PATH}" \
            --log_path "${RESULT_DIR}" \
            --weight_path "${weight}"
    ) > "${log_path}" 2>&1
    return_code=$?

    after="$(find "${RESULT_DIR}/test" -maxdepth 1 -type d -name 'test_*' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
    [[ "${after}" == "${before}" ]] && after=""
    report_path="${RESULT_DIR}/test/${after}/report.txt"
    full_miou="NA"
    if [[ -n "${after}" && -f "${report_path}" ]]; then
        full_miou="$(extract_miou "${report_path}")"
    else
        report_path=""
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${name}" "${internal_epoch}" "${sha256}" "${return_code}" \
        "${after}" "${full_miou}" "${report_path}" "${log_path}" >> "${MANIFEST}"
    printf '%s return_code=%s test_dir=%s full_mIoU=%s\n' \
        "${name}" "${return_code}" "${after}" "${full_miou}"
done

touch "${EVAL_DIR}/evaluation_complete"
printf 'evaluation_complete=%s\n' "${EVAL_DIR}/evaluation_complete"
