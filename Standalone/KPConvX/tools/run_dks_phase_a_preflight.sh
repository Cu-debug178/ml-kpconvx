#!/usr/bin/env bash

# DKS Phase-A GPU preflight: paired L0/init-identity full-room evaluation,
# followed by 100-step identity-control and learned runs. It validates artifacts,
# finite training, and that the learned gate leaves zero. It does not launch the
# 3-seed mechanism screen.

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
PYTHON_BIN="${KP_CONVX_PYTHON:-python3}"
DATASET_PATH="${S3DIS_DATASET_PATH:-<DATASET_PATH>}"
L0_RUN="${DKS_L0_RUN:-${RESULTS_DIR}/s3dis_litept_l0_b12a2_seed57106803}"
L0_CHECKPOINT="${DKS_L0_CHECKPOINT:-${L0_RUN}/checkpoints/chkp_0210.tar}"
ROOT_OUT="${DKS_PREFLIGHT_ROOT:-${RESULTS_DIR}/dks_phase_a_preflight_seed57106803}"
BASELINE_OUT="${ROOT_OUT}/baseline_full_identity"
IDENTITY_OUT="${ROOT_OUT}/init_identity_full_identity"
CONTROL_RUN="dks_preflight_l0_head_seed57106803_e1_s100"
LEARNED_RUN="dks_preflight_learned_seed57106803_e1_s100"

mkdir -p "${ROOT_OUT}"
exec >>"${ROOT_OUT}/preflight.log" 2>&1

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

[[ -x "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]] || die "Python is unavailable: ${PYTHON_BIN}"
[[ -d "${DATASET_PATH}/Area_5" ]] || die "S3DIS Area_5 is unavailable: ${DATASET_PATH}/Area_5"
[[ -f "${L0_RUN}/parameters.json" ]] || die "L0 parameters are unavailable: ${L0_RUN}"
[[ -f "${L0_CHECKPOINT}" ]] || die "L0 checkpoint is unavailable: ${L0_CHECKPOINT}"

run_eval() {
    local mode="$1" output="$2"
    "${PYTHON_BIN}" "${PROJECT_DIR}/tools/evaluate_s3dis_dks_ablation.py" \
        --source-log "${L0_RUN}" \
        --checkpoint "${L0_CHECKPOINT}" \
        --dataset-path "${DATASET_PATH}" \
        --output-dir "${output}" \
        --mode "${mode}" \
        --seed 57106803
}

log "Starting paired deterministic full-room identity check."
run_eval baseline "${BASELINE_OUT}"
run_eval init_identity "${IDENTITY_OUT}"

"${PYTHON_BIN}" - "${BASELINE_OUT}/result.json" "${IDENTITY_OUT}/result.json" <<'PY'
import json
import sys

baseline = json.load(open(sys.argv[1], encoding="utf-8"))
identity = json.load(open(sys.argv[2], encoding="utf-8"))
if baseline.get("status") != "completed" or identity.get("status") != "completed":
    raise SystemExit("paired identity evaluations are incomplete")
if baseline.get("confusion") != identity.get("confusion"):
    raise SystemExit("fresh DKS alpha=1 changed the full-room confusion matrix")
if baseline.get("miou_pct") != identity.get("miou_pct"):
    raise SystemExit("fresh DKS alpha=1 changed full-room mIoU")
print("paired identity evaluation is exactly equal")
PY

log "Starting 100-step frozen-identity control."
DKS_STEPS_PER_EPOCH=100 \
DKS_MONITOR_INTERVAL=10 \
DKS_BATCH_SIZE=24 \
DKS_ACCUM_BATCH=1 \
KP_CONVX_PYTHON="${PYTHON_BIN}" \
S3DIS_DATASET_PATH="${DATASET_PATH}" \
DKS_L0_CHECKPOINT="${L0_CHECKPOINT}" \
    "${PROJECT_DIR}/tools/run_stage2_dks_candidate.sh" \
        l0_head 57106803 "${CONTROL_RUN}" 1

log "Starting 100-step learned DKS run."
DKS_STEPS_PER_EPOCH=100 \
DKS_MONITOR_INTERVAL=10 \
DKS_BATCH_SIZE=24 \
DKS_ACCUM_BATCH=1 \
KP_CONVX_PYTHON="${PYTHON_BIN}" \
S3DIS_DATASET_PATH="${DATASET_PATH}" \
DKS_L0_CHECKPOINT="${L0_CHECKPOINT}" \
    "${PROJECT_DIR}/tools/run_stage2_dks_candidate.sh" \
        learned 57106803 "${LEARNED_RUN}" 1

"${PYTHON_BIN}" - \
    "${RESULTS_DIR}/${CONTROL_RUN}" \
    "${RESULTS_DIR}/${LEARNED_RUN}" \
    "${ROOT_OUT}/summary.json" <<'PY'
import csv
import json
import math
import os
import sys

control, learned, output = sys.argv[1:]
for run in (control, learned):
    required = [
        os.path.join(run, "val_IoUs.txt"),
        os.path.join(run, "training.txt"),
        os.path.join(run, "checkpoints", "current_chkp.tar"),
        os.path.join(run, "checkpoints", "best_val_chkp.tar"),
        os.path.join(run, "dks_alpha_stats.csv"),
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise SystemExit("preflight artifacts are missing: " + repr(missing))

with open(os.path.join(learned, "dks_alpha_stats.csv"), newline="") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit("learned DKS alpha diagnostics are empty")
last = rows[-1]
gate = float(last["gate"])
std = float(last["std"])
if not math.isfinite(gate) or abs(gate) <= 1e-8:
    raise SystemExit("learned DKS gate did not leave zero")
if not math.isfinite(std) or std <= 0.0:
    raise SystemExit("learned DKS alpha distribution stayed constant")

payload = {
    "status": "completed",
    "protocol": "paired_full_identity_plus_100_step_control_and_learned",
    "seed": 57106803,
    "control_run": os.path.basename(control),
    "learned_run": os.path.basename(learned),
    "learned_last_gate": gate,
    "learned_last_alpha_std": std,
    "limitations": [
        "This proves implementation readiness, not DKS mechanism benefit.",
        "The 3-seed fixed/random/learned screen has not run yet."
    ],
}
temporary = output + ".tmp"
with open(temporary, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.replace(temporary, output)
PY

printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" > "${ROOT_OUT}/success.tsv"
log "DKS Phase-A preflight completed successfully."
