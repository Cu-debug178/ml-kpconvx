#!/usr/bin/env bash

# Build and run the independent 3-seed x 6-arm DKS Phase-A screen through the
# resilient queue. Failed arms keep logs/artifacts; later independent arms run
# until the queue circuit breaker detects repeated systemic failures.

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${DKS_QUEUE_STATE_DIR:-${PROJECT_DIR}/results/dks_stage2_queue}"
QUEUE_DIR="${STATE_DIR}/jobs"
QUEUE_RUNNER="${PROJECT_DIR}/tools/run_resilient_queue.sh"
EPOCHS="${DKS_SCREEN_EPOCHS:-10}"

[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'DKS_SCREEN_EPOCHS must be a positive integer\n' >&2
    exit 2
}
mkdir -p "${QUEUE_DIR}"

job_index=0
PROJECT_DIR_LITERAL="$(printf '%q' "${PROJECT_DIR}")"
MANIFEST_PATH="${STATE_DIR}/queue_manifest.tsv"
mkdir -p "${STATE_DIR}"
printf 'job\tseed\tarm\trun_name\tepochs\tprepared_at\n' > "${MANIFEST_PATH}"
for seed in 57106803 12345 98765; do
    for arm in l0_head fixed_1.0 fixed_0.8 fixed_1.15 random learned; do
        job_index=$((job_index + 1))
        job_path="${QUEUE_DIR}/$(printf '%02d' "${job_index}")_${seed}_${arm//./p}.sh"
        run_name="stage2_dks_${arm//./p}_seed${seed}_e${EPOCHS}"
        # Keep the temporary file in the jobs directory so the final rename
        # is atomic even when /tmp is a different filesystem.
        patch_file="$(mktemp "${QUEUE_DIR}/.job.XXXXXX")"
        printf '%s\n' \
            '#!/usr/bin/env bash' \
            'set -Eeuo pipefail' \
            "PROJECT_DIR=${PROJECT_DIR_LITERAL}" \
            'exec "${PROJECT_DIR}/tools/run_stage2_dks_candidate.sh" '"${arm}"' '"${seed}"' '"${run_name}"' '"${EPOCHS}" \
            > "${patch_file}"
        if [[ ! -f "${job_path}" ]] || ! cmp -s "${patch_file}" "${job_path}"; then
            # Replace the path atomically.  An in-place cp can corrupt a job
            # script that is already being interpreted by a running queue.
            chmod 0755 "${patch_file}"
            mv -f -- "${patch_file}" "${job_path}"
        fi
        rm -f -- "${patch_file}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(basename -- "${job_path}" .sh)" "${seed}" "${arm}" "${run_name}" \
            "${EPOCHS}" "$(date --iso-8601=seconds)" >> "${MANIFEST_PATH}"
    done
done

if [[ "${DKS_QUEUE_PREPARE_ONLY:-0}" == "1" ]]; then
    printf 'Prepared %s DKS jobs under %s\n' "${job_index}" "${QUEUE_DIR}"
    exit 0
fi

exec "${QUEUE_RUNNER}" \
    --jobs-dir "${QUEUE_DIR}" \
    --state-dir "${STATE_DIR}" \
    --max-consecutive-failures 2 \
    --retry 0
