#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/../tools/run_resilient_queue.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${TEST_ROOT}"' EXIT

make_job() {
    local path="$1"
    local body="$2"
    printf '#!/usr/bin/env bash\n%s\n' "${body}" > "${path}"
    chmod +x "${path}"
}

continue_jobs="${TEST_ROOT}/continue-jobs"
continue_state="${TEST_ROOT}/continue-state"
mkdir -p "${continue_jobs}"
make_job "${continue_jobs}/01_success.sh" 'echo first >> "${TEST_OUTPUT}"'
make_job "${continue_jobs}/02_failure.sh" 'echo failed >&2; exit 7'
make_job "${continue_jobs}/03_success.sh" 'echo third >> "${TEST_OUTPUT}"'

export TEST_OUTPUT="${TEST_ROOT}/continue-output.txt"
set +e
"${RUNNER}" \
    --jobs-dir "${continue_jobs}" \
    --state-dir "${continue_state}" \
    --max-consecutive-failures 2
continue_rc=$?
set -e

[[ "${continue_rc}" -eq 1 ]]
grep -Fxq first "${TEST_OUTPUT}"
grep -Fxq third "${TEST_OUTPUT}"
[[ -f "${continue_state}/markers/01_success.success" ]]
[[ -f "${continue_state}/markers/02_failure.failed" ]]
[[ -f "${continue_state}/markers/03_success.success" ]]
[[ -f "${continue_state}/diagnostics/02_failure.attempt-1.txt" ]]

circuit_jobs="${TEST_ROOT}/circuit-jobs"
circuit_state="${TEST_ROOT}/circuit-state"
mkdir -p "${circuit_jobs}"
make_job "${circuit_jobs}/01_failure.sh" 'exit 8'
make_job "${circuit_jobs}/02_should_not_run.sh" 'echo unexpected > "${CIRCUIT_OUTPUT}"'

export CIRCUIT_OUTPUT="${TEST_ROOT}/circuit-output.txt"
set +e
"${RUNNER}" \
    --jobs-dir "${circuit_jobs}" \
    --state-dir "${circuit_state}" \
    --max-consecutive-failures 1
circuit_rc=$?
set -e

[[ "${circuit_rc}" -eq 1 ]]
[[ ! -e "${CIRCUIT_OUTPUT}" ]]

printf 'resilient queue tests passed\n'
