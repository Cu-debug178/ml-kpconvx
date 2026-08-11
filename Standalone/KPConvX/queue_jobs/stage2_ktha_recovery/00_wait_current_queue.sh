#!/usr/bin/env bash

set -uo pipefail

QUEUE_PID=37677
PROC_STAT="/proc/${QUEUE_PID}/stat"

[[ -r "${PROC_STAT}" ]] || exit 0
EXPECTED_START_TIME="$(awk '{print $22}' "${PROC_STAT}" 2>/dev/null)" || exit 0

while [[ -r "${PROC_STAT}" ]]; do
    CURRENT_START_TIME="$(awk '{print $22}' "${PROC_STAT}" 2>/dev/null)" || break
    [[ "${CURRENT_START_TIME}" == "${EXPECTED_START_TIME}" ]] || break
    sleep 20
done
