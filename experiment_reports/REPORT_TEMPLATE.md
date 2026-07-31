# <Dataset / experiment name>

## Summary

- Status: `<completed | interrupted | failed | partial>`
- Run ID: `<stable identifier>`
- Exact training Git commit: `<40-character SHA | not recorded>`
- Dataset and split: `<dataset, version, train/validation/test split>`
- Model: `<architecture and important variant>`
- Seed(s): `<values>`
- Primary result: `<metric, value, protocol>`

## Configuration

Record the important model, optimizer, schedule, batching, augmentation, and data
settings. Commands must use placeholders such as `<DATASET_PATH>` and
`<OUTPUT_PATH>` instead of host-specific absolute paths.

## Protocol

Define every reported metric, vote/augmentation count, checkpoint selection rule,
and whether selection and final evaluation used independent splits.

## Environment and resources

Record verified GPU/CPU, framework and CUDA versions, parameter count, elapsed or
active training time, peak memory, and the source used to derive each value. Label
directory-name hardware tags as labels unless telemetry independently confirms them.

## Results

Keep complete machine-readable values in `metrics.csv`; summarize only the most
important comparisons here. Checkpoint names are identifiers only. Do not copy the
weight files.

## Limitations and anomalies

Disclose provenance gaps, single-seed uncertainty, selection bias, restarts,
adaptive batch limits, skipped batches, hardware mismatches, and failed attempts.
