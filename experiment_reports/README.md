# Experiment reports

This directory contains lightweight, reviewable summaries of completed training
and evaluation runs. Raw runtime output remains in
`Standalone/KPConvX/results/` and is intentionally excluded from Git.

## Index

| Dataset | Run | Main result | Status |
|---|---|---|---|
| S3DIS Area 5 | [KPConvX-L, seed 57106803](./s3dis/kpconvx-l-area5-seed57106803/) | final 10-vote full-cloud mIoU 71.4% | completed |
| ScanObjectNN main split | [four-model checkpoint study, seed 57106803](./scanobjectnn/full-checkpoint-study-seed57106803/) | best observed 10-vote OA 89.3% | completed |

## Storage policy

Commit small Markdown, CSV, and sanitized JSON/YAML summaries only. Do not commit
checkpoints, weights, datasets, raw console logs, generated point clouds, process
state, or machine-specific paths. Checkpoint file names may be recorded solely as
identifiers.

Each run directory must include:

- `README.md`: provenance, protocol, interpretation, and limitations;
- `metrics.csv`: machine-readable results with units in column names;
- optional small tables for class metrics, resource use, or sanitized settings.

Use [REPORT_TEMPLATE.md](./REPORT_TEMPLATE.md) for new runs and validate changes
with:

```bash
python3 tools/validate_experiment_reports.py
```

The reports preserve what can be verified from local artifacts. When an exact
training commit was not captured during the run, the report says so explicitly;
the repository state at publication time is not a substitute for run provenance.
