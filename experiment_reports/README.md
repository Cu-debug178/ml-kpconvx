# Experiment reports

This directory contains lightweight, reviewable summaries of completed training
and evaluation runs. Raw runtime output remains in
`Standalone/KPConvX/results/` and is intentionally excluded from Git.

## Index

| Dataset | Run | Main result | Status |
|---|---|---|---|
| S3DIS Area 5 | [KPConvX-L, seed 57106803](./s3dis/kpconvx-l-area5-seed57106803/) | final 10-vote full-cloud mIoU 71.4% | completed |
| S3DIS Area 5 | [LitePT L0, seed 57106803](./s3dis/litept-l0-450-seed57106803/) | best tested 10-vote full-cloud mIoU 72.1% at epoch 210 | completed |
| S3DIS Area 5 | [LitePT L0D, seed 57106803](./s3dis/litept-l0d-250-seed57106803/) | best tested 10-vote full-cloud mIoU 71.8% at epoch 150 | completed |
| S3DIS Area 5 | [LitePT L1, seed 57106803](./s3dis/litept-l1-250-seed57106803/) | best tested 10-vote full-cloud mIoU 69.6% at checkpoint epoch 130 | completed |
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

## Downloadable CSV

- [Existing normalized checkpoint tests](./all_test_results.csv): baseline S3DIS
  and ScanObjectNN checkpoint table. The LitePT L0/L0D/L1 run-specific tables are
  kept in their respective report directories above.
- [Paper versus current summary](./paper_vs_current_summary.csv): headline values
  from arXiv:2405.13194 and current runs, with explicit comparability labels.

Use [REPORT_TEMPLATE.md](./REPORT_TEMPLATE.md) for new runs and validate changes
with:

```bash
python3 tools/validate_experiment_reports.py
```

The reports preserve what can be verified from local artifacts. When an exact
training commit was not captured during the run, the report says so explicitly;
the repository state at publication time is not a substitute for run provenance.
