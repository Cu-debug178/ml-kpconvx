# ScanObjectNN main split: full checkpoint study, seed 57106803

## Summary

- Status: completed.
- Dataset: ScanObjectNN `main_split`, 15 classes.
- Seed: `57106803` for all four training runs.
- Protocol: each of 42 distinct internal checkpoint epochs completed 10 test votes.
- Best observed 10-vote result: KPConvX-L without FastAdapter at epoch 199,
  OA 89.3% and mAcc 88.2%.
- Exact training Git commit: not recorded.

The missing run commit prevents an exact code-to-result mapping. The commit that
publishes this report is not evidence of the code revision that produced these
weights or measurements.

## Configuration

All four runs used 250 epochs, AdamW with learning rate 0.0005 and weight decay
0.01, batch size 32, gradient accumulation 2, and 10 test votes. The two
FastAdapter runs used joint training with 64 FPS anchors, geometry dimension 16,
attention dimension 64, four heads, chunk size 4096, and cross-layer plus spatial
attention enabled.

| ID | Model | FastAdapter | Parameters | Active training time | Training environment label |
|---|---|---:|---:|---:|---|
| D | KPConvD-L | no | 7,146,735 | 15,097.069 s (4.19 h) | 4090D-24G |
| X+FA | KPConvX-L | yes | 13,139,393 | 36,720.519 s (10.20 h) | 4090D-24G |
| D+FA | KPConvD-L | yes | 7,896,617 | 35,204.849 s (9.78 h) | 4080S-32G |
| X | KPConvX-L | no | 12,389,511 | 29,553.054 s (8.21 h) | 4080S-32G |

The environment labels are retained from experiment directory names. They are not
independent hardware telemetry. All reported test throughput values were measured
by the same test pipeline on an NVIDIA GeForce RTX 4080 SUPER; throughput includes
input wait, device transfer, forward inference, and result handling, but excludes
backpropagation.

## Results

`metrics.csv` contains the complete 42-checkpoint table. `single_oa_pct` and
`single_macc_pct` are the tenth augmentation pass by itself; the `vote10_*`
columns are the cumulative prediction across votes 0 through 9 and are the values
used for comparison.

| Rank | Model | Checkpoint epoch | 10-vote OA | 10-vote mAcc | Throughput (instances/s) |
|---:|---|---:|---:|---:|---:|
| 1 | KPConvX-L | 199 | 89.3% | 88.2% | 243.2 |
| 2 | KPConvD-L + FastAdapter | 250 | 89.1% | 87.8% | 135.1 |
| 3 | KPConvD-L | 249/250 | 88.9% | 87.3% | 437.9/437.4 |
| 4 | KPConvX-L + FastAdapter | 190 | 88.7% | 87.3% | 110.6 |

Checkpoint names in the CSV are identifiers only. No checkpoint, tensor, raw log,
or test prediction is stored in Git.

## Limitations and anomalies

- Results are from one seed only. Small differences do not establish a stable
  ranking across independent runs.
- The best checkpoints were selected after testing all 42 candidates on the same
  test split. This introduces test-set selection bias. A final comparison should
  select on validation data and evaluate the selected checkpoint once on test.
- Training hardware labels and common test hardware differ; do not compare their
  training speeds to the reported test throughput.
- The KPConvX-L + FastAdapter run had one CUDA out-of-memory event at epoch 0.
  The program reduced the batch limit and restarted the epoch; 12 oversized
  batches were subsequently skipped. Its saved pre-OOM configuration does not
  prove the final effective batch limit.
- The exact training commits, full software environments, and raw configurations
  were not captured in a shareable provenance record.
