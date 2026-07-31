# S3DIS Area 5: KPConvX-L, seed 57106803

## Summary

- Status: completed after one interruption and checkpoint resume.
- Dataset protocol: S3DIS, train on Areas 1-4 and 6; validate/evaluate on Area 5.
- Model: KPConvX-L, 18,288,909 parameters, FastAdapter disabled.
- Seed: `57106803`.
- Best validation checkpoint: epoch 348, validation mIoU 75.7385%.
- Final checkpoint: 10-vote full-cloud mIoU 71.4%.
- Exact training Git commit: not recorded.

The missing run commit is a provenance gap. The repository commit that publishes
this report must not be interpreted as the exact code revision used for training.

## Configuration

| Setting | Value |
|---|---:|
| Epochs | 450 |
| Steps per epoch | 300 |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Weight decay | 0.05 |
| Batch size | 4 |
| Gradient accumulation | 6 |
| Input radius | 2.1 |
| Training workers | 10 |
| Test votes | 10 |
| Test batch size | 1 |
| Test input radius | 100 |

The run used one GPU. Telemetry identifies it as an NVIDIA GeForce RTX 4090 D
with 24,564 MiB memory. Across 4,074 monitor samples, device-memory use averaged
12,731.41 MiB and peaked at 14,805 MiB; utilization averaged 62.74% and peaked at
100%; power averaged 226.04 W and peaked at 280.50 W.

## Training history

Training was interrupted during epoch 282 and resumed from a checkpoint. The two
recorded active-time counters are 84,705.045 s and 46,732.528 s. Their sum is
131,437.573 s (36.51 h). This is recorded active training time, not independently
measured wall-clock duration; epoch 282 was partially repeated after the resume.

The final log reaches epoch 449, step 299, while the validation/checkpoint workflow
records completion epoch 450. The final row in `metrics.csv` uses that completion
counter; it was not re-read from the checkpoint binary during report preparation.

## Evaluation protocol

Both published checkpoints were evaluated with 10 votes. `subcloud_miou_pct` is
computed on sampled sub-cloud predictions; `fullcloud_miou_pct` is computed after
projection to full clouds. Detailed class IoUs are in `class_iou.csv`.

The epoch-348 checkpoint was selected by the best validation mIoU on Area 5.
Because the 10-vote evaluation also uses Area 5, the selection and reported
evaluation are not independent. The final checkpoint scores higher in the 10-vote
full-cloud evaluation despite having a lower final validation snapshot mIoU.

## Results

| Checkpoint role | Epoch | Validation mIoU | 10-vote sub-cloud mIoU | 10-vote full-cloud mIoU |
|---|---:|---:|---:|---:|
| Best validation | 348 | 75.7385% | 70.9% | 71.1% |
| Final | 450 | 72.0% | 71.1% | 71.4% |

Checkpoint file names in the CSV are identifiers only. No checkpoint data is
included in this repository.

## Limitations

- This is a single-seed result, so run-to-run variance is unknown.
- Area 5 was used for both checkpoint selection and reported evaluation.
- The exact training Git commit and complete software-version snapshot were not
  captured during the run.
- Active time comes from two reset-on-resume counters and is not end-to-end wall
  time.
- The interruption and repeated portion of epoch 282 can affect exact
  reproducibility even with the recorded seed.
