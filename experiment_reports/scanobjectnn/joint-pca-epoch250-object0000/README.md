# ScanObjectNN epoch-250 joint-PCA diagnostic

## Scope

- Dataset and split: ScanObjectNN `main_split`, test set.
- Fixed object: `object_index=0`, true class index `0`.
- Seed: `57106803`.
- Protocol: one deterministic forward per model on the same input pyramid; test augmentation was disabled.
- Hardware: NVIDIA GeForce RTX 4080 SUPER.
- Completion: completed for all six requested forwards.
- Exact Git commit: not recorded during the run.

The six forwards were:

| Model | Mode | Checkpoint | Internal epoch |
|---|---|---|---:|
| KPConvD Grid-only | Grid-only | `current_chkp.tar` | 250 |
| KPConvD FastAdapter | FastAdapter bypass | `current_chkp.tar` | 250 |
| KPConvD FastAdapter | FastAdapter enabled | `current_chkp.tar` | 250 |
| KPConvX Grid-only | Grid-only | `current_chkp.tar` | 250 |
| KPConvX FastAdapter | FastAdapter bypass | `current_chkp.tar` | 250 |
| KPConvX FastAdapter | FastAdapter enabled | `current_chkp.tar` | 250 |

The two periodic files whose internal epochs are 249 and 199 were deliberately
excluded. The comparison is therefore matched at internal epoch 250.

## Joint-PCA result

All six models had the same feature width at each stage, so one shared PCA
basis was fitted across all six representations per stage. The stage widths
were `96, 128, 192, 256, 256`; the first three PCA components explained the
following fractions of the joint fitted variance:

| Stage | Fit points | Channels | Explained variance ratio, 3 PCs |
|---:|---:|---:|---:|
| 0 | 6144 | 96 | 0.315524 |
| 1 | 5982 | 128 | 0.249329 |
| 2 | 3480 | 192 | 0.351614 |
| 3 | 1134 | 256 | 0.492844 |
| 4 | 324 | 256 | 0.410761 |

The shared colours are a visualization alignment only. They do not establish
accuracy improvement or causal FastAdapter behavior.

## One-object classifier output

All six forwards predicted class index `0` correctly for this object. The
top-1 probabilities were:

| Model and mode | Top-1 probability |
|---|---:|
| KPConvD Grid-only | 0.6526 |
| KPConvD FastAdapter bypass | 0.8125 |
| KPConvD FastAdapter enabled | 0.6353 |
| KPConvX Grid-only | 0.6057 |
| KPConvX FastAdapter bypass | 0.5679 |
| KPConvX FastAdapter enabled | 0.7702 |

These are confidence values for one object, not OA or mAcc. The full test-set
10-vote results remain in the checkpoint study report.

## FastAdapter effect against bypass

The mean feature L2 differences between FastAdapter-enabled and bypass
forwards, by stage, were:

| Architecture | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---:|---:|---:|---:|---:|
| KPConvD | 5.9755 | 5.0546 | 9.3975 | 8.2064 | 1.8901 |
| KPConvX | 1.5404 | 1.2616 | 24.7403 | 1.7547 | 1.7033 |

The differences are nonzero, but their size alone is not evidence of a
benefit. In particular, this is one object and one seed, and bypassing the
adapter changes the forward path while retaining the FastAdapter checkpoint.

## Artifacts

The full runtime artifacts are local under
`Standalone/KPConvX/results/scanobjectnn_joint_pca/epoch250_object0000/`.
They include `representation_metrics.csv`, `adapter_response.csv`,
`feature_delta_vs_bypass.csv`, `logits.csv`, `checkpoints.csv`, the shared-colour
stage PLY files, and `stage_representation_comparison.png`.

This report intentionally contains no checkpoint, dataset, raw log, or point
cloud binary. It is a one-object diagnostic and should not be used as a
dataset-level mechanism claim without expanding to a fixed multi-object
sample and reporting an independently specified aggregation protocol.
