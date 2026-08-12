# KTHA same-checkpoint inference ablations

## Question

Test whether M1 uses the token-to-kernel-geometry correspondence, only room-level
signature statistics, only the additional feature MLP, or none of the KTHA branch.

## Fixed protocol

- Source weights: the completed true-geometry M1 joint-screen best checkpoint,
  internal epoch 7.
- Dataset: S3DIS Area 5.
- Evaluation: deterministic `full_identity`, one complete single view, no test
  augmentation and no voting.
- Seed: `57106803` for every intervention.
- The checkpoint SHA-256, checkpoint epoch, seed and protocol must match across
  every completed result before summary generation succeeds.
- No training or checkpoint mutation occurs. Per-room PLY output is disabled.

## Interventions

| Mode | Intervention | What it isolates |
|---|---|---|
| `true` | Keep every signature aligned with its token | Reference |
| `shuffled` | Permute whole signature rows inside each packed room | Token-geometry correspondence |
| `zero` | Feed zero geometry while keeping the trained feature MLP and scale | Feature-only residual path |
| `room_mean` | Give every token the mean signature of its room | Room-level first-moment context |
| `branch_off` | Set trained concat residual scales to zero in memory | Reliance on the entire KTHA residual |

`zero` is the feature-only concat control because a zero geometry vector removes
the geometry input while preserving the MLP, its bias and the learned residual
scale. `branch_off` is therefore required as a distinct fifth intervention.

## Interpretation

- `true > shuffled` supports value in token-specific geometry alignment.
- `true ≈ shuffled`, with both above `room_mean` and `zero`, suggests signature
  distributional information beyond its first moment, but not correct alignment.
- `true ≈ shuffled ≈ room_mean > zero` suggests room-level mean context is enough.
- `zero > branch_off` suggests the trained feature-only residual MLP contributes.
- `true ≈ zero ≈ branch_off` suggests little inference-time reliance on KTHA.

These are descriptive single-checkpoint contrasts. They do not estimate seed
variance or statistical significance. `branch_off` uses a backbone co-trained
with KTHA and is not equivalent to evaluating the original L0 checkpoint.

## Launch after GPU mode is enabled

Run the launcher with the local dataset root:

```bash
Standalone/KPConvX/run_ktha_m1_inference_ablations.sh <DATASET_PATH>
```

The resilient queue runs the five independent evaluations in the table order and
then writes `summary.csv` and `summary.json`. A completed mode is skipped on a
rerun. An incomplete mode directory is preserved and requires manual diagnosis.
Before checking CUDA or creating queue state, the launcher performs a CPU-only
strict checkpoint/configuration preflight.
