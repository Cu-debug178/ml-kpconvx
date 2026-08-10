# S3DIS LitePT L0 + FastAdapter checkpoint evaluation

## Scope and status

- Dataset and split: S3DIS, train on Areas 1-4 and 6, evaluate on Area 5.
- Model: KPNeXt with `kp_mode=kpconvx`, LitePT enabled, three leading
  configured-convolution stages, no handover stage, and FastAdapter joint
  training enabled.
- Seed: `57106803`.
- Training budget: 250 epochs, 300 steps per epoch, batch size 12 with gradient
  accumulation 2, AdamW, initial learning rate 0.0001, weight decay 0.05.
- Numerical mode: full precision. This run predates the new AMP configuration
  fields and its saved parameters do not enable AMP.
- Checkpoint interval: 10 epochs.
- Hardware: one NVIDIA GeForce RTX 4080 SUPER with 32760 MiB reported memory.
- Exact Git commit during the run: not recorded.
- Status at report creation: training resumed after interruptions and was still
  running; the latest completed checkpoint had internal epoch 234. The planned
  250-epoch result was not yet available.

The experiment was interrupted more than once, including once when the AutoDL
instance was terminated externally. It resumed from format-v2 checkpoints that
contain model, optimizer, learning-rate scheduler, RNG streams, and adaptive
train/test batch limits. No claim is made that the interrupted and uninterrupted
wall-clock trajectories are identical outside those captured states.

## Checkpoint selection

Periodic checkpoints from internal epochs 100 through 230 were evaluated with
one deterministic Identity view. Identity disables rotation, scale, and flip
augmentation and uses fixed sampling. The highest observed Identity Area-5 mIoU
was 70.7% at `chkp_0120.tar`.

The selected epoch-120 checkpoint was then evaluated using two protocols:

| Protocol | Area-5 full-cloud mIoU | Area-5 sub-cloud mIoU |
|---|---:|---:|
| Deterministic Identity | 70.7% | not recorded in the selection manifest |
| Standalone 10-vote | 71.7% | 71.3% |
| Fixed Pointcept-style 13-TTA | 70.8% | 70.5% |

The 13-TTA transform signature was `rot4-scale3-xflip-v2`, with fixed sampling
points reused between transforms. Both final evaluation processes returned code
0. The beam IoU remained near zero: 0.1% for 10-vote and 0.2% for 13-TTA.

## Artifacts

- [metrics.csv](metrics.csv) records checkpoint-level Identity results and the
  two final evaluation protocols.
- [checkpoint_inventory.csv](checkpoint_inventory.csv) records the physically
  stored checkpoints for this run at report creation and their resumability
  fields.
- [class_iou.csv](class_iou.csv) records the final per-class full-cloud IoUs.

Raw logs, predictions, parameters with machine-local paths, and checkpoint
binaries remain under `Standalone/KPConvX/results/` and are not versioned.

## Interpretation limits

- This is one seed, so it does not estimate run-to-run variance.
- Area 5 was used both to select epoch 120 and to report final metrics. The
  10-vote and 13-TTA numbers therefore have checkpoint-selection bias and are
  not independent test estimates.
- Identity, repeated stochastic voting, and fixed TTA are different inference
  protocols. Their values measure both model behavior and protocol differences.
- The best Identity score occurred at epoch 120, but only epochs divisible by
  10 in the 100-230 range were compared. This does not prove epoch 120 is the
  best possible training state.
- Training was incomplete when this report was created. Epochs 240 and 250 had
  not yet been evaluated, so the selection result can still change.
- The very low beam IoU materially depresses and destabilizes mean IoU because
  all 13 classes receive equal weight.
