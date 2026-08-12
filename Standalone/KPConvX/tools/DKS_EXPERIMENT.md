# Dynamic Kernel Scale (DKS)

DKS predicts one bounded isotropic scale per query point and divides centred
KPConv neighbour coordinates by that scale. It keeps the precomputed KNN set
fixed while changing nearest-kernel assignment and influence weights. The
default learned path starts at exactly `alpha=1` and can warm-start L0.

## Scope

Phase A is implemented. DKS is mutually exclusive with KTHA and GLSKF and
currently requires `share_kp=True`, a KPConvD/KPConvX stage, and non-MLP kernel
influence. Phase B teacher distillation and Phase C oracle mining are gated on
Phase A experimental evidence and are not part of this implementation.

The specification says five arms but names six independent runs. Preserve all
six: `l0_head`, `fixed_1.0`, `fixed_0.8`, `fixed_1.15`, `random`, and `learned`.
Use seeds `57106803`, `12345`, and `98765`; compare the fixed epoch-10 metric,
not each run's best epoch.

## Run one warm-start arm

```bash
tools/run_stage2_dks_candidate.sh learned 57106803 \
  stage2_dks_learned_seed57106803 10
```

Set `DKS_L0_CHECKPOINT` and `S3DIS_DATASET_PATH`; `KP_CONVX_PYTHON` defaults to
`python3`. The script skips verified complete output, preserves incomplete
directories, and writes runtime artifacts only under `results/`.

After the one-arm acceptance checks pass, run the reproducible queue with
`tools/run_stage2_dks_queue.sh`. Reusing the default queue-state directory skips
verified successes; each failure records its own log and diagnostics, and two
consecutive failures open the systemic-failure circuit breaker.

When `dks_log_stats=True`, monitored optimizer steps append per-stage alpha
statistics to `dks_alpha_stats.csv`.

## Same-checkpoint intervention

```bash
python tools/evaluate_s3dis_dks_ablation.py \
  --source-log results/<DKS_RUN> \
  --checkpoint results/<DKS_RUN>/checkpoints/best_val_chkp.tar \
  --dataset-path <DATASET_PATH> \
  --output-dir results/<OUTPUT> \
  --mode shuffle
```

Modes are `true`, `identity`, `shuffle`, `room_mean`, and `random`. `baseline`
is an external L0 reference and is not a same-checkpoint intervention.

Mechanism support requires `learned` to exceed both the best fixed-scale arm
and `random` beyond the predeclared noise floor. `learned ~= fixed_best` means
global radius tuning explains the result; `learned ~= random` means arbitrary
per-point perturbation or regularization remains a sufficient explanation.
