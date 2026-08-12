# KTHA V2

`pairwise_bias_v2` transfers geometry without a semantic-feature bypass.

## Geometry signature

For `K` KPConv kernels, each token receives `3K + 2` geometry-only channels:

- normalized kernel occupancy;
- per-kernel normalized residual-distance mean and standard deviation;
- valid-neighbor ratio and mean KP influence mass.

V2 pooling applies a per-channel mean/max blend and does not renormalize the
heterogeneous channels together.

## Attention handover

The geometry-only projection produces per-head embeddings. Patch-centering
removes constant geometry. A zero-initialized diagonal pairwise metric adds a
learned squared-difference bias to attention logits. Consequently, zero and
room-mean signatures produce zero geometry bias by construction.

`matched_mlp_v2` has exactly the same trainable parameter count per attention
block but consumes semantic features only. It retains V2 signature computation
to provide a closer latency control.

## Short screen

```bash
Standalone/KPConvX/run_ktha_v2_screen.sh
```

The queue starts three independent 10-epoch runs from the L0 epoch-210
checkpoint: true geometry, room-local shuffled geometry, and parameter-matched
MLP. Runtime state and outputs remain under `Standalone/KPConvX/results/`.

Do not promote V2 to long training unless true geometry beats both controls and
same-checkpoint zero/shuffle/branch-off interventions show a material effect.
