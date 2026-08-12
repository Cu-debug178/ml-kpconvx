# GLSKF effectiveness screen

GLSKF tests whether deep semantic features from stages 4/5 can spatially
modulate the effective KPConvD kernel used by the stage-3 decoder skip.
KTHA and GLSKF are mutually exclusive in one run so the two causal paths do
not confound each other.

## Stage 2 warm-start

All runs start from the same L0 epoch-210 checkpoint and use the same
`full_identity` Area_5 single-view validation protocol:

| Run | Configuration | Purpose |
| --- | --- | --- |
| `l0_head` | `glskf_mode=none`, `head_only` | Head-only training control |
| `true` | `kernel_gate`, `context_control=none`, `module_head` | Deep semantic gate |
| `shuffled` | `kernel_gate`, `context_control=shuffle` | Break point/context correspondence while preserving room statistics |
| `room_mean` | `kernel_gate`, `context_control=room_mean` | Keep room semantics but remove point-local correspondence |
| `matched_mlp` | `matched_mlp`, `module_head` | Match added capacity without deep context |

Launch only after the GPU is free:

```text
run_stage2_glskf_queue.sh
```

The queue records per-job logs, exit codes, and diagnostics. It is a short
screen, not a 250-epoch result. Promote GLSKF only when it beats the L0
control and `matched_mlp`, and `true - shuffled` is at least 0.3 percentage
point. A single seed and best-epoch selection remain exploratory evidence.

## Same-checkpoint interventions

Choose the best `true` checkpoint and set `GLSKF_SOURCE_RUN` to that run directory
before executing. Set `GLSKF_L0_CHECKPOINT` to the original L0 epoch-210 weight
so the `baseline` job uses the correct checkpoint:

```text
run_glskf_inference_ablations.sh
```

The queue first evaluates the original L0 epoch-210 checkpoint (`baseline`),
then evaluates the same GLSKF checkpoint under six modes:

- `true`: learned gate and aligned context;
- `shuffled`: context rows shuffled inside each room;
- `room_mean`: room-level context only;
- `zero_context`: context replaced by zero;
- `neutral_gate`: gate forced to one, residual KPConvD retained;
- `branch_off`: complete GLSKF residual bypass.

Interpretation is causal and conditional: `true > shuffled` supports spatial
correspondence; `true > neutral_gate` supports kernel modulation beyond an
ordinary residual convolution; `true > branch_off` shows inference-time use;
`true > baseline` is the direct positive-gain check. These contrasts do not
estimate seed variance or statistical significance.
