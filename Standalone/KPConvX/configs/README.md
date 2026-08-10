# Python experiment configs

This directory follows Pointcept's declarative Python-config pattern while
keeping KPConvX's current dataset `my_config()` as the implicit root base. This
avoids migrating or changing any existing model, data, or optimizer defaults.

Each config contains section dictionaries and only lists differences:

```python
model = dict(
    kp_mode='kpconvd',
    fa_enabled=True,
)

train = dict(
    amp_enabled=True,
    amp_dtype='bfloat16',
)
```

File names are arbitrary. Subdirectories only organize experiments. Run a file
by its path relative to this directory; the `.py` suffix is optional:

```bash
python3 experiments/S3DIS/train_S3DIS.py \
  --config-file s3dis/fastadapter_bf16 \
  --dataset_path ../data/s3dis
```

`--config` is a shorter alias for the Pointcept-style `--config-file` name.
The generic launchers also forward extra arguments, for example from the
`Standalone` directory:

```bash
bash train_S3DIS.sh --config-file s3dis/fastadapter_bf16
```

Specialized FastAdapter/LitePT launchers deliberately provide named preset
arguments; those named arguments have higher priority than a config file.

Validation and checkpoint controls are independent. `train.validation_mode` is
`partial` by default; `full_identity` is an explicit full-room Identity
protocol and is not selected by default. `save_latest_val` keeps the latest
recoverable validation state, while `save_best_val` keeps a single-validation
best when enabled. For S3DIS, `save_best_val_cycle` is enabled by default and
selects `best_cycle_chkp.tar` only after every index in a complete regular
validation vote has been observed. `save_fraction_checkpoints` keeps the 1/5
through 4/5 milestones, and `save_periodic_checkpoints` enables legacy
`checkpoint_gap` copies.

A regular-validation cycle may span adjacent training epochs, so its samples
can come from different network states. `best_cycle_chkp.tar` is the current
network at the moment that selection heuristic completes. Final reported
metrics must still come from a fixed checkpoint evaluated by the unchanged
full Area 5 multi-vote test.

For a one-off override, use Pointcept-style dotted options instead of creating
another file:

```bash
python3 experiments/S3DIS/train_S3DIS.py \
  --config-file s3dis/fastadapter_bf16 \
  --options train.amp_enabled=false model.fa_num_anchors=80 \
  --dataset_path ../data/s3dis
```

Option values accept Python literals such as `true`, `false`, `None`, numbers,
lists, tuples, and dictionaries. Unknown or duplicate keys are rejected. Named
arguments such as `--amp_enabled` are applied after `--options` and therefore
have the highest priority.

## Inheriting another experiment config

Use Pointcept-style `_base_` paths. Paths are relative to the child file, and
inheritance can be recursive:

```python
_base_ = '../_base_/fastadapter.py'

model = dict(kp_mode='kpconvd')
train = dict(amp_enabled=True, amp_dtype='bfloat16')
```

Multiple sibling bases may set different parameters. If two sibling bases set
the same `section.key`, loading fails instead of silently depending on list
order; a child config may still intentionally override any inherited value.

Nested dictionaries are merged. Add `_delete_=True` when a dictionary-valued
parameter must be replaced instead of merged:

```python
train = dict(
    lr_decays=dict(_delete_=True, **{'100': 0.1}),
)
```

The precedence from lowest to highest is:

1. Dataset-specific `my_config()` defaults.
2. `_base_` files from parent to child.
3. The selected Python config.
4. Generic `--options section.key=value` overrides.
5. Existing named command-line overrides.

The complete effective configuration is still saved as `parameters.json` in
the experiment directory. Its `exp.config_file`, `exp.config_sources`, and
`exp.config_options` fields record the selected file, every inherited source's
SHA-256 hash, and one-off CLI overrides. Dataset and result paths should
normally remain machine-local command-line arguments. Resumed training always
uses its saved `parameters.json`; `--config` and `--options` are deliberately
rejected with `--resume_path`.

Python configs execute as Python code, so only use trusted files. Unknown
sections and parameter names are rejected to catch spelling mistakes.
