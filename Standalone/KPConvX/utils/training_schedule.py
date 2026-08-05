"""Pure helpers for training schedules, checkpoints, and monitor cadence."""

from __future__ import annotations

from typing import Any, Dict


def rebuild_cyclic_lr(train_cfg: Any) -> Dict[str, float]:
    """Rebuild the epoch-wise cyclic LR multipliers from the final config.

    This function must be called after command-line overrides are applied.  It
    intentionally returns the generated mapping as well as updating
    ``train_cfg`` so launchers can be tested without constructing a dataset.
    """

    lr0 = float(train_cfg.cyc_lr0)
    lr1 = float(train_cfg.cyc_lr1)
    raise_epochs = int(train_cfg.cyc_raise_n)
    decrease_epochs_per_decade = int(train_cfg.cyc_decrease10)
    plateau_epochs = int(train_cfg.cyc_plateau)
    max_epoch = int(train_cfg.max_epoch)
    if lr0 <= 0 or lr1 <= 0:
        raise ValueError("cyc_lr0 and cyc_lr1 must be positive")
    if raise_epochs <= 0 or decrease_epochs_per_decade <= 0:
        raise ValueError("cyc_raise_n and cyc_decrease10 must be positive")
    if plateau_epochs < 0 or max_epoch < 1:
        raise ValueError("cyc_plateau must be non-negative and max_epoch positive")

    raise_rate = (lr1 / lr0) ** (1.0 / raise_epochs)
    decrease_rate = 0.1 ** (1.0 / decrease_epochs_per_decade)
    decays = {str(epoch): raise_rate for epoch in range(1, raise_epochs + 1)}
    decrease_start = raise_epochs + 1 + plateau_epochs
    for epoch in range(decrease_start, max_epoch):
        decays[str(epoch)] = decrease_rate

    train_cfg.lr = lr0
    train_cfg.lr_decays = decays
    return decays


def periodic_checkpoint_due(
    completed_epoch: int,
    checkpoint_gap: int,
    checkpoint_start: int | None = None,
) -> bool:
    """Return whether a periodic copy is due after ``completed_epoch``.

    Checkpoint payloads and filenames use the number of fully completed epochs,
    so ``chkp_0200.tar`` is the model after epoch 200, not a zero-based index.
    """

    completed_epoch = int(completed_epoch)
    checkpoint_gap = int(checkpoint_gap)
    if checkpoint_gap <= 0:
        return False
    if checkpoint_start is None:
        return completed_epoch % checkpoint_gap == 0
    checkpoint_start = int(checkpoint_start)
    return (
        completed_epoch >= checkpoint_start
        and (completed_epoch - checkpoint_start) % checkpoint_gap == 0
    )


def optimizer_step_monitor_due(
    enabled: bool,
    mini_step: int,
    optimizer_step: int,
    accum_batch: int,
    interval: int,
) -> bool:
    """Select the final mini-batch of requested optimizer updates.

    ``mini_step`` and ``optimizer_step`` are zero-based.  Monitoring therefore
    captures the first optimizer update and then updates ``interval`` apart.
    """

    accum_batch = int(accum_batch)
    interval = int(interval)
    if accum_batch < 1 or interval < 1:
        raise ValueError("accum_batch and monitor interval must be positive")
    return bool(
        enabled
        and (int(mini_step) + 1) % accum_batch == 0
        and int(optimizer_step) % interval == 0
    )
