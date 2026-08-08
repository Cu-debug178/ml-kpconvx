"""Pure helpers for training schedules, checkpoints, and monitor cadence."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping


class EpochMultiplicativeLRScheduler:
    """Checkpointable form of KPConvX's existing epoch LR updates.

    The original training loop multiplied every optimizer learning rate by
    ``train_cfg.lr_decays[str(epoch)]`` after a successfully completed epoch.
    This object preserves that exact update order while exposing a strict
    ``state_dict``/``load_state_dict`` interface for interruption recovery.
    """

    STATE_VERSION = 1

    def __init__(self, optimizer: Any, lr_decays: Mapping[str, float]) -> None:
        self.optimizer = optimizer
        self.lr_decays = {
            int(epoch): float(factor) for epoch, factor in lr_decays.items()
        }
        if any(epoch < 0 for epoch in self.lr_decays):
            raise ValueError("learning-rate decay epochs must be non-negative")
        if any(not math.isfinite(factor) or factor <= 0 for factor in self.lr_decays.values()):
            raise ValueError("learning-rate decay factors must be finite and positive")
        self.last_epoch = -1
        self._last_lr = self.get_last_lr()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def step(self, epoch: int | None = None) -> None:
        next_epoch = self.last_epoch + 1 if epoch is None else int(epoch)
        if next_epoch != self.last_epoch + 1:
            raise ValueError(
                "scheduler epochs must be consecutive: expected {:d}, got {:d}".format(
                    self.last_epoch + 1, next_epoch
                )
            )
        factor = self.lr_decays.get(next_epoch, 1.0)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= factor
        self.last_epoch = next_epoch
        self._last_lr = self.get_last_lr()

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "last_epoch": self.last_epoch,
            "last_lr": list(self._last_lr),
            "lr_decays": dict(self.lr_decays),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        version = int(state_dict.get("version", 0))
        if version != self.STATE_VERSION:
            raise ValueError("unsupported scheduler state version: {:d}".format(version))
        saved_decays = {
            int(epoch): float(factor)
            for epoch, factor in dict(state_dict["lr_decays"]).items()
        }
        if saved_decays != self.lr_decays:
            raise ValueError("checkpoint learning-rate schedule differs from the run config")

        saved_lrs = [float(value) for value in state_dict["last_lr"]]
        current_lrs = self.get_last_lr()
        if len(saved_lrs) != len(current_lrs) or any(
            not math.isclose(saved, current, rel_tol=1e-12, abs_tol=0.0)
            for saved, current in zip(saved_lrs, current_lrs)
        ):
            raise ValueError(
                "optimizer learning rates do not match the scheduler checkpoint: "
                "optimizer={!r}, scheduler={!r}".format(current_lrs, saved_lrs)
            )
        self.last_epoch = int(state_dict["last_epoch"])
        self._last_lr = saved_lrs

    def align_legacy_checkpoint(self, next_epoch: int) -> None:
        """Align state for a checkpoint written before scheduler serialization."""

        next_epoch = int(next_epoch)
        if next_epoch < 0:
            raise ValueError("checkpoint epoch must be non-negative")
        self.last_epoch = next_epoch - 1
        self._last_lr = self.get_last_lr()


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
