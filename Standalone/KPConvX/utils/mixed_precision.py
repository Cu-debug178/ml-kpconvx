"""Mixed-precision configuration, autocast, and checkpoint helpers."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping

import torch


SUPPORTED_AMP_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
MIXED_PRECISION_STATE_VERSION = 1


@dataclass(frozen=True)
class MixedPrecisionSettings:
    """Resolved AMP settings for one training process."""

    enabled: bool
    dtype_name: str
    dtype: torch.dtype

    @property
    def grad_scaler_enabled(self) -> bool:
        return self.enabled and self.dtype is torch.float16


def normalize_amp_dtype(value: Any) -> str:
    """Return a canonical AMP dtype name or raise for an unsupported value."""

    dtype_name = str(value).strip().lower()
    if dtype_name not in SUPPORTED_AMP_DTYPES:
        raise ValueError(
            "train.amp_dtype must be one of {}; got {!r}".format(
                sorted(SUPPORTED_AMP_DTYPES), value
            )
        )
    return dtype_name


def resolve_mixed_precision(train_cfg: Any, device: torch.device) -> MixedPrecisionSettings:
    """Validate the training configuration against the selected device."""

    dtype_name = normalize_amp_dtype(
        getattr(train_cfg, "amp_dtype", "bfloat16")
    )
    enabled = bool(getattr(train_cfg, "amp_enabled", False))
    if enabled and device.type != "cuda":
        raise RuntimeError("Mixed precision training currently requires a CUDA device")
    if enabled and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 mixed precision is not supported by this CUDA device")
    return MixedPrecisionSettings(
        enabled=enabled,
        dtype_name=dtype_name,
        dtype=SUPPORTED_AMP_DTYPES[dtype_name],
    )


def autocast_context(
    settings: MixedPrecisionSettings,
    device: torch.device,
) -> AbstractContextManager:
    """Create an autocast context matching the resolved settings."""

    return torch.autocast(
        device_type=device.type,
        dtype=settings.dtype,
        enabled=settings.enabled,
    )


def create_grad_scaler(
    settings: MixedPrecisionSettings,
    device: torch.device,
) -> torch.amp.GradScaler:
    """Use dynamic loss scaling only for float16 AMP."""

    return torch.amp.GradScaler(
        device.type,
        enabled=settings.grad_scaler_enabled,
    )


def mixed_precision_state_dict(
    settings: MixedPrecisionSettings,
    grad_scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    """Serialize the numerical mode together with any scaler state."""

    return {
        "version": MIXED_PRECISION_STATE_VERSION,
        "enabled": settings.enabled,
        "dtype": settings.dtype_name,
        "grad_scaler": grad_scaler.state_dict(),
    }


def restore_mixed_precision_state(
    state: Mapping[str, Any] | None,
    settings: MixedPrecisionSettings,
    grad_scaler: torch.amp.GradScaler,
) -> None:
    """Restore AMP state and reject silent numerical-mode changes on resume."""

    if state is None:
        if settings.enabled:
            raise RuntimeError(
                "Checkpoint has no mixed-precision state but AMP is enabled in its config"
            )
        return

    version = int(state.get("version", 0))
    if version != MIXED_PRECISION_STATE_VERSION:
        raise ValueError("unsupported mixed-precision state version: {:d}".format(version))

    saved_enabled = bool(state.get("enabled", False))
    saved_dtype = normalize_amp_dtype(state.get("dtype", ""))
    if saved_enabled != settings.enabled or saved_dtype != settings.dtype_name:
        raise RuntimeError(
            "Checkpoint mixed precision differs from the run config: "
            "checkpoint=({}, {}), config=({}, {})".format(
                saved_enabled,
                saved_dtype,
                settings.enabled,
                settings.dtype_name,
            )
        )

    scaler_state = state.get("grad_scaler", {})
    if not isinstance(scaler_state, Mapping):
        raise ValueError("checkpoint grad_scaler state must be a mapping")
    grad_scaler.load_state_dict(dict(scaler_state))
