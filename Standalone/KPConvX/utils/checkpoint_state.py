"""Checkpoint helpers for reproducible epoch-boundary training recovery."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


RNG_STATE_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    """Capture process RNG streams used by training and DataLoader seeding."""

    state: dict[str, Any] = {
        "version": RNG_STATE_VERSION,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG streams captured by :func:`capture_rng_state`."""

    version = int(state.get("version", 0))
    if version != RNG_STATE_VERSION:
        raise ValueError("unsupported RNG state version: {:d}".format(version))
    required = ("python", "numpy", "torch_cpu", "torch_cuda")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError("RNG checkpoint is missing: {:s}".format(", ".join(missing)))

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_states = state["torch_cuda"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from the checkpoint: saved {:d}, current {:d}".format(
                    len(cuda_states), torch.cuda.device_count()
                )
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def atomic_torch_save(value: Any, path: str | os.PathLike[str]) -> None:
    """Write a torch checkpoint without exposing a partially written target."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        ".{}.tmp-{}-{}".format(destination.name, os.getpid(), time.monotonic_ns())
    )
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
