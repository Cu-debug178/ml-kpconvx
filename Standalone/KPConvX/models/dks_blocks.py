"""Dynamic Kernel Scale (DKS) utilities.

DKS predicts one bounded isotropic scale per query point.  KPConv applies the
scale to centred neighbour coordinates, so the fixed KNN set is preserved while
the effective kernel radius and influence width change together.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


DKS_MODES = ("none", "learned", "fixed", "random")
DKS_ABLATIONS = ("none", "identity", "shuffle", "room_mean", "random")


def _validate_packed_rows(x: Tensor, lengths: Tensor) -> list[int]:
    packed_lengths = [int(value) for value in lengths.detach().cpu().tolist()]
    packed_total = sum(packed_lengths)
    row_count = int(x.shape[0])
    if packed_total != row_count:
        raise ValueError(
            "sum(lengths) ({:d}) must match x rows ({:d})".format(
                packed_total, row_count
            )
        )
    if any(length < 0 for length in packed_lengths):
        raise ValueError("packed lengths must be non-negative")
    return packed_lengths


def shuffle_packed_rows(
    x: Tensor,
    lengths: Tensor,
    *,
    seed: int = 0,
) -> Tensor:
    """Shuffle rows independently inside each packed cloud.

    A private CPU generator is used deliberately: this intervention must not
    advance the model, data-loader, CPU, or CUDA global RNG streams.
    """

    packed_lengths = _validate_packed_rows(x, lengths)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    shuffled = torch.empty_like(x)
    start = 0
    for length in packed_lengths:
        if length:
            permutation = torch.randperm(
                length, generator=generator, device="cpu"
            ).to(x.device)
            shuffled[start : start + length] = x[start : start + length][
                permutation
            ]
        start += length
    return shuffled


def room_mean_broadcast(x: Tensor, lengths: Tensor) -> Tensor:
    """Replace each row by the mean of its packed cloud."""

    packed_lengths = _validate_packed_rows(x, lengths)
    pooled = torch.empty_like(x)
    start = 0
    for length in packed_lengths:
        if length:
            segment = x[start : start + length]
            pooled[start : start + length] = segment.mean(
                dim=0, keepdim=True
            )
        start += length
    return pooled


def alpha_stats(alpha: Tensor, lengths: Optional[Tensor] = None) -> dict[str, float]:
    """Return detached global scale diagnostics for one packed batch."""

    if alpha.ndim != 1:
        raise ValueError("alpha must have shape (N,)")
    if lengths is not None:
        _validate_packed_rows(alpha, lengths)
    if alpha.numel() == 0:
        return {
            key: float("nan")
            for key in (
                "mean",
                "std",
                "p05",
                "p95",
                "frac_below_0.9",
                "frac_above_1.1",
            )
        }
    values = alpha.detach().float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p05": float(torch.quantile(values, 0.05).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "frac_below_0.9": float((values < 0.9).float().mean().item()),
        "frac_above_1.1": float((values > 1.1).float().mean().item()),
    }


class DynamicKernelScale(nn.Module):
    """Predict a bounded per-point scale with exact identity initialization."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 32,
        alpha_min: float = 0.5,
        alpha_max: float = 1.2,
        mode: str = "learned",
        fixed_alpha: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in DKS_MODES or mode == "none":
            raise ValueError("dks mode must be one of {}".format(DKS_MODES[1:]))
        if in_channels < 1 or hidden_dim < 1:
            raise ValueError("in_channels and hidden_dim must be positive")
        if not 0.0 < alpha_min < 1.0 < alpha_max:
            raise ValueError("DKS bounds must satisfy 0 < alpha_min < 1 < alpha_max")
        if mode == "fixed" and not alpha_min <= fixed_alpha <= alpha_max:
            raise ValueError("fixed_alpha must lie inside [alpha_min, alpha_max]")

        self.mode = mode
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.fixed_alpha = float(fixed_alpha)

        if mode == "learned":
            self.predictor = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.predictor[-1].bias)
            nn.init.normal_(self.predictor[-1].weight, std=0.02)
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.predictor = None
            self.register_parameter("gate", None)

        self.register_buffer(
            "_rng_seed", torch.tensor(int(seed), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_rng_step", torch.zeros((), dtype=torch.long), persistent=False
        )
        self._diagnostics_enabled = False
        self.last_stats: dict[str, float] = {}

    def set_diagnostics(self, enabled: bool) -> None:
        self._diagnostics_enabled = bool(enabled)
        if not enabled:
            self.last_stats = {}

    def _next_seed(self) -> int:
        seed = int(self._rng_seed.item()) + int(self._rng_step.item())
        self._rng_step.add_(1)
        return seed

    def _random_alpha(self, features: Tensor) -> Tensor:
        # A per-call device generator avoids a large CPU-to-GPU copy while
        # remaining completely isolated from every global RNG stream.
        generator = torch.Generator(device=features.device)
        generator.manual_seed(self._next_seed())
        values = torch.rand(
            (features.shape[0],),
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        )
        return values.mul(self.alpha_max - self.alpha_min).add(self.alpha_min)

    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        ablation: str = "none",
    ) -> Tensor:
        """Return ``alpha`` with shape ``(N,)`` and the feature dtype/device."""

        if features.ndim != 2:
            raise ValueError("features must have shape (N, C)")
        ablation = str(ablation).strip().lower()
        if ablation not in DKS_ABLATIONS:
            raise ValueError(
                "dks ablation must be one of {}; got {!r}".format(
                    DKS_ABLATIONS, ablation
                )
            )
        if lengths is not None:
            _validate_packed_rows(features, lengths)

        if self.mode == "fixed":
            alpha = features.new_full((features.shape[0],), self.fixed_alpha)
        elif self.mode == "random":
            alpha = self._random_alpha(features)
        else:
            raw = self.predictor(features).squeeze(-1)
            gate = self.gate.to(dtype=raw.dtype)
            signed = torch.tanh(gate * raw)
            # This equals the requested asymmetric ReLU expression away from
            # zero.  Writing it as t + abs(t) selects a non-zero midpoint
            # subgradient at t=0; the direct ReLU form has zero PyTorch gradient
            # at the identity initialization and can never train its gate.
            mean_slope = 0.5 * (self.alpha_max - self.alpha_min)
            asymmetry = 0.5 * (self.alpha_max + self.alpha_min - 2.0)
            alpha = 1.0 + mean_slope * signed + asymmetry * signed.abs()
            alpha = alpha.to(dtype=features.dtype)

        if ablation == "identity":
            alpha = torch.ones_like(alpha)
        elif ablation == "shuffle":
            if lengths is None:
                raise ValueError("shuffle ablation requires packed lengths")
            alpha = shuffle_packed_rows(alpha, lengths, seed=self._next_seed())
        elif ablation == "room_mean":
            if lengths is None:
                raise ValueError("room_mean ablation requires packed lengths")
            alpha = room_mean_broadcast(alpha, lengths)
        elif ablation == "random":
            alpha = self._random_alpha(features)

        alpha = alpha.to(dtype=features.dtype, device=features.device)
        if self._diagnostics_enabled:
            self.last_stats = alpha_stats(alpha, lengths)
            self.last_stats["gate"] = (
                float(self.gate.detach().float().item())
                if self.gate is not None
                else float("nan")
            )
            self._diagnostics_enabled = False
        return alpha


__all__ = [
    "DKS_ABLATIONS",
    "DKS_MODES",
    "DynamicKernelScale",
    "alpha_stats",
    "room_mean_broadcast",
    "shuffle_packed_rows",
]
