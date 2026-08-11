"""Kernel-to-token geometry handover utilities.

The producer deliberately uses the same nearest-kernel partition as KPConvD.
It emits a compact, feature-independent occupancy signature so geometry can be
shuffled independently from semantic features in causal-control experiments.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from kernels.kernel_points import load_kernels
from models.generic_blocks import index_select


class KernelOccupancySignature(nn.Module):
    """Build normalized per-point occupancy over the KPConv kernel basis."""

    def __init__(
        self,
        shell_sizes: Sequence[int],
        radius: float,
        sigma: float,
        dimension: int = 3,
        influence_mode: str = "linear",
        fixed_kernel_points: str = "center",
        kernel_points: Tensor | None = None,
    ):
        super().__init__()
        if influence_mode not in {"constant", "linear", "gaussian"}:
            raise ValueError(
                "KTHA occupancy requires constant, linear, or gaussian KP influence"
            )
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.num_kernels = int(sum(shell_sizes))
        self.sigma = float(sigma)
        self.influence_mode = influence_mode
        if kernel_points is None:
            loaded_points = load_kernels(
                radius,
                list(shell_sizes),
                dimension=dimension,
                fixed=fixed_kernel_points,
            )
            kernel_points = torch.from_numpy(loaded_points).float()
        elif tuple(kernel_points.shape) != (self.num_kernels, dimension):
            raise ValueError(
                "kernel_points must have shape ({}, {})".format(
                    self.num_kernels, dimension
                )
            )
        self.register_buffer("kernel_points", kernel_points.detach().clone().float())

    def _assign_geometry(self, q_pts: Tensor, s_pts: Tensor, neighbor_indices: Tensor):
        padded_points = torch.cat(
            [s_pts, torch.zeros_like(s_pts[:1]) + 1e6], dim=0
        )
        relative = index_select(padded_points, neighbor_indices, dim=0)
        relative = relative - q_pts.unsqueeze(1)
        sq_distances = torch.sum(
            (relative.unsqueeze(2) - self.kernel_points) ** 2,
            dim=-1,
        )
        nearest_sq_distance, nearest_kernel = torch.min(sq_distances, dim=2)

        if self.influence_mode == "constant":
            influence = torch.ones_like(nearest_sq_distance)
        elif self.influence_mode == "linear":
            influence = torch.clamp(
                1.0 - torch.sqrt(nearest_sq_distance) / self.sigma,
                min=0.0,
            )
        else:
            gaussian_sigma = self.sigma * 0.3
            influence = torch.exp(
                -nearest_sq_distance / (2.0 * gaussian_sigma * gaussian_sigma)
            )
        return influence, nearest_kernel

    def forward(
        self,
        q_pts: Tensor,
        s_pts: Tensor,
        neighbor_indices: Tensor,
        cached_geometry: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        if q_pts.shape[0] == 0:
            return q_pts.new_empty((0, self.num_kernels))

        use_cache = (
            cached_geometry is not None
            and cached_geometry.get("neighb_1nn") is not None
            and cached_geometry.get("neighb_1nn").shape == neighbor_indices.shape
        )
        if use_cache:
            nearest_kernel = cached_geometry["neighb_1nn"]
            influence = cached_geometry.get("infl_w")
            if influence is None:
                influence = torch.ones_like(nearest_kernel, dtype=q_pts.dtype)
        else:
            influence, nearest_kernel = self._assign_geometry(
                q_pts, s_pts, neighbor_indices
            )

        # Shadow neighbors must never contribute, including in constant mode.
        valid = neighbor_indices < s_pts.shape[0]
        weights = influence.to(q_pts.dtype) * valid.to(q_pts.dtype)
        occupancy = q_pts.new_zeros((q_pts.shape[0], self.num_kernels))
        occupancy.scatter_add_(1, nearest_kernel, weights)
        return occupancy / occupancy.sum(dim=1, keepdim=True).clamp_min(1e-6)


def pool_kernel_signature(signature: Tensor, pool_indices: Tensor) -> Tensor:
    """Pool a signature with an equal mean/max blend, preserving its width."""

    if signature.ndim != 2:
        raise ValueError("signature must have shape [N, K]")
    if pool_indices.ndim != 2:
        raise ValueError("pool_indices must have shape [M, H]")
    if pool_indices.shape[0] == 0:
        return signature.new_empty((0, signature.shape[1]))

    valid = pool_indices < signature.shape[0]
    padded = torch.cat([signature, torch.zeros_like(signature[:1])], dim=0)
    gathered = index_select(padded, pool_indices, dim=0)
    valid_f = valid.unsqueeze(-1).to(signature.dtype)
    mean = (gathered * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
    maximum = gathered.masked_fill(~valid.unsqueeze(-1), -math.inf).amax(dim=1)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    pooled = 0.5 * (mean + maximum)
    return pooled / pooled.sum(dim=1, keepdim=True).clamp_min(1e-6)


def shuffle_packed_signature(signature: Tensor, lengths: Tensor) -> Tensor:
    """Randomly permute geometry within each packed room, preserving its marginal."""

    shuffled = torch.empty_like(signature)
    start = 0
    for length in lengths.detach().cpu().tolist():
        length = int(length)
        if length > 0:
            permutation = torch.randperm(length, device=signature.device)
            shuffled[start : start + length] = signature[
                start : start + length
            ][permutation]
        start += length
    if start != signature.shape[0]:
        raise ValueError("sum(lengths) must match signature rows")
    return shuffled
