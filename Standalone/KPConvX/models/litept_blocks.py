"""LitePT-inspired stage-specialized attention blocks for packed point clouds.

This module is a dependency-light adaptation for the standalone KPConvX codebase.
It follows the LitePT design principle (convolution at high resolution, token
attention at low resolution) and the PointROPE equations, while reusing the
existing KPConvX point pyramid and packed variable-length batches.

The implementation intentionally does not import LitePT's spconv/FlashAttention
stack.  Local patches are formed with Morton (Z-order) serialization and are
processed with PyTorch scaled-dot-product attention, which can automatically use
an optimized CUDA kernel when the installed PyTorch build supports it.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.generic_blocks import DropPathPack


_SUPPORTED_ORDERS = {"z", "z-trans"}


def _round_attention_dim(channels: int, num_heads: int, ratio: float) -> int:
    """Return an attention width whose per-head dimension is divisible by six.

    PointROPE splits every head into x/y/z subspaces and applies pairwise rotary
    embedding in each subspace.  Consequently, the per-head width must be a
    multiple of six.  Rounding to the nearest valid width keeps the adapter close
    to the requested channel ratio without forcing KPConvX stage widths to change.
    """

    if channels <= 0:
        raise ValueError("channels must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if ratio <= 0:
        raise ValueError("attention ratio must be positive")

    multiple = 6 * num_heads
    requested = max(1, int(round(channels * ratio)))
    units = max(1, int(round(requested / multiple)))
    return units * multiple


def _quantize_cloud(points: Tensor, voxel_size: float) -> Tensor:
    """Quantize one cloud to non-negative integer coordinates."""

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if points.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.long, device=points.device)

    # Account for the precision of the stored absolute coordinates.  Without
    # this tolerance, translating float32 points can move values that lie on a
    # voxel boundary into the preceding voxel.  Keep the original dtype because
    # FP64 coordinate transforms are disproportionately slow on consumer GPUs.
    origin = points.amin(dim=0, keepdim=True)
    scaled = (points - origin) / voxel_size
    if torch.is_floating_point(points):
        magnitude = torch.maximum(points.abs(), origin.abs())
        tolerance = 4.0 * torch.finfo(points.dtype).eps * magnitude / voxel_size
        scaled = scaled + tolerance
    return torch.floor(scaled).to(torch.long)


def _morton_code(coords: Tensor, order: str = "z", max_bits: int = 20) -> Tensor:
    """Encode non-negative integer coordinates into sortable Morton codes.

    ``z-trans`` swaps x and y before interleaving.  Cycling between the two
    orders changes local patch boundaries without introducing another dependency.
    Coordinates that exceed ``max_bits`` are uniformly right-shifted, preserving
    coarse spatial order while avoiding int64 overflow.
    """

    if order not in _SUPPORTED_ORDERS:
        raise ValueError(
            "Unsupported serialization order {!r}; expected one of {}".format(
                order, sorted(_SUPPORTED_ORDERS)
            )
        )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [N, 3]")
    if not 1 <= max_bits <= 21:
        raise ValueError("max_bits must be between 1 and 21 for int64 Morton codes")
    if coords.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=coords.device)
    # GPU coordinates come from ``_quantize_packed_clouds`` in the hot path.
    # Avoid a validation reduction that would synchronize the device.
    if coords.device.type == "cpu" and torch.any(coords < 0):
        raise ValueError("Morton coordinates must be non-negative")

    xyz = coords
    if order == "z-trans":
        xyz = coords[:, [1, 0, 2]]

    # Keep the bit-width decision on-device.  Calling ``item()`` here would
    # synchronize CUDA once for every cloud and serialization order.
    max_coord = xyz.amax().clamp(min=1)
    bits_needed = torch.floor(torch.log2(max_coord.to(torch.float64))).to(torch.long) + 1
    shift = torch.clamp(bits_needed - max_bits, min=0)
    xyz = torch.bitwise_right_shift(xyz, shift)

    code = torch.zeros((xyz.shape[0],), dtype=torch.long, device=xyz.device)
    for bit in range(max_bits):
        code |= ((xyz[:, 0] >> bit) & 1) << (3 * bit)
        code |= ((xyz[:, 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((xyz[:, 2] >> bit) & 1) << (3 * bit + 2)
    return code


def _packed_length_values(points: Tensor, lengths: Tensor) -> Tuple[int, ...]:
    """Validate a packed layout and copy its small length vector once."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if lengths.ndim != 1:
        raise ValueError("lengths must have shape [B]")
    if lengths.dtype == torch.bool or torch.is_floating_point(lengths) or torch.is_complex(lengths):
        raise ValueError("lengths must use an integer dtype")

    length_values = tuple(int(length) for length in lengths.detach().cpu().tolist())
    if any(length < 0 for length in length_values):
        raise ValueError("lengths must be non-negative")
    if sum(length_values) != points.shape[0]:
        raise ValueError("sum(lengths) must match the packed point count")
    return length_values


def _quantize_packed_clouds(
    points: Tensor,
    length_values: Sequence[int],
    voxel_size: float,
) -> Tensor:
    """Quantize packed clouds independently so translations do not affect ROPE."""

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    quantized: List[Tensor] = []
    start = 0
    for length in length_values:
        if length > 0:
            quantized.append(_quantize_cloud(points[start : start + length], voxel_size))
        start += length
    if not quantized:
        return torch.empty((0, 3), dtype=torch.long, device=points.device)
    return torch.cat(quantized, dim=0)


def _build_patches_from_quantized(
    points: Tensor,
    length_values: Sequence[int],
    quantized: Tensor,
    patch_size: int,
    order: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build one serialization layout from reusable quantized coordinates."""

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if order not in _SUPPORTED_ORDERS:
        raise ValueError("Unsupported serialization order: {}".format(order))
    if quantized.shape != points.shape or quantized.dtype != torch.long:
        raise ValueError("quantized coordinates must be int64 with shape [N, 3]")
    if quantized.device != points.device:
        raise ValueError("points and quantized coordinates must be on the same device")

    all_indices: List[Tensor] = []
    all_masks: List[Tensor] = []
    all_coords: List[Tensor] = []
    start = 0

    for length in length_values:
        if length == 0:
            continue

        grid = quantized[start : start + length]
        code = _morton_code(grid, order=order)
        permutation = torch.argsort(code, stable=True)

        sorted_global = permutation + start
        sorted_grid = grid[permutation]
        num_patches = (length + patch_size - 1) // patch_size
        padded_length = num_patches * patch_size
        padding = padded_length - length

        if padding > 0:
            sorted_global = torch.cat(
                [sorted_global, sorted_global[-1:].expand(padding)], dim=0
            )
            sorted_grid = torch.cat(
                [sorted_grid, sorted_grid[-1:].expand(padding, -1)], dim=0
            )

        valid = torch.arange(padded_length, device=points.device) < length
        all_indices.append(sorted_global.view(num_patches, patch_size))
        all_masks.append(valid.view(num_patches, patch_size))
        all_coords.append(sorted_grid.view(num_patches, patch_size, 3))
        start += length

    if not all_indices:
        empty_i = torch.empty((0, patch_size), dtype=torch.long, device=points.device)
        empty_m = torch.empty((0, patch_size), dtype=torch.bool, device=points.device)
        empty_c = torch.empty((0, patch_size, 3), dtype=torch.long, device=points.device)
        return empty_i, empty_m, empty_c

    return (
        torch.cat(all_indices, dim=0),
        torch.cat(all_masks, dim=0),
        torch.cat(all_coords, dim=0),
    )


@torch.no_grad()
def build_serialized_patches(
    points: Tensor,
    lengths: Tensor,
    patch_size: int,
    voxel_size: float,
    order: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build fixed-size local patches for a packed point batch.

    Returns:
        patch_indices: ``[num_patches, patch_size]`` indices into packed points.
        valid_mask: ``[num_patches, patch_size]``; false entries are padding.
        grid_coords: quantized coordinates aligned with ``patch_indices``.

    Every real point occurs exactly once.  The last patch of each cloud is padded
    by repeating its final valid index, but those entries are masked in attention.
    """

    length_values = _packed_length_values(points, lengths)
    quantized = _quantize_packed_clouds(points, length_values, voxel_size)
    return _build_patches_from_quantized(
        points=points,
        length_values=length_values,
        quantized=quantized,
        patch_size=patch_size,
        order=order,
    )


class SerializedPatchCache:
    """Per-stage, per-forward cache for serialization and patch metadata.

    A LitePT stage normally contains multiple attention blocks that cycle over a
    small set of serialization orders.  Sorting in every block would erase much
    of the intended speed benefit, so all blocks in a stage share this cache.
    ``KPNeXt.forward`` clears it before processing a new batch.
    """

    def __init__(self):
        self._entries = {}
        self._quantized_entries = {}

    def clear(self):
        self._entries.clear()
        self._quantized_entries.clear()

    @property
    def quantization_count(self) -> int:
        return len(self._quantized_entries)

    @property
    def layout_count(self) -> int:
        return len(self._entries)

    @torch.no_grad()
    def get(self, points: Tensor, lengths: Tensor, patch_size: int,
            voxel_size: float, order: str) -> Tuple[Tensor, Tensor, Tensor]:
        geometry_key = (
            float(voxel_size),
            int(points.data_ptr()),
            tuple(points.shape),
            int(lengths.data_ptr()),
            tuple(lengths.shape),
        )
        if geometry_key not in self._quantized_entries:
            length_values = _packed_length_values(points, lengths)
            quantized = _quantize_packed_clouds(points, length_values, voxel_size)
            self._quantized_entries[geometry_key] = (length_values, quantized)

        key = (order, int(patch_size), geometry_key)
        if key not in self._entries:
            length_values, quantized = self._quantized_entries[geometry_key]
            self._entries[key] = _build_patches_from_quantized(
                points=points,
                length_values=length_values,
                quantized=quantized,
                patch_size=patch_size,
                order=order,
            )
        return self._entries[key]


class PointROPE(nn.Module):
    """Parameter-free 3D rotary positional embedding for query/key tensors.

    Inputs have shape ``[patches, heads, tokens, head_dim]`` and positions have
    shape ``[patches, tokens, 3]``.  ``head_dim`` must be divisible by six.
    """

    def __init__(self, base: float = 100.0):
        super().__init__()
        if base <= 1:
            raise ValueError("PointROPE base must be greater than one")
        self.base = float(base)

    @staticmethod
    def _rotate_axis(axis_features: Tensor, axis_position: Tensor, base: float) -> Tensor:
        axis_dim = axis_features.shape[-1]
        if axis_dim % 2 != 0:
            raise ValueError("Each PointROPE axis subspace must have even width")

        pair_dim = axis_dim // 2
        frequency_index = torch.arange(
            pair_dim, device=axis_features.device, dtype=torch.float32
        )
        inv_frequency = base ** (-frequency_index / max(pair_dim, 1))
        angles = axis_position.to(torch.float32).unsqueeze(-1) * inv_frequency
        cos = torch.cos(angles).unsqueeze(1).to(axis_features.dtype)
        sin = torch.sin(angles).unsqueeze(1).to(axis_features.dtype)

        first, second = axis_features.split(pair_dim, dim=-1)
        return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)

    def forward(self, features: Tensor, positions: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape [P, H, K, D]")
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [P, K, 3]")
        if features.shape[0] != positions.shape[0] or features.shape[2] != positions.shape[1]:
            raise ValueError("features and positions have incompatible patch dimensions")

        head_dim = features.shape[-1]
        if head_dim % 6 != 0:
            raise ValueError("PointROPE head dimension must be divisible by 6")
        axis_dim = head_dim // 3

        x_features, y_features, z_features = features.split(axis_dim, dim=-1)
        x_rotated = self._rotate_axis(x_features, positions[..., 0], self.base)
        y_rotated = self._rotate_axis(y_features, positions[..., 1], self.base)
        z_rotated = self._rotate_axis(z_features, positions[..., 2], self.base)
        return torch.cat([x_rotated, y_rotated, z_rotated], dim=-1)


class SerializedPointROPEAttention(nn.Module):
    """Local token attention over Morton-serialized point patches."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        patch_size: int = 128,
        attention_ratio: float = 1.0,
        rope_base: float = 100.0,
        rope_enabled: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        order: str = "z",
        patch_cache: SerializedPatchCache | None = None,
    ):
        super().__init__()
        if order not in _SUPPORTED_ORDERS:
            raise ValueError("Unsupported serialization order: {}".format(order))
        if not 0 <= attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0 <= projection_dropout < 1:
            raise ValueError("projection_dropout must be in [0, 1)")

        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.patch_size = int(patch_size)
        self.order = order
        self.attention_dim = _round_attention_dim(
            self.channels, self.num_heads, attention_ratio
        )
        self.head_dim = self.attention_dim // self.num_heads
        self.attention_dropout = float(attention_dropout)
        self.rope_enabled = bool(rope_enabled)
        self.patch_cache = patch_cache

        self.qkv = nn.Linear(self.channels, 3 * self.attention_dim, bias=True)
        self.projection = nn.Linear(self.attention_dim, self.channels, bias=True)
        self.projection_dropout = nn.Dropout(projection_dropout)
        self.rope = PointROPE(base=rope_base)

    def forward(
        self,
        points: Tensor,
        features: Tensor,
        lengths: Tensor,
        voxel_size: float,
    ) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.channels:
            raise ValueError(
                "features must have shape [N, {}], got {}".format(
                    self.channels, tuple(features.shape)
                )
            )
        if features.shape[0] != points.shape[0]:
            raise ValueError("points and features must have the same packed length")
        if features.shape[0] == 0:
            return features

        if self.patch_cache is None:
            patch_indices, valid_mask, grid_coords = build_serialized_patches(
                points=points,
                lengths=lengths,
                patch_size=self.patch_size,
                voxel_size=voxel_size,
                order=self.order,
            )
        else:
            patch_indices, valid_mask, grid_coords = self.patch_cache.get(
                points=points,
                lengths=lengths,
                patch_size=self.patch_size,
                voxel_size=voxel_size,
                order=self.order,
            )

        patch_features = features[patch_indices]
        num_patches = patch_features.shape[0]
        qkv = self.qkv(patch_features)
        qkv = qkv.view(
            num_patches,
            self.patch_size,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        if self.rope_enabled:
            query = self.rope(query, grid_coords)
            key = self.rope(key, grid_coords)

        # SDPA bool masks use True for entries that participate in attention.
        attention_mask = valid_mask[:, None, None, :]
        dropout_p = self.attention_dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            num_patches, self.patch_size, self.attention_dim
        )
        attended = self.projection_dropout(self.projection(attended))

        # Each real point appears exactly once; padded outputs are intentionally ignored.
        output = torch.empty_like(features)
        flat_valid = valid_mask.reshape(-1)
        flat_indices = patch_indices.reshape(-1)[flat_valid]
        flat_output = attended.reshape(-1, self.channels)[flat_valid]
        output.index_copy_(0, flat_indices, flat_output)
        return output


class LitePointTransformerBlock(nn.Module):
    """Pre-norm PointROPE attention + MLP block compatible with KPNeXt forward."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        voxel_size: float,
        num_heads: int = 8,
        patch_size: int = 128,
        attention_ratio: float = 1.0,
        mlp_ratio: float = 4.0,
        rope_base: float = 100.0,
        rope_enabled: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        drop_path: float = 0.0,
        order: str = "z",
        patch_cache: SerializedPatchCache | None = None,
    ):
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.voxel_size = float(voxel_size)
        self.input_projection = (
            nn.Linear(self.in_channels, self.out_channels, bias=False)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )
        self.norm1 = nn.LayerNorm(self.out_channels, eps=1e-6)
        self.attention = SerializedPointROPEAttention(
            channels=self.out_channels,
            num_heads=num_heads,
            patch_size=patch_size,
            attention_ratio=attention_ratio,
            rope_base=rope_base,
            rope_enabled=rope_enabled,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            order=order,
            patch_cache=patch_cache,
        )
        self.norm2 = nn.LayerNorm(self.out_channels, eps=1e-6)
        hidden_channels = max(1, int(round(self.out_channels * mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(self.out_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(hidden_channels, self.out_channels),
            nn.Dropout(projection_dropout),
        )
        self.drop_path = DropPathPack(drop_path)

    def forward(
        self,
        q_pts: Tensor,
        s_pts: Tensor,
        s_feats: Tensor,
        neighb_inds: Tensor,
        lengths: Tensor,
        upcut: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor | None]:
        del q_pts, neighb_inds  # Kept in the signature for KPNeXt block compatibility.
        features = self.input_projection(s_feats)
        features = features + self.drop_path(
            self.attention(s_pts, self.norm1(features), lengths, self.voxel_size),
            lengths,
        )
        features = features + self.drop_path(self.mlp(self.norm2(features)), lengths)
        return features, upcut


class LiteHandoverBlock(nn.Module):
    """A convolution block followed by PointROPE attention in one U-Net stage."""

    def __init__(self, convolution: nn.Module, attention: LitePointTransformerBlock):
        super().__init__()
        self.convolution = convolution
        self.attention = attention

    def forward(
        self,
        q_pts: Tensor,
        s_pts: Tensor,
        s_feats: Tensor,
        neighb_inds: Tensor,
        lengths: Tensor,
        upcut: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor | None]:
        features, next_upcut = self.convolution(
            q_pts,
            s_pts,
            s_feats,
            neighb_inds,
            lengths,
            upcut=upcut,
        )
        features, _ = self.attention(
            q_pts,
            s_pts,
            features,
            neighb_inds,
            lengths,
            upcut=None,
        )
        return features, next_upcut


def parse_serialization_orders(value: str | Sequence[str]) -> Tuple[str, ...]:
    """Normalize a comma-separated or sequence-valued serialization setting."""

    if isinstance(value, str):
        orders = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        orders = tuple(str(item).strip() for item in value if str(item).strip())
    if not orders:
        raise ValueError("At least one serialization order is required")
    unsupported = sorted(set(orders) - _SUPPORTED_ORDERS)
    if unsupported:
        raise ValueError(
            "Unsupported serialization orders {}; supported orders are {}".format(
                unsupported, sorted(_SUPPORTED_ORDERS)
            )
        )
    return orders
