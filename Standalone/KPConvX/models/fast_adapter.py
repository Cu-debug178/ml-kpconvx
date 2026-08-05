#
# FastAdapter-inspired module for KPConvX.
#
# This is an independent implementation of the P2A/A2P ideas described in:
# "Mitigating Geometric Degradation in Fast DownSampling via FastAdapter for
# Point Cloud Segmentation" (ICCV 2025).
#

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class FastAdapterState:
    """Per-forward state shared by all adapter layers."""

    anchor_points: Tensor
    anchor_lengths: Tensor
    previous_anchor_features: Optional[Tensor] = None


class PointMLP(nn.Module):
    """Small MLP used by the geometry-aware P2A and A2P branches."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AnchorSelfAttention(nn.Module):
    """Self-attention over the small, padded anchor set of every cloud."""

    def __init__(
        self,
        channels: int,
        attention_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()

        attention_dim = max(1, min(channels, attention_dim))
        num_heads = max(1, min(num_heads, attention_dim))
        while attention_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1

        self.in_proj = nn.Linear(channels, attention_dim)
        self.norm = nn.LayerNorm(attention_dim)
        self.attention = nn.MultiheadAttention(
            attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(attention_dim, channels)
        self.dropout = nn.Dropout(dropout)

        # Keep the new branch close to identity at initialization.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, anchor_features: Tensor, anchor_lengths: Tensor) -> Tensor:
        if anchor_features.numel() == 0:
            return anchor_features

        batch_size = int(anchor_lengths.shape[0])
        max_anchors = int(anchor_lengths.max().item())
        channels = int(anchor_features.shape[1])

        padded = anchor_features.new_zeros((batch_size, max_anchors, channels))
        padding_mask = torch.ones(
            (batch_size, max_anchors),
            dtype=torch.bool,
            device=anchor_features.device,
        )

        start = 0
        for batch_index, length_tensor in enumerate(anchor_lengths):
            length = int(length_tensor.item())
            padded[batch_index, :length] = anchor_features[start:start + length]
            padding_mask[batch_index, :length] = False
            start += length

        projected = self.norm(self.in_proj(padded))
        attended, _ = self.attention(
            projected,
            projected,
            projected,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        padded = padded + self.dropout(self.out_proj(attended))

        outputs = []
        for batch_index, length_tensor in enumerate(anchor_lengths):
            length = int(length_tensor.item())
            outputs.append(padded[batch_index, :length])

        return torch.cat(outputs, dim=0)


class FastAdapterLayer(nn.Module):
    """One P2A Aggregator + A2P Adapter operating on a packed point batch."""

    def __init__(
        self,
        channels: int,
        previous_channels: Optional[int],
        geometry_dim: int,
        attention_dim: int,
        attention_heads: int,
        use_cross_layer: bool,
        use_spatial_attention: bool,
        dropout: float,
        residual_init: float,
    ):
        super().__init__()

        self.channels = channels
        self.geometry_dim = geometry_dim
        self.use_cross_layer = use_cross_layer and previous_channels is not None

        self.geometry_mlp = PointMLP(3, geometry_dim, geometry_dim)
        self.p2a_weight_mlp = PointMLP(2 * geometry_dim, geometry_dim, 1)
        self.offset_mlp = PointMLP(3, max(geometry_dim, 16), channels)
        self.a2p_gate_mlp = PointMLP(2 * geometry_dim, geometry_dim, 1)

        if self.use_cross_layer:
            self.cross_layer_mlp = nn.Sequential(
                nn.Linear(channels + int(previous_channels), channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )
        else:
            self.cross_layer_mlp = None

        if use_spatial_attention:
            self.spatial_attention = AnchorSelfAttention(
                channels=channels,
                attention_dim=attention_dim,
                num_heads=attention_heads,
                dropout=dropout,
            )
        else:
            self.spatial_attention = None

        self.residual_scale = nn.Parameter(
            torch.full((channels,), float(residual_init), dtype=torch.float32)
        )

    @staticmethod
    def _scatter_mean(
        values: Tensor,
        indices: Tensor,
        output_size: int,
    ) -> Tuple[Tensor, Tensor]:
        output_shape = (output_size,) + tuple(values.shape[1:])
        sums = values.new_zeros(output_shape)

        if values.ndim == 1:
            sums.scatter_add_(0, indices, values)
        else:
            expanded_indices = indices.view(
                -1, *([1] * (values.ndim - 1))
            ).expand_as(values)
            sums.scatter_add_(0, expanded_indices, values)

        counts = values.new_zeros((output_size, 1))
        counts.scatter_add_(
            0,
            indices.unsqueeze(1),
            values.new_ones((values.shape[0], 1)),
        )
        safe_counts = counts.clamp_min(1.0)

        if values.ndim == 1:
            means = sums / safe_counts.squeeze(1)
        else:
            means = sums / safe_counts.view(
                output_size, *([1] * (values.ndim - 1))
            )

        return means, counts

    def forward(
        self,
        features: Tensor,
        assignments: Tensor,
        point_to_anchor_offsets: Tensor,
        state: FastAdapterState,
        capture_summary: bool = False,
        capture_full: bool = False,
    ) -> Tuple[Tensor, FastAdapterState, Dict[str, Any]]:
        anchor_count = int(state.anchor_points.shape[0])
        input_features = features

        # P2A: geometry-aware local aggregation.
        point_geometry = self.geometry_mlp(point_to_anchor_offsets)
        anchor_geometry, anchor_counts = self._scatter_mean(
            point_geometry,
            assignments,
            anchor_count,
        )
        local_geometry = anchor_geometry.index_select(0, assignments)

        p2a_weights = torch.sigmoid(
            self.p2a_weight_mlp(
                torch.cat([point_geometry, local_geometry], dim=1)
            )
        )
        anchor_features, _ = self._scatter_mean(
            features * p2a_weights,
            assignments,
            anchor_count,
        )

        # Empty deep anchors can still retain prior-stage context.
        if self.use_cross_layer and state.previous_anchor_features is not None:
            anchor_features = anchor_features + self.cross_layer_mlp(
                torch.cat(
                    [anchor_features, state.previous_anchor_features],
                    dim=1,
                )
            )

        if self.spatial_attention is not None:
            anchor_features = self.spatial_attention(
                anchor_features,
                state.anchor_lengths,
            )

        # A2P: return refined anchor context to every point.
        point_anchor_features = anchor_features.index_select(0, assignments)
        offsets = self.offset_mlp(point_to_anchor_offsets)
        a2p_gate = torch.sigmoid(
            self.a2p_gate_mlp(
                torch.cat([point_geometry, local_geometry], dim=1)
            )
        )
        correction = a2p_gate * (point_anchor_features + offsets)
        scaled_correction = correction * self.residual_scale.to(features.dtype)
        features = features + scaled_correction

        diagnostics: Dict[str, Any] = {}
        if capture_summary or capture_full:
            feature_norm = input_features.detach().norm(dim=1).clamp_min(1e-8)
            correction_norm = scaled_correction.detach().norm(dim=1)
            correction_ratio = correction_norm / feature_norm
            summary = {
                "point_count": torch.tensor(
                    float(features.shape[0]), device=features.device
                ),
                "anchor_count": torch.tensor(
                    float(anchor_count), device=features.device
                ),
                "residual_scale_mean_abs": self.residual_scale.detach().abs().mean(),
                "residual_scale_max_abs": self.residual_scale.detach().abs().max(),
                "p2a_gate_mean": p2a_weights.detach().mean(),
                "p2a_gate_std": p2a_weights.detach().std(unbiased=False),
                "a2p_gate_mean": a2p_gate.detach().mean(),
                "a2p_gate_std": a2p_gate.detach().std(unbiased=False),
                "a2p_gate_low_fraction": (a2p_gate.detach() < 0.05).float().mean(),
                "a2p_gate_high_fraction": (a2p_gate.detach() > 0.95).float().mean(),
                "correction_ratio_mean": correction_ratio.mean(),
                "correction_ratio_p90": torch.quantile(correction_ratio, 0.9),
                "anchor_occupancy_mean": anchor_counts.detach().mean(),
                "anchor_occupancy_max": anchor_counts.detach().max(),
            }
            if self.spatial_attention is not None:
                summary["spatial_out_proj_norm"] = (
                    self.spatial_attention.out_proj.weight.detach().norm()
                )
            diagnostics["summary"] = summary

            if capture_full:
                diagnostics["full"] = {
                    "assignments": assignments.detach(),
                    "point_to_anchor_offsets": point_to_anchor_offsets.detach(),
                    "p2a_weights": p2a_weights.detach().squeeze(1),
                    "a2p_gate": a2p_gate.detach().squeeze(1),
                    "correction_norm": correction_norm,
                    "correction_ratio": correction_ratio,
                    "anchor_features": anchor_features.detach(),
                    "anchor_counts": anchor_counts.detach().squeeze(1),
                }

        state.previous_anchor_features = anchor_features
        return features, state, diagnostics


class FastAdapterStack(nn.Module):
    """
    Sampler-agnostic FastAdapter stack for KPConvX packed batches.

    Anchors are selected once from the input scene/object and remain fixed for
    the full forward pass. Every encoder stage assigns its current points to the
    same anchors, so cross-layer fusion works with any pyramid sampler.
    """

    def __init__(self, layer_channels: Sequence[int], cfg_model):
        super().__init__()

        self.num_anchors = int(getattr(cfg_model, "fa_num_anchors", 100))
        self.anchor_mode = str(
            getattr(cfg_model, "fa_anchor_mode", "fps")
        ).lower()
        self.anchor_level = int(getattr(cfg_model, "fa_anchor_level", 0))
        self.assignment_chunk_size = int(
            getattr(cfg_model, "fa_chunk_size", 16384)
        )

        geometry_dim = int(getattr(cfg_model, "fa_geometry_dim", 16))
        attention_dim = int(getattr(cfg_model, "fa_attention_dim", 64))
        attention_heads = int(getattr(cfg_model, "fa_attention_heads", 4))
        use_cross_layer = bool(getattr(cfg_model, "fa_cross_layer", True))
        use_spatial_attention = bool(getattr(cfg_model, "fa_spatial", True))
        dropout = float(getattr(cfg_model, "fa_dropout", 0.0))
        residual_init = float(getattr(cfg_model, "fa_residual_init", 1e-3))

        if self.num_anchors < 1:
            raise ValueError("fa_num_anchors must be at least 1")
        if self.anchor_mode not in {"fps", "random", "stride", "pyramid"}:
            raise ValueError(
                "fa_anchor_mode must be one of: fps, random, stride, pyramid"
            )

        layers = []
        previous_channels = None
        for channels in layer_channels:
            layers.append(
                FastAdapterLayer(
                    channels=int(channels),
                    previous_channels=previous_channels,
                    geometry_dim=geometry_dim,
                    attention_dim=attention_dim,
                    attention_heads=attention_heads,
                    use_cross_layer=use_cross_layer,
                    use_spatial_attention=use_spatial_attention,
                    dropout=dropout,
                    residual_init=residual_init,
                )
            )
            previous_channels = int(channels)
        self.layers = nn.ModuleList(layers)
        self._capture_summary = False
        self._capture_full = False
        self._last_diagnostics: Dict[str, Any] = {}

    def set_diagnostics_mode(
        self,
        summary: bool = False,
        full: bool = False,
    ) -> None:
        """Enable one-forward diagnostics without changing normal training cost."""

        self._capture_full = bool(full)
        self._capture_summary = bool(summary or full)
        self._last_diagnostics = {}

    def diagnostics(self) -> Dict[str, Any]:
        return self._last_diagnostics

    @staticmethod
    @torch.no_grad()
    def _fps_indices(points: Tensor, sample_count: int) -> Tensor:
        point_count = int(points.shape[0])
        if sample_count >= point_count:
            return torch.arange(
                point_count,
                device=points.device,
                dtype=torch.long,
            )

        selected = torch.empty(
            (sample_count,),
            device=points.device,
            dtype=torch.long,
        )
        center = points.mean(dim=0, keepdim=True)
        min_distances = torch.sum((points - center) ** 2, dim=1)
        farthest = torch.argmax(min_distances)
        min_distances.fill_(torch.finfo(points.dtype).max)

        for index in range(sample_count):
            selected[index] = farthest
            distances = torch.sum((points - points[farthest]) ** 2, dim=1)
            min_distances = torch.minimum(min_distances, distances)
            farthest = torch.argmax(min_distances)

        return selected

    @torch.no_grad()
    def _select_single_cloud_anchors(self, points: Tensor) -> Tensor:
        point_count = int(points.shape[0])
        sample_count = min(self.num_anchors, point_count)
        if sample_count < 1:
            raise ValueError("FastAdapter received an empty point cloud")

        if self.anchor_mode in {"fps", "pyramid"}:
            indices = self._fps_indices(points, sample_count)
        elif self.anchor_mode == "random":
            indices = torch.randperm(
                point_count,
                device=points.device,
            )[:sample_count]
        else:  # stride
            if sample_count == point_count:
                indices = torch.arange(point_count, device=points.device)
            else:
                indices = torch.linspace(
                    0,
                    point_count - 1,
                    sample_count,
                    device=points.device,
                ).round().long()

        return points.index_select(0, indices).detach()

    @torch.no_grad()
    def initialize_state(
        self,
        pyramid_points: Sequence[Tensor],
        pyramid_lengths: Sequence[Tensor],
    ) -> FastAdapterState:
        if self.anchor_mode == "pyramid":
            level = self.anchor_level
            if level < 0:
                level += len(pyramid_points)
            level = max(0, min(level, len(pyramid_points) - 1))
        else:
            level = 0

        source_points = pyramid_points[level]
        source_lengths = pyramid_lengths[level]

        anchors: List[Tensor] = []
        anchor_lengths: List[int] = []
        start = 0
        for length_tensor in source_lengths:
            length = int(length_tensor.item())
            cloud = source_points[start:start + length]
            cloud_anchors = self._select_single_cloud_anchors(cloud)
            anchors.append(cloud_anchors)
            anchor_lengths.append(int(cloud_anchors.shape[0]))
            start += length

        if start != int(source_points.shape[0]):
            raise ValueError("Pyramid lengths do not match the packed points")

        state = FastAdapterState(
            anchor_points=torch.cat(anchors, dim=0),
            anchor_lengths=torch.tensor(
                anchor_lengths,
                dtype=torch.long,
                device=source_points.device,
            ),
        )
        if self._capture_summary:
            self._last_diagnostics = {
                "anchor_points": state.anchor_points.detach() if self._capture_full else None,
                "anchor_lengths": state.anchor_lengths.detach(),
                "layers": {},
            }
        return state

    @torch.no_grad()
    def _assign_points_to_anchors(
        self,
        points: Tensor,
        point_lengths: Tensor,
        state: FastAdapterState,
    ) -> Tuple[Tensor, Tensor]:
        if len(point_lengths) != len(state.anchor_lengths):
            raise ValueError("Point and anchor batches have different sizes")

        assignments: List[Tensor] = []
        offsets: List[Tensor] = []
        point_start = 0
        anchor_start = 0
        chunk_size = max(1, self.assignment_chunk_size)

        for point_length_tensor, anchor_length_tensor in zip(
            point_lengths,
            state.anchor_lengths,
        ):
            point_length = int(point_length_tensor.item())
            anchor_length = int(anchor_length_tensor.item())
            if point_length < 1 or anchor_length < 1:
                raise ValueError("FastAdapter received an empty point cloud")

            cloud_points = points[point_start:point_start + point_length]
            cloud_anchors = state.anchor_points[
                anchor_start:anchor_start + anchor_length
            ]

            local_assignments = []
            for chunk_start in range(0, point_length, chunk_size):
                chunk = cloud_points[chunk_start:chunk_start + chunk_size]
                squared_distances = torch.sum(
                    (chunk[:, None, :] - cloud_anchors[None, :, :]) ** 2,
                    dim=2,
                )
                local_assignments.append(torch.argmin(squared_distances, dim=1))

            local_assignments_tensor = torch.cat(local_assignments, dim=0)
            assignments.append(local_assignments_tensor + anchor_start)
            offsets.append(
                cloud_points
                - cloud_anchors.index_select(0, local_assignments_tensor)
            )

            point_start += point_length
            anchor_start += anchor_length

        if point_start != int(points.shape[0]):
            raise ValueError("Point lengths do not match the packed points")

        return torch.cat(assignments, dim=0), torch.cat(offsets, dim=0)

    def forward_layer(
        self,
        layer_index: int,
        points: Tensor,
        point_lengths: Tensor,
        features: Tensor,
        state: FastAdapterState,
    ) -> Tuple[Tensor, FastAdapterState]:
        assignments, offsets = self._assign_points_to_anchors(
            points,
            point_lengths,
            state,
        )
        features, state, diagnostics = self.layers[layer_index](
            features,
            assignments,
            offsets,
            state,
            capture_summary=self._capture_summary,
            capture_full=self._capture_full,
        )
        if self._capture_summary:
            if not self._last_diagnostics:
                self._last_diagnostics = {
                    "anchor_points": state.anchor_points.detach() if self._capture_full else None,
                    "anchor_lengths": state.anchor_lengths.detach(),
                    "layers": {},
                }
            self._last_diagnostics["layers"][int(layer_index)] = diagnostics
        return features, state
