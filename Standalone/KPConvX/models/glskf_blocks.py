"""Global-to-Local Semantic Kernel Feedback (GLSKF).

KPConvX generates its kernel modulation from the *current* stage feature, so
the operator only ever modulates itself with bottom-up evidence.  GLSKF is the
top-down counterpart: deep stages (large receptive field, reliable category
evidence) produce a spatially aligned gate that rescales the *effective*
depthwise kernel weights of a shallower stage, and a zero-initialised residual
conv writes the correction back into the skip connection that the decoder uses.

Two properties are load-bearing for the experiments this module exists for:

1. The base ``nn.Parameter`` weights are never mutated.  The gate multiplies a
   temporary copy of the gathered per-neighbour weights, so autograd and the
   optimizer state stay intact and no room can leak into the next one.
2. At initialisation the block is *exactly* the identity, because the outer
   per-channel ``scale`` starts at zero.  An L0 checkpoint therefore keeps its
   logits bit-for-bit, and ``scale`` frozen at zero is a free causal control.

The gate itself is initialised in its normal ``1 + tanh(raw)`` regime rather
than at zero, otherwise the first optimizer steps would see no gate gradient at
all and the module would begin its life as a plain residual convolution.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from models.generic_blocks import NormBlock
from models.kpnext_blocks import KPConvD, apply_kernel_gate

GLSKF_MODES = ("none", "film", "kernel_gate", "matched_mlp")
GLSKF_CONTEXT_CONTROLS = ("none", "shuffle", "room_mean")
GLSKF_INFERENCE_ABLATIONS = (
    "none",
    "shuffle",
    "room_mean",
    "zero_context",
    "neutral_gate",
    "branch_off",
)

# ``apply_kernel_gate`` lives next to KPConvD (it manipulates gathered kernel
# weights) but belongs to this mechanism, so it is re-exported here.
__all__ = [
    "GLSKF_MODES",
    "GLSKF_CONTEXT_CONTROLS",
    "GLSKF_INFERENCE_ABLATIONS",
    "GlskfFeedback",
    "apply_kernel_gate",
    "room_mean_broadcast",
    "shuffle_packed_rows",
]


def shuffle_packed_rows(features: Tensor, lengths: Tensor) -> Tensor:
    """Permute rows inside each packed cloud, never across clouds."""

    shuffled = torch.empty_like(features)
    start = 0
    for length in lengths.detach().cpu().tolist():
        length = int(length)
        if length > 0:
            permutation = torch.randperm(length, device=features.device)
            shuffled[start:start + length] = features[start:start + length][permutation]
        start += length
    if start != features.shape[0]:
        raise ValueError("sum(lengths) must match the number of rows")
    return shuffled


def room_mean_broadcast(features: Tensor, lengths: Tensor) -> Tensor:
    """Replace every row by its own cloud mean.

    This keeps whatever room-level semantics the deep stages produced while
    destroying the point-to-context correspondence, which is a much tighter
    control than an in-room shuffle (a shuffle also destroys the marginal
    statistics of any single point's context).
    """

    pooled = torch.empty_like(features)
    start = 0
    for length in lengths.detach().cpu().tolist():
        length = int(length)
        if length > 0:
            window = features[start:start + length]
            pooled[start:start + length] = window.mean(dim=0, keepdim=True)
        start += length
    if start != features.shape[0]:
        raise ValueError("sum(lengths) must match the number of rows")
    return pooled


class GlskfFeedback(nn.Module):
    """Correct one encoder skip connection using deep semantic context."""

    def __init__(
        self,
        refine_channels: int,
        context_channels: Sequence[int],
        shell_sizes: Sequence[int],
        radius: float,
        sigma: float,
        mode: str = "kernel_gate",
        groups: int = 8,
        hidden_dim: int = 64,
        matched_hidden_dim: int = 0,
        dimension: int = 3,
        influence_mode: str = "linear",
        fixed_kernel_points: str = "center",
        norm_type: str = "batch",
        bn_momentum: float = 0.1,
        shared_kp_data=None,
        detach_context: bool = False,
        deep_residual: bool = False,
        context_control: str = "none",
        inference_ablation: str = "none",
    ):
        super().__init__()

        mode = str(mode).lower()
        context_control = str(context_control).lower()
        inference_ablation = str(inference_ablation).lower()
        if mode not in GLSKF_MODES or mode == "none":
            raise ValueError(
                "glskf_mode must be one of {}".format(GLSKF_MODES[1:])
            )
        if context_control not in GLSKF_CONTEXT_CONTROLS:
            raise ValueError(
                "glskf_context_control must be one of {}".format(GLSKF_CONTEXT_CONTROLS)
            )
        if inference_ablation not in GLSKF_INFERENCE_ABLATIONS:
            raise ValueError(
                "glskf_inference_ablation must be one of {}".format(
                    GLSKF_INFERENCE_ABLATIONS
                )
            )
        if context_control != "none" and inference_ablation != "none":
            raise ValueError(
                "training context control and inference ablation cannot be combined"
            )
        if (
            inference_ablation in {"shuffle", "room_mean", "zero_context"}
            and mode == "matched_mlp"
        ):
            raise ValueError(
                "context interventions require a GLSKF mode that reads deep context"
            )
        if inference_ablation == "neutral_gate" and mode != "kernel_gate":
            raise ValueError("neutral_gate is only defined for glskf_mode='kernel_gate'")
        if inference_ablation == "neutral_gate" and deep_residual:
            raise ValueError("neutral_gate requires glskf_deep_residual=False")
        if influence_mode == "mlp":
            raise ValueError("GLSKF needs nearest-kernel assignment, not kp_influence='mlp'")
        if deep_residual and mode != "kernel_gate":
            # Feeding the deep context in as features would defeat both the FiLM
            # ablation and the matched control, which must not see it at all.
            raise ValueError("glskf_deep_residual is only defined for glskf_mode='kernel_gate'")
        if mode == "matched_mlp" and context_control != "none":
            raise ValueError("the matched control ignores the deep context, so it takes no context control")
        if refine_channels < 1:
            raise ValueError("refine_channels must be positive")
        if hidden_dim < 1:
            raise ValueError("glskf_hidden_dim must be positive")
        context_width = int(sum(int(width) for width in context_channels))
        if context_width < 1:
            raise ValueError("GLSKF needs at least one context stage")

        self.mode = mode
        self.context_control = context_control
        self.inference_ablation = inference_ablation
        self.detach_context = bool(detach_context)
        self.deep_residual = bool(deep_residual)
        self.refine_channels = int(refine_channels)
        self.context_width = context_width
        self.num_kernels = int(np.sum(shell_sizes))

        if mode == "film":
            self.groups = 0
            gate_out = 2 * self.refine_channels
        else:
            groups = int(groups)
            if groups < 1:
                raise ValueError("glskf_groups must be positive")
            if self.refine_channels % groups != 0:
                raise ValueError(
                    "glskf_groups ({:d}) must divide the refined stage width ({:d})".format(
                        groups, self.refine_channels
                    )
                )
            self.groups = groups
            gate_out = self.num_kernels * groups

        # The matched control reads the refined stage feature instead of the deep
        # context.  Its hidden width is enlarged so that the added parameter
        # count matches the real kernel-gate branch as closely as an integer
        # allows: capacity is then held fixed and only the *source* of the gate
        # differs, which is the comparison the experiment is about.
        gate_in = context_width
        gate_hidden = int(hidden_dim)
        if mode == "matched_mlp":
            gate_in = self.refine_channels
            if int(matched_hidden_dim) > 0:
                gate_hidden = int(matched_hidden_dim)
            else:
                target = context_width * hidden_dim + hidden_dim * gate_out
                gate_hidden = max(1, round(target / (gate_in + gate_out)))
        self.gate_in = gate_in
        self.gate_hidden = gate_hidden

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(gate_hidden, gate_out),
        )
        # Small initial gate deviations keep the first steps well conditioned
        # while still giving the gate a non-zero gradient.
        nn.init.zeros_(self.gate_mlp[-1].bias)
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.02)

        if mode == "film":
            self.conv = None
            self.conv_norm = None
            self.activation = None
            self.down_mlp = None
            self.in_mlp = None
        else:
            conv_in = self.refine_channels
            self.in_mlp = None
            if self.deep_residual:
                # The deep context also enters as *features*, not only as a
                # gate, so the two pathways can be told apart.
                self.in_mlp = nn.Linear(
                    self.refine_channels + context_width, conv_in, bias=False
                )
            self.conv = KPConvD(
                conv_in,
                list(shell_sizes),
                radius,
                sigma,
                shared_kp_data=shared_kp_data,
                dimension=dimension,
                influence_mode=influence_mode,
                fixed_kernel_points=fixed_kernel_points,
                norm_type=norm_type,
                bn_momentum=bn_momentum,
            )
            self.conv_norm = NormBlock(conv_in, norm_type, bn_momentum)
            self.activation = nn.LeakyReLU(0.1)
            self.down_mlp = nn.Linear(conv_in, self.refine_channels, bias=False)

        # Identity switch.  Zero here makes the whole module a no-op at load
        # time, which is what lets an L0 checkpoint reproduce its logits.
        self.scale = nn.Parameter(torch.zeros(self.refine_channels))

        self.last_stats = {}
        self._collect_stats = False

    def set_diagnostics(self, enabled: bool):
        self._collect_stats = bool(enabled)

    def added_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def build_gate(self, gate_source: Tensor) -> Tensor:
        """Map the gate source to ``(M, K, G)`` multiplicative kernel weights."""

        raw = self.gate_mlp(gate_source)
        gate = 1.0 + torch.tanh(raw)
        return gate.view(gate_source.shape[0], self.num_kernels, self.groups)

    def forward(
        self,
        q_pts: Tensor,
        refine_feats: Tensor,
        context_feats: Tensor | None,
        neighb_inds: Tensor,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """Return the corrected skip feature for the refined stage."""

        self.last_stats = {}
        if self.training and self.inference_ablation != "none":
            raise RuntimeError("glskf_inference_ablation is evaluation-only")

        if refine_feats.shape[1] != self.refine_channels:
            raise ValueError(
                "GLSKF expected {:d} refine channels, got {:d}".format(
                    self.refine_channels, refine_feats.shape[1]
                )
            )
        if refine_feats.shape[0] == 0:
            return refine_feats
        if self.inference_ablation == "branch_off":
            if self._collect_stats:
                self.last_stats = {
                    "inference_ablation": "branch_off",
                    "scale_abs_mean": self.scale.abs().mean().item(),
                    "correction_ratio": 0.0,
                }
            return refine_feats

        if self.mode == "matched_mlp":
            gate_source = refine_feats
        else:
            if context_feats is None:
                raise ValueError("GLSKF requires deep context features")
            if context_feats.shape[1] != self.context_width:
                raise ValueError(
                    "GLSKF expected {:d} context channels, got {:d}".format(
                        self.context_width, context_feats.shape[1]
                    )
                )
            gate_source = context_feats
            context_control = self.context_control
            if self.inference_ablation in {"shuffle", "room_mean"}:
                context_control = self.inference_ablation
            if context_control != "none":
                if lengths is None:
                    raise ValueError("context controls need packed cloud lengths")
                if context_control == "shuffle":
                    gate_source = shuffle_packed_rows(gate_source, lengths)
                else:
                    gate_source = room_mean_broadcast(gate_source, lengths)
            elif self.inference_ablation == "zero_context":
                gate_source = torch.zeros_like(gate_source)
            if self.detach_context:
                gate_source = gate_source.detach()

        if self.mode == "film":
            film = self.gate_mlp(gate_source)
            gamma, beta = film.chunk(2, dim=1)
            correction = refine_feats * torch.tanh(gamma) + beta
        else:
            if self.inference_ablation == "neutral_gate":
                gate = torch.ones(
                    (refine_feats.shape[0], self.num_kernels, self.groups),
                    dtype=refine_feats.dtype,
                    device=refine_feats.device,
                )
            else:
                gate = self.build_gate(gate_source)
            conv_feats = refine_feats
            if self.in_mlp is not None:
                conv_feats = self.in_mlp(
                    torch.cat([refine_feats, gate_source], dim=1)
                )
            correction = self.conv(
                q_pts, q_pts, conv_feats, neighb_inds, kernel_gate=gate
            )
            correction = self.activation(self.conv_norm(correction))
            correction = self.down_mlp(correction)
            if self._collect_stats:
                with torch.no_grad():
                    self.last_stats = {
                        "gate_mean": gate.mean().item(),
                        "gate_std": gate.std().item(),
                        "gate_min": gate.amin().item(),
                        "gate_max": gate.amax().item(),
                    }

        scale = self.scale.to(correction.dtype)
        refined = refine_feats + scale.unsqueeze(0) * correction
        if self._collect_stats:
            with torch.no_grad():
                base = refine_feats.float().norm() + 1e-6
                delta = (scale.unsqueeze(0) * correction).float().norm()
                self.last_stats["scale_abs_mean"] = self.scale.abs().mean().item()
                self.last_stats["correction_ratio"] = (delta / base).item()
                self.last_stats["inference_ablation"] = self.inference_ablation
        return refined

    def __repr__(self):
        return (
            "GlskfFeedback(mode: {:s}, C: {:d}, ctx: {:d}, K: {:d}, G: {:d}, "
            "hidden: {:d}, control: {:s}, ablation: {:s})".format(
                self.mode,
                self.refine_channels,
                self.context_width,
                self.num_kernels,
                self.groups,
                self.gate_hidden,
                self.context_control,
                self.inference_ablation,
            )
        )
