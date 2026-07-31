import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fast_adapter import FastAdapterStack


def make_cfg(anchor_mode="fps"):
    return SimpleNamespace(
        fa_num_anchors=8,
        fa_anchor_mode=anchor_mode,
        fa_anchor_level=1,
        fa_chunk_size=7,
        fa_geometry_dim=8,
        fa_attention_dim=16,
        fa_attention_heads=4,
        fa_cross_layer=True,
        fa_spatial=True,
        fa_dropout=0.0,
        fa_residual_init=1e-3,
    )


def run_packed_forward_backward(anchor_mode):
    torch.manual_seed(7)
    points0 = torch.randn(35, 3)
    lengths0 = torch.tensor([20, 15], dtype=torch.long)
    points1 = torch.cat([points0[:10], points0[20:28]], dim=0)
    lengths1 = torch.tensor([10, 8], dtype=torch.long)
    points2 = torch.cat([points1[:5], points1[10:14]], dim=0)
    lengths2 = torch.tensor([5, 4], dtype=torch.long)

    stack = FastAdapterStack([12, 16, 24], make_cfg(anchor_mode))
    state = stack.initialize_state(
        [points0, points1, points2],
        [lengths0, lengths1, lengths2],
    )
    assert state.anchor_lengths.tolist() == [8, 8]

    feature_sets = [
        torch.randn(35, 12, requires_grad=True),
        torch.randn(18, 16, requires_grad=True),
        torch.randn(9, 24, requires_grad=True),
    ]

    outputs = []
    for layer_index, (points, lengths, features) in enumerate(zip(
        [points0, points1, points2],
        [lengths0, lengths1, lengths2],
        feature_sets,
    )):
        output, state = stack.forward_layer(
            layer_index,
            points,
            lengths,
            features,
            state,
        )
        assert output.shape == features.shape
        assert torch.isfinite(output).all()
        outputs.append(output)

    sum(output.square().mean() for output in outputs).backward()
    assert all(features.grad is not None for features in feature_sets)
    assert all(torch.isfinite(features.grad).all() for features in feature_sets)
    assert any(parameter.grad is not None for parameter in stack.parameters())


class FastAdapterTest(unittest.TestCase):

    def test_all_anchor_modes_forward_backward(self):
        for anchor_mode in ["fps", "pyramid", "random", "stride"]:
            with self.subTest(anchor_mode=anchor_mode):
                run_packed_forward_backward(anchor_mode)

    def test_rejects_mismatched_packed_lengths(self):
        stack = FastAdapterStack([8], make_cfg())
        points = torch.randn(6, 3)
        with self.assertRaisesRegex(ValueError, "Pyramid lengths"):
            stack.initialize_state([points], [torch.tensor([5])])


if __name__ == "__main__":
    unittest.main()
