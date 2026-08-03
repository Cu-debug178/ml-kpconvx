"""Small CPU integration tests for the LitePT/KPNeXt wiring.

The production project builds point pyramids with compiled operators.  These
checks inject a complete synthetic pyramid, so they exercise the actual stem,
KPConvD stages, PointROPE attention stages, pooling, heads and backward pass
without requiring the C++ extensions in a lightweight CI environment.
"""

import os
import sys
import types
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


try:
    from easydict import EasyDict
except ImportError:
    class EasyDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    easydict_module = types.ModuleType("easydict")
    easydict_module.EasyDict = EasyDict
    sys.modules["easydict"] = easydict_module

# KPNext imports fill_pyramid at module import time.  The tests always supply a
# complete pyramid, therefore importing the compiled pyramid stack is unnecessary.
torch_pyramid_module = types.ModuleType("utils.torch_pyramid")

def _unexpected_fill(*args, **kwargs):
    raise AssertionError("the synthetic test pyramid should already be complete")

torch_pyramid_module.fill_pyramid = _unexpected_fill
sys.modules.setdefault("utils.torch_pyramid", torch_pyramid_module)

from models.KPNext import KPNeXt  # noqa: E402
from models.litept_blocks import LiteHandoverBlock, LitePointTransformerBlock  # noqa: E402
from utils.config import init_cfg  # noqa: E402


def _make_config(task, handover_stage=0):
    cfg = init_cfg()
    cfg.data.task = task
    cfg.data.dim = 3
    cfg.data.init_sub_size = 0.1
    cfg.data.label_values = list(range(4))
    cfg.data.ignored_labels = []

    cfg.model.in_sub_size = 0.1
    cfg.model.in_sub_mode = "grid"
    cfg.model.kp_radius = 2.1
    cfg.model.kp_sigma = 2.1
    cfg.model.radius_scaling = 2.0
    cfg.model.neighbor_limits = [4] * 5
    cfg.model.layer_blocks = (1, 1, 1, 1, 1)
    cfg.model.input_channels = 5 if task == "cloud_segmentation" else 4
    cfg.model.init_channels = 16
    cfg.model.channel_scaling = 1.41
    cfg.model.kp_mode = "kpconvx"
    cfg.model.shell_sizes = [1, 14, 28]
    cfg.model.kp_influence = "linear"
    cfg.model.kp_aggregation = "nearest"
    cfg.model.conv_groups = -1
    cfg.model.share_kp = True
    cfg.model.grid_pool = True
    cfg.model.decoder_layer = True
    cfg.model.drop_path_rate = 0.0
    cfg.model.first_inv_layer = 1
    cfg.model.inv_groups = 4
    cfg.model.inv_grp_norm = True
    cfg.model.inv_act = "sigmoid"
    cfg.model.kpx_upcut = False
    cfg.model.upsample_n = 1
    cfg.model.norm = "batch"
    cfg.model.bn_momentum = 0.1

    cfg.model.litept_enabled = True
    cfg.model.litept_conv_stages = 2 if handover_stage else 3
    cfg.model.litept_handover_stage = handover_stage
    cfg.model.litept_patch_size = 8
    cfg.model.litept_num_heads = 2
    cfg.model.litept_attention_ratio = 1.0
    cfg.model.litept_mlp_ratio = 2.0
    cfg.model.litept_rope_base = 100.0
    cfg.model.litept_rope_enabled = True
    cfg.model.litept_attention_dropout = 0.0
    cfg.model.litept_projection_dropout = 0.0
    cfg.model.litept_orders = "z,z-trans"
    cfg.model.litept_light_decoder = task == "cloud_segmentation"

    cfg.model.fa_enabled = False
    cfg.model.fa_train_mode = "joint"
    cfg.model.fa_num_anchors = 4
    cfg.model.fa_anchor_mode = "stride"
    cfg.model.fa_anchor_level = 0
    cfg.model.fa_geometry_dim = 8
    cfg.model.fa_attention_dim = 16
    cfg.model.fa_attention_heads = 4
    cfg.model.fa_chunk_size = 32
    cfg.model.fa_cross_layer = True
    cfg.model.fa_spatial = True
    cfg.model.fa_dropout = 0.0
    cfg.model.fa_residual_init = 1e-3

    cfg.train.smooth_labels = False
    cfg.train.class_w = []
    cfg.train.deform_loss_factor = 0.0
    cfg.train.deform_fit_rep_ratio = 0.0
    return cfg


def _make_batch(input_channels):
    level_counts = [(24, 16), (12, 8), (6, 4), (3, 2), (2, 1)]
    points = []
    lengths = []
    neighbors = []
    for level, counts in enumerate(level_counts):
        clouds = [
            torch.randn(count, 3) * (0.2 + level * 0.1) + batch_index * 3
            for batch_index, count in enumerate(counts)
        ]
        level_points = torch.cat(clouds, dim=0)
        points.append(level_points)
        lengths.append(torch.tensor(counts, dtype=torch.long))
        neighbors.append(
            torch.arange(level_points.shape[0], dtype=torch.long)
            .unsqueeze(1)
            .repeat(1, 4)
        )

    pools = []
    upsamples = []
    for level in range(4):
        pool_rows = []
        upsample_rows = []
        previous_offset = 0
        next_offset = 0
        for previous_count, next_count in zip(
            level_counts[level], level_counts[level + 1]
        ):
            selected = torch.linspace(0, previous_count - 1, next_count).long()
            pool_rows.append(
                torch.stack([selected, selected], dim=1) + previous_offset
            )
            assignments = torch.div(
                torch.arange(previous_count) * next_count,
                previous_count,
                rounding_mode="floor",
            ).clamp(max=next_count - 1)
            upsample_rows.append((assignments + next_offset).unsqueeze(1))
            previous_offset += previous_count
            next_offset += next_count
        pools.append(torch.cat(pool_rows, dim=0))
        upsamples.append(torch.cat(upsample_rows, dim=0))

    in_dict = EasyDict(
        points=points,
        lengths=lengths,
        neighbors=neighbors,
        pools=pools,
        upsamples=upsamples,
        up_distances=[],
        features=torch.randn(sum(level_counts[0]), input_channels),
    )
    return types.SimpleNamespace(in_dict=in_dict)


class KPNeXtLitePTIntegrationTests(unittest.TestCase):

    def test_handover_must_follow_convolution_only_stages(self):
        cfg = _make_config("classification", handover_stage=3)
        cfg.model.litept_conv_stages = 3
        with self.assertRaisesRegex(ValueError, "handover_stage = conv_stages \\+ 1"):
            KPNeXt(cfg)

    def test_segmentation_path_uses_attention_and_light_decoder(self):
        torch.manual_seed(10)
        model = KPNeXt(_make_config("cloud_segmentation"))
        batch = _make_batch(input_channels=5)
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (40, 4))
        self.assertFalse(model.add_decoder_layer)
        self.assertIsInstance(model.encoder_4[0], LitePointTransformerBlock)
        self.assertEqual(model.decoder_unary_4.norm.norm_type, "layer")
        logits.square().mean().backward()
        attention_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "attention" in name
        ]
        self.assertTrue(any(grad is not None for grad in attention_grads))
        self.assertTrue(torch.isfinite(logits).all())

    def test_classification_handover_path(self):
        torch.manual_seed(11)
        model = KPNeXt(_make_config("classification", handover_stage=3))
        batch = _make_batch(input_channels=4)
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertIsInstance(model.encoder_3[0], LiteHandoverBlock)
        logits.mean().backward()
        self.assertTrue(torch.isfinite(logits).all())

    def test_litept_and_fastadapter_compose_in_one_forward(self):
        torch.manual_seed(12)
        cfg = _make_config("cloud_segmentation")
        cfg.model.fa_enabled = True
        model = KPNeXt(cfg)
        batch = _make_batch(input_channels=5)
        logits = model(batch)
        logits.abs().mean().backward()
        adapter_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith("fast_adapter.")
        ]
        self.assertTrue(any(grad is not None for grad in adapter_grads))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
