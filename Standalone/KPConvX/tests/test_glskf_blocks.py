"""CPU unit and integration tests for GLSKF (top-down semantic kernel feedback).

The production pyramid needs compiled operators, so a complete synthetic
pyramid is injected exactly like ``test_litept_kpnext_integration`` does. That
keeps the real encoder, the real KPConvD gate path, the real chained upsamples
and the real decoder in the loop without any C++ extension.
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

torch_pyramid_module = types.ModuleType("utils.torch_pyramid")


def _unexpected_fill(*args, **kwargs):
    raise AssertionError("the synthetic test pyramid should already be complete")


torch_pyramid_module.fill_pyramid = _unexpected_fill
sys.modules.setdefault("utils.torch_pyramid", torch_pyramid_module)

from models.KPNext import KPNeXt  # noqa: E402
from models.glskf_blocks import (GlskfFeedback,  # noqa: E402
                                 apply_kernel_gate,
                                 room_mean_broadcast,
                                 shuffle_packed_rows)
from models.kpnext_blocks import KPConvD  # noqa: E402
from utils.config import init_cfg  # noqa: E402
from utils.mixed_precision import (MixedPrecisionSettings,  # noqa: E402
                                   autocast_context)


LEVEL_COUNTS = [(24, 16), (12, 8), (6, 4), (3, 2), (2, 1)]


def _make_config(glskf_mode="none"):
    cfg = init_cfg()
    cfg.data.task = "cloud_segmentation"
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
    cfg.model.input_channels = 5
    cfg.model.init_channels = 16
    cfg.model.channel_scaling = 1.41
    cfg.model.kp_mode = "kpconvx"
    cfg.model.shell_sizes = [1, 14]
    cfg.model.kp_influence = "linear"
    cfg.model.kp_aggregation = "nearest"
    cfg.model.conv_groups = -1
    cfg.model.share_kp = True
    cfg.model.grid_pool = True
    cfg.model.decoder_layer = False
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
    cfg.model.litept_conv_stages = 3
    cfg.model.litept_handover_stage = 0
    cfg.model.litept_patch_size = 8
    cfg.model.litept_num_heads = 2
    cfg.model.litept_attention_ratio = 1.0
    cfg.model.litept_mlp_ratio = 2.0
    cfg.model.litept_rope_base = 100.0
    cfg.model.litept_rope_enabled = True
    cfg.model.litept_attention_dropout = 0.0
    cfg.model.litept_projection_dropout = 0.0
    cfg.model.litept_orders = "z,z-trans"
    cfg.model.litept_light_decoder = True

    cfg.model.fa_enabled = False
    cfg.model.fa_train_mode = "joint"

    cfg.model.glskf_mode = glskf_mode
    cfg.model.glskf_refine_stage = 3
    cfg.model.glskf_context_stages = "4,5"
    cfg.model.glskf_groups = 8
    cfg.model.glskf_hidden_dim = 16
    cfg.model.glskf_train_mode = "joint"

    cfg.train.smooth_labels = False
    cfg.train.class_w = []
    cfg.train.deform_loss_factor = 0.0
    cfg.train.deform_fit_rep_ratio = 0.0
    return cfg


def _make_batch(input_channels=5, level_counts=None):
    level_counts = level_counts or LEVEL_COUNTS
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
    for level in range(len(level_counts) - 1):
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


class KernelGateTests(unittest.TestCase):
    """The gate must scale effective weights without touching the parameters."""

    def test_gate_of_ones_is_a_no_op(self):
        torch.manual_seed(0)
        conv = KPConvD(8, [1, 14], 1.0, 1.0)
        points = torch.randn(6, 3) * 0.2
        feats = torch.randn(6, 8)
        neighbors = torch.arange(6).unsqueeze(1).repeat(1, 4)

        baseline = conv(points, points, feats, neighbors)
        gate = torch.ones(6, conv.K, 4)
        gated = conv(points, points, feats, neighbors, kernel_gate=gate)
        self.assertTrue(torch.allclose(baseline, gated, atol=1e-6))

    def test_gate_scales_grouped_channels(self):
        torch.manual_seed(1)
        channels = 8
        groups = 4
        conv = KPConvD(channels, [1, 14], 1.0, 1.0)
        points = torch.randn(5, 3) * 0.2
        feats = torch.randn(5, channels)
        neighbors = torch.arange(5).unsqueeze(1).repeat(1, 4)

        baseline = conv(points, points, feats, neighbors)
        gate = torch.ones(5, conv.K, groups)
        gate[:, :, 1] = 3.0
        gated = conv(points, points, feats, neighbors, kernel_gate=gate)

        width = channels // groups
        expected = baseline.clone()
        expected[:, width:2 * width] *= 3.0
        self.assertTrue(torch.allclose(expected, gated, atol=1e-5))

    def test_base_weights_are_never_mutated(self):
        torch.manual_seed(2)
        conv = KPConvD(8, [1, 14], 1.0, 1.0)
        before = conv.weights.detach().clone()
        points = torch.randn(5, 3) * 0.2
        feats = torch.randn(5, 8)
        neighbors = torch.arange(5).unsqueeze(1).repeat(1, 4)
        gate = 1.0 + torch.rand(5, conv.K, 2)
        conv(points, points, feats, neighbors, kernel_gate=gate).sum().backward()
        self.assertTrue(torch.equal(before, conv.weights.detach()))
        self.assertIsNotNone(conv.weights.grad)

    def test_shadow_neighbors_never_contribute(self):
        torch.manual_seed(3)
        channels = 8
        conv = KPConvD(channels, [1, 14], 1.0, 1.0)
        points = torch.randn(4, 3) * 0.2
        feats = torch.randn(4, channels)
        # Index 4 is out of range for 4 support points: a shadow neighbor.
        neighbors = torch.full((4, 4), 4, dtype=torch.long)
        gate = 1.0 + 5.0 * torch.rand(4, conv.K, 4)
        output = conv(points, points, feats, neighbors, kernel_gate=gate)
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)))

    def test_invalid_gate_shapes_raise(self):
        conv = KPConvD(8, [1, 14], 1.0, 1.0)
        points = torch.randn(4, 3) * 0.2
        feats = torch.randn(4, 8)
        neighbors = torch.arange(4).unsqueeze(1).repeat(1, 4)

        with self.assertRaises(ValueError):
            conv(points, points, feats, neighbors,
                 kernel_gate=torch.ones(4, conv.K + 1, 4))
        with self.assertRaises(ValueError):
            # 3 does not divide 8 channels.
            conv(points, points, feats, neighbors,
                 kernel_gate=torch.ones(4, conv.K, 3))
        with self.assertRaises(ValueError):
            conv(points, points, feats, neighbors,
                 kernel_gate=torch.ones(4, conv.K))

    def test_mlp_influence_rejects_a_gate(self):
        conv = KPConvD(8, [1, 14], 1.0, 1.0, influence_mode='mlp')
        points = torch.randn(4, 3) * 0.2
        feats = torch.randn(4, 8)
        neighbors = torch.arange(4).unsqueeze(1).repeat(1, 4)
        with self.assertRaises(ValueError):
            conv(points, points, feats, neighbors,
                 kernel_gate=torch.ones(4, conv.K, 4))

    def test_apply_kernel_gate_matches_a_dense_reference(self):
        torch.manual_seed(4)
        points, neighbors_count, channels, groups, num_kernels = 5, 4, 8, 4, 15
        weights = torch.randn(points, neighbors_count, channels)
        gate = torch.rand(points, num_kernels, groups) + 0.5
        assignment = torch.randint(0, num_kernels, (points, neighbors_count))

        gated = apply_kernel_gate(weights, gate, assignment)
        width = channels // groups
        reference = weights.clone()
        for m in range(points):
            for h in range(neighbors_count):
                for c in range(channels):
                    reference[m, h, c] *= gate[m, assignment[m, h], c // width]
        self.assertTrue(torch.allclose(gated, reference, atol=1e-6))


class PackedControlTests(unittest.TestCase):

    def test_shuffle_stays_inside_each_cloud(self):
        torch.manual_seed(5)
        lengths = torch.tensor([3, 4])
        features = torch.cat([
            torch.zeros(3, 2),
            torch.ones(4, 2),
        ], dim=0)
        shuffled = shuffle_packed_rows(features, lengths)
        self.assertTrue(torch.equal(shuffled[:3], torch.zeros(3, 2)))
        self.assertTrue(torch.equal(shuffled[3:], torch.ones(4, 2)))

    def test_shuffle_is_a_permutation_of_each_cloud(self):
        torch.manual_seed(6)
        lengths = torch.tensor([5, 2])
        features = torch.arange(7, dtype=torch.float32).unsqueeze(1)
        shuffled = shuffle_packed_rows(features, lengths)
        self.assertEqual(
            sorted(shuffled[:5, 0].tolist()), [0.0, 1.0, 2.0, 3.0, 4.0]
        )
        self.assertEqual(sorted(shuffled[5:, 0].tolist()), [5.0, 6.0])

    def test_room_mean_keeps_the_cloud_mean_and_drops_positions(self):
        lengths = torch.tensor([2, 3])
        features = torch.tensor(
            [[0.0], [2.0], [3.0], [3.0], [6.0]]
        )
        pooled = room_mean_broadcast(features, lengths)
        self.assertTrue(torch.allclose(pooled[:2], torch.full((2, 1), 1.0)))
        self.assertTrue(torch.allclose(pooled[2:], torch.full((3, 1), 4.0)))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            shuffle_packed_rows(torch.zeros(4, 2), torch.tensor([2, 1]))
        with self.assertRaises(ValueError):
            room_mean_broadcast(torch.zeros(4, 2), torch.tensor([2, 1]))


def _make_feedback(mode="kernel_gate", groups=4, **kwargs):
    return GlskfFeedback(
        refine_channels=8,
        context_channels=[8, 12],
        shell_sizes=[1, 14],
        radius=1.0,
        sigma=1.0,
        mode=mode,
        groups=groups,
        hidden_dim=16,
        norm_type="batch",
        **kwargs,
    )


def _feedback_inputs(num_points=6, num_clouds=2):
    points = torch.randn(num_points, 3) * 0.2
    refine = torch.randn(num_points, 8)
    context = torch.randn(num_points, 20)
    neighbors = torch.arange(num_points).unsqueeze(1).repeat(1, 4)
    base = num_points // num_clouds
    lengths = torch.tensor(
        [base] * (num_clouds - 1) + [num_points - base * (num_clouds - 1)],
        dtype=torch.long,
    )
    return points, refine, context, neighbors, lengths


class GlskfFeedbackTests(unittest.TestCase):

    def test_zero_scale_is_strict_identity(self):
        torch.manual_seed(7)
        for mode in ("kernel_gate", "film", "matched_mlp"):
            with self.subTest(mode=mode):
                module = _make_feedback(mode).eval()
                points, refine, context, neighbors, lengths = _feedback_inputs()
                output = module(points, refine, context, neighbors, lengths=lengths)
                self.assertTrue(torch.equal(output, refine))

    def test_non_zero_scale_changes_the_output(self):
        torch.manual_seed(8)
        module = _make_feedback().eval()
        with torch.no_grad():
            module.scale.fill_(0.5)
        points, refine, context, neighbors, lengths = _feedback_inputs()
        output = module(points, refine, context, neighbors, lengths=lengths)
        self.assertFalse(torch.allclose(output, refine))
        self.assertTrue(torch.isfinite(output).all())

    def test_gate_shape_and_positive_range(self):
        torch.manual_seed(9)
        module = _make_feedback(groups=4)
        gate = module.build_gate(torch.randn(6, 20))
        self.assertEqual(tuple(gate.shape), (6, module.num_kernels, 4))
        self.assertTrue((gate > 0).all())
        self.assertTrue((gate < 2).all())

    def test_cpu_backward_reaches_gate_and_scale(self):
        torch.manual_seed(10)
        module = _make_feedback()
        with torch.no_grad():
            module.scale.fill_(0.1)
        points, refine, context, neighbors, lengths = _feedback_inputs()
        module(points, refine, context, neighbors, lengths=lengths).square().mean().backward()
        self.assertIsNotNone(module.scale.grad)
        self.assertTrue(module.scale.grad.abs().sum() > 0)
        self.assertIsNotNone(module.gate_mlp[-1].weight.grad)
        self.assertTrue(module.gate_mlp[-1].weight.grad.abs().sum() > 0)

    def test_gate_has_gradient_even_at_zero_scale_after_one_step(self):
        # The identity switch lives in the outer scale only, so the gate keeps a
        # usable gradient as soon as the scale leaves zero.  Verifying that the
        # gate is *not* also zero-initialized guards the design decision.
        torch.manual_seed(11)
        module = _make_feedback()
        gate = module.build_gate(torch.randn(4, 20))
        self.assertFalse(torch.allclose(gate, torch.ones_like(gate)))

    def test_detached_context_blocks_gradient_flow(self):
        torch.manual_seed(12)
        module = _make_feedback(detach_context=True)
        with torch.no_grad():
            module.scale.fill_(0.1)
        points, refine, context, neighbors, lengths = _feedback_inputs()
        context = context.clone().requires_grad_(True)
        module(points, refine, context, neighbors, lengths=lengths).square().mean().backward()
        self.assertIsNone(context.grad)

    def test_context_flows_when_not_detached(self):
        torch.manual_seed(13)
        module = _make_feedback()
        with torch.no_grad():
            module.scale.fill_(0.1)
        points, refine, context, neighbors, lengths = _feedback_inputs()
        context = context.clone().requires_grad_(True)
        module(points, refine, context, neighbors, lengths=lengths).square().mean().backward()
        self.assertIsNotNone(context.grad)
        self.assertTrue(context.grad.abs().sum() > 0)

    def test_empty_point_cloud_returns_input(self):
        module = _make_feedback()
        empty = torch.zeros(0, 8)
        output = module(
            torch.zeros(0, 3),
            empty,
            torch.zeros(0, 20),
            torch.zeros(0, 4, dtype=torch.long),
            lengths=torch.tensor([0, 0]),
        )
        self.assertEqual(tuple(output.shape), (0, 8))

    def test_matched_control_matches_the_parameter_budget(self):
        torch.manual_seed(14)
        real = _make_feedback("kernel_gate")
        matched = _make_feedback("matched_mlp")
        real_count = real.added_parameter_count()
        matched_count = matched.added_parameter_count()
        self.assertLess(abs(real_count - matched_count) / real_count, 0.05)

    def test_matched_control_ignores_the_context(self):
        torch.manual_seed(15)
        module = _make_feedback("matched_mlp")
        with torch.no_grad():
            module.scale.fill_(0.1)
        module.eval()
        points, refine, context, neighbors, lengths = _feedback_inputs()
        first = module(points, refine, context, neighbors, lengths=lengths)
        second = module(
            points, refine, torch.randn_like(context), neighbors, lengths=lengths
        )
        self.assertTrue(torch.equal(first, second))

    def test_room_mean_control_is_constant_inside_each_cloud(self):
        torch.manual_seed(16)
        module = _make_feedback(context_control="room_mean")
        points, refine, context, neighbors, lengths = _feedback_inputs()
        pooled = room_mean_broadcast(context, lengths)
        gate = module.build_gate(pooled)
        self.assertTrue(torch.allclose(gate[0], gate[1], atol=1e-6))
        self.assertFalse(torch.allclose(gate[0], gate[-1], atol=1e-6))

    def test_invalid_configurations_raise(self):
        with self.assertRaises(ValueError):
            _make_feedback(mode="none")
        with self.assertRaises(ValueError):
            # 3 does not divide the 8 refined channels.
            _make_feedback(groups=3)
        with self.assertRaises(ValueError):
            _make_feedback(groups=0)
        with self.assertRaises(ValueError):
            _make_feedback(context_control="wrong")
        with self.assertRaises(ValueError):
            _make_feedback(mode="film", deep_residual=True)
        with self.assertRaises(ValueError):
            _make_feedback(mode="matched_mlp", context_control="shuffle")
        with self.assertRaises(ValueError):
            GlskfFeedback(
                refine_channels=8,
                context_channels=[],
                shell_sizes=[1, 14],
                radius=1.0,
                sigma=1.0,
            )

    def test_deep_residual_adds_a_context_feature_path(self):
        torch.manual_seed(17)
        module = _make_feedback(deep_residual=True)
        self.assertIsNotNone(module.in_mlp)
        with torch.no_grad():
            module.scale.fill_(0.1)
        module.eval()
        points, refine, context, neighbors, lengths = _feedback_inputs()
        first = module(points, refine, context, neighbors, lengths=lengths)
        second = module(
            points, refine, torch.randn_like(context), neighbors, lengths=lengths
        )
        self.assertFalse(torch.allclose(first, second))


class GlskfKPNeXtIntegrationTests(unittest.TestCase):

    def test_all_modes_warm_start_to_the_baseline_logits(self):
        torch.manual_seed(20)
        batch = _make_batch()
        baseline = KPNeXt(_make_config()).eval()
        baseline_logits = baseline(batch)
        self.assertIsNone(baseline.glskf)

        for mode in ("kernel_gate", "film", "matched_mlp"):
            with self.subTest(mode=mode):
                cfg = _make_config(mode)
                candidate = KPNeXt(cfg)
                incompatible = candidate.load_state_dict(
                    baseline.state_dict(), strict=False
                )
                self.assertEqual(list(incompatible.unexpected_keys), [])
                self.assertTrue(
                    all(key.startswith('glskf.') for key in incompatible.missing_keys)
                )
                candidate.eval()
                logits = candidate(batch)
                self.assertTrue(torch.allclose(logits, baseline_logits, atol=1e-6))

    def test_refined_stage_skip_uses_the_expected_width(self):
        cfg = _make_config("kernel_gate")
        model = KPNeXt(cfg)
        # grid pooling expands the last block of each stage, so the stage-3 skip
        # already carries the stage-4 width.
        expected = model.glskf.refine_channels
        batch = _make_batch()
        model.eval()
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (40, 4))
        self.assertEqual(expected % cfg.model.glskf_groups, 0)

    def test_non_zero_scale_changes_logits_and_backpropagates(self):
        torch.manual_seed(21)
        cfg = _make_config("kernel_gate")
        model = KPNeXt(cfg)
        batch = _make_batch()
        model.eval()
        identity_logits = model(batch)
        with torch.no_grad():
            model.glskf.scale.fill_(0.5)
        gated_logits = model(batch)
        self.assertFalse(torch.allclose(identity_logits, gated_logits, atol=1e-5))
        self.assertTrue(torch.isfinite(gated_logits).all())

        model.train()
        logits = model(batch)
        logits.square().mean().backward()
        glskf_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith('glskf.')
        ]
        self.assertTrue(all(grad is not None for grad in glskf_grads))
        self.assertTrue(any(grad.abs().sum() > 0 for grad in glskf_grads))

    def test_module_head_trains_only_glskf_and_head(self):
        cfg = _make_config("kernel_gate")
        cfg.model.glskf_train_mode = "module_head"
        model = KPNeXt(cfg)
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(any(name.startswith('glskf.') for name in trainable))
        self.assertTrue(any(name.startswith('head.') for name in trainable))
        self.assertTrue(all(
            name.startswith('glskf.') or name.startswith('head.')
            for name in trainable
        ))

        # The frozen backbone must also stop updating its BN statistics.
        model.train()
        self.assertFalse(model.encoder_1.training)
        self.assertFalse(model.stem.training)
        self.assertTrue(model.glskf.training)
        self.assertTrue(model.head.training)

    def test_module_head_backward_leaves_the_backbone_without_grads(self):
        torch.manual_seed(22)
        cfg = _make_config("kernel_gate")
        cfg.model.glskf_train_mode = "module_head"
        model = KPNeXt(cfg)
        with torch.no_grad():
            model.glskf.scale.fill_(0.2)
        batch = _make_batch()
        model.train()
        model(batch).square().mean().backward()
        for name, parameter in model.named_parameters():
            if name.startswith('glskf.') or name.startswith('head.'):
                continue
            self.assertIsNone(parameter.grad, msg=name)

    def test_checkpoint_round_trip_preserves_the_module(self):
        torch.manual_seed(23)
        cfg = _make_config("kernel_gate")
        model = KPNeXt(cfg)
        with torch.no_grad():
            model.glskf.scale.fill_(0.3)
        batch = _make_batch()
        model.eval()
        expected = model(batch)

        state = {key: value.clone() for key, value in model.state_dict().items()}
        restored = KPNeXt(_make_config("kernel_gate"))
        restored.load_state_dict(state)
        restored.eval()
        self.assertTrue(torch.allclose(restored(batch), expected, atol=1e-6))

    def test_glskf_and_ktha_cannot_be_combined(self):
        cfg = _make_config("kernel_gate")
        cfg.model.ktha_mode = "relation_bias"
        cfg.model.ktha_source_stage = 3
        cfg.model.ktha_target_stages = "4"
        with self.assertRaisesRegex(ValueError, "cannot be enabled in the same run"):
            KPNeXt(cfg)

    def test_invalid_stage_configurations_raise(self):
        cfg = _make_config("kernel_gate")
        cfg.model.glskf_refine_stage = 5
        with self.assertRaises(ValueError):
            KPNeXt(cfg)

        cfg = _make_config("kernel_gate")
        cfg.model.glskf_context_stages = "2,5"
        with self.assertRaises(ValueError):
            KPNeXt(cfg)

        cfg = _make_config("kernel_gate")
        cfg.model.glskf_refine_stage = 4
        with self.assertRaises(ValueError):
            # Stage 4 is an attention stage: it carries no KP basis.
            KPNeXt(cfg)

        cfg = _make_config("kernel_gate")
        cfg.model.glskf_groups = 7
        with self.assertRaises(ValueError):
            KPNeXt(cfg)

        cfg = _make_config("kernel_gate")
        cfg.model.share_kp = False
        with self.assertRaises(ValueError):
            KPNeXt(cfg)

        cfg = _make_config("kernel_gate")
        cfg.data.task = "classification"
        with self.assertRaises(ValueError):
            KPNeXt(cfg)

        cfg = _make_config("kernel_gate")
        cfg.model.litept_enabled = False
        cfg.model.kp_mode = "kpconv"
        with self.assertRaisesRegex(ValueError, "kp_mode='kpconv' is not supported"):
            KPNeXt(cfg)

    def test_context_controls_run_end_to_end(self):
        torch.manual_seed(24)
        batch = _make_batch()
        for control in ("shuffle", "room_mean"):
            with self.subTest(control=control):
                cfg = _make_config("kernel_gate")
                cfg.model.glskf_context_control = control
                model = KPNeXt(cfg)
                with torch.no_grad():
                    model.glskf.scale.fill_(0.2)
                model.eval()
                logits = model(batch)
                self.assertEqual(tuple(logits.shape), (40, 4))
                self.assertTrue(torch.isfinite(logits).all())

    def test_bfloat16_autocast_forward_and_backward(self):
        torch.manual_seed(25)
        cfg = _make_config("kernel_gate")
        model = KPNeXt(cfg)
        with torch.no_grad():
            model.glskf.scale.fill_(0.2)
        batch = _make_batch()
        device = torch.device("cpu")
        settings = MixedPrecisionSettings(True, "bfloat16", torch.bfloat16)
        with autocast_context(settings, device):
            logits = model(batch)
            loss = logits.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.float()).all())
        self.assertIsNotNone(model.glskf.scale.grad)
        self.assertTrue(torch.isfinite(model.glskf.scale.grad).all())

    def test_feedback_conv_reuses_the_stage_kernel_geometry(self):
        cfg = _make_config("kernel_gate")
        model = KPNeXt(cfg)
        refine_l = cfg.model.glskf_refine_stage - 1
        cached = model.shared_kp[refine_l]["k_pts"]

        # Not the first KP conv of the layer: it consumes the cached
        # nearest-kernel assignment instead of recomputing (M, H, K, 3).
        self.assertFalse(model.glskf.conv.first_kp)
        self.assertTrue(model.glskf.conv.share_kp)

        # Same tensor object, so the gated kernel index space is by
        # construction the one the stage-3 convolutions already use.
        self.assertIs(model.glskf.conv.kernel_points, cached)
        self.assertEqual(model.glskf.conv.K, model.glskf.num_kernels)

    def test_gate_uses_the_cached_nearest_kernel_assignment(self):
        torch.manual_seed(27)
        cfg = _make_config("kernel_gate")
        model = KPNeXt(cfg)
        with torch.no_grad():
            model.glskf.scale.fill_(0.25)
        batch = _make_batch()
        model.eval()
        model(batch)
        refine_l = cfg.model.glskf_refine_stage - 1
        assignment = model.shared_kp[refine_l]["neighb_1nn"]
        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.shape[0], batch.in_dict.points[refine_l].shape[0])
        self.assertTrue(int(assignment.max()) < model.glskf.num_kernels)

    def test_single_stage_contexts_run_end_to_end(self):
        torch.manual_seed(28)
        batch = _make_batch()
        for stages in ("4", "5", "4,5"):
            with self.subTest(stages=stages):
                cfg = _make_config("kernel_gate")
                cfg.model.glskf_context_stages = stages
                model = KPNeXt(cfg)
                with torch.no_grad():
                    model.glskf.scale.fill_(0.2)
                model.eval()
                logits = model(batch)
                self.assertEqual(tuple(logits.shape), (40, 4))
                self.assertTrue(torch.isfinite(logits).all())

    def test_runtime_monitoring_reports_gate_statistics(self):
        torch.manual_seed(26)
        cfg = _make_config("kernel_gate")
        model = KPNeXt(cfg)
        with torch.no_grad():
            model.glskf.scale.fill_(0.2)
        batch = _make_batch()
        model.eval()
        model.set_runtime_monitoring(True)
        model(batch)
        stats = model.runtime_monitoring_stats()
        self.assertIn('glskf', stats)
        self.assertIn('gate_mean', stats['glskf'])
        self.assertIn('correction_ratio', stats['glskf'])


if __name__ == "__main__":
    unittest.main()
