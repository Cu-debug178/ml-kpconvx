"""CPU tests for Dynamic Kernel Scale and its KPNeXt integration."""

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
from models.dks_blocks import (DynamicKernelScale, alpha_stats,  # noqa: E402
                               room_mean_broadcast,
                               shuffle_packed_rows)
from models.kpnext_blocks import KPConvD, KPConvX  # noqa: E402
from utils.config import init_cfg  # noqa: E402


LEVEL_COUNTS = [(16, 12), (8, 6), (4, 3), (2, 2), (1, 1)]


def _make_config(dks_mode="none", train_mode="joint"):
    cfg = init_cfg()
    cfg.exp.seed = 57106803
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
    cfg.model.litept_patch_size = 4
    cfg.model.litept_num_heads = 2
    cfg.model.litept_attention_ratio = 1.0
    cfg.model.litept_mlp_ratio = 2.0
    cfg.model.litept_rope_base = 100.0
    cfg.model.litept_rope_enabled = True
    cfg.model.litept_attention_dropout = 0.0
    cfg.model.litept_projection_dropout = 0.0
    cfg.model.litept_orders = "z,z-trans"
    cfg.model.litept_light_decoder = True
    cfg.model.litept_legacy_kpconvd_encoder = True

    cfg.model.fa_enabled = False
    cfg.model.fa_train_mode = "joint"
    cfg.model.ktha_mode = "none"
    cfg.model.ktha_train_mode = "joint"
    cfg.model.glskf_mode = "none"
    cfg.model.glskf_train_mode = "joint"

    cfg.model.dks_mode = dks_mode
    cfg.model.dks_stages = "3"
    cfg.model.dks_hidden_dim = 8
    cfg.model.dks_alpha_min = 0.5
    cfg.model.dks_alpha_max = 1.2
    cfg.model.dks_fixed_alpha = 1.0
    cfg.model.dks_inference_ablation = "none"
    cfg.model.dks_train_mode = train_mode
    cfg.model.dks_log_stats = True

    cfg.train.smooth_labels = False
    cfg.train.class_w = []
    cfg.train.deform_loss_factor = 0.0
    cfg.train.deform_fit_rep_ratio = 0.0
    return cfg


def _make_batch(input_channels=5):
    points = []
    lengths = []
    neighbors = []
    for level, counts in enumerate(LEVEL_COUNTS):
        clouds = [
            torch.randn(count, 3) * (0.1 + level * 0.05) + cloud * 2.0
            for cloud, count in enumerate(counts)
        ]
        level_points = torch.cat(clouds, dim=0)
        points.append(level_points)
        lengths.append(torch.tensor(counts, dtype=torch.long))
        neighbors.append(
            torch.arange(level_points.shape[0]).unsqueeze(1).repeat(1, 4)
        )

    pools = []
    upsamples = []
    for old_counts, new_counts in zip(LEVEL_COUNTS[:-1], LEVEL_COUNTS[1:]):
        pool_rows = []
        up_rows = []
        old_offset = 0
        new_offset = 0
        for old_count, new_count in zip(old_counts, new_counts):
            selected = torch.linspace(0, old_count - 1, new_count).long()
            pool_rows.append(torch.stack([selected, selected], dim=1) + old_offset)
            assignment = torch.div(
                torch.arange(old_count) * new_count,
                old_count,
                rounding_mode="floor",
            ).clamp(max=new_count - 1)
            up_rows.append((assignment + new_offset).unsqueeze(1))
            old_offset += old_count
            new_offset += new_count
        pools.append(torch.cat(pool_rows))
        upsamples.append(torch.cat(up_rows))

    in_dict = EasyDict(
        points=points,
        lengths=lengths,
        neighbors=neighbors,
        pools=pools,
        upsamples=upsamples,
        up_distances=[],
        features=torch.randn(sum(LEVEL_COUNTS[0]), input_channels),
    )
    return types.SimpleNamespace(in_dict=in_dict)


class DynamicKernelScaleUnitTests(unittest.TestCase):
    def test_initial_alpha_is_exact_identity_and_gate_can_leave_zero(self):
        torch.manual_seed(1)
        module = DynamicKernelScale(8, hidden_dim=4)
        features = torch.randn(6, 8)
        alpha = module(features, torch.tensor([2, 4]))
        self.assertTrue(torch.equal(alpha, torch.ones_like(alpha)))
        alpha[0].backward()
        self.assertIsNotNone(module.gate.grad)
        self.assertNotEqual(float(module.gate.grad.item()), 0.0)

    def test_fixed_has_no_parameters_and_random_is_rng_isolated(self):
        fixed = DynamicKernelScale(8, mode="fixed", fixed_alpha=0.8)
        self.assertEqual(sum(p.numel() for p in fixed.parameters()), 0)
        x = torch.randn(7, 8)
        self.assertTrue(torch.equal(fixed(x), x.new_full((7,), 0.8)))

        random_scale = DynamicKernelScale(8, mode="random", seed=17)
        torch.manual_seed(99)
        before = torch.random.get_rng_state().clone()
        first = random_scale(x)
        after = torch.random.get_rng_state()
        second = random_scale(x)
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(torch.equal(first, second))
        self.assertGreaterEqual(float(first.min()), 0.5)
        self.assertLessEqual(float(first.max()), 1.2)

    def test_packed_ablations_do_not_advance_global_rng(self):
        x = torch.arange(12, dtype=torch.float32)
        lengths = torch.tensor([5, 7])
        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()
        shuffled = shuffle_packed_rows(x, lengths, seed=2)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertCountEqual(shuffled[:5].tolist(), x[:5].tolist())
        self.assertCountEqual(shuffled[5:].tolist(), x[5:].tolist())
        means = room_mean_broadcast(x, lengths)
        self.assertTrue(torch.equal(means[:5], torch.full((5,), 2.0)))
        self.assertTrue(torch.equal(means[5:], torch.full((7,), 8.0)))

    def test_packed_length_mismatch_reports_both_counts(self):
        with self.assertRaisesRegex(ValueError, r"5.*6"):
            shuffle_packed_rows(torch.zeros(6), torch.tensor([2, 3]))
        with self.assertRaisesRegex(ValueError, r"7.*6"):
            room_mean_broadcast(torch.zeros(6), torch.tensor([3, 4]))

    def test_alpha_stats_and_bfloat16(self):
        module = DynamicKernelScale(8, hidden_dim=4)
        features = torch.randn(6, 8, dtype=torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            alpha = module(features, torch.tensor([3, 3]))
        self.assertEqual(alpha.dtype, features.dtype)
        self.assertTrue(torch.isfinite(alpha).all())
        stats = alpha_stats(alpha, torch.tensor([3, 3]))
        self.assertEqual(stats["mean"], 1.0)
        self.assertEqual(stats["std"], 0.0)


class KernelScaleOperatorTests(unittest.TestCase):
    def test_identity_scale_matches_existing_kpconvd_and_kpconvx_paths(self):
        for conv_type in (KPConvD, KPConvX):
            with self.subTest(conv=conv_type.__name__):
                torch.manual_seed(2)
                kwargs = {}
                if conv_type is KPConvX:
                    kwargs["attention_groups"] = 4
                conv = conv_type(8, [1, 14], 1.0, 1.0, **kwargs)
                points = torch.randn(7, 3) * 0.25
                feats = torch.randn(7, 8)
                neighbors = torch.arange(7).unsqueeze(1).repeat(1, 4)
                expected = conv(points, points, feats, neighbors)
                actual = conv(
                    points,
                    points,
                    feats,
                    neighbors,
                    kernel_scale=torch.ones(7),
                )
                self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_alpha_expands_receptive_field(self):
        torch.manual_seed(0)
        conv = KPConvD(8, [1, 14], 1.0, 1.0, influence_mode="linear")
        query = torch.zeros(1, 3)
        support = torch.tensor([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
        neighbors = torch.tensor([[0, 1]])
        base, _, _ = conv.get_neighbors_influences(query, support, neighbors)
        wide, _, _ = conv.get_neighbors_influences(
            query, support, neighbors, kernel_scale=torch.tensor([2.0])
        )
        self.assertEqual(float(base[0, 1]), 0.0)
        self.assertGreater(float(wide[0, 1]), 0.05)

    def test_self_and_shadow_neighbors_have_safe_gradients(self):
        conv = KPConvD(8, [1, 14], 1.0, 1.0, influence_mode="linear")
        conv.kernel_points[0].zero_()
        query = torch.zeros(1, 3)
        support = torch.zeros(1, 3)
        neighbors = torch.tensor([[0, 1]])
        scale = torch.tensor([1.0], requires_grad=True)
        influences, _, _ = conv.get_neighbors_influences(
            query, support, neighbors, kernel_scale=scale
        )
        self.assertEqual(float(influences[0, 1]), 0.0)
        influences.sum().backward()
        self.assertTrue(torch.isfinite(scale.grad).all())

        support = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.1, 0.0]])
        scale = torch.tensor([1.0], requires_grad=True)
        influences, _, _ = conv.get_neighbors_influences(
            query, support, torch.tensor([[0, 1]]), kernel_scale=scale
        )
        influences.sum().backward()
        self.assertTrue(torch.isfinite(scale.grad).all())
        self.assertNotEqual(float(scale.grad.item()), 0.0)

    def test_shared_second_operator_rejects_scale(self):
        shared = {}
        first = KPConvD(8, [1, 14], 1.0, 1.0, shared_kp_data=shared)
        second = KPConvD(8, [1, 14], 1.0, 1.0, shared_kp_data=shared)
        points = torch.randn(4, 3) * 0.1
        neighbors = torch.arange(4).unsqueeze(1).repeat(1, 2)
        first.get_neighbors_influences(
            points, points, neighbors, kernel_scale=torch.ones(4)
        )
        with self.assertRaisesRegex(ValueError, "first KPConv"):
            second.get_neighbors_influences(
                points, points, neighbors, kernel_scale=torch.ones(4)
            )


class DksNetworkTests(unittest.TestCase):
    def test_initial_network_is_equivalent_and_cache_is_cleared(self):
        torch.manual_seed(7)
        baseline = KPNeXt(_make_config("none")).eval()
        torch.manual_seed(7)
        dks = KPNeXt(_make_config("learned")).eval()
        incompatible = dks.load_state_dict(baseline.state_dict(), strict=False)
        self.assertTrue(all(key.startswith("dks.") for key in incompatible.missing_keys))
        self.assertFalse(incompatible.unexpected_keys)

        torch.manual_seed(8)
        batch = _make_batch()
        with torch.no_grad():
            expected = baseline(batch)
            actual, trace = dks(batch, return_intermediates=True)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5))
        self.assertTrue(torch.equal(trace["dks_alpha"][3], torch.ones(7)))
        for shared in dks.shared_kp:
            # Attention-only stages have no KP runtime or fixed kernel basis.
            if shared:
                self.assertIn("k_pts", shared)
            self.assertNotIn("infl_w", shared)
            self.assertNotIn("neighb_p", shared)
            self.assertNotIn("neighb_1nn", shared)

    def test_module_head_trainability_is_exact(self):
        model = KPNeXt(_make_config("learned", train_mode="module_head"))
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("dks.") or name.startswith("head.") for name in trainable)
        )
        self.assertTrue(any(name.startswith("dks.") for name in trainable))
        self.assertTrue(any(name.startswith("head.") for name in trainable))
        model.train()
        self.assertTrue(model.dks.training)
        self.assertTrue(model.head.training)
        self.assertFalse(model.stem.training)

    def test_mutually_exclusive_mechanisms_are_rejected(self):
        cfg = _make_config("learned")
        cfg.model.glskf_mode = "kernel_gate"
        with self.assertRaisesRegex(ValueError, "must not be combined"):
            KPNeXt(cfg)

        cfg = _make_config("learned")
        cfg.model.kp_influence = "constant"
        with self.assertRaisesRegex(ValueError, "no differentiable scale path"):
            KPNeXt(cfg)

    def test_inference_ablation_is_explicit_and_eval_only(self):
        cfg = _make_config("learned")
        cfg.model.dks_inference_ablation = "identity"
        model = KPNeXt(cfg)
        module = model.dks["3"]
        with torch.no_grad():
            module.gate.fill_(1.0)
        batch = _make_batch()
        model.train()
        _, train_trace = model(batch, return_intermediates=True)
        self.assertFalse(torch.equal(train_trace["dks_alpha"][3], torch.ones(7)))
        model.eval()
        with torch.no_grad():
            _, eval_trace = model(batch, return_intermediates=True)
        self.assertTrue(torch.equal(eval_trace["dks_alpha"][3], torch.ones(7)))


if __name__ == "__main__":
    unittest.main()
