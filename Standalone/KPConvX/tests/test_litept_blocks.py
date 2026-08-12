import math
import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.litept_blocks import (  # noqa: E402
    LitePointTransformerBlock,
    PointROPE,
    SerializedPatchCache,
    SerializedPointROPEAttention,
    _morton_code,
    build_serialized_patches,
    parse_serialization_orders,
)
from models.ktha_blocks import (KernelGeometrySignatureV2,
                                KernelOccupancySignature,
                                ablate_packed_signature,
                                pool_kernel_geometry_signature_v2,
                                pool_kernel_signature,
                                shuffle_packed_signature)  # noqa: E402


class PointROPETests(unittest.TestCase):

    def test_zero_position_is_identity_and_backward_is_finite(self):
        features = torch.randn(2, 4, 7, 24, requires_grad=True)
        positions = torch.zeros(2, 7, 3, dtype=torch.long)
        output = PointROPE(base=100.0)(features, positions)
        self.assertTrue(torch.allclose(output, features, atol=1e-6, rtol=1e-6))
        output.square().mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_invalid_head_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            PointROPE()(torch.randn(1, 2, 3, 16), torch.zeros(1, 3, 3))


class SerializationTests(unittest.TestCase):

    def test_vectorized_morton_matches_loop_reference(self):
        torch.manual_seed(1)
        coordinate_sets = [
            torch.randint(0, 2 ** 20, (257, 3), dtype=torch.long),
            torch.randint(0, 2 ** 27, (257, 3), dtype=torch.long),
        ]
        for coords in coordinate_sets:
            for order in ("z", "z-trans"):
                for max_bits in (5, 13, 20, 21):
                    xyz = coords if order == "z" else coords[:, [1, 0, 2]]
                    max_coord = xyz.amax().clamp(min=1)
                    bits_needed = (
                        torch.floor(torch.log2(max_coord.to(torch.float64))).to(torch.long)
                        + 1
                    )
                    shift = torch.clamp(bits_needed - max_bits, min=0)
                    shifted = torch.bitwise_right_shift(xyz, shift)
                    reference = torch.zeros(coords.shape[0], dtype=torch.long)
                    for bit in range(max_bits):
                        reference |= ((shifted[:, 0] >> bit) & 1) << (3 * bit)
                        reference |= ((shifted[:, 1] >> bit) & 1) << (3 * bit + 1)
                        reference |= ((shifted[:, 2] >> bit) & 1) << (3 * bit + 2)
                    self.assertTrue(
                        torch.equal(
                            _morton_code(coords, order=order, max_bits=max_bits),
                            reference,
                        )
                    )

    def test_packed_clouds_are_partitioned_exactly_once(self):
        torch.manual_seed(2)
        points = torch.randn(12, 3)
        lengths = torch.tensor([5, 7], dtype=torch.long)
        indices, valid, coords = build_serialized_patches(
            points, lengths, patch_size=4, voxel_size=0.1, order="z"
        )
        real_indices = indices.reshape(-1)[valid.reshape(-1)]
        self.assertEqual(real_indices.numel(), points.shape[0])
        self.assertTrue(torch.equal(torch.sort(real_indices).values, torch.arange(12)))
        self.assertEqual(tuple(coords.shape), (4, 4, 3))

    def test_cloud_local_quantization_is_translation_invariant(self):
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [3.0, 3.0, 3.0], [3.2, 3.1, 3.4]]
        )
        lengths = torch.tensor([2, 2], dtype=torch.long)
        translated = points.clone()
        translated[:2] += torch.tensor([100.0, -40.0, 8.0])
        translated[2:] += torch.tensor([-20.0, 70.0, 12.0])

        original_layout = build_serialized_patches(
            points, lengths, patch_size=2, voxel_size=0.1, order="z"
        )
        translated_layout = build_serialized_patches(
            translated, lengths, patch_size=2, voxel_size=0.1, order="z"
        )
        for original, shifted in zip(original_layout, translated_layout):
            self.assertTrue(torch.equal(original, shifted))

    def test_rejects_non_integer_lengths(self):
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            build_serialized_patches(
                torch.randn(3, 3),
                torch.tensor([3.0]),
                patch_size=2,
                voxel_size=0.1,
                order="z",
            )

    def test_order_parser(self):
        self.assertEqual(parse_serialization_orders("z,z-trans"), ("z", "z-trans"))
        with self.assertRaises(ValueError):
            parse_serialization_orders("hilbert")

    def test_stage_cache_reuses_patch_metadata_until_cleared(self):
        points = torch.randn(9, 3)
        lengths = torch.tensor([4, 5], dtype=torch.long)
        cache = SerializedPatchCache(profile_enabled=True)
        first = cache.get(points, lengths, 4, 0.1, "z")
        second = cache.get(points, lengths, 4, 0.1, "z")
        self.assertIs(first, second)
        cache.get(points, lengths, 4, 0.1, "z-trans")
        self.assertEqual(cache.quantization_count, 1)
        self.assertEqual(cache.layout_count, 2)
        self.assertEqual(cache.profile_stats["quantization_count"], 1)
        self.assertEqual(cache.profile_stats["layout_count"], 2)
        self.assertGreaterEqual(cache.profile_stats["total_ms"], 0.0)
        cache.clear()
        self.assertEqual(cache.quantization_count, 0)
        self.assertEqual(cache.layout_count, 0)
        self.assertEqual(cache.profile_stats["total_ms"], 0.0)
        third = cache.get(points, lengths, 4, 0.1, "z")
        self.assertIsNot(first, third)


class AttentionTests(unittest.TestCase):

    def test_opt_in_attention_diagnostics_report_entropy_and_distance(self):
        torch.manual_seed(30)
        points = torch.randn(12, 3)
        features = torch.randn(12, 48)
        lengths = torch.tensor([5, 7], dtype=torch.long)
        attention = SerializedPointROPEAttention(
            channels=48,
            num_heads=2,
            patch_size=4,
            order="z",
        ).eval()
        attention.set_diagnostics_mode(True, max_queries=6)
        attention(points, features, lengths, voxel_size=0.2)
        diagnostics = attention.diagnostics()
        self.assertEqual(diagnostics["query_count"], 6)
        self.assertEqual(diagnostics["head_observation_count"], 12)
        self.assertTrue(math.isfinite(diagnostics["entropy_normalized_mean"]))
        self.assertGreaterEqual(diagnostics["entropy_normalized_mean"], 0.0)
        self.assertLessEqual(diagnostics["entropy_normalized_mean"], 1.0 + 1e-6)
        self.assertTrue(math.isfinite(diagnostics["distance_m_mean"]))
        self.assertGreaterEqual(diagnostics["distance_m_mean"], 0.0)

    def test_block_diagnostics_report_residual_ratios(self):
        torch.manual_seed(29)
        points = torch.randn(10, 3)
        features = torch.randn(10, 48)
        lengths = torch.tensor([4, 6], dtype=torch.long)
        block = LitePointTransformerBlock(
            in_channels=48,
            out_channels=48,
            voxel_size=0.2,
            num_heads=2,
            patch_size=4,
            drop_path=0.0,
        ).eval()
        block.set_diagnostics_mode(True, max_queries=5)
        block(points, points, features, torch.empty((10, 0), dtype=torch.long), lengths)
        diagnostics = block.diagnostics()
        self.assertEqual(diagnostics["token_count"], 10)
        for key in (
            "attention_residual_ratio",
            "mlp_residual_ratio",
            "total_residual_ratio",
        ):
            self.assertTrue(math.isfinite(diagnostics[key]["mean"]))
            self.assertGreaterEqual(diagnostics[key]["mean"], 0.0)

    def test_variable_length_attention_forward_backward(self):
        torch.manual_seed(3)
        points = torch.randn(17, 3)
        lengths = torch.tensor([3, 9, 5], dtype=torch.long)
        features = torch.randn(17, 192, requires_grad=True)
        attention = SerializedPointROPEAttention(
            channels=192,
            num_heads=8,
            patch_size=8,
            attention_ratio=1.0,
            order="z-trans",
        )
        output = attention(points, features, lengths, voxel_size=0.2)
        self.assertEqual(tuple(output.shape), (17, 192))
        self.assertTrue(torch.isfinite(output).all())
        output.mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_clouds_do_not_exchange_attention_features(self):
        torch.manual_seed(31)
        points = torch.randn(12, 3)
        lengths = torch.tensor([5, 7], dtype=torch.long)
        features = torch.randn(12, 48)
        attention = SerializedPointROPEAttention(
            channels=48,
            num_heads=4,
            patch_size=8,
            order="z",
        ).eval()

        reference = attention(points, features, lengths, voxel_size=0.2)
        changed = features.clone()
        changed[5:] = changed[5:] * 1000 + 500
        changed_output = attention(points, changed, lengths, voxel_size=0.2)
        self.assertTrue(
            torch.allclose(reference[:5], changed_output[:5], atol=1e-6, rtol=1e-6)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_attention_forward_backward(self):
        torch.manual_seed(32)
        device = torch.device("cuda")
        points = torch.randn(19, 3, device=device)
        lengths = torch.tensor([8, 11], dtype=torch.long, device=device)
        features = torch.randn(19, 96, device=device, requires_grad=True)
        attention = SerializedPointROPEAttention(
            channels=96,
            num_heads=4,
            patch_size=8,
            order="z-trans",
        ).to(device)

        output = attention(points, features, lengths, voxel_size=0.2)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(features.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_attention_supports_both_amp_dtypes(self):
        torch.manual_seed(33)
        device = torch.device("cuda")
        points = torch.randn(19, 3, device=device)
        lengths = torch.tensor([8, 11], dtype=torch.long, device=device)
        attention = SerializedPointROPEAttention(
            channels=96,
            num_heads=4,
            patch_size=8,
            order="z-trans",
        ).to(device)

        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                features = torch.randn(
                    19,
                    96,
                    device=device,
                    requires_grad=True,
                )
                attention.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    output = attention(points, features, lengths, voxel_size=0.2)
                    loss = output.float().square().mean()
                loss.backward()
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(features.grad).all())

    def test_non_divisible_kpconvx_width_uses_valid_inner_dimension(self):
        attention = SerializedPointROPEAttention(
            channels=256,
            num_heads=8,
            patch_size=4,
            attention_ratio=1.0,
        )
        self.assertEqual(attention.attention_dim, 240)
        self.assertEqual(attention.head_dim % 6, 0)

    def test_block_supports_channel_transition_and_preserves_upcut(self):
        torch.manual_seed(4)
        points = torch.randn(10, 3)
        features = torch.randn(10, 192, requires_grad=True)
        lengths = torch.tensor([6, 4], dtype=torch.long)
        neighbors = torch.empty((10, 0), dtype=torch.long)
        upcut = torch.randn(10, 3)
        block = LitePointTransformerBlock(
            in_channels=192,
            out_channels=256,
            voxel_size=0.4,
            num_heads=8,
            patch_size=8,
            drop_path=0.0,
        )
        output, returned_upcut = block(
            points, points, features, neighbors, lengths, upcut=upcut
        )
        self.assertEqual(tuple(output.shape), (10, 256))
        self.assertIs(returned_upcut, upcut)
        output.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())


class KernelGeometryHandoverTests(unittest.TestCase):

    def test_occupancy_pool_and_room_shuffle_preserve_distributions(self):
        torch.manual_seed(7)
        points = torch.randn(9, 3) * 0.1
        neighbors = torch.arange(9).unsqueeze(1).repeat(1, 3)
        producer = KernelOccupancySignature(
            shell_sizes=[1, 4],
            radius=0.4,
            sigma=0.3,
            influence_mode="linear",
        )
        signature = producer(points, points, neighbors)
        self.assertEqual(tuple(signature.shape), (9, 5))
        self.assertTrue(torch.allclose(signature.sum(1), torch.ones(9), atol=1e-6))

        pools = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        pooled = pool_kernel_signature(signature, pools)
        self.assertEqual(tuple(pooled.shape), (3, 5))
        self.assertTrue(torch.allclose(pooled.sum(1), torch.ones(3), atol=1e-6))

        shuffled = shuffle_packed_signature(signature, torch.tensor([4, 5]))
        self.assertTrue(
            torch.allclose(
                signature[:4].sort(dim=0).values,
                shuffled[:4].sort(dim=0).values,
            )
        )
        self.assertTrue(
            torch.allclose(
                signature[4:].sort(dim=0).values,
                shuffled[4:].sort(dim=0).values,
            )
        )

    def test_signature_ablations_are_room_local_and_shape_preserving(self):
        signature = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.2, 0.8],
                [0.8, 0.2],
            ]
        )
        lengths = torch.tensor([2, 3])

        self.assertIs(ablate_packed_signature(signature, lengths, "none"), signature)
        self.assertTrue(
            torch.equal(
                ablate_packed_signature(signature, lengths, "zero"),
                torch.zeros_like(signature),
            )
        )
        room_mean = ablate_packed_signature(signature, lengths, "room_mean")
        self.assertTrue(torch.allclose(room_mean[:2], torch.tensor([[0.5, 0.5]]).repeat(2, 1)))
        expected_second = signature[2:].mean(dim=0, keepdim=True).repeat(3, 1)
        self.assertTrue(torch.allclose(room_mean[2:], expected_second))

        torch.manual_seed(17)
        shuffled = ablate_packed_signature(signature, lengths, "shuffle")
        self.assertTrue(
            torch.equal(
                shuffled[:2].sort(dim=0).values,
                signature[:2].sort(dim=0).values,
            )
        )
        with self.assertRaisesRegex(ValueError, "signature ablation mode"):
            ablate_packed_signature(signature, lengths, "unknown")

    def test_occupancy_can_reuse_the_models_exact_kernel_basis(self):
        kernel_points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, 0.2],
                [-0.2, 0.0, 0.0],
            ]
        )
        producer = KernelOccupancySignature(
            shell_sizes=[1, 4],
            radius=0.4,
            sigma=0.3,
            kernel_points=kernel_points,
        )
        self.assertTrue(torch.equal(producer.kernel_points, kernel_points))

    def test_v2_signature_preserves_mass_and_distance_channels(self):
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ]
        )
        neighbors = torch.tensor(
            [
                [0, 1, 3],
                [1, 0, 2],
                [2, 0, 3],
            ]
        )
        producer = KernelGeometrySignatureV2(
            shell_sizes=[1, 4],
            radius=0.4,
            sigma=0.3,
            influence_mode="linear",
        )
        signature = producer(points, points, neighbors)
        self.assertEqual(tuple(signature.shape), (3, 17))
        self.assertTrue(torch.isfinite(signature).all())
        self.assertTrue(
            torch.allclose(signature[:, :5].sum(1), torch.ones(3), atol=1e-6)
        )
        self.assertTrue(torch.all((signature[:, -2] >= 0) & (signature[:, -2] <= 1)))
        self.assertTrue(torch.all((signature[:, -1] >= 0) & (signature[:, -1] <= 1)))
        self.assertLess(float(signature[0, -2]), float(signature[1, -2]))

        pools = torch.tensor([[0, 1], [2, 3]])
        pooled = pool_kernel_geometry_signature_v2(signature, pools)
        self.assertEqual(tuple(pooled.shape), (2, 17))
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertGreater(float(pooled[0, -2]), 0.0)

        influence, nearest_kernel, _ = producer._assign_geometry(
            points, points, neighbors
        )
        cached = producer(
            points,
            points,
            neighbors,
            cached_geometry={
                "infl_w": influence,
                "neighb_1nn": nearest_kernel,
            },
        )
        self.assertTrue(torch.allclose(cached, signature, atol=1e-6, rtol=1e-6))

    def test_all_ktha_candidates_are_identity_initialized_and_trainable(self):
        torch.manual_seed(8)
        points = torch.randn(11, 3)
        features = torch.randn(11, 48)
        lengths = torch.tensor([5, 6])
        signature = torch.softmax(torch.randn(11, 5), dim=-1)
        baseline = SerializedPointROPEAttention(
            channels=48, num_heads=2, patch_size=4, geometry_mode="none"
        )
        baseline.eval()
        expected = baseline(points, features, lengths, voxel_size=0.2)

        for mode in (
            "concat",
            "qk",
            "relation_bias",
            "pairwise_bias_v2",
            "matched_mlp",
            "matched_mlp_v2",
        ):
            with self.subTest(mode=mode):
                candidate = SerializedPointROPEAttention(
                    channels=48,
                    num_heads=2,
                    patch_size=4,
                    geometry_mode=mode,
                    geometry_signature_dim=5,
                    geometry_relation_dim=3,
                )
                candidate.load_state_dict(baseline.state_dict(), strict=False)
                candidate.eval()
                actual = candidate(
                    points,
                    features,
                    lengths,
                    voxel_size=0.2,
                    kernel_signature=signature,
                )
                self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))

                candidate.train()
                candidate.zero_grad(set_to_none=True)
                actual.square().mean().backward()
                ktha_grads = [
                    parameter.grad
                    for name, parameter in candidate.named_parameters()
                    if "ktha" in name
                ]
                self.assertTrue(any(grad is not None for grad in ktha_grads))
                if mode == "pairwise_bias_v2":
                    metric_grad = candidate.ktha["pairwise_weight"]["value"].grad
                    self.assertIsNotNone(metric_grad)
                    self.assertGreater(float(metric_grad.abs().sum()), 0.0)

    def test_v2_pairwise_bias_has_no_semantic_or_constant_geometry_bypass(self):
        torch.manual_seed(18)
        points = torch.randn(11, 3)
        features = torch.randn(11, 48)
        lengths = torch.tensor([5, 6])
        signature = torch.randn(11, 17)
        baseline = SerializedPointROPEAttention(
            channels=48,
            num_heads=2,
            patch_size=4,
            geometry_mode="none",
        ).eval()
        candidate = SerializedPointROPEAttention(
            channels=48,
            num_heads=2,
            patch_size=4,
            geometry_mode="pairwise_bias_v2",
            geometry_signature_dim=17,
            geometry_relation_dim=3,
        ).eval()
        candidate.load_state_dict(baseline.state_dict(), strict=False)
        with torch.no_grad():
            candidate.ktha["pairwise_weight"]["value"].fill_(-0.5)
        candidate.set_diagnostics_mode(True, max_queries=32)

        expected = baseline(points, features, lengths, voxel_size=0.2)
        actual = candidate(
            points,
            features,
            lengths,
            voxel_size=0.2,
            kernel_signature=signature,
        )
        self.assertGreater(candidate.diagnostics()["geometry_bias_rms"], 0.0)
        zeros = candidate(
            points,
            features,
            lengths,
            voxel_size=0.2,
            kernel_signature=torch.zeros_like(signature),
        )
        self.assertEqual(candidate.diagnostics()["geometry_bias_rms"], 0.0)
        constant = candidate(
            points,
            features,
            lengths,
            voxel_size=0.2,
            kernel_signature=signature.mean(0, keepdim=True).expand_as(signature),
        )
        self.assertFalse(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(zeros, expected, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(constant, expected, atol=2e-5, rtol=2e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_v2_pairwise_bias_supports_bfloat16_backward(self):
        torch.manual_seed(19)
        device = torch.device("cuda")
        attention = SerializedPointROPEAttention(
            channels=48,
            num_heads=2,
            patch_size=8,
            geometry_mode="pairwise_bias_v2",
            geometry_signature_dim=17,
            geometry_relation_dim=3,
        ).to(device).train()
        points = torch.randn(12, 3, device=device)
        features = torch.randn(12, 48, device=device, requires_grad=True)
        lengths = torch.tensor([7, 5], device=device)
        signature = torch.randn(12, 17, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = attention(
                points,
                features,
                lengths,
                voxel_size=0.2,
                kernel_signature=signature,
            )
            loss = output.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(features.grad).all())
        metric_grad = attention.ktha["pairwise_weight"]["value"].grad
        self.assertIsNotNone(metric_grad)
        self.assertTrue(torch.isfinite(metric_grad).all())
        self.assertGreater(float(metric_grad.abs().sum()), 0.0)

    def test_matched_mlp_has_relation_bias_parameter_budget(self):
        kwargs = dict(
            channels=192,
            num_heads=8,
            patch_size=8,
            geometry_signature_dim=43,
            geometry_relation_dim=8,
        )
        relation = SerializedPointROPEAttention(
            geometry_mode="relation_bias", **kwargs
        )
        control = SerializedPointROPEAttention(
            geometry_mode="matched_mlp", **kwargs
        )
        relation_parameters = sum(p.numel() for p in relation.ktha.parameters())
        control_parameters = sum(p.numel() for p in control.ktha.parameters())
        self.assertEqual(control_parameters, relation_parameters)

    def test_v2_matched_mlp_has_exact_pairwise_bias_parameter_budget(self):
        kwargs = dict(
            channels=192,
            num_heads=8,
            patch_size=8,
            geometry_signature_dim=131,
            geometry_relation_dim=8,
        )
        relation = SerializedPointROPEAttention(
            geometry_mode="pairwise_bias_v2", **kwargs
        )
        control = SerializedPointROPEAttention(
            geometry_mode="matched_mlp_v2", **kwargs
        )
        relation_parameters = sum(p.numel() for p in relation.ktha.parameters())
        control_parameters = sum(p.numel() for p in control.ktha.parameters())
        self.assertEqual(control_parameters, relation_parameters)


if __name__ == "__main__":
    unittest.main()
