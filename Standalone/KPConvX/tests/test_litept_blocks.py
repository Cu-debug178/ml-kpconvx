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
    build_serialized_patches,
    parse_serialization_orders,
)


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
        cache = SerializedPatchCache()
        first = cache.get(points, lengths, 4, 0.1, "z")
        second = cache.get(points, lengths, 4, 0.1, "z")
        self.assertIs(first, second)
        cache.get(points, lengths, 4, 0.1, "z-trans")
        self.assertEqual(cache.quantization_count, 1)
        self.assertEqual(cache.layout_count, 2)
        cache.clear()
        self.assertEqual(cache.quantization_count, 0)
        self.assertEqual(cache.layout_count, 0)
        third = cache.get(points, lengths, 4, 0.1, "z")
        self.assertIsNot(first, third)


class AttentionTests(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
