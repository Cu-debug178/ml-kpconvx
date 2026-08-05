import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.stage_diagnostics import (
    compose_ancestor_maps,
    fit_joint_pca_rgb,
    spearman_correlation,
    stage_cell_statistics,
)
from tools.analyze_stage_representations import masked_prediction_metrics


class StageDiagnosticsTests(unittest.TestCase):

    def test_joint_pca_uses_comparable_rgb_ranges(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(50, 8))
        b = a + 0.1
        rgb, info = fit_joint_pca_rgb([a, b], max_fit_points=100)
        self.assertEqual(rgb[0].shape, (50, 3))
        self.assertEqual(rgb[1].shape, (50, 3))
        self.assertTrue(np.all((rgb[0] >= 0) & (rgb[0] <= 1)))
        self.assertGreater(info["explained_variance_ratio_3"], 0)

    def test_composed_grid_maps_and_entropy(self):
        maps = [np.array([0, 0, 1, 1]), np.array([0, 0])]
        ancestors = compose_ancestor_maps(maps, [4, 2, 1])
        np.testing.assert_array_equal(ancestors[2], np.zeros(4, dtype=np.int64))
        stats = stage_cell_statistics(
            np.array([0, 0, 1, 1]), ancestors[1], stage_size=2, num_classes=2
        )
        self.assertFalse(stats["mixed"].any())
        stats2 = stage_cell_statistics(
            np.array([0, 1, 0, 1]), ancestors[1], stage_size=2, num_classes=2
        )
        self.assertTrue(stats2["mixed"].all())
        self.assertTrue(np.all(stats2["label_entropy"] > 0))

    def test_spearman(self):
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [30, 20, 10]), -1.0)

    def test_masked_metrics_report_subset_improvement(self):
        targets = np.array([0, 0, 1, 1])
        baseline = np.array([0, 1, 1, 0])
        improved = np.array([0, 0, 1, 0])
        mask = np.array([True, True, True, False])
        base_metrics = masked_prediction_metrics(baseline, targets, 2, mask)
        new_metrics = masked_prediction_metrics(improved, targets, 2, mask)
        self.assertEqual(base_metrics["point_count"], 3)
        self.assertGreater(new_metrics["OA"], base_metrics["OA"])


if __name__ == "__main__":
    unittest.main()
