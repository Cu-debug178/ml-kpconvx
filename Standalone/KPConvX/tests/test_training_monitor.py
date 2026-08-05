import os
import sys
import types
import unittest

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.training_monitor import (capture_parameter_samples,
                                    collect_parameter_statistics,
                                    collect_sampled_update_statistics,
                                    merge_update_statistics)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


class TrainingMonitorTests(unittest.TestCase):

    def test_collects_disjoint_gradient_statistics(self):
        model = TinyModel()
        loss = model(torch.randn(5, 4)).square().mean()
        loss.backward()
        stats = collect_parameter_statistics(model)
        self.assertIn("global", stats)
        self.assertGreater(stats["global"]["grad_norm"], 0)
        self.assertGreater(stats["head"]["parameter_count"], 0)
        self.assertGreater(stats["backbone"]["parameter_count"], 0)

    def test_sampled_update_ratio_detects_optimizer_step(self):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = model(torch.randn(5, 4)).square().mean()
        loss.backward()
        stats = collect_parameter_statistics(model)
        snapshot = capture_parameter_samples(model, max_samples_per_tensor=8)
        optimizer.step()
        merged = merge_update_statistics(
            stats,
            collect_sampled_update_statistics(snapshot),
        )
        self.assertGreater(merged["global"]["sampled_update_rms"], 0)
        self.assertGreater(merged["global"]["sampled_update_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
