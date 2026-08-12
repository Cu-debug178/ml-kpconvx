import os
import sys
import types
import csv
import tempfile
import unittest

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.training_monitor import (append_dks_monitor,
                                    capture_parameter_samples,
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

    def test_dks_monitor_writes_the_stable_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            append_dks_monitor(
                directory,
                epoch=2,
                optimizer_step=123,
                diagnostics={
                    3: {
                        "mean": 1.0,
                        "std": 0.05,
                        "p05": 0.9,
                        "p95": 1.1,
                        "frac_below_0.9": 0.2,
                        "frac_above_1.1": 0.1,
                        "gate": 0.4,
                    }
                },
            )
            with open(os.path.join(directory, "dks_alpha_stats.csv"), newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], "123")
            self.assertEqual(rows[0]["stage"], "3")
            self.assertEqual(rows[0]["frac_lt_0p9"], "0.2")


if __name__ == "__main__":
    unittest.main()
