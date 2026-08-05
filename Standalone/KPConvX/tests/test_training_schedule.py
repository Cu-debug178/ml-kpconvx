import math
import os
import sys
import unittest

from easydict import EasyDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.training_schedule import (
    optimizer_step_monitor_due,
    periodic_checkpoint_due,
    rebuild_cyclic_lr,
)


class TrainingScheduleTests(unittest.TestCase):

    @staticmethod
    def make_train_cfg(max_epoch, decrease10):
        return EasyDict(
            cyc_lr0=1e-4,
            cyc_lr1=5e-3,
            cyc_raise_n=30,
            cyc_decrease10=decrease10,
            cyc_plateau=5,
            max_epoch=max_epoch,
        )

    @staticmethod
    def final_lr(train_cfg):
        value = train_cfg.lr
        for epoch in range(train_cfg.max_epoch):
            value *= train_cfg.lr_decays.get(str(epoch), 1.0)
        return value

    def test_250_epoch_schedule_matches_original_final_lr(self):
        original = self.make_train_cfg(450, 120)
        compressed = self.make_train_cfg(250, 62)
        rebuild_cyclic_lr(original)
        rebuild_cyclic_lr(compressed)
        ratio = self.final_lr(compressed) / self.final_lr(original)
        self.assertTrue(math.isclose(ratio, 1.0, rel_tol=0.02), ratio)
        self.assertNotIn("250", compressed.lr_decays)
        self.assertEqual(max(map(int, compressed.lr_decays)), 249)

    def test_schedule_is_rebuilt_from_overridden_values(self):
        train_cfg = self.make_train_cfg(450, 120)
        rebuild_cyclic_lr(train_cfg)
        train_cfg.max_epoch = 250
        train_cfg.cyc_decrease10 = 62
        decays = rebuild_cyclic_lr(train_cfg)
        self.assertNotIn("449", decays)
        self.assertAlmostEqual(decays["249"], 0.1 ** (1 / 62))

    def test_periodic_checkpoint_names_are_completed_epochs(self):
        due = [
            epoch
            for epoch in range(1, 251)
            if periodic_checkpoint_due(epoch, 10, 200)
        ]
        self.assertEqual(due, [200, 210, 220, 230, 240, 250])

    def test_monitor_uses_optimizer_steps_not_mini_batches(self):
        selected = [
            mini_step
            for mini_step in range(0, 18)
            if optimizer_step_monitor_due(
                True,
                mini_step,
                (mini_step + 1) // 6 - 1,
                accum_batch=6,
                interval=2,
            )
        ]
        self.assertEqual(selected, [5, 17])


if __name__ == "__main__":
    unittest.main()
