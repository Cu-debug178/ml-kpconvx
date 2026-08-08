import math
import os
import sys
import unittest

import torch
from easydict import EasyDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.training_schedule import (
    EpochMultiplicativeLRScheduler,
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

    def test_checkpointable_scheduler_matches_original_epoch_updates(self):
        train_cfg = self.make_train_cfg(250, 62)
        decays = rebuild_cyclic_lr(train_cfg)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=train_cfg.lr)
        scheduler = EpochMultiplicativeLRScheduler(optimizer, decays)

        expected_lr = train_cfg.lr
        for epoch in range(train_cfg.max_epoch):
            expected_lr *= decays.get(str(epoch), 1.0)
            scheduler.step(epoch)
            self.assertTrue(
                math.isclose(optimizer.param_groups[0]['lr'], expected_lr, rel_tol=1e-12)
            )

    def test_scheduler_state_restores_next_epoch_exactly(self):
        decays = {'1': 2.0, '2': 0.5, '4': 0.1}
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
        scheduler = EpochMultiplicativeLRScheduler(optimizer, decays)
        for epoch in range(3):
            scheduler.step(epoch)

        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        restored_parameter = torch.nn.Parameter(torch.tensor(1.0))
        restored_optimizer = torch.optim.SGD(
            [restored_parameter], lr=0.1, momentum=0.9
        )
        restored_scheduler = EpochMultiplicativeLRScheduler(
            restored_optimizer, decays
        )
        restored_optimizer.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)

        scheduler.step(3)
        restored_scheduler.step(3)
        self.assertEqual(restored_scheduler.last_epoch, scheduler.last_epoch)
        self.assertEqual(
            restored_optimizer.param_groups[0]['lr'],
            optimizer.param_groups[0]['lr'],
        )


if __name__ == "__main__":
    unittest.main()
