import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from easydict import EasyDict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tasks.trainval import train_and_validate


class _Loader:

    def __init__(self):
        self.dataset = EasyDict(
            b_lim=32,
            reg_sampling_i=torch.zeros(1, dtype=torch.long),
            reg_votes=torch.zeros(1, dtype=torch.long),
        )


def _config(log_dir):
    return EasyDict(
        exp=EasyDict(saving=True, log_dir=log_dir),
        train=EasyDict(
            max_epoch=5,
            optimizer='SGD',
            lr=0.1,
            deform_lr_factor=1.0,
            sgd_momentum=0.0,
            weight_decay=0.0,
            lr_decays={},
            amp_enabled=False,
            amp_dtype='bfloat16',
            validation_mode='partial',
            save_best_val=False,
            save_best_val_cycle=True,
            save_latest_val=True,
            save_fraction_checkpoints=True,
            save_periodic_checkpoints=False,
            checkpoint_gap=1,
        ),
    )


class CheckpointPolicyTests(unittest.TestCase):

    def test_cycle_best_latest_and_four_fractions_are_independent(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-policy-') as temp:
            cfg = _config(temp)
            net = torch.nn.Linear(1, 1)

            def validation_result(epoch, *args, **kwargs):
                cycles = []
                if epoch in (2, 4):
                    cycles.append({
                        'vote_id': epoch // 2,
                        'miou': 80.0 - epoch,
                        'ious': [80.0 - epoch, 80.0 - epoch],
                        'start_epoch': epoch - 1,
                        'end_epoch': epoch,
                    })
                return {'metric': float(epoch), 'completed_cycles': cycles}

            with patch('tasks.trainval.training_epoch', return_value=True), patch(
                'tasks.trainval.validation_epoch', side_effect=validation_result
            ):
                summary = train_and_validate(
                    net,
                    _Loader(),
                    _Loader(),
                    cfg,
                    on_gpu=False,
                )

            checkpoint_dir = Path(temp) / 'checkpoints'
            self.assertTrue(summary.completed)
            self.assertEqual(summary.best_cycle_miou, 78.0)
            self.assertEqual(summary.best_cycle_vote, 1)
            self.assertFalse((checkpoint_dir / 'best_val_chkp.tar').exists())
            self.assertFalse((checkpoint_dir / 'chkp_0001.tar').exists())
            self.assertTrue((checkpoint_dir / 'current_chkp.tar').exists())
            self.assertTrue((checkpoint_dir / 'best_cycle_chkp.tar').exists())
            for fraction in range(1, 5):
                self.assertTrue((checkpoint_dir / f'chkp_{fraction}of5.tar').exists())

            best_state = torch.load(
                checkpoint_dir / 'best_cycle_chkp.tar',
                map_location='cpu',
                weights_only=False,
            )
            latest_state = torch.load(
                checkpoint_dir / 'current_chkp.tar',
                map_location='cpu',
                weights_only=False,
            )
            self.assertEqual(best_state['epoch'], 2)
            self.assertEqual(best_state['best_cycle_miou'], 78.0)
            self.assertEqual(latest_state['epoch'], 5)

            resume_validation_loader = _Loader()
            resumed = train_and_validate(
                torch.nn.Linear(1, 1),
                _Loader(),
                resume_validation_loader,
                cfg,
                chkp_path=str(checkpoint_dir / 'current_chkp.tar'),
                on_gpu=False,
            )
            self.assertEqual(resumed.best_cycle_miou, 78.0)
            self.assertEqual(resumed.best_cycle_vote, 1)
            self.assertEqual(resume_validation_loader.dataset.reg_votes.item(), 1)


if __name__ == '__main__':
    unittest.main()
