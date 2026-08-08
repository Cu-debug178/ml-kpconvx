import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.checkpoint_state import (
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
)


class CheckpointStateTests(unittest.TestCase):

    def test_cpu_rng_streams_resume_exactly(self):
        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)
        state = capture_rng_state()

        expected_python = [random.random() for _ in range(4)]
        expected_numpy = np.random.rand(4)
        expected_torch = torch.rand(4)

        random.seed(1)
        np.random.seed(2)
        torch.manual_seed(3)
        restore_rng_state(state)

        self.assertEqual([random.random() for _ in range(4)], expected_python)
        np.testing.assert_array_equal(np.random.rand(4), expected_numpy)
        torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)

    def test_atomic_torch_save_replaces_target_without_temporary_files(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-state-test-') as temp:
            path = Path(temp) / 'current_chkp.tar'
            atomic_torch_save({'epoch': 1}, path)
            atomic_torch_save({'epoch': 2}, path)

            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual([item.name for item in path.parent.iterdir()], [path.name])


if __name__ == '__main__':
    unittest.main()
