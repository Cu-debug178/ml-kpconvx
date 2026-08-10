import os
import sys
import types
import unittest
from unittest import mock

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from utils.mixed_precision import (  # noqa: E402
    MixedPrecisionSettings,
    autocast_context,
    create_grad_scaler,
    mixed_precision_state_dict,
    normalize_amp_dtype,
    resolve_mixed_precision,
    restore_mixed_precision_state,
)


class MixedPrecisionTests(unittest.TestCase):

    def test_only_explicit_supported_dtypes_are_accepted(self):
        self.assertEqual(normalize_amp_dtype("bfloat16"), "bfloat16")
        self.assertEqual(normalize_amp_dtype("FLOAT16"), "float16")
        with self.assertRaisesRegex(ValueError, "train.amp_dtype"):
            normalize_amp_dtype("float32")

    def test_disabled_mode_is_valid_on_cpu(self):
        cfg = types.SimpleNamespace(amp_enabled=False, amp_dtype="bfloat16")
        settings = resolve_mixed_precision(cfg, torch.device("cpu"))
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.dtype, torch.bfloat16)
        self.assertFalse(create_grad_scaler(settings, torch.device("cpu")).is_enabled())

    def test_enabled_training_rejects_non_cuda_device(self):
        cfg = types.SimpleNamespace(amp_enabled=True, amp_dtype="bfloat16")
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
            resolve_mixed_precision(cfg, torch.device("cpu"))

    def test_autocast_switch_changes_eligible_operator_dtype(self):
        settings = MixedPrecisionSettings(True, "bfloat16", torch.bfloat16)
        linear = torch.nn.Linear(4, 4)
        with autocast_context(settings, torch.device("cpu")):
            output = linear(torch.randn(2, 4))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_bfloat16_autocast_performs_a_real_optimizer_update(self):
        settings = MixedPrecisionSettings(True, "bfloat16", torch.bfloat16)
        device = torch.device("cpu")
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = create_grad_scaler(settings, device)
        before = model.weight.detach().clone()

        with autocast_context(settings, device):
            loss = model(torch.randn(8, 4)).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        self.assertFalse(torch.equal(before, model.weight.detach()))

    def test_float16_uses_and_restores_grad_scaler(self):
        settings = MixedPrecisionSettings(True, "float16", torch.float16)
        with mock.patch('torch.cuda.is_available', return_value=True):
            scaler = create_grad_scaler(settings, torch.device("cuda"))
        self.assertTrue(scaler.is_enabled())
        state = mixed_precision_state_dict(settings, scaler)

        with mock.patch('torch.cuda.is_available', return_value=True):
            restored = create_grad_scaler(settings, torch.device("cuda"))
        restore_mixed_precision_state(state, settings, restored)
        self.assertEqual(restored.state_dict(), scaler.state_dict())

    def test_resume_rejects_numerical_mode_change(self):
        old_settings = MixedPrecisionSettings(False, "bfloat16", torch.bfloat16)
        old_scaler = create_grad_scaler(old_settings, torch.device("cpu"))
        state = mixed_precision_state_dict(old_settings, old_scaler)

        new_settings = MixedPrecisionSettings(False, "float16", torch.float16)
        new_scaler = create_grad_scaler(new_settings, torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "differs from the run config"):
            restore_mixed_precision_state(state, new_settings, new_scaler)


if __name__ == "__main__":
    unittest.main()
