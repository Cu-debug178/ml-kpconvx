"""Static tests for the DKS full-room intervention entry point."""

import importlib.util
import os
import sys
import types
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from easydict import EasyDict
except ImportError:
    class EasyDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    module = types.ModuleType("easydict")
    module.EasyDict = EasyDict
    sys.modules["easydict"] = module

from utils.config import init_cfg  # noqa: E402


SCRIPT = os.path.join(ROOT, "tools", "evaluate_s3dis_dks_ablation.py")
SPEC = importlib.util.spec_from_file_location("evaluate_s3dis_dks_ablation", SCRIPT)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class DksAblationToolTests(unittest.TestCase):
    def test_baseline_resets_disabled_module_training_policy(self):
        cfg = init_cfg()
        cfg.model.dks_mode = "learned"
        cfg.model.dks_train_mode = "module_head"
        cfg.model.dks_inference_ablation = "shuffle"
        TOOL.configure_model_intervention(cfg, "baseline")
        self.assertEqual(cfg.model.dks_mode, "none")
        self.assertEqual(cfg.model.dks_train_mode, "joint")
        self.assertEqual(cfg.model.dks_inference_ablation, "none")

    def test_same_checkpoint_modes_map_to_explicit_ablations(self):
        for mode in ("true", "identity", "shuffle", "room_mean", "random"):
            with self.subTest(mode=mode):
                cfg = init_cfg()
                cfg.model.dks_mode = "learned"
                cfg.model.dks_train_mode = "module_head"
                TOOL.configure_model_intervention(cfg, mode)
                expected = "none" if mode == "true" else mode
                self.assertEqual(cfg.model.dks_inference_ablation, expected)
                self.assertEqual(cfg.model.dks_train_mode, "joint")

    def test_intervention_rejects_a_non_dks_source(self):
        cfg = init_cfg()
        with self.assertRaisesRegex(ValueError, "DKS source"):
            TOOL.configure_model_intervention(cfg, "shuffle")

    def test_init_identity_builds_learned_dks_from_a_baseline_source(self):
        cfg = init_cfg()
        TOOL.configure_model_intervention(cfg, "init_identity")
        self.assertEqual(cfg.model.dks_mode, "learned")
        self.assertEqual(cfg.model.dks_stages, "3")
        self.assertEqual(cfg.model.dks_train_mode, "joint")
        self.assertEqual(cfg.model.dks_inference_ablation, "none")

        cfg = init_cfg()
        cfg.model.dks_mode = "learned"
        with self.assertRaisesRegex(ValueError, "DKS-disabled"):
            TOOL.configure_model_intervention(cfg, "init_identity")

    def test_legacy_encoder_flag_is_inferred_from_checkpoint_keys(self):
        cfg = init_cfg()
        cfg.model.litept_enabled = True
        cfg.model.litept_conv_stages = 3
        cfg.model.litept_legacy_kpconvd_encoder = False
        TOOL.align_legacy_encoder_with_checkpoint(
            cfg, {"encoder_1.0.conv.weights": object()}
        )
        self.assertTrue(cfg.model.litept_legacy_kpconvd_encoder)

        TOOL.align_legacy_encoder_with_checkpoint(
            cfg, {"encoder_1.0.conv.alpha_mlp.0.weight": object()}
        )
        self.assertFalse(cfg.model.litept_legacy_kpconvd_encoder)


if __name__ == "__main__":
    unittest.main()
