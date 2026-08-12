import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_s3dis_glskf_ablation import configure_model_intervention  # noqa: E402
from models.KPNext import KPNeXt  # noqa: E402
from tests.test_glskf_blocks import _make_config  # noqa: E402


class GLSKFAblationToolTests(unittest.TestCase):
    def test_baseline_from_module_head_config_can_construct_network(self):
        cfg = _make_config("kernel_gate")
        cfg.model.glskf_train_mode = "module_head"
        configure_model_intervention(cfg, "baseline")
        self.assertEqual(cfg.model.glskf_mode, "none")
        self.assertEqual(cfg.model.glskf_train_mode, "joint")
        KPNeXt(cfg)

    def test_true_intervention_clears_training_context_control(self):
        cfg = _make_config("kernel_gate")
        cfg.model.glskf_context_control = "shuffle"
        cfg.model.glskf_train_mode = "module_head"
        configure_model_intervention(cfg, "true")
        self.assertEqual(cfg.model.glskf_context_control, "none")
        self.assertEqual(cfg.model.glskf_inference_ablation, "none")
        self.assertEqual(cfg.model.glskf_train_mode, "joint")

    def test_public_shuffled_label_maps_to_model_shuffle_enum(self):
        cfg = _make_config("kernel_gate")
        configure_model_intervention(cfg, "shuffled")
        self.assertEqual(cfg.model.glskf_inference_ablation, "shuffle")


if __name__ == "__main__":
    unittest.main()
