import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_s3dis_ktha_ablation import (  # noqa: E402
    completed_result,
    disable_concat_branch,
    disable_ktha_branch,
)
from summarize_s3dis_ktha_ablations import MODES, main as summarize_main  # noqa: E402


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ktha = torch.nn.ModuleDict(
            {
                "feature_scale": torch.nn.ParameterDict(
                    {"value": torch.nn.Parameter(torch.tensor(0.25))}
                )
            }
        )


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()


class _Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])


class _V2Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ktha = torch.nn.ModuleDict(
            {
                "pairwise_weight": torch.nn.ParameterDict(
                    {"value": torch.nn.Parameter(torch.full((2, 3), 0.5))}
                )
            }
        )


class KTHAAblationToolTests(unittest.TestCase):
    def test_branch_off_zeros_only_discovered_feature_scales(self):
        network = _Network()
        names, values = disable_concat_branch(network)
        self.assertEqual(len(names), 2)
        self.assertEqual(values, [0.25, 0.25])
        self.assertTrue(
            all(
                torch.equal(parameter, torch.zeros_like(parameter))
                for name, parameter in network.named_parameters()
                if name.endswith(".ktha.feature_scale.value")
            )
        )

    def test_branch_off_disables_v2_pairwise_metric(self):
        network = _V2Attention()
        names, values = disable_ktha_branch(network)
        self.assertEqual(names, ["ktha.pairwise_weight.value"])
        self.assertEqual(values, [0.5])
        self.assertTrue(
            torch.equal(
                network.ktha["pairwise_weight"]["value"],
                torch.zeros(2, 3),
            )
        )

    def test_completed_result_requires_explicit_completed_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(completed_result(root))
            (root / "result.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8"
            )
            self.assertFalse(completed_result(root))
            (root / "result.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            self.assertTrue(completed_result(root))

    def test_summary_checks_invariants_and_writes_contrasts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, mode in enumerate(MODES):
                mode_dir = root / mode
                mode_dir.mkdir()
                result = {
                    "status": "completed",
                    "mode": mode,
                    "checkpoint_sha256": "abc123",
                    "checkpoint_epoch": 7,
                    "seed": 57106803,
                    "protocol": "deterministic_full_identity_single_view",
                    "miou_pct": 70.0 + index,
                    "elapsed_seconds": 10.0,
                    "peak_cuda_allocated_bytes": 100,
                }
                (mode_dir / "result.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )

            with patch.object(sys, "argv", ["summary", "--root", str(root)]):
                self.assertEqual(summarize_main(), 0)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["contrasts"]["shuffled_minus_true_pct_point"], 1.0)
            self.assertTrue((root / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
