#!/usr/bin/env python3
"""CPU-only compatibility checks for the KTHA inference-ablation queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.KPNext import KPNeXt  # noqa: E402
from tools.evaluate_s3dis_ktha_ablation import is_ktha_branch_parameter  # noqa: E402
from utils.config import load_cfg  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    args = parser.parse_args()

    source_log = args.source_log.resolve()
    checkpoint_path = args.checkpoint.resolve()
    dataset_path = args.dataset_path.resolve()
    required = (
        (source_log / "parameters.json", "source parameters"),
        (checkpoint_path, "checkpoint"),
        (dataset_path / "Area_5", "S3DIS Area_5 directory"),
    )
    for path, description in required:
        if not path.exists():
            raise FileNotFoundError("{} not found: {}".format(description, path))

    cfg = load_cfg(str(source_log))
    if str(cfg.model.ktha_mode).lower() == "none":
        raise ValueError("source configuration does not enable KTHA")
    cfg.model.ktha_shuffle_geometry = False
    cfg.model.ktha_signature_ablation = "none"
    network = KPNeXt(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    gate_names = [
        name
        for name, _ in network.named_parameters()
        if is_ktha_branch_parameter(name)
    ]
    if not gate_names:
        raise RuntimeError("checkpoint has no recognized KTHA gate parameters")

    summary = {
        "status": "ready",
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "ktha_mode": str(cfg.model.ktha_mode),
        "ktha_gate_parameter_count": len(gate_names),
        "dataset_split": "S3DIS Area_5",
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
