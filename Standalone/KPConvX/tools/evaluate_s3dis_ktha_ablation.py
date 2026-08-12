#!/usr/bin/env python3
"""Evaluate one KTHA intervention with deterministic full-identity validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODES = ("true", "shuffled", "zero", "room_mean", "branch_off")
MODE_TO_SIGNATURE_ABLATION = {
    "true": "none",
    "shuffled": "shuffle",
    "zero": "zero",
    "room_mean": "room_mean",
    "branch_off": "none",
}
INTERVENTION_DEFINITIONS = {
    "true": "unaltered token-aligned kernel signature",
    "shuffled": "whole signature rows randomly permuted within each packed room",
    "zero": "zero signature with trained KTHA parameters retained",
    "room_mean": "each token receives its room's mean signature",
    "branch_off": "trained KTHA gate or metric parameters are set to zero in memory",
}

BRANCH_PARAMETER_SUFFIXES = (
    "ktha.feature_scale.value",
    "ktha.q_scale.value",
    "ktha.k_scale.value",
    "ktha.bias_scale.value",
    "ktha.pairwise_weight.value",
)


def is_ktha_branch_parameter(name: str) -> bool:
    return any(
        name == suffix or name.endswith("." + suffix)
        for suffix in BRANCH_PARAMETER_SUFFIXES
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--seed", type=int, default=57106803)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disable_ktha_branch(network: torch.nn.Module) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    with torch.no_grad():
        for name, parameter in network.named_parameters():
            if is_ktha_branch_parameter(name):
                names.append(name)
                values.append(float(parameter.detach().float().abs().mean().cpu()))
                parameter.zero_()
    if not names:
        raise RuntimeError("branch_off requires at least one recognized KTHA gate")
    return names, values


def disable_concat_branch(network: torch.nn.Module) -> tuple[list[str], list[float]]:
    """Backward-compatible alias for the original M1-only helper name."""

    return disable_ktha_branch(network)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def completed_result(output_dir: Path) -> bool:
    result_path = output_dir / "result.json"
    if not result_path.is_file():
        return False
    with result_path.open(encoding="utf-8") as stream:
        return json.load(stream).get("status") == "completed"


def main() -> int:
    args = parse_args()
    source_log = args.source_log.resolve()
    checkpoint_path = args.checkpoint.resolve()
    dataset_path = args.dataset_path.resolve()
    output_dir = args.output_dir.resolve()

    if completed_result(output_dir):
        print("Completed result already exists; skipping: {}".format(output_dir))
        return 0
    if output_dir.exists():
        raise RuntimeError(
            "incomplete output directory requires manual diagnosis: {}".format(output_dir)
        )
    for path, description in (
        (source_log / "parameters.json", "source parameters"),
        (checkpoint_path, "checkpoint"),
        (dataset_path, "dataset"),
    ):
        if not path.exists():
            raise FileNotFoundError("{} not found: {}".format(description, path))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full Area 5 ablation evaluation")

    from easydict import EasyDict
    from torch.utils.data import DataLoader

    from data_handlers.scene_seg import SceneSegCollate, SceneSegSampler
    from experiments.S3DIS.S3DIS_rooms import S3DIR_cfg, S3DIRDataset
    from experiments.S3DIS.train_S3DIS import configure_validation_mode
    from models.KPNext import KPNeXt
    from tasks.validation import validation_epoch
    from utils.config import load_cfg
    from utils.gpu_init import init_gpu
    from utils.mixed_precision import resolve_mixed_precision

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard_limit), hard_limit))
    set_seed(args.seed)

    cfg = load_cfg(str(source_log))
    S3DIR_cfg(cfg, dataset_path=str(dataset_path))
    if str(cfg.model.ktha_mode).lower() == "none":
        raise ValueError("this experiment requires a KTHA checkpoint")
    cfg.model.ktha_shuffle_geometry = False
    cfg.model.ktha_signature_ablation = MODE_TO_SIGNATURE_ABLATION[args.mode]
    cfg.train.validation_mode = "full_identity"
    cfg.train.save_best_val_cycle = False
    cfg.test.save_validation_clouds = False
    cfg.exp.log_dir = str(output_dir)
    cfg.exp.results_dir = str(output_dir.parent)
    cfg.exp.date = output_dir.name
    cfg.exp.seed = args.seed
    cfg.exp.saving = False
    configure_validation_mode(cfg)

    output_dir.mkdir(parents=True)
    run_config = {
        "status": "running",
        "mode": args.mode,
        "intervention": INTERVENTION_DEFINITIONS[args.mode],
        "dataset": "S3DIS",
        "split": "Area_5",
        "protocol": "deterministic_full_identity_single_view",
        "seed": args.seed,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "amp_enabled": bool(cfg.train.amp_enabled),
        "amp_dtype": str(cfg.train.amp_dtype),
    }
    write_json_atomic(output_dir / "run_config.json", run_config)

    dataset = S3DIRDataset(cfg, chosen_set="validation", precompute_pyramid=True)
    dataset.b_n = cfg.test.batch_size
    dataset.b_lim = cfg.test.batch_limit
    sampler = SceneSegSampler(dataset)
    sampler.N = dataset.get_reg_sampling_size()
    loader_kwargs = {
        "batch_size": 1,
        "sampler": sampler,
        "collate_fn": SceneSegCollate,
        "num_workers": cfg.test.num_workers,
        "pin_memory": True,
    }
    if cfg.test.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    network = KPNeXt(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    disabled_names: list[str] = []
    original_scales: list[float] = []
    if args.mode == "branch_off":
        disabled_names, original_scales = disable_ktha_branch(network)

    device = init_gpu(args.gpu)
    # Architecture construction consumes different RNG amounts across KTHA
    # variants.  Keep subsequent sampling aligned with the requested seed.
    set_seed(args.seed)
    amp_settings = resolve_mixed_precision(cfg.train, device)
    network.to(device)
    network.eval()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.no_grad():
        validation = validation_epoch(
            0,
            network,
            loader,
            cfg,
            EasyDict(),
            device,
            amp_settings,
        )
    torch.cuda.synchronize(device)
    elapsed_seconds = time.monotonic() - started

    class_ious = [100.0 * float(value) for value in validation["ious"]]
    result = {
        **run_config,
        "status": "completed",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "miou_pct": float(validation["metric"]),
        "class_iou_pct": dict(zip(cfg.data.label_names, class_ious)),
        "confusion": validation["confusion"],
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "disabled_gate_parameter_count": len(disabled_names),
        "original_disabled_gate_mean_abs": original_scales,
        # Retain the original field names for existing M1 summaries.
        "disabled_scale_count": len(disabled_names),
        "original_disabled_scales": original_scales,
    }
    write_json_atomic(output_dir / "result.json", result)
    print(json.dumps({"mode": args.mode, "miou_pct": result["miou_pct"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
