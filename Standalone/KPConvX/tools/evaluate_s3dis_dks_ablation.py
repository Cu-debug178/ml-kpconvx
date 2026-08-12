#!/usr/bin/env python3
"""Evaluate same-checkpoint DKS interventions on deterministic Area_5 rooms."""

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


MODES = (
    "baseline",
    "init_identity",
    "true",
    "identity",
    "shuffle",
    "room_mean",
    "random",
)
DEFINITIONS = {
    "baseline": "external L0 reference with DKS disabled; not a same-checkpoint intervention",
    "init_identity": "L0 checkpoint loaded into freshly initialized learned DKS; pre-training identity check",
    "true": "learned per-point scales without intervention",
    "identity": "all DKS scales forced to one",
    "shuffle": "learned scales shuffled independently inside each packed room",
    "room_mean": "every point receives its packed-room mean learned scale",
    "random": "learned scales replaced by private-RNG uniform random scales",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_result(output_dir: Path) -> bool:
    result = output_dir / "result.json"
    if not result.is_file():
        return False
    try:
        with result.open(encoding="utf-8") as stream:
            return json.load(stream).get("status") == "completed"
    except (OSError, ValueError):
        return False


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def configure_model_intervention(cfg, mode: str):
    """Apply one DKS evaluation mode without leaving a training-only policy."""

    source_mode = str(getattr(cfg.model, "dks_mode", "none")).lower()
    if mode == "baseline":
        cfg.model.dks_mode = "none"
        cfg.model.dks_inference_ablation = "none"
        cfg.model.dks_train_mode = "joint"
        return cfg
    if mode == "init_identity":
        if source_mode != "none":
            raise ValueError("init_identity requires a DKS-disabled source configuration")
        cfg.model.dks_mode = "learned"
        cfg.model.dks_stages = "3"
        cfg.model.dks_hidden_dim = 32
        cfg.model.dks_alpha_min = 0.5
        cfg.model.dks_alpha_max = 1.2
        cfg.model.dks_fixed_alpha = 1.0
        cfg.model.dks_inference_ablation = "none"
        cfg.model.dks_train_mode = "joint"
        cfg.model.dks_log_stats = True
        return cfg
    if source_mode == "none":
        raise ValueError("DKS intervention requires a DKS source configuration")
    cfg.model.dks_inference_ablation = "none" if mode == "true" else mode
    cfg.model.dks_train_mode = "joint"
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_legacy_encoder_with_checkpoint(cfg, state_dict: dict) -> None:
    """Infer whether early LitePT stages use KPConvD from checkpoint keys.

    Some early L0 parameter files predate the explicit legacy flag even though
    their checkpoint contains KPConvD early stages.  The checkpoint is the
    primary source: KPConvX early blocks contain ``alpha_mlp`` parameters,
    whereas KPConvD blocks do not.
    """

    if not bool(getattr(cfg.model, "litept_enabled", False)):
        return
    conv_stages = int(getattr(cfg.model, "litept_conv_stages", 0))
    prefixes = tuple("encoder_{}.".format(stage) for stage in range(1, conv_stages + 1))
    early_keys = [key for key in state_dict if key.startswith(prefixes)]
    if not early_keys:
        return
    cfg.model.litept_legacy_kpconvd_encoder = not any(
        ".conv.alpha_mlp." in key for key in early_keys
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-log", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--seed", type=int, default=57106803)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    source_log = args.source_log.resolve()
    checkpoint_path = args.checkpoint.resolve()
    dataset_path = args.dataset_path.resolve()
    output_dir = args.output_dir.resolve()
    if completed_result(output_dir):
        print("Completed result already exists; skipping: {}".format(output_dir))
        return 0
    if output_dir.exists():
        raise RuntimeError(
            "incomplete output directory requires manual diagnosis: {}".format(
                output_dir
            )
        )
    for path, description in (
        (source_log / "parameters.json", "source parameters"),
        (checkpoint_path, "checkpoint"),
        (dataset_path / "Area_5", "S3DIS Area_5 directory"),
    ):
        if not path.exists():
            raise FileNotFoundError("{} not found: {}".format(description, path))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full Area_5 DKS evaluation")

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

    _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard_limit), hard_limit))
    set_seed(args.seed)

    cfg = load_cfg(str(source_log))
    S3DIR_cfg(cfg, dataset_path=str(dataset_path))
    configure_model_intervention(cfg, args.mode)
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
        "intervention": DEFINITIONS[args.mode],
        "dataset": "S3DIS",
        "split": "Area_5",
        "protocol": "deterministic_full_identity_single_view",
        "seed": args.seed,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_run": source_log.name,
        "amp_enabled": bool(cfg.train.amp_enabled),
        "amp_dtype": str(cfg.train.amp_dtype),
    }
    write_json_atomic(output_dir / "run_config.json", run_config)

    dataset = S3DIRDataset(cfg, chosen_set="validation", precompute_pyramid=True)
    dataset.b_n = cfg.test.batch_size
    dataset.b_lim = cfg.test.batch_limit
    sampler = SceneSegSampler(dataset)
    sampler.N = dataset.get_reg_sampling_size()
    loader_args = {
        "batch_size": 1,
        "sampler": sampler,
        "collate_fn": SceneSegCollate,
        "num_workers": cfg.test.num_workers,
        "pin_memory": True,
    }
    if cfg.test.num_workers > 0:
        loader_args["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_args)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    align_legacy_encoder_with_checkpoint(cfg, checkpoint["model_state_dict"])
    network = KPNeXt(cfg)
    if args.mode == "init_identity":
        incompatible = network.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if unexpected or not missing or not all(key.startswith("dks.") for key in missing):
            raise RuntimeError(
                "L0 checkpoint is not a clean DKS warm start; missing={!r}, unexpected={!r}".format(
                    missing, unexpected
                )
            )
        run_config["warm_start_missing_keys"] = missing
        write_json_atomic(output_dir / "run_config.json", run_config)
    else:
        network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = init_gpu(args.gpu)
    # Architecture construction consumes different RNG counts across modes.
    # Reset here so evaluation sampling is paired across interventions.
    set_seed(args.seed)
    amp_settings = resolve_mixed_precision(cfg.train, device)
    network.to(device)
    network.eval()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.no_grad():
        validation = validation_epoch(
            0, network, loader, cfg, EasyDict(), device, amp_settings
        )
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    class_ious = [100.0 * float(value) for value in validation["ious"]]
    result = {
        **run_config,
        "status": "completed",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "miou_pct": float(validation["metric"]),
        "class_iou_pct": dict(zip(cfg.data.label_names, class_ious)),
        "confusion": validation["confusion"],
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    write_json_atomic(output_dir / "result.json", result)
    print(json.dumps({"mode": args.mode, "miou_pct": result["miou_pct"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
