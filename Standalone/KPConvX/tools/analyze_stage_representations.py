#!/usr/bin/env python3
"""Compare stage representations for two S3DIS KPNeXt checkpoints.

Typical use compares grid-only KPConvX/LitePT against the same hierarchy with
FastAdapter. The script fits one PCA basis per stage across both models, exports
PLY files and a paper-style montage, then quantifies grid-cell semantic mixing
and whether FastAdapter corrections concentrate in degraded regions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ply import write_ply
from utils.stage_diagnostics import (
    compose_ancestor_maps,
    fit_joint_pca_rgb,
    representation_geometry_metrics,
    spearman_correlation,
    stage_cell_statistics,
    summarize_cell_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_log", required=True)
    parser.add_argument("--comparison_log", required=True)
    parser.add_argument("--baseline_checkpoint")
    parser.add_argument("--comparison_checkpoint")
    parser.add_argument("--baseline_name", default="Grid")
    parser.add_argument("--comparison_name", default="Grid+FastAdapter")
    parser.add_argument(
        "--include_adapter_bypass",
        action="store_true",
        help="Also evaluate the comparison checkpoint with FastAdapter bypassed at inference.",
    )
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--scene_index",
        type=int,
        default=None,
        help="Exact Area-5 room index. When omitted, the seeded regular sampler chooses a room.",
    )
    parser.add_argument("--cloud_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=57106803)
    parser.add_argument("--max_plot_points", type=int, default=30000)
    parser.add_argument("--max_metric_points", type=int, default=1024)
    parser.add_argument("--max_pca_fit_points", type=int, default=50000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(log_path: Path, explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = [
        "best_val_miou.tar",
        "best_mIoU_chkp.tar",
        "current_chkp.tar",
    ]
    checkpoint_dir = log_path / "checkpoints"
    for name in candidates:
        path = checkpoint_dir / name
        if path.is_file():
            return path
    periodic = sorted(checkpoint_dir.glob("chkp_*.tar"))
    if periodic:
        return periodic[-1]
    raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")


def deterministic_test_cfg(log_path: Path, dataset_path: str):
    from utils.config import load_cfg

    cfg = load_cfg(str(log_path))
    cfg.data.path = dataset_path
    cfg.test.batch_limit = 1
    cfg.test.in_radius = 100.0
    cfg.test.max_steps_per_epoch = 1
    cfg.test.max_votes = 1
    cfg.test.num_workers = 0
    cfg.augment_test.anisotropic = False
    cfg.augment_test.scale = [1.0, 1.0]
    cfg.augment_test.flips = [0.0, 0.0, 0.0]
    cfg.augment_test.rotations = "none"
    cfg.augment_test.jitter = 0.0
    cfg.augment_test.color_drop = 0.0
    cfg.augment_test.pts_drop_p = 0.0
    cfg.augment_test.pts_drop_reg = False
    cfg.augment_test.chromatic_contrast = False
    cfg.augment_test.chromatic_all = False
    return cfg


def validate_hierarchy_compatibility(cfg_a, cfg_b) -> None:
    keys = [
        ("model.in_sub_size", cfg_a.model.in_sub_size, cfg_b.model.in_sub_size),
        ("model.radius_scaling", cfg_a.model.radius_scaling, cfg_b.model.radius_scaling),
        ("model.grid_pool", cfg_a.model.grid_pool, cfg_b.model.grid_pool),
        ("num_stages", len(cfg_a.model.layer_blocks), len(cfg_b.model.layer_blocks)),
    ]
    mismatches = [f"{name}: {a} != {b}" for name, a, b in keys if a != b]
    if mismatches:
        raise ValueError(
            "The two checkpoints do not share the same point hierarchy:\n" + "\n".join(mismatches)
        )


def load_model(cfg, checkpoint_path: Path, device: torch.device):
    from models.KPNext import KPNeXt

    if cfg.model.kp_mode not in {"kpconvx", "kpconvd"}:
        raise ValueError("This analyzer currently supports KPNeXt kpconvx/kpconvd logs")
    model = KPNeXt(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def packed_slice(lengths: torch.Tensor, cloud_index: int) -> slice:
    lengths_cpu = lengths.detach().cpu().numpy().astype(np.int64)
    if not 0 <= cloud_index < len(lengths_cpu):
        raise IndexError(f"cloud_index={cloud_index} outside packed batch of size {len(lengths_cpu)}")
    start = int(lengths_cpu[:cloud_index].sum())
    return slice(start, start + int(lengths_cpu[cloud_index]))


def extract_cloud_trace(trace: Dict, cloud_index: int) -> Dict:
    points = []
    features = []
    pre_features = []
    stage_slices = []
    for stage, (point_tensor, length_tensor, stage_data) in enumerate(
        zip(trace["points"], trace["lengths"], trace["stages"])
    ):
        stage_slice = packed_slice(length_tensor, cloud_index)
        stage_slices.append(stage_slice)
        points.append(point_tensor[stage_slice].detach().cpu().numpy())
        features.append(stage_data["post_adapter_features"][stage_slice].detach().cpu().numpy())
        pre_features.append(stage_data["pre_adapter_features"][stage_slice].detach().cpu().numpy())

    upsample_maps = []
    for level, upsample in enumerate(trace["upsamples"]):
        fine_slice = stage_slices[level]
        coarse_slice = stage_slices[level + 1]
        mapping = upsample[fine_slice].reshape(-1).detach().cpu().numpy().astype(np.int64)
        mapping -= coarse_slice.start
        upsample_maps.append(mapping)

    labels = None
    if trace.get("labels") is not None:
        labels = trace["labels"][stage_slices[0]].detach().cpu().numpy().astype(np.int64)

    adapter = trace.get("adapter", {})
    adapter_layers = {}
    for stage, layer_data in adapter.get("layers", {}).items():
        full = layer_data.get("full")
        if not full:
            continue
        stage_slice = stage_slices[int(stage)]
        adapter_layers[int(stage)] = {
            key: value[stage_slice].detach().cpu().numpy()
            if value.ndim > 0 and value.shape[0] == trace["points"][int(stage)].shape[0]
            else value.detach().cpu().numpy()
            for key, value in full.items()
        }
    return {
        "points": points,
        "features": features,
        "pre_features": pre_features,
        "upsamples": upsample_maps,
        "labels": labels,
        "adapter_layers": adapter_layers,
    }


def map_valid_labels(labels: np.ndarray, valid_labels: Sequence[int]) -> np.ndarray:
    mapped = np.full(labels.shape, -1, dtype=np.int64)
    for index, value in enumerate(valid_labels):
        mapped[labels == value] = index
    return mapped


def confusion_metrics(logits: torch.Tensor, labels: torch.Tensor, valid_labels: Sequence[int]) -> Dict[str, float]:
    predictions = logits.argmax(dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    target = map_valid_labels(labels_np, valid_labels)
    valid = target >= 0
    class_count = len(valid_labels)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (target[valid], predictions[valid]), 1)
    tp = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - tp
    iou = tp / np.maximum(union, 1)
    recall = tp / np.maximum(confusion.sum(axis=1), 1)
    return {
        "mIoU": float(100.0 * iou.mean()),
        "mAcc": float(100.0 * recall.mean()),
        "OA": float(100.0 * tp.sum() / max(confusion.sum(), 1)),
    }


def masked_prediction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_count: int,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Task metrics on a diagnostic subset of points.

    ``mIoU_present`` and ``mAcc_present`` average only classes represented in
    the selected subset, avoiding an artificial penalty from absent classes.
    """

    predictions = np.asarray(predictions).reshape(-1).astype(np.int64, copy=False)
    targets = np.asarray(targets).reshape(-1).astype(np.int64, copy=False)
    mask = np.asarray(mask).reshape(-1).astype(bool, copy=False)
    valid = mask & (targets >= 0) & (targets < class_count)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    if np.any(valid):
        np.add.at(confusion, (targets[valid], predictions[valid]), 1)
    tp = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - tp
    support = confusion.sum(axis=1)
    present_iou = union > 0
    present_recall = support > 0
    iou = tp / np.maximum(union, 1)
    recall = tp / np.maximum(support, 1)
    point_count = int(valid.sum())
    correct = int(tp.sum())
    return {
        "point_count": float(point_count),
        "mIoU_present": float(100.0 * iou[present_iou].mean()) if np.any(present_iou) else float("nan"),
        "mAcc_present": float(100.0 * recall[present_recall].mean()) if np.any(present_recall) else float("nan"),
        "OA": float(100.0 * correct / max(point_count, 1)),
        "error_rate": float(100.0 * (point_count - correct) / max(point_count, 1)),
        "present_classes": float(present_iou.sum()),
    }


def boundary_mask(points: np.ndarray, labels: np.ndarray, k: int = 8) -> np.ndarray:
    """Mark points whose local spatial neighbourhood contains another class."""

    from sklearn.neighbors import KDTree

    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    if points.shape[0] < 2:
        return np.zeros(points.shape[0], dtype=bool)
    k_eff = min(max(2, k + 1), points.shape[0])
    neighbors = KDTree(points).query(points, k=k_eff, return_distance=False)
    neighbors = neighbors[:, 1:]
    return np.any(labels[neighbors] != labels[:, None], axis=1)


def diagnostic_masks(
    labels: np.ndarray,
    boundary: np.ndarray,
    ancestors: Sequence[np.ndarray],
    cell_stats: Sequence[Dict[str, np.ndarray]],
) -> List[Tuple[int, str, np.ndarray]]:
    """Build fixed, interpretable subsets for mechanism evidence."""

    valid = labels >= 0
    masks: List[Tuple[int, str, np.ndarray]] = [
        (0, "all_points", valid),
        (0, "semantic_boundary", valid & boundary),
        (0, "non_boundary", valid & ~boundary),
    ]
    for stage in range(1, len(ancestors)):
        ancestor = ancestors[stage]
        occupancy = cell_stats[stage]["occupancy"][ancestor]
        entropy = cell_stats[stage]["label_entropy"][ancestor]
        mixed = entropy > 1e-12
        masks.extend([
            (stage, "pure_cells", valid & ~mixed),
            (stage, "mixed_cells", valid & mixed),
        ])
        valid_occupancy = occupancy[valid]
        if valid_occupancy.size:
            q25, q75 = np.percentile(valid_occupancy, [25, 75])
            masks.extend([
                (stage, "low_occupancy_quartile", valid & (occupancy <= q25)),
                (stage, "high_occupancy_quartile", valid & (occupancy >= q75)),
            ])
        mixed_entropy = entropy[valid & mixed]
        if mixed_entropy.size:
            q75_entropy = np.percentile(mixed_entropy, 75)
            masks.append(
                (stage, "highest_mixed_entropy_quartile", valid & mixed & (entropy >= q75_entropy))
            )
        masks.append((stage, "mixed_boundary_points", valid & mixed & boundary))
    return masks


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def robust_scalar_to_uint8(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    result = np.zeros(values.shape, dtype=np.uint8)
    if not np.any(finite):
        return result
    lo, hi = np.percentile(values[finite], [1, 99])
    normalized = np.clip((values - lo) / max(hi - lo, 1e-12), 0, 1)
    result[finite] = np.round(255 * normalized[finite]).astype(np.uint8)
    return result


def save_stage_ply(
    path: Path,
    points: np.ndarray,
    rgb: np.ndarray,
    extra_fields: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    colors = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    arrays: List[np.ndarray] = [points.astype(np.float32), colors]
    names = ["x", "y", "z", "red", "green", "blue"]
    for name, values in (extra_fields or {}).items():
        values = np.asarray(values).reshape(-1)
        if values.shape[0] != points.shape[0]:
            continue
        arrays.append(values.astype(np.float32))
        names.append(name)
    write_ply(str(path), arrays, names)


def projected_xy(points: np.ndarray, azimuth_degrees: float = -50.0, elevation_degrees: float = 28.0) -> np.ndarray:
    az = math.radians(azimuth_degrees)
    el = math.radians(elevation_degrees)
    rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(el), -math.sin(el)], [0, math.sin(el), math.cos(el)]])
    centered = points - points.mean(axis=0, keepdims=True)
    rotated = centered @ rz.T @ rx.T
    return rotated[:, :2]


def render_montage(
    path: Path,
    names: Sequence[str],
    point_sets: Sequence[Sequence[np.ndarray]],
    rgb_sets: Sequence[Sequence[np.ndarray]],
    max_points: int,
    seed: int,
) -> None:
    rows = len(names)
    stages = len(point_sets[0])
    fig, axes = plt.subplots(rows, stages, figsize=(3.2 * stages, 3.0 * rows), squeeze=False)
    rng = np.random.default_rng(seed)
    for row in range(rows):
        for stage in range(stages):
            ax = axes[row][stage]
            points = point_sets[row][stage]
            colors = rgb_sets[row][stage]
            if max_points > 0 and points.shape[0] > max_points:
                indices = rng.choice(points.shape[0], size=max_points, replace=False)
                points = points[indices]
                colors = colors[indices]
            xy = projected_xy(points)
            point_size = max(0.15, min(24.0, 12000.0 / max(points.shape[0], 1))) * (1.25 ** stage)
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=point_size, linewidths=0, rasterized=True)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Stage {stage}", fontsize=14)
            if stage == 0:
                ax.text(-0.08, 0.5, names[row], transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=13)
    fig.suptitle("Joint-PCA stage representations (shared basis per stage)", fontsize=16)
    fig.tight_layout(rect=[0.03, 0.02, 1, 0.95])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    # Delay project-heavy imports until after argparse so --help works in a
    # lightweight environment and import errors point to missing runtime deps.
    from data_handlers.scene_seg import SceneSegCollate, SceneSegSampler
    from experiments.S3DIS.S3DIS_rooms import S3DIRDataset
    from utils.gpu_init import init_gpu

    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    baseline_log = Path(args.baseline_log).expanduser().resolve()
    comparison_log = Path(args.comparison_log).expanduser().resolve()
    baseline_cfg = deterministic_test_cfg(baseline_log, args.dataset_path)
    comparison_cfg = deterministic_test_cfg(comparison_log, args.dataset_path)
    validate_hierarchy_compatibility(baseline_cfg, comparison_cfg)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        device = init_gpu()
    else:
        device = init_gpu() if torch.cuda.is_available() else torch.device("cpu")

    # Area 5 is the held-out split for both validation and testing. The generic
    # dataset class hides labels in ``test`` mode, so diagnostics deliberately
    # use ``validation`` to retain the ground truth needed for cell entropy and
    # boundary analyses.
    dataset = S3DIRDataset(baseline_cfg, chosen_set="validation", precompute_pyramid=True)
    if args.scene_index is not None:
        if not 0 <= args.scene_index < len(dataset.scene_names):
            raise IndexError(
                f"scene_index={args.scene_index} outside Area-5 range "
                f"[0, {len(dataset.scene_names) - 1}]"
            )
        points = np.asarray(dataset.input_trees[args.scene_index].data, dtype=np.float32)
        dataset.reg_sample_pts = torch.from_numpy(points.mean(axis=0, keepdims=True))
        dataset.reg_sample_clouds = torch.tensor([args.scene_index], dtype=torch.long)
        dataset.reg_sampling_i.zero_()
        dataset.reg_votes.zero_()
    sampler = SceneSegSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=SceneSegCollate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    batch = next(iter(loader))
    cloud_id = args.cloud_index
    if "cloud_inds" in batch.in_dict:
        cloud_values = batch.in_dict.cloud_inds.detach().cpu().numpy().reshape(-1)
        if 0 <= args.cloud_index < cloud_values.shape[0]:
            cloud_id = int(cloud_values[args.cloud_index])
    if device.type == "cuda":
        batch.to(device)

    baseline_checkpoint = resolve_checkpoint(baseline_log, args.baseline_checkpoint)
    comparison_checkpoint = resolve_checkpoint(comparison_log, args.comparison_checkpoint)
    specs = [(args.baseline_name, baseline_cfg, baseline_checkpoint, False)]
    if args.include_adapter_bypass:
        specs.append((f"{args.comparison_name} (bypass)", comparison_cfg, comparison_checkpoint, True))
    specs.append((args.comparison_name, comparison_cfg, comparison_checkpoint, False))
    traces = []
    logits_list = []
    checkpoint_rows = []
    for name, cfg, checkpoint, bypass_adapter in specs:
        model = load_model(cfg, checkpoint, device)
        if bypass_adapter:
            if model.fast_adapter is None:
                raise ValueError("--include_adapter_bypass requires a comparison checkpoint with FastAdapter")
            model.fast_adapter = None
        with torch.no_grad():
            logits, trace = model(
                batch,
                return_intermediates=True,
                capture_adapter_details=True,
            )
        traces.append(extract_cloud_trace(trace, args.cloud_index))
        cloud_slice = packed_slice(trace["lengths"][0], args.cloud_index)
        logits_list.append(logits[cloud_slice].detach().cpu())
        checkpoint_rows.append({
            "model": name,
            "log_path": str(cfg.exp.log_dir),
            "checkpoint": str(checkpoint),
            "adapter_bypassed": int(bypass_adapter),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for stage in range(len(traces[0]["points"])):
        for trace in traces[1:]:
            if traces[0]["points"][stage].shape != trace["points"][stage].shape or not np.allclose(
                traces[0]["points"][stage], trace["points"][stage], atol=1e-5
            ):
                raise ValueError(f"Stage {stage} point sets differ; use checkpoints with identical grid hierarchy")

    labels_raw = traces[0]["labels"]
    valid_labels = sorted(
        label for label in baseline_cfg.data.label_values
        if label not in baseline_cfg.data.ignored_labels
    )
    labels = map_valid_labels(labels_raw, valid_labels)
    semantic_boundary = boundary_mask(traces[0]["points"][0], labels)
    stage_sizes = [points.shape[0] for points in traces[0]["points"]]
    ancestors = compose_ancestor_maps(traces[0]["upsamples"], stage_sizes)
    cell_stats = [
        stage_cell_statistics(labels, ancestor, stage_size, len(valid_labels))
        for ancestor, stage_size in zip(ancestors, stage_sizes)
    ]

    grid_rows = []
    for stage, stats in enumerate(cell_stats):
        row = {
            "cloud_id": cloud_id,
            "stage": stage,
            "point_count": stage_sizes[stage],
            "reduction_from_input": stage_sizes[stage] / max(stage_sizes[0], 1),
            **summarize_cell_statistics(stats),
        }
        grid_rows.append(row)
    write_csv(output_dir / "grid_degradation.csv", grid_rows)

    all_rgb: List[List[np.ndarray]] = [[] for _ in traces]
    representation_rows = []
    for stage in range(len(stage_sizes)):
        stage_features = [trace["features"][stage] for trace in traces]
        rgb, pca_info = fit_joint_pca_rgb(
            stage_features,
            max_fit_points=args.max_pca_fit_points,
            seed=args.seed + stage,
        )
        for model_index, (name, _, _, _) in enumerate(specs):
            all_rgb[model_index].append(rgb[model_index])
            metrics = representation_geometry_metrics(
                traces[model_index]["points"][stage],
                stage_features[model_index],
                max_points=args.max_metric_points,
                seed=args.seed + stage,
            )
            representation_rows.append({
                "cloud_id": cloud_id,
                "model": name,
                "stage": stage,
                "point_count": stage_sizes[stage],
                **pca_info,
                **metrics,
            })
            extras = {
                "cell_occupancy": cell_stats[stage]["occupancy"],
                "cell_entropy": cell_stats[stage]["label_entropy"],
                "minority_fraction": cell_stats[stage]["minority_fraction"],
            }
            if stage == 0:
                extras["semantic_boundary"] = semantic_boundary.astype(np.float32)
            adapter_layer = traces[model_index]["adapter_layers"].get(stage)
            if adapter_layer:
                for key in ("p2a_weights", "a2p_gate", "correction_norm", "correction_ratio"):
                    if key in adapter_layer:
                        extras[key] = adapter_layer[key]
            safe_name = name.lower().replace("+", "_plus_").replace(" ", "_")
            save_stage_ply(
                output_dir / f"{safe_name}_stage_{stage}.ply",
                traces[model_index]["points"][stage],
                rgb[model_index],
                extras,
            )
    write_csv(output_dir / "representation_metrics.csv", representation_rows)

    render_montage(
        output_dir / "stage_representation_comparison.png",
        [spec[0] for spec in specs],
        [trace["points"] for trace in traces],
        all_rgb,
        args.max_plot_points,
        args.seed,
    )

    adapter_rows = []
    comparison_trace = traces[-1]
    for stage, diag in sorted(comparison_trace["adapter_layers"].items()):
        correction_ratio = np.asarray(diag["correction_ratio"]).reshape(-1)
        current_occupancy = cell_stats[stage]["occupancy"]
        current_entropy = cell_stats[stage]["label_entropy"]
        row = {
            "cloud_id": cloud_id,
            "stage": stage,
            "point_count": correction_ratio.size,
            "correction_ratio_mean": float(np.mean(correction_ratio)),
            "correction_ratio_p90": float(np.percentile(correction_ratio, 90)),
            "a2p_gate_mean": float(np.mean(diag["a2p_gate"])),
            "a2p_gate_std": float(np.std(diag["a2p_gate"])),
            "corr_correction_current_occupancy": spearman_correlation(correction_ratio, current_occupancy),
            "corr_correction_current_entropy": spearman_correlation(correction_ratio, current_entropy),
        }
        if stage < len(stage_sizes) - 1:
            mapping = comparison_trace["upsamples"][stage]
            next_entropy = cell_stats[stage + 1]["label_entropy"][mapping]
            next_occupancy = cell_stats[stage + 1]["occupancy"][mapping]
            mixed = next_entropy > 0
            row.update({
                "corr_correction_next_occupancy": spearman_correlation(correction_ratio, next_occupancy),
                "corr_correction_next_entropy": spearman_correlation(correction_ratio, next_entropy),
                "correction_mixed_next_cells": float(correction_ratio[mixed].mean()) if np.any(mixed) else float("nan"),
                "correction_pure_next_cells": float(correction_ratio[~mixed].mean()) if np.any(~mixed) else float("nan"),
            })
        adapter_rows.append(row)
    write_csv(output_dir / "adapter_response.csv", adapter_rows)

    metric_rows = []
    predictions = []
    label_tensor = torch.from_numpy(labels_raw)
    for (name, _, _, _), logits in zip(specs, logits_list):
        predictions.append(logits.argmax(dim=1).numpy().astype(np.int64))
        metric_rows.append({
            "cloud_id": cloud_id,
            "model": name,
            **confusion_metrics(logits, label_tensor, valid_labels),
        })
    write_csv(output_dir / "single_cloud_metrics.csv", metric_rows)

    degradation_rows = []
    gain_rows = []
    for stage, group_name, mask in diagnostic_masks(
        labels,
        semantic_boundary,
        ancestors,
        cell_stats,
    ):
        group_metrics = []
        for model_index, (name, _, _, _) in enumerate(specs):
            metrics = masked_prediction_metrics(
                predictions[model_index],
                labels,
                len(valid_labels),
                mask,
            )
            group_metrics.append(metrics)
            degradation_rows.append({
                "cloud_id": cloud_id,
                "stage": stage,
                "group": group_name,
                "model": name,
                **metrics,
            })
        gain_rows.append({
            "cloud_id": cloud_id,
            "stage": stage,
            "group": group_name,
            "point_count": group_metrics[0]["point_count"],
            "delta_mIoU_present": group_metrics[-1]["mIoU_present"] - group_metrics[0]["mIoU_present"],
            "delta_mAcc_present": group_metrics[-1]["mAcc_present"] - group_metrics[0]["mAcc_present"],
            "delta_OA": group_metrics[-1]["OA"] - group_metrics[0]["OA"],
            "error_rate_reduction": group_metrics[0]["error_rate"] - group_metrics[-1]["error_rate"],
        })
    write_csv(output_dir / "degradation_task_metrics.csv", degradation_rows)
    write_csv(output_dir / "fastadapter_gain_by_degradation.csv", gain_rows)

    per_class_rows = []
    class_names = list(getattr(baseline_cfg.data, "label_names", valid_labels))
    for model_index, (name, _, _, _) in enumerate(specs):
        prediction = predictions[model_index]
        for class_index, label_value in enumerate(valid_labels):
            target_class = labels == class_index
            predicted_class = prediction == class_index
            intersection = np.logical_and(target_class, predicted_class).sum()
            union = np.logical_or(target_class, predicted_class).sum()
            support = target_class.sum()
            per_class_rows.append({
                "cloud_id": cloud_id,
                "model": name,
                "class_index": class_index,
                "label_value": label_value,
                "class_name": class_names[class_index] if class_index < len(class_names) else str(label_value),
                "support": int(support),
                "IoU": float(100.0 * intersection / max(union, 1)),
                "accuracy": float(100.0 * intersection / max(support, 1)),
            })
    write_csv(output_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(output_dir / "checkpoints.csv", checkpoint_rows)

    summary = [
        "# Stage representation and grid/FastAdapter diagnostic",
        "",
        f"This run uses identical deterministic S3DIS cloud {cloud_id} for both checkpoints.",
        "PCA is fitted jointly per stage, so colours are directly comparable across rows.",
        "",
        "## Single-cloud task metrics",
        "",
    ]
    for row in metric_rows:
        summary.append(
            f"- {row['model']}: mIoU={row['mIoU']:.2f}, mAcc={row['mAcc']:.2f}, OA={row['OA']:.2f}"
        )
    summary.extend([
        "",
        "## Interpretation rules",
        "",
        "- `grid_degradation.csv` proves what the grid hierarchy destroys: occupancy, mixed-cell ratio and label entropy.",
        "- `representation_metrics.csv` describes the geometry-to-semantics transition; it is not itself a task score.",
        "- `adapter_response.csv` tests whether correction magnitude is spatially targeted at mixed/high-entropy cells.",
        "- `degradation_task_metrics.csv` reports both models on boundary, mixed-cell and high-degradation subsets.",
        "- `fastadapter_gain_by_degradation.csv` is the direct mechanism test: gains should be larger in mixed/high-entropy regions.",
        "- The montage is qualitative evidence only. Use multiple rooms and seeds before making a paper claim.",
        "",
    ])
    (output_dir / "README.md").write_text("\n".join(summary), encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Diagnostics written to: {output_dir}")


if __name__ == "__main__":
    main()
