#!/usr/bin/env python3
"""Three-layer S3DIS diagnosis: dataset, one model, and paired models.

Examples are documented in ``tools/S3DIS_DIAGNOSTICS.md``.  Large prediction
artifacts should be written below ``Standalone/KPConvX/results/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.s3dis_diagnostics import (  # noqa: E402
    S3DIS_CLASS_NAMES,
    confusion_matrix,
    difficulty_masks,
    estimate_quantile_thresholds,
    fixed_radius_geometry,
    index_prediction_artifacts,
    load_prediction_artifact,
    metrics_from_confusion,
    patch_neighbor_recall,
    per_class_metrics,
    radius_suffix,
    save_prediction_artifact,
    validate_paired_artifacts,
)
from utils.stage_diagnostics import compose_ancestor_maps, stage_cell_statistics  # noqa: E402


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_scene_filename(scene_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", scene_name) + ".npz"


def parse_area_index(name: str) -> Optional[int]:
    match = re.search(r"Area[_-]?(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def discover_s3dis_scenes(dataset_path: Path) -> List[Dict[str, object]]:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_dir():
        raise NotADirectoryError(dataset_path)
    records = []
    for area_path in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        area_index = parse_area_index(area_path.name)
        if area_index is None:
            continue
        split = "validation" if area_index == 5 else "training"
        for scene_path in sorted(area_path.iterdir()):
            supported = scene_path.is_dir() or scene_path.suffix.lower() in {".ply", ".npy"}
            if not supported:
                continue
            if scene_path.is_dir() and not (
                (scene_path / "coord.npy").is_file()
                and (scene_path / "segment.npy").is_file()
            ):
                continue
            records.append(
                {
                    "area": area_index,
                    "split": split,
                    "scene_name": "{}_{}".format(area_path.name, scene_path.stem),
                    "path": scene_path,
                }
            )
    if not records:
        raise FileNotFoundError(
            "No S3DIS rooms found. Expected Area_*/<room> with coord.npy and segment.npy, "
            "or room .ply/.npy files under {}".format(dataset_path)
        )
    if not any(record["area"] == 5 for record in records):
        raise FileNotFoundError("Area 5 was not found under {}".format(dataset_path))
    return records


def load_raw_scene(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    path = Path(path)
    instances = None
    if path.is_dir():
        points = np.load(path / "coord.npy")
        labels = np.load(path / "segment.npy")
        instance_path = path / "instance.npy"
        if instance_path.is_file():
            instances = np.load(instance_path)
    elif path.suffix.lower() == ".npy":
        data = np.load(path)
        if data.ndim != 2 or data.shape[1] < 7:
            raise ValueError("{} must contain xyzrgb+label columns".format(path))
        points = data[:, :3]
        labels = data[:, 6]
    elif path.suffix.lower() == ".ply":
        from utils.ply import read_ply

        data = read_ply(str(path))
        fields = set(data.dtype.fields)
        required = {"x", "y", "z", "class"}
        if not required.issubset(fields):
            raise ValueError("{} lacks PLY fields {}".format(path, sorted(required - fields)))
        points = np.column_stack((data["x"], data["y"], data["z"]))
        labels = data["class"]
        for instance_field in ("instance", "instance_id", "object"):
            if instance_field in fields:
                instances = data[instance_field]
                break
    else:
        raise ValueError("Unsupported scene path: {}".format(path))
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
    if points.shape != (labels.shape[0], 3):
        raise ValueError("{} has non-aligned points and labels".format(path))
    if instances is not None:
        instances = np.asarray(instances).reshape(-1).astype(np.int64, copy=False)
        if instances.shape[0] != labels.shape[0]:
            raise ValueError("{} has non-aligned instance ids".format(path))
    return points, labels, instances


def sample_query_indices(point_count: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or point_count <= maximum:
        return np.arange(point_count, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(point_count, maximum, replace=False))


def summarize_attribute(values: np.ndarray, prefix: str) -> Dict[str, object]:
    values = np.asarray(values).reshape(-1)
    finite = values[np.isfinite(values)].astype(np.float64, copy=False)
    if finite.size == 0:
        return {
            prefix + "_valid_count": 0,
            prefix + "_mean": float("nan"),
            prefix + "_p25": float("nan"),
            prefix + "_p50": float("nan"),
            prefix + "_p75": float("nan"),
        }
    return {
        prefix + "_valid_count": int(finite.size),
        prefix + "_mean": float(np.mean(finite)),
        prefix + "_p25": float(np.quantile(finite, 0.25)),
        prefix + "_p50": float(np.quantile(finite, 0.50)),
        prefix + "_p75": float(np.quantile(finite, 0.75)),
    }


def run_dataset_audit(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    records = discover_s3dis_scenes(Path(args.dataset_path))
    class_count = len(S3DIS_CLASS_NAMES)

    room_rows = []
    composition_rows = []
    geometry_rows = []
    instance_rows = []
    scope_counts: Dict[Tuple[str, str], np.ndarray] = {}
    scope_room_counts: Dict[Tuple[str, str], np.ndarray] = {}
    scope_total_rooms: Dict[Tuple[str, str], int] = defaultdict(int)
    invalid_label_count = 0

    for scene_i, record in enumerate(records):
        points, labels, instances = load_raw_scene(record["path"])
        valid = (labels >= 0) & (labels < class_count)
        invalid_label_count += int((~valid).sum())
        counts = np.bincount(labels[valid], minlength=class_count)
        extents = np.ptp(points, axis=0) if points.shape[0] else np.zeros(3)
        room_rows.append(
            {
                "scene_name": record["scene_name"],
                "area": record["area"],
                "split": record["split"],
                "point_count": int(points.shape[0]),
                "valid_label_count": int(valid.sum()),
                "class_count_present": int(np.count_nonzero(counts)),
                "extent_x_m": float(extents[0]),
                "extent_y_m": float(extents[1]),
                "extent_z_m": float(extents[2]),
                "has_true_instance_ids": int(instances is not None),
            }
        )
        for class_index, class_name in enumerate(S3DIS_CLASS_NAMES):
            composition_rows.append(
                {
                    "scene_name": record["scene_name"],
                    "area": record["area"],
                    "split": record["split"],
                    "class_index": class_index,
                    "class_name": class_name,
                    "point_count": int(counts[class_index]),
                    "point_ratio": float(counts[class_index] / max(valid.sum(), 1)),
                }
            )

        for scope in ((record["split"], "all"), ("area", str(record["area"]))):
            scope_counts.setdefault(scope, np.zeros(class_count, dtype=np.int64))
            scope_room_counts.setdefault(scope, np.zeros(class_count, dtype=np.int64))
            scope_counts[scope] += counts
            scope_room_counts[scope] += counts > 0
            scope_total_rooms[scope] += 1

        if instances is not None:
            for instance_id in np.unique(instances):
                if instance_id < 0:
                    continue
                instance_mask = instances == instance_id
                instance_points = points[instance_mask]
                instance_labels = labels[instance_mask]
                valid_instance_labels = instance_labels[
                    (instance_labels >= 0) & (instance_labels < class_count)
                ]
                if valid_instance_labels.size:
                    semantic_class = int(
                        np.bincount(valid_instance_labels, minlength=class_count).argmax()
                    )
                else:
                    semantic_class = -1
                instance_extent = np.ptp(instance_points, axis=0)
                instance_rows.append(
                    {
                        "scene_name": record["scene_name"],
                        "instance_id": int(instance_id),
                        "semantic_class": semantic_class,
                        "class_name": S3DIS_CLASS_NAMES[semantic_class]
                        if semantic_class >= 0
                        else "invalid",
                        "point_count": int(instance_mask.sum()),
                        "extent_x_m": float(instance_extent[0]),
                        "extent_y_m": float(instance_extent[1]),
                        "extent_z_m": float(instance_extent[2]),
                    }
                )

        geometry_enabled = args.geometry_split == "all" or (
            args.geometry_split == "validation" and record["split"] == "validation"
        )
        if geometry_enabled:
            query_indices = sample_query_indices(
                points.shape[0], args.geometry_max_points_per_room, args.seed + scene_i
            )
            attributes = fixed_radius_geometry(
                points,
                labels,
                radii=args.radii,
                pca_radius=args.pca_radius,
                query_indices=query_indices,
                chunk_size=args.geometry_chunk_size,
                max_pca_neighbors=args.max_pca_neighbors,
            )
            row = {
                "scene_name": record["scene_name"],
                "area": record["area"],
                "split": record["split"],
                "room_point_count": int(points.shape[0]),
                "query_point_count": int(query_indices.size),
                "query_fraction": float(query_indices.size / max(points.shape[0], 1)),
                "pca_neighbor_cap": args.max_pca_neighbors,
            }
            for name, values in attributes.items():
                row.update(summarize_attribute(values, name))
            full_neighbors = attributes["pca_neighbor_count_full"]
            row["pca_neighbor_cap_saturation_ratio"] = (
                float(np.mean(full_neighbors > args.max_pca_neighbors))
                if full_neighbors.size and args.max_pca_neighbors > 0
                else 0.0
            )
            geometry_rows.append(row)

    class_rows = []
    for (scope_type, scope_value), counts in sorted(scope_counts.items()):
        total_points = int(counts.sum())
        total_rooms = scope_total_rooms[(scope_type, scope_value)]
        room_counts = scope_room_counts[(scope_type, scope_value)]
        for class_index, class_name in enumerate(S3DIS_CLASS_NAMES):
            class_rows.append(
                {
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "class_index": class_index,
                    "class_name": class_name,
                    "point_count": int(counts[class_index]),
                    "point_ratio": float(counts[class_index] / max(total_points, 1)),
                    "room_count": int(room_counts[class_index]),
                    "room_coverage": float(room_counts[class_index] / max(total_rooms, 1)),
                    "total_rooms": int(total_rooms),
                }
            )

    write_csv(output_dir / "dataset_class_stats.csv", class_rows)
    write_csv(output_dir / "room_summary.csv", room_rows)
    write_csv(output_dir / "room_class_composition.csv", composition_rows)
    write_csv(output_dir / "room_geometry_summary.csv", geometry_rows)
    write_csv(output_dir / "instance_stats.csv", instance_rows)
    write_json(
        output_dir / "dataset_summary.json",
        {
            "dataset_path": "<DATASET_PATH>",
            "room_count": len(records),
            "training_room_count": sum(record["split"] == "training" for record in records),
            "area5_room_count": sum(record["area"] == 5 for record in records),
            "invalid_label_count": invalid_label_count,
            "rooms_with_true_instance_ids": sum(row["has_true_instance_ids"] for row in room_rows),
            "geometry_split": args.geometry_split,
            "geometry_radii_m": list(args.radii),
            "pca_radius_m": args.pca_radius,
            "geometry_max_points_per_room": args.geometry_max_points_per_room,
            "density_definition": "exact count of other room points inside a fixed physical radius",
            "boundary_definition": "at least one differently-labelled neighbour inside a fixed physical radius",
            "instance_note": "instance_stats.csv contains only true supplied instance IDs; no semantic connected-component proxy is substituted",
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# S3DIS dataset audit",
                "",
                "This is evidence about dataset composition and geometric difficulty, not model performance.",
                "",
                "- `dataset_class_stats.csv`: point imbalance and room coverage for training/Area 5.",
                "- `room_class_composition.csv`: room-by-class distribution shift evidence.",
                "- `room_geometry_summary.csv`: fixed-radius density/boundary and local PCA summaries.",
                "- `instance_stats.csv`: real instances only; an empty file means IDs were unavailable.",
                "- Geometry queries use the complete room as support even when query points are sampled.",
            ]
        ),
        encoding="utf-8",
    )
    print("Dataset audit written to {}".format(output_dir))


def geometry_attribute_names(radii: Sequence[float]) -> List[str]:
    names = ["linearity", "planarity", "curvature", "anisotropy"]
    for radius in radii:
        suffix = radius_suffix(radius)
        names.extend(
            [
                "density_count_" + suffix,
                "boundary_" + suffix,
                "boundary_fraction_" + suffix,
            ]
        )
    return names


def prepare_profile_artifacts(
    prediction_dir: Path,
    output_dir: Path,
    compute_geometry: bool,
    radii: Sequence[float],
    pca_radius: float,
    chunk_size: int,
    max_pca_neighbors: int,
    geometry_workers: int = 1,
) -> Dict[str, Path]:
    indexed = index_prediction_artifacts(prediction_dir)
    if not compute_geometry:
        return indexed
    required = set(geometry_attribute_names(radii))
    cache_dir = output_dir / "predictions_with_geometry"
    prepared = {}
    pending = []
    for scene_name, source_path in indexed.items():
        artifact = load_prediction_artifact(source_path)
        if not required.difference(artifact["attributes"]):
            prepared[scene_name] = source_path
            continue
        pending.append((scene_name, source_path))

    worker_count = max(1, min(int(geometry_workers), len(pending) or 1))
    tasks = [
        (
            scene_name,
            str(source_path),
            str(cache_dir / safe_scene_filename(scene_name)),
            tuple(radii),
            float(pca_radius),
            int(chunk_size),
            int(max_pca_neighbors),
        )
        for scene_name, source_path in pending
    ]
    if worker_count == 1:
        results = map(compute_and_cache_geometry, tasks)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = executor.map(compute_and_cache_geometry, tasks)
    for scene_name, cached_path in results:
        prepared[scene_name] = Path(cached_path)
    return prepared


def compute_and_cache_geometry(task) -> Tuple[str, str]:
    """Compute one room independently so profile geometry can use multiple CPU cores."""

    (
        scene_name,
        source_path,
        cached_path,
        radii,
        pca_radius,
        chunk_size,
        max_pca_neighbors,
    ) = task
    artifact = load_prediction_artifact(Path(source_path))
    attributes = dict(artifact["attributes"])
    attributes.update(
        fixed_radius_geometry(
            artifact["points"],
            artifact["labels"],
            radii=radii,
            pca_radius=pca_radius,
            chunk_size=chunk_size,
            max_pca_neighbors=max_pca_neighbors,
        )
    )
    save_prediction_artifact(
        Path(cached_path),
        scene_name,
        artifact["points"],
        artifact["labels"],
        artifact["predictions"],
        probabilities=artifact["probabilities"],
        attributes=attributes,
        metadata=artifact["metadata"],
    )
    return scene_name, cached_path


def attribute_stream(indexed: Mapping[str, Path]) -> Iterable[Mapping[str, np.ndarray]]:
    for path in indexed.values():
        yield load_prediction_artifact(path)["attributes"]


def confusion_rows(confusion: np.ndarray) -> List[Dict[str, object]]:
    rows = []
    for target, class_name in enumerate(S3DIS_CLASS_NAMES):
        row = {"target_class": target, "target_name": class_name}
        row.update(
            {
                "pred_{}_{}".format(index, name): int(confusion[target, index])
                for index, name in enumerate(S3DIS_CLASS_NAMES)
            }
        )
        rows.append(row)
    return rows


def check_prediction_coverage(
    artifact: Mapping[str, object], allow_incomplete: bool
) -> str:
    attributes = artifact["attributes"]
    if "prediction_covered" not in attributes:
        return "not_recorded"
    covered = np.asarray(attributes["prediction_covered"]).reshape(-1).astype(bool)
    coverage = float(np.mean(covered)) if covered.size else 1.0
    if coverage < 1.0 and not allow_incomplete:
        raise ValueError(
            "{} has only {:.3f}% prediction coverage; rerun inference or pass "
            "--allow_incomplete_coverage for a deliberately incomplete diagnostic".format(
                artifact["scene_name"], 100.0 * coverage
            )
        )
    return "{:.8f}".format(coverage)


def run_profile(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    indexed = prepare_profile_artifacts(
        Path(args.prediction_dir).expanduser().resolve(),
        output_dir,
        args.compute_geometry,
        args.radii,
        args.pca_radius,
        args.geometry_chunk_size,
        args.max_pca_neighbors,
        getattr(args, "geometry_workers", 1),
    )
    thresholds = estimate_quantile_thresholds(
        attribute_stream(indexed), seed=args.seed
    )
    write_json(output_dir / "difficulty_thresholds.json", thresholds)

    class_count = len(S3DIS_CLASS_NAMES)
    overall_confusion = np.zeros((class_count, class_count), dtype=np.int64)
    subset_confusions: Dict[str, np.ndarray] = {}
    room_rows = []
    total_points = 0
    correct_points = 0
    coverage_states = []
    pca_query_count = 0
    pca_capped_count = 0
    for scene_name, path in indexed.items():
        artifact = load_prediction_artifact(path)
        coverage_states.append(
            check_prediction_coverage(
                artifact, getattr(args, "allow_incomplete_coverage", False)
            )
        )
        labels = artifact["labels"]
        predictions = artifact["predictions"]
        attributes = artifact["attributes"]
        if {
            "pca_neighbor_count_full",
            "pca_neighbor_count_used",
        }.issubset(attributes):
            pca_full = np.asarray(attributes["pca_neighbor_count_full"]).reshape(-1)
            pca_used = np.asarray(attributes["pca_neighbor_count_used"]).reshape(-1)
            pca_query_count += int(pca_full.size)
            pca_capped_count += int(np.sum(pca_full > pca_used))
        masks = difficulty_masks(
            attributes, thresholds, point_count=labels.shape[0]
        )
        room_confusion = confusion_matrix(labels, predictions, class_count)
        overall_confusion += room_confusion
        room_rows.append({"scene_name": scene_name, **metrics_from_confusion(room_confusion)})
        valid = (labels >= 0) & (labels < class_count)
        total_points += int(valid.sum())
        correct_points += int(np.sum(valid & (labels == predictions)))
        for subset_name, mask in masks.items():
            subset_confusions.setdefault(
                subset_name, np.zeros((class_count, class_count), dtype=np.int64)
            )
            subset_confusions[subset_name] += confusion_matrix(
                labels, predictions, class_count, mask=mask
            )

    overall_rows = [{"scope": "Area5", **metrics_from_confusion(overall_confusion)}]
    class_rows = per_class_metrics(overall_confusion, S3DIS_CLASS_NAMES)
    subset_rows = [
        {"subset": name, **metrics_from_confusion(confusion)}
        for name, confusion in sorted(subset_confusions.items())
    ]
    subset_class_rows = []
    for subset_name, subset_confusion in sorted(subset_confusions.items()):
        for row in per_class_metrics(subset_confusion, S3DIS_CLASS_NAMES):
            subset_class_rows.append({"subset": subset_name, **row})
    write_csv(output_dir / "model_overall_metrics.csv", overall_rows)
    write_csv(output_dir / "per_class_metrics.csv", class_rows)
    write_csv(output_dir / "confusion_matrix.csv", confusion_rows(overall_confusion))
    write_csv(output_dir / "subset_metrics.csv", subset_rows)
    write_csv(output_dir / "per_class_subset_metrics.csv", subset_class_rows)
    write_csv(output_dir / "per_room_metrics.csv", room_rows)
    write_json(
        output_dir / "profile_summary.json",
        {
            "prediction_dir": "<PREDICTION_DIR>",
            "scene_count": len(indexed),
            "valid_point_count": total_points,
            "correct_point_count": correct_points,
            "compute_geometry": args.compute_geometry,
            "geometry_workers": getattr(args, "geometry_workers", 1),
            "max_pca_neighbors": args.max_pca_neighbors,
            "pca_query_count": pca_query_count,
            "pca_neighbor_cap_saturation_ratio": (
                float(pca_capped_count / pca_query_count) if pca_query_count else None
            ),
            "geometry_cache_created": (output_dir / "predictions_with_geometry").is_dir(),
            "coverage_states": sorted(set(coverage_states)),
            "warning": "Subset mIoU_present averages only represented classes. Use point counts and confusion together; do not compare tiny subsets as if they were full Area-5 mIoU.",
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Single-model S3DIS error profile",
                "",
                "The overall/per-class/confusion files show where this model fails. `subset_metrics.csv` tests pre-defined difficulties (boundary, density, PCA, grid mixing, or patch cut) without defining difficulty from correctness.",
                "",
                "A weak class score alone does not establish the cause. Match it with dataset support/coverage and the corresponding subset/confusion evidence.",
            ]
        ),
        encoding="utf-8",
    )
    print("Single-model profile written to {}".format(output_dir))


def paired_transition_counts(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    class_count: int,
) -> Dict[str, int]:
    valid = mask & (labels >= 0) & (labels < class_count)
    baseline_correct = baseline == labels
    candidate_correct = candidate == labels
    return {
        "point_count": int(valid.sum()),
        "both_correct": int(np.sum(valid & baseline_correct & candidate_correct)),
        "fixed_by_candidate": int(np.sum(valid & ~baseline_correct & candidate_correct)),
        "regressed_by_candidate": int(np.sum(valid & baseline_correct & ~candidate_correct)),
        "both_wrong": int(np.sum(valid & ~baseline_correct & ~candidate_correct)),
    }


def bootstrap_paired_confusions(
    baseline_room_confusions: Sequence[np.ndarray],
    candidate_room_confusions: Sequence[np.ndarray],
    repeats: int,
    seed: int,
) -> List[Dict[str, object]]:
    if repeats <= 0 or len(baseline_room_confusions) < 2:
        return []
    rng = np.random.default_rng(seed)
    room_count = len(baseline_room_confusions)
    samples = defaultdict(list)
    for _ in range(repeats):
        selected = rng.integers(0, room_count, size=room_count)
        baseline_confusion = np.sum(
            [baseline_room_confusions[index] for index in selected], axis=0
        )
        candidate_confusion = np.sum(
            [candidate_room_confusions[index] for index in selected], axis=0
        )
        baseline_metrics = metrics_from_confusion(baseline_confusion)
        candidate_metrics = metrics_from_confusion(candidate_confusion)
        for metric in ("mIoU_all", "mIoU_present", "mAcc_present", "OA", "error_rate"):
            samples[metric].append(candidate_metrics[metric] - baseline_metrics[metric])
    rows = []
    for metric, values in samples.items():
        values = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "metric": metric,
                "candidate_minus_baseline_mean": float(np.mean(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_repeats": repeats,
                "resampling_unit": "room",
            }
        )
    return rows


def run_compare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    baseline_index = prepare_profile_artifacts(
        Path(args.baseline_predictions).expanduser().resolve(),
        output_dir,
        args.compute_geometry,
        args.radii,
        args.pca_radius,
        args.geometry_chunk_size,
        args.max_pca_neighbors,
        getattr(args, "geometry_workers", 1),
    )
    candidate_index = index_prediction_artifacts(
        Path(args.candidate_predictions).expanduser().resolve()
    )
    if set(baseline_index) != set(candidate_index):
        missing_candidate = sorted(set(baseline_index) - set(candidate_index))
        missing_baseline = sorted(set(candidate_index) - set(baseline_index))
        raise ValueError(
            "Prediction scene sets differ; missing candidate={}, missing baseline={}".format(
                missing_candidate, missing_baseline
            )
        )
    thresholds = estimate_quantile_thresholds(
        attribute_stream(baseline_index), seed=args.seed
    )
    write_json(output_dir / "difficulty_thresholds.json", thresholds)

    class_count = len(S3DIS_CLASS_NAMES)
    baseline_subset_confusions: Dict[str, np.ndarray] = {}
    candidate_subset_confusions: Dict[str, np.ndarray] = {}
    transition_totals: Dict[str, Dict[str, int]] = {}
    room_rows = []
    baseline_room_confusions = []
    candidate_room_confusions = []

    for scene_name in sorted(baseline_index):
        baseline = load_prediction_artifact(baseline_index[scene_name])
        candidate = load_prediction_artifact(candidate_index[scene_name])
        check_prediction_coverage(
            baseline, getattr(args, "allow_incomplete_coverage", False)
        )
        check_prediction_coverage(
            candidate, getattr(args, "allow_incomplete_coverage", False)
        )
        validate_paired_artifacts(baseline, candidate, coordinate_atol=args.coordinate_atol)
        labels = baseline["labels"]
        baseline_predictions = baseline["predictions"]
        candidate_predictions = candidate["predictions"]
        masks = difficulty_masks(
            baseline["attributes"], thresholds, point_count=labels.shape[0]
        )
        baseline_room = confusion_matrix(labels, baseline_predictions, class_count)
        candidate_room = confusion_matrix(labels, candidate_predictions, class_count)
        baseline_room_confusions.append(baseline_room)
        candidate_room_confusions.append(candidate_room)
        baseline_metrics = metrics_from_confusion(baseline_room)
        candidate_metrics = metrics_from_confusion(candidate_room)
        room_rows.append(
            {
                "scene_name": scene_name,
                **{"baseline_" + key: value for key, value in baseline_metrics.items()},
                **{"candidate_" + key: value for key, value in candidate_metrics.items()},
                "delta_mIoU_all": candidate_metrics["mIoU_all"]
                - baseline_metrics["mIoU_all"],
                "delta_OA": candidate_metrics["OA"] - baseline_metrics["OA"],
                "error_rate_reduction": baseline_metrics["error_rate"]
                - candidate_metrics["error_rate"],
            }
        )
        for subset_name, mask in masks.items():
            baseline_subset_confusions.setdefault(
                subset_name, np.zeros((class_count, class_count), dtype=np.int64)
            )
            candidate_subset_confusions.setdefault(
                subset_name, np.zeros((class_count, class_count), dtype=np.int64)
            )
            baseline_subset_confusions[subset_name] += confusion_matrix(
                labels, baseline_predictions, class_count, mask=mask
            )
            candidate_subset_confusions[subset_name] += confusion_matrix(
                labels, candidate_predictions, class_count, mask=mask
            )
            transitions = paired_transition_counts(
                labels, baseline_predictions, candidate_predictions, mask, class_count
            )
            total = transition_totals.setdefault(
                subset_name,
                {
                    "point_count": 0,
                    "both_correct": 0,
                    "fixed_by_candidate": 0,
                    "regressed_by_candidate": 0,
                    "both_wrong": 0,
                },
            )
            for key, value in transitions.items():
                total[key] += value

    subset_rows = []
    for subset_name in sorted(baseline_subset_confusions):
        baseline_metrics = metrics_from_confusion(baseline_subset_confusions[subset_name])
        candidate_metrics = metrics_from_confusion(candidate_subset_confusions[subset_name])
        subset_rows.append(
            {
                "subset": subset_name,
                **{"baseline_" + key: value for key, value in baseline_metrics.items()},
                **{"candidate_" + key: value for key, value in candidate_metrics.items()},
                "delta_mIoU_all": candidate_metrics["mIoU_all"]
                - baseline_metrics["mIoU_all"],
                "delta_mIoU_present": candidate_metrics["mIoU_present"]
                - baseline_metrics["mIoU_present"],
                "delta_mAcc_present": candidate_metrics["mAcc_present"]
                - baseline_metrics["mAcc_present"],
                "delta_OA": candidate_metrics["OA"] - baseline_metrics["OA"],
                "error_rate_reduction": baseline_metrics["error_rate"]
                - candidate_metrics["error_rate"],
            }
        )

    all_baseline = baseline_subset_confusions["all"]
    all_candidate = candidate_subset_confusions["all"]
    baseline_classes = per_class_metrics(all_baseline, S3DIS_CLASS_NAMES)
    candidate_classes = per_class_metrics(all_candidate, S3DIS_CLASS_NAMES)
    class_rows = []
    for baseline_row, candidate_row in zip(baseline_classes, candidate_classes):
        class_rows.append(
            {
                "class_index": baseline_row["class_index"],
                "class_name": baseline_row["class_name"],
                "support": baseline_row["support"],
                "baseline_IoU": baseline_row["IoU"],
                "candidate_IoU": candidate_row["IoU"],
                "delta_IoU": candidate_row["IoU"] - baseline_row["IoU"],
                "baseline_recall": baseline_row["recall"],
                "candidate_recall": candidate_row["recall"],
                "delta_recall": candidate_row["recall"] - baseline_row["recall"],
                "baseline_precision": baseline_row["precision"],
                "candidate_precision": candidate_row["precision"],
                "delta_precision": candidate_row["precision"] - baseline_row["precision"],
            }
        )

    subset_class_rows = []
    for subset_name in sorted(baseline_subset_confusions):
        baseline_classes = per_class_metrics(
            baseline_subset_confusions[subset_name], S3DIS_CLASS_NAMES
        )
        candidate_classes = per_class_metrics(
            candidate_subset_confusions[subset_name], S3DIS_CLASS_NAMES
        )
        for baseline_row, candidate_row in zip(baseline_classes, candidate_classes):
            subset_class_rows.append(
                {
                    "subset": subset_name,
                    "class_index": baseline_row["class_index"],
                    "class_name": baseline_row["class_name"],
                    "support": baseline_row["support"],
                    "baseline_IoU": baseline_row["IoU"],
                    "candidate_IoU": candidate_row["IoU"],
                    "delta_IoU": candidate_row["IoU"] - baseline_row["IoU"],
                    "baseline_recall": baseline_row["recall"],
                    "candidate_recall": candidate_row["recall"],
                    "delta_recall": candidate_row["recall"] - baseline_row["recall"],
                }
            )

    transition_rows = []
    for subset_name, totals in sorted(transition_totals.items()):
        points = max(totals["point_count"], 1)
        transition_rows.append(
            {
                "subset": subset_name,
                **totals,
                "fixed_rate": 100.0 * totals["fixed_by_candidate"] / points,
                "regression_rate": 100.0 * totals["regressed_by_candidate"] / points,
                "net_fixed_points": totals["fixed_by_candidate"]
                - totals["regressed_by_candidate"],
            }
        )

    baseline_metrics = metrics_from_confusion(all_baseline)
    candidate_metrics = metrics_from_confusion(all_candidate)
    paired_summary = [
        {
            "baseline_name": args.baseline_name,
            "candidate_name": args.candidate_name,
            "scene_count": len(baseline_index),
            **{"baseline_" + key: value for key, value in baseline_metrics.items()},
            **{"candidate_" + key: value for key, value in candidate_metrics.items()},
            "delta_mIoU_all": candidate_metrics["mIoU_all"] - baseline_metrics["mIoU_all"],
            "delta_OA": candidate_metrics["OA"] - baseline_metrics["OA"],
            "error_rate_reduction": baseline_metrics["error_rate"]
            - candidate_metrics["error_rate"],
        }
    ]
    bootstrap_rows = bootstrap_paired_confusions(
        baseline_room_confusions,
        candidate_room_confusions,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    write_csv(output_dir / "paired_summary.csv", paired_summary)
    write_csv(output_dir / "gain_by_class.csv", class_rows)
    write_csv(output_dir / "gain_by_subset.csv", subset_rows)
    write_csv(output_dir / "gain_by_class_subset.csv", subset_class_rows)
    write_csv(output_dir / "gain_by_room.csv", room_rows)
    write_csv(output_dir / "transitions_by_subset.csv", transition_rows)
    write_csv(output_dir / "paired_room_bootstrap.csv", bootstrap_rows)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Paired S3DIS comparison",
                "",
                "Both models were checked against identical scene names, GT labels, point counts, and coordinates before comparison.",
                "",
                "- `gain_by_subset.csv` tests whether gains concentrate on a pre-labelled difficulty.",
                "- `transitions_by_subset.csv` distinguishes fixed errors from new regressions.",
                "- `gain_by_room.csv` prevents one large room from hiding room-to-room instability.",
                "- `paired_room_bootstrap.csv` gives a room-resampled 95% interval; it is empty when fewer than two rooms are available.",
                "",
                "Do not claim that the new module solves a difficulty unless the paired gain, fixed/regressed counts, and multiple-room evidence agree.",
            ]
        ),
        encoding="utf-8",
    )
    print("Paired comparison written to {}".format(output_dir))


def packed_slices(lengths) -> List[slice]:
    lengths = np.asarray(lengths).reshape(-1).astype(np.int64, copy=False)
    starts = np.concatenate(([0], np.cumsum(lengths)))
    return [slice(int(starts[index]), int(starts[index + 1])) for index in range(lengths.size)]


def patch_ids_from_layout(indices, valid, point_count: int) -> np.ndarray:
    indices = np.asarray(indices)
    valid = np.asarray(valid, dtype=bool)
    patch_ids = np.full(point_count, -1, dtype=np.int64)
    rows = np.broadcast_to(np.arange(indices.shape[0])[:, None], indices.shape)
    patch_ids[indices[valid]] = rows[valid]
    if np.any(patch_ids < 0):
        raise ValueError("serialized patch layout did not cover every point")
    return patch_ids


def hierarchy_attributes_for_batch(batch, trace, cfg) -> List[Dict[str, np.ndarray]]:
    """Map actual grid/patch diagnostics from every stage back to stage 0."""

    import torch
    from models.litept_blocks import build_serialized_patches, parse_serialization_orders

    stage_points = [tensor.detach().cpu().numpy() for tensor in trace["points"]]
    stage_lengths = [tensor.detach().cpu().numpy() for tensor in trace["lengths"]]
    stage_slices = [packed_slices(lengths) for lengths in stage_lengths]
    upsample_maps = [tensor.detach().cpu().numpy() for tensor in trace["upsamples"]]
    labels = trace["labels"].detach().cpu().numpy().astype(np.int64)
    cloud_count = len(stage_slices[0])
    per_cloud: List[Dict[str, np.ndarray]] = []

    patch_stage_attributes: Dict[int, Dict[str, np.ndarray]] = {}
    if bool(getattr(cfg.model, "litept_enabled", False)):
        orders = parse_serialization_orders(getattr(cfg.model, "litept_orders", "z,z-trans"))
        conv_stages = int(getattr(cfg.model, "litept_conv_stages", 3))
        handover_stage = int(getattr(cfg.model, "litept_handover_stage", 0))
        for stage in range(len(stage_points)):
            one_based = stage + 1
            if not (one_based > conv_stages or one_based == handover_stage):
                continue
            points_tensor = trace["points"][stage]
            lengths_tensor = trace["lengths"][stage]
            voxel_size = max(
                float(cfg.model.in_sub_size) * float(cfg.model.radius_scaling) ** stage,
                1e-6,
            )
            patch_ids_by_order = {}
            for order in orders:
                indices, valid, _ = build_serialized_patches(
                    points_tensor,
                    lengths_tensor,
                    patch_size=int(cfg.model.litept_patch_size),
                    voxel_size=voxel_size,
                    order=order,
                )
                patch_ids_by_order[order] = patch_ids_from_layout(
                    indices.detach().cpu().numpy(),
                    valid.detach().cpu().numpy(),
                    stage_points[stage].shape[0],
                )
            neighbors = batch.in_dict.neighbors[stage].detach().cpu().numpy()
            patch_stage_attributes[stage] = patch_neighbor_recall(
                neighbors,
                patch_ids_by_order,
                shadow_index=stage_points[stage].shape[0],
            )

    for cloud_index in range(cloud_count):
        cloud_stage_sizes = [
            stage_slice[cloud_index].stop - stage_slice[cloud_index].start
            for stage_slice in stage_slices
        ]
        cloud_upsamples = []
        for stage, mapping in enumerate(upsample_maps):
            fine_slice = stage_slices[stage][cloud_index]
            coarse_slice = stage_slices[stage + 1][cloud_index]
            cloud_mapping = mapping[fine_slice].reshape(-1).astype(np.int64)
            cloud_upsamples.append(cloud_mapping - coarse_slice.start)
        ancestors = compose_ancestor_maps(cloud_upsamples, cloud_stage_sizes)
        stage0_slice = stage_slices[0][cloud_index]
        cloud_labels = labels[stage0_slice]
        attributes: Dict[str, np.ndarray] = {}
        for stage, (ancestor, stage_size) in enumerate(zip(ancestors, cloud_stage_sizes)):
            stats = stage_cell_statistics(
                cloud_labels, ancestor, stage_size, num_classes=len(S3DIS_CLASS_NAMES)
            )
            attributes["grid_stage{}_cell_occupancy".format(stage)] = stats["occupancy"][ancestor]
            attributes["grid_stage{}_cell_entropy".format(stage)] = stats["label_entropy"][ancestor]
            attributes["grid_stage{}_mixed_cell".format(stage)] = stats["mixed"][ancestor]
            # Internal categorical assignment used by ``run_infer`` to recompute
            # the same cells with full-resolution GT after test projection.
            attributes["__grid_stage{}_cell_local_id".format(stage)] = ancestor
            if stage in patch_stage_attributes:
                stage_slice = stage_slices[stage][cloud_index]
                for name, values in patch_stage_attributes[stage].items():
                    local_values = values[stage_slice]
                    attributes["stage{}_{}".format(stage, name)] = local_values[ancestor]
        per_cloud.append(attributes)
    return per_cloud


def resolve_checkpoint(log_path: Path, explicit: Optional[str]) -> Path:
    if explicit:
        checkpoint = Path(explicit).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    checkpoint_dir = log_path / "checkpoints"
    for name in ("best_val_miou.tar", "best_mIoU_chkp.tar", "current_chkp.tar"):
        checkpoint = checkpoint_dir / name
        if checkpoint.is_file():
            return checkpoint
    periodic = sorted(checkpoint_dir.glob("chkp_*.tar"))
    if periodic:
        return periodic[-1]
    raise FileNotFoundError("No checkpoint found under {}".format(checkpoint_dir))


def deterministic_inference_cfg(log_path: Path, dataset_path: Path, in_radius: float):
    from utils.config import load_cfg

    cfg = load_cfg(str(log_path))
    cfg.data.path = str(dataset_path)
    cfg.test.data_sampler = "regular"
    cfg.test.in_radius = float(in_radius)
    cfg.test.batch_limit = 1
    cfg.test.batch_size = 1
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


def infer_checkpoint_kp_mode(state_dict: Dict[str, object]) -> str:
    """Infer the KP operator layout encoded by a KPNeXt checkpoint."""

    modulation_keys = [
        key
        for key in state_dict
        if ".conv.alpha_mlp." in key or ".conv.grpnorm." in key
    ]
    if any(key.startswith("encoder_") or key.startswith("pooling_") for key in modulation_keys):
        return "kpconvx"
    if any(key.startswith("decoder_layer_") for key in modulation_keys):
        return "legacy_kpconvd_encoder_kpconvx_decoder"
    return "kpconvd"


def resolve_device(device_name: str):
    import torch

    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()):
        from utils.gpu_init import init_gpu

        return init_gpu()
    return torch.device("cpu")


def parse_gpu_compute_processes(output: str) -> List[Dict[str, object]]:
    processes = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        memory_match = re.search(r"(\d+)", fields[2])
        processes.append(
            {
                "pid": pid,
                "process_name": fields[1],
                "used_memory_mib": int(memory_match.group(1)) if memory_match else None,
            }
        )
    return processes


def refuse_gpu_contention_unless_allowed(allow_gpu_contention: bool) -> None:
    """Protect an active training job from accidental diagnostic inference."""

    if allow_gpu_contention:
        return
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return
    other_processes = [
        process
        for process in parse_gpu_compute_processes(result.stdout)
        if process["pid"] != os.getpid()
    ]
    if other_processes:
        summary = ", ".join(
            "pid={pid} memory={used_memory_mib}MiB".format(**process)
            for process in other_processes
        )
        raise RuntimeError(
            "GPU already has compute processes ({}). Refusing diagnostic inference "
            "to protect active training. Wait for training to finish; use "
            "--allow_gpu_contention only after explicitly accepting the risk.".format(summary)
        )


def run_infer(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    from data_handlers.scene_seg import SceneSegCollate, SceneSegSampler
    from experiments.S3DIS.S3DIS_rooms import S3DIRDataset
    from models.KPNext import KPNeXt

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = Path(args.log_path).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    checkpoint_path = resolve_checkpoint(log_path, args.checkpoint)
    cfg = deterministic_inference_cfg(log_path, dataset_path, args.in_radius)
    if cfg.model.kp_mode not in {"kpconvx", "kpconvd"}:
        raise ValueError("infer currently supports KPNeXt kpconvx/kpconvd logs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    recorded_kp_mode = cfg.model.kp_mode
    effective_kp_mode = infer_checkpoint_kp_mode(model_state_dict)
    if effective_kp_mode == "legacy_kpconvd_encoder_kpconvx_decoder":
        cfg.model.litept_legacy_kpconvd_encoder = True
        expected_recorded_mode = "kpconvx"
    else:
        cfg.model.kp_mode = effective_kp_mode
        expected_recorded_mode = effective_kp_mode
    kp_mode_mismatch = recorded_kp_mode != expected_recorded_mode or (
        effective_kp_mode == "legacy_kpconvd_encoder_kpconvx_decoder"
    )
    if kp_mode_mismatch:
        print(
            "Warning: recorded kp_mode={} conflicts with checkpoint structure; "
            "using {} for inference.".format(recorded_kp_mode, effective_kp_mode)
        )
    if args.device != "cpu":
        refuse_gpu_contention_unless_allowed(args.allow_gpu_contention)
    device = resolve_device(args.device)
    dataset = S3DIRDataset(cfg, chosen_set="validation", precompute_pyramid=True)
    class_count = len(S3DIS_CLASS_NAMES)
    model = KPNeXt(cfg)
    model.load_state_dict(model_state_dict, strict=True)
    if int(model.num_logits) != class_count:
        raise ValueError(
            "checkpoint predicts {} classes, expected {} for S3DIS".format(
                model.num_logits, class_count
            )
        )
    checkpoint_epoch = checkpoint.get("epoch", -1)
    if hasattr(checkpoint_epoch, "item"):
        checkpoint_epoch = checkpoint_epoch.item()
    del checkpoint
    model.to(device)
    model.eval()

    probability_sums = [
        np.zeros((labels.shape[0], class_count), dtype=np.float64)
        for labels in dataset.input_labels
    ]
    visit_counts = [np.zeros(labels.shape[0], dtype=np.int32) for labels in dataset.input_labels]
    hierarchy_sums: List[Dict[str, np.ndarray]] = [dict() for _ in dataset.input_labels]
    hierarchy_counts: List[Dict[str, np.ndarray]] = [dict() for _ in dataset.input_labels]
    hierarchy_cell_ids: List[Dict[int, np.ndarray]] = [dict() for _ in dataset.input_labels]
    hierarchy_cell_offsets: List[Dict[int, int]] = [dict() for _ in dataset.input_labels]
    softmax = torch.nn.Softmax(dim=1)

    for vote in range(args.votes):
        if vote > 0:
            dataset.reg_sampling_i.zero_()
            dataset.new_reg_sampling_pts()
            dataset.reg_votes += 1
        sampler = SceneSegSampler(dataset)
        # Process every regular centre exactly once.  The generic validation
        # sampler intentionally uses only a fraction per epoch for training-time
        # validation, which is not sufficient for a diagnostic export.
        sampler.N = dataset.get_reg_sampling_size()
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=SceneSegCollate,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        print("Inference vote {}/{} ({} regular centres)".format(vote + 1, args.votes, sampler.N))
        with torch.no_grad():
            for step, batch in enumerate(loader):
                if len(batch.in_dict.points) < 1:
                    break
                if device.type == "cuda":
                    batch.to(device)
                capture = bool(args.capture_hierarchy and vote == 0)
                if capture:
                    logits, trace = model(batch, return_intermediates=True)
                    hierarchy = hierarchy_attributes_for_batch(batch, trace, cfg)
                else:
                    logits = model(batch)
                    hierarchy = None
                probabilities = softmax(logits).detach().cpu().numpy()
                lengths = batch.in_dict.lengths[0].detach().cpu().numpy()
                lengths0 = batch.in_dict.lengths0.detach().cpu().numpy()
                input_inds = batch.in_dict.input_inds.detach().cpu().numpy()
                input_invs = batch.in_dict.input_invs.detach().cpu().numpy()
                cloud_inds = batch.in_dict.cloud_inds.detach().cpu().numpy()
                stage_start = 0
                input_start = 0
                for batch_index, (length, length0, cloud_index) in enumerate(
                    zip(lengths, lengths0, cloud_inds)
                ):
                    length = int(length)
                    length0 = int(length0)
                    cloud_index = int(cloud_index)
                    region_probabilities = probabilities[stage_start : stage_start + length]
                    indices = input_inds[input_start : input_start + length0]
                    inverse = input_invs[input_start : input_start + length0]
                    lifted_probabilities = region_probabilities[inverse]
                    probability_sums[cloud_index][indices] += lifted_probabilities
                    visit_counts[cloud_index][indices] += 1
                    if hierarchy is not None:
                        for name, stage0_values in hierarchy[batch_index].items():
                            lifted_values = np.asarray(stage0_values)[inverse]
                            cell_match = re.fullmatch(
                                r"__grid_stage(\d+)_cell_local_id", name
                            )
                            if cell_match:
                                stage = int(cell_match.group(1))
                                if stage not in hierarchy_cell_ids[cloud_index]:
                                    hierarchy_cell_ids[cloud_index][stage] = np.full(
                                        dataset.input_labels[cloud_index].shape[0],
                                        -1,
                                        dtype=np.int64,
                                    )
                                    hierarchy_cell_offsets[cloud_index][stage] = 0
                                offset = hierarchy_cell_offsets[cloud_index][stage]
                                lifted_ids = lifted_values.astype(np.int64) + offset
                                target_ids = hierarchy_cell_ids[cloud_index][stage]
                                first_assignment = target_ids[indices] < 0
                                target_ids[indices[first_assignment]] = lifted_ids[first_assignment]
                                hierarchy_cell_offsets[cloud_index][stage] = offset + int(
                                    np.max(stage0_values)
                                ) + 1
                                continue
                            finite = np.isfinite(lifted_values)
                            if name not in hierarchy_sums[cloud_index]:
                                hierarchy_sums[cloud_index][name] = np.zeros(
                                    dataset.input_labels[cloud_index].shape[0], dtype=np.float64
                                )
                                hierarchy_counts[cloud_index][name] = np.zeros(
                                    dataset.input_labels[cloud_index].shape[0], dtype=np.int32
                                )
                            np.add.at(
                                hierarchy_sums[cloud_index][name],
                                indices[finite],
                                lifted_values[finite],
                            )
                            np.add.at(
                                hierarchy_counts[cloud_index][name], indices[finite], 1
                            )
                    stage_start += length
                    input_start += length0
                if (step + 1) % 10 == 0:
                    print("  processed {} / {} centres".format(step + 1, sampler.N))

    prediction_dir = output_dir / "predictions"
    coverage_rows = []
    for cloud_index, scene_name in enumerate(dataset.scene_names):
        full_points, _, loaded_labels = dataset.load_scene_file(dataset.scene_files[cloud_index])
        if cloud_index < len(dataset.val_labels):
            labels = np.asarray(dataset.val_labels[cloud_index]).reshape(-1).astype(np.int64)
        elif loaded_labels is not None:
            labels = np.asarray(loaded_labels).reshape(-1).astype(np.int64)
        else:
            raise ValueError("full-resolution GT is unavailable for {}".format(scene_name))
        if loaded_labels is not None and not np.array_equal(
            labels, np.asarray(loaded_labels).reshape(-1)
        ):
            raise ValueError("full-resolution GT alignment failed for {}".format(scene_name))
        if cloud_index < len(dataset.test_proj):
            projection = np.asarray(dataset.test_proj[cloud_index]).reshape(-1).astype(np.int64)
        else:
            projection = np.arange(labels.shape[0], dtype=np.int64)
            if projection.shape[0] != dataset.input_labels[cloud_index].shape[0]:
                raise ValueError("identity projection is invalid for {}".format(scene_name))
        counts = visit_counts[cloud_index]
        averaged = np.divide(
            probability_sums[cloud_index],
            counts[:, None],
            out=np.zeros_like(probability_sums[cloud_index]),
            where=counts[:, None] > 0,
        )
        sub_predictions = dataset.probs_to_preds(averaged)
        full_predictions = sub_predictions[projection]
        attributes: Dict[str, np.ndarray] = {
            "prediction_covered": (counts[projection] > 0),
            "prediction_visit_count": counts[projection],
        }
        for name, sums in hierarchy_sums[cloud_index].items():
            attr_counts = hierarchy_counts[cloud_index][name]
            sub_values = np.divide(
                sums,
                attr_counts,
                out=np.full_like(sums, np.nan),
                where=attr_counts > 0,
            )
            attributes[name] = sub_values[projection].astype(np.float32)
        # Recompute semantic mixing with full-resolution GT rather than calling
        # the stage-0-label summary an original-point statistic.  Cell IDs are
        # produced by the actual model pooling hierarchy above.
        for stage, sub_cell_ids in hierarchy_cell_ids[cloud_index].items():
            full_cell_ids = sub_cell_ids[projection]
            valid_cells = (full_cell_ids >= 0) & (labels >= 0) & (labels < class_count)
            if not np.any(valid_cells):
                continue
            _, compact_ids = np.unique(full_cell_ids[valid_cells], return_inverse=True)
            full_stats = stage_cell_statistics(
                labels[valid_cells],
                compact_ids,
                int(compact_ids.max()) + 1,
                num_classes=class_count,
            )
            for statistic, values in (
                ("occupancy", full_stats["occupancy"]),
                ("entropy", full_stats["label_entropy"]),
                ("mixed_cell", full_stats["mixed"]),
            ):
                full_values = np.full(labels.shape[0], np.nan, dtype=np.float32)
                full_values[valid_cells] = values[compact_ids]
                attributes[
                    "grid_stage{}_full_cell_{}".format(stage, statistic)
                ] = full_values
        if args.compute_geometry:
            attributes.update(
                fixed_radius_geometry(
                    full_points,
                    labels,
                    radii=args.radii,
                    pca_radius=args.pca_radius,
                    chunk_size=args.geometry_chunk_size,
                    max_pca_neighbors=args.max_pca_neighbors,
                )
            )
        full_probabilities = averaged[projection].astype(np.float16) if args.save_probabilities else None
        save_prediction_artifact(
            prediction_dir / safe_scene_filename(scene_name),
            scene_name,
            full_points,
            labels,
            full_predictions,
            probabilities=full_probabilities,
            attributes=attributes,
            metadata={
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_epoch": int(checkpoint_epoch),
                "votes": args.votes,
                "in_radius_m": args.in_radius,
            },
        )
        coverage_rows.append(
            {
                "scene_name": scene_name,
                "subsampled_point_count": int(counts.size),
                "subsampled_covered_count": int(np.sum(counts > 0)),
                "subsampled_coverage": float(np.mean(counts > 0)),
                "full_point_count": int(labels.size),
                "full_coverage": float(np.mean(counts[projection] > 0)),
            }
        )
    write_csv(output_dir / "coverage.csv", coverage_rows)
    write_json(
        output_dir / "inference_config.json",
        {
            "log_dir_name": log_path.name,
            "checkpoint_name": checkpoint_path.name,
            "dataset_path": "<DATASET_PATH>",
            "votes": args.votes,
            "device": str(device),
            "in_radius_m": args.in_radius,
            "capture_hierarchy": args.capture_hierarchy,
            "compute_geometry": args.compute_geometry,
            "recorded_kp_mode": recorded_kp_mode,
            "effective_kp_mode": effective_kp_mode,
            "kp_mode_mismatch": kp_mode_mismatch,
            "note": "Prediction artifacts are full-resolution via dataset.test_proj. Check coverage.csv before profiling.",
        },
    )
    print("Full-resolution prediction artifacts written to {}".format(prediction_dir))


def add_geometry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--radii", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--pca_radius", type=float, default=0.10)
    parser.add_argument("--geometry_chunk_size", type=int, default=4096)
    parser.add_argument(
        "--max_pca_neighbors",
        type=int,
        default=0,
        help="PCA neighbour cap; 0 keeps every point inside pca_radius",
    )
    parser.add_argument(
        "--geometry_workers",
        type=int,
        default=1,
        help="independent room workers for CPU geometry computation",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evidence-first S3DIS dataset/model/paired diagnostics"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset", help="Layer 1: audit S3DIS itself")
    dataset.add_argument("--dataset_path", required=True)
    dataset.add_argument("--output_dir", required=True)
    dataset.add_argument(
        "--geometry_split", choices=["none", "validation", "all"], default="validation"
    )
    dataset.add_argument("--geometry_max_points_per_room", type=int, default=100000)
    dataset.add_argument("--seed", type=int, default=57106803)
    add_geometry_arguments(dataset)
    dataset.set_defaults(func=run_dataset_audit)

    profile = subparsers.add_parser("profile", help="Layer 2: profile one model")
    profile.add_argument("--prediction_dir", required=True)
    profile.add_argument("--output_dir", required=True)
    profile.add_argument(
        "--compute_geometry", action=argparse.BooleanOptionalAction, default=True
    )
    profile.add_argument("--allow_incomplete_coverage", action="store_true")
    profile.add_argument("--seed", type=int, default=57106803)
    add_geometry_arguments(profile)
    profile.set_defaults(func=run_profile)

    compare = subparsers.add_parser("compare", help="Layer 3: paired model comparison")
    compare.add_argument("--baseline_predictions", required=True)
    compare.add_argument("--candidate_predictions", required=True)
    compare.add_argument("--output_dir", required=True)
    compare.add_argument("--baseline_name", default="baseline")
    compare.add_argument("--candidate_name", default="candidate")
    compare.add_argument(
        "--compute_geometry", action=argparse.BooleanOptionalAction, default=True
    )
    compare.add_argument("--allow_incomplete_coverage", action="store_true")
    compare.add_argument("--coordinate_atol", type=float, default=1e-5)
    compare.add_argument("--bootstrap_repeats", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=57106803)
    add_geometry_arguments(compare)
    compare.set_defaults(func=run_compare)

    infer = subparsers.add_parser("infer", help="Export full Area-5 predictions")
    infer.add_argument("--log_path", required=True)
    infer.add_argument("--checkpoint")
    infer.add_argument("--dataset_path", required=True)
    infer.add_argument("--output_dir", required=True)
    infer.add_argument("--votes", type=int, default=1)
    infer.add_argument("--in_radius", type=float, default=100.0)
    infer.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    infer.add_argument(
        "--allow_gpu_contention",
        action="store_true",
        help="override the default refusal when another GPU compute process is active",
    )
    infer.add_argument("--save_probabilities", action="store_true")
    infer.add_argument("--capture_hierarchy", action="store_true")
    infer.add_argument("--compute_geometry", action="store_true")
    add_geometry_arguments(infer)
    infer.set_defaults(func=run_infer)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "votes") and args.votes < 1:
        parser.error("--votes must be positive")
    if hasattr(args, "geometry_workers") and args.geometry_workers < 1:
        parser.error("--geometry_workers must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
