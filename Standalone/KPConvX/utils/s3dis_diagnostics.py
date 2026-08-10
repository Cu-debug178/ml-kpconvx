"""Reusable S3DIS dataset and prediction diagnostics.

The module deliberately separates three questions:

1. what difficulties exist in the dataset;
2. where one model makes errors;
3. which errors a candidate fixes relative to the same baseline points.

Large point-wise artifacts are meant to stay under ``results/``.  The helpers
only depend on NumPy and scikit-learn so prediction profiling does not require a
checkpoint or a GPU.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import KDTree


SCHEMA_VERSION = 1
S3DIS_CLASS_NAMES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "chair",
    "table",
    "bookcase",
    "sofa",
    "board",
    "clutter",
)


def radius_suffix(radius: float) -> str:
    """Return a stable, filename-safe physical-radius suffix."""

    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be positive")
    return "r{:04d}mm".format(int(round(1000.0 * radius)))


def validate_points_labels(points, labels) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points)
    labels = np.asarray(labels).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if points.shape[0] != labels.shape[0]:
        raise ValueError("points and labels must contain the same number of rows")
    if not np.all(np.isfinite(points)):
        raise ValueError("points contain non-finite coordinates")
    return points.astype(np.float32, copy=False), labels.astype(np.int64, copy=False)


def fixed_radius_geometry(
    points,
    labels,
    radii: Sequence[float] = (0.05, 0.10, 0.20),
    pca_radius: float = 0.10,
    query_indices: Optional[np.ndarray] = None,
    chunk_size: int = 4096,
    max_pca_neighbors: int = 0,
) -> Dict[str, np.ndarray]:
    """Measure density, semantic boundary and local PCA at physical radii.

    Density is the exact number of *other* input points within each radius; it
    is not a fixed-k KNN count.  Boundary membership is true when at least one
    valid neighbour within the radius has a different semantic label.  PCA may
    cap neighbours for runtime, and exposes both the full and used counts.

    ``query_indices`` permits a representative room sample to be analysed
    against the complete room without corrupting its neighbourhood density.
    """

    points, labels = validate_points_labels(points, labels)
    radii = tuple(sorted({float(radius) for radius in radii}))
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("radii must contain positive values")
    if pca_radius <= 0:
        raise ValueError("pca_radius must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if 0 < max_pca_neighbors < 3:
        raise ValueError("max_pca_neighbors must be 0 (unlimited) or at least 3")

    if query_indices is None:
        query_indices = np.arange(points.shape[0], dtype=np.int64)
    else:
        query_indices = np.asarray(query_indices).reshape(-1).astype(np.int64, copy=False)
        if query_indices.size and (
            query_indices.min() < 0 or query_indices.max() >= points.shape[0]
        ):
            raise IndexError("query_indices contains an out-of-range point index")

    query_count = query_indices.shape[0]
    attributes: Dict[str, np.ndarray] = {}
    for radius in radii:
        suffix = radius_suffix(radius)
        attributes["density_count_" + suffix] = np.zeros(query_count, dtype=np.int32)
        attributes["boundary_" + suffix] = np.zeros(query_count, dtype=bool)
        attributes["boundary_fraction_" + suffix] = np.full(
            query_count, np.nan, dtype=np.float32
        )

    attributes["pca_neighbor_count_full"] = np.zeros(query_count, dtype=np.int32)
    attributes["pca_neighbor_count_used"] = np.zeros(query_count, dtype=np.int32)
    for name in ("linearity", "planarity", "curvature", "anisotropy"):
        attributes[name] = np.full(query_count, np.nan, dtype=np.float32)

    if points.shape[0] == 0 or query_count == 0:
        return attributes

    tree = KDTree(points, leaf_size=32)
    search_radius = max(max(radii), float(pca_radius))
    eps = 1e-7
    for start in range(0, query_count, chunk_size):
        stop = min(start + chunk_size, query_count)
        source_indices = query_indices[start:stop]
        neighbor_rows, distance_rows = tree.query_radius(
            points[source_indices],
            r=search_radius + eps,
            return_distance=True,
            sort_results=True,
        )
        for local_i, (source_i, neighbors, distances) in enumerate(
            zip(source_indices, neighbor_rows, distance_rows)
        ):
            out_i = start + local_i
            # Remove only the query point itself.  Coincident but distinct
            # points remain legitimate neighbours.
            keep_other = neighbors != source_i
            neighbors = neighbors[keep_other]
            distances = distances[keep_other]

            for radius in radii:
                suffix = radius_suffix(radius)
                within = distances <= radius + eps
                radius_neighbors = neighbors[within]
                attributes["density_count_" + suffix][out_i] = radius_neighbors.size
                valid = labels[radius_neighbors] >= 0
                if np.any(valid) and labels[source_i] >= 0:
                    different = labels[radius_neighbors[valid]] != labels[source_i]
                    attributes["boundary_" + suffix][out_i] = bool(np.any(different))
                    attributes["boundary_fraction_" + suffix][out_i] = float(
                        np.mean(different)
                    )

            pca_mask = distances <= pca_radius + eps
            pca_neighbors = neighbors[pca_mask]
            attributes["pca_neighbor_count_full"][out_i] = pca_neighbors.size
            if max_pca_neighbors > 0 and pca_neighbors.size > max_pca_neighbors:
                pca_neighbors = pca_neighbors[:max_pca_neighbors]
            attributes["pca_neighbor_count_used"][out_i] = pca_neighbors.size
            # Include the centre; at least three neighbours plus the centre
            # makes the covariance less degenerate for local surface analysis.
            if pca_neighbors.size < 3:
                continue
            local_points = np.vstack((points[source_i], points[pca_neighbors])).astype(
                np.float64, copy=False
            )
            centered = local_points - local_points.mean(axis=0, keepdims=True)
            covariance = centered.T @ centered / max(local_points.shape[0] - 1, 1)
            eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0.0)
            lambda_1, lambda_2, lambda_3 = eigenvalues
            if lambda_1 <= 1e-15:
                continue
            attributes["linearity"][out_i] = (lambda_1 - lambda_2) / lambda_1
            attributes["planarity"][out_i] = (lambda_2 - lambda_3) / lambda_1
            attributes["anisotropy"][out_i] = (lambda_1 - lambda_3) / lambda_1
            attributes["curvature"][out_i] = lambda_3 / max(eigenvalues.sum(), 1e-15)

    return attributes


def confusion_matrix(labels, predictions, num_classes: int, mask=None) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
    predictions = np.asarray(predictions).reshape(-1).astype(np.int64, copy=False)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have equal shape")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    valid = (labels >= 0) & (labels < num_classes)
    if mask is not None:
        mask = np.asarray(mask).reshape(-1).astype(bool, copy=False)
        if mask.shape != labels.shape:
            raise ValueError("mask must have the same shape as labels")
        valid &= mask
    invalid_prediction = valid & ((predictions < 0) | (predictions >= num_classes))
    if np.any(invalid_prediction):
        bad = np.unique(predictions[invalid_prediction])
        raise ValueError("predictions contain invalid class indices: {}".format(bad.tolist()))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    if np.any(valid):
        np.add.at(confusion, (labels[valid], predictions[valid]), 1)
    return confusion


def metrics_from_confusion(confusion) -> Dict[str, float]:
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = support + predicted - true_positive
    iou = np.divide(true_positive, union, out=np.zeros_like(union), where=union > 0)
    recall = np.divide(
        true_positive, support, out=np.zeros_like(support), where=support > 0
    )
    present_iou = union > 0
    present_recall = support > 0
    total = int(confusion.sum())
    correct = int(true_positive.sum())
    return {
        "point_count": total,
        "mIoU_all": float(100.0 * iou.mean()) if iou.size else float("nan"),
        "mIoU_present": (
            float(100.0 * iou[present_iou].mean()) if np.any(present_iou) else float("nan")
        ),
        "mAcc_present": (
            float(100.0 * recall[present_recall].mean())
            if np.any(present_recall)
            else float("nan")
        ),
        "OA": float(100.0 * correct / total) if total else float("nan"),
        "error_rate": float(100.0 * (total - correct) / total) if total else float("nan"),
    }


def per_class_metrics(confusion, class_names: Sequence[str]) -> List[Dict[str, float]]:
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (len(class_names), len(class_names)):
        raise ValueError("class_names length does not match confusion")
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    rows = []
    for class_index, class_name in enumerate(class_names):
        union = support[class_index] + predicted[class_index] - true_positive[class_index]
        rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "support": int(support[class_index]),
                "predicted_count": int(predicted[class_index]),
                "true_positive": int(true_positive[class_index]),
                "IoU": float(100.0 * true_positive[class_index] / union)
                if union > 0
                else float("nan"),
                "recall": float(100.0 * true_positive[class_index] / support[class_index])
                if support[class_index] > 0
                else float("nan"),
                "precision": float(100.0 * true_positive[class_index] / predicted[class_index])
                if predicted[class_index] > 0
                else float("nan"),
            }
        )
    return rows


def subset_metrics(
    labels,
    predictions,
    masks: Mapping[str, np.ndarray],
    num_classes: int,
) -> Tuple[List[Dict[str, float]], Dict[str, np.ndarray]]:
    rows = []
    confusions = {}
    for name, mask in masks.items():
        confusion = confusion_matrix(labels, predictions, num_classes, mask=mask)
        confusions[name] = confusion
        rows.append({"subset": name, **metrics_from_confusion(confusion)})
    return rows, confusions


def estimate_quantile_thresholds(
    attribute_sets: Iterable[Mapping[str, np.ndarray]],
    max_values_per_attribute: int = 1_000_000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Estimate reusable difficulty thresholds from one or more rooms."""

    rng = np.random.default_rng(seed)
    samples: Dict[str, List[np.ndarray]] = {}
    for attributes in attribute_sets:
        for name, values in attributes.items():
            if not (
                name.startswith("density_count_")
                or name in {"linearity", "planarity", "curvature", "anisotropy"}
                or "cell_entropy" in name
                or "patch_cut_ratio" in name
            ):
                continue
            values = np.asarray(values).reshape(-1)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            per_room_cap = max(1, max_values_per_attribute // 32)
            if finite.size > per_room_cap:
                finite = finite[rng.choice(finite.size, per_room_cap, replace=False)]
            samples.setdefault(name, []).append(finite.astype(np.float64, copy=False))

    thresholds = {}
    for name, parts in samples.items():
        values = np.concatenate(parts)
        if values.size > max_values_per_attribute:
            values = values[rng.choice(values.size, max_values_per_attribute, replace=False)]
        q25, q50, q75 = np.quantile(values, (0.25, 0.50, 0.75))
        thresholds[name] = {
            "q25": float(q25),
            "q50": float(q50),
            "q75": float(q75),
            "sample_count": int(values.size),
        }
    return thresholds


def difficulty_masks(
    attributes: Mapping[str, np.ndarray],
    thresholds: Mapping[str, Mapping[str, float]],
    point_count: int,
) -> Dict[str, np.ndarray]:
    """Create named, point-aligned masks without using model correctness."""

    masks: Dict[str, np.ndarray] = {"all": np.ones(point_count, dtype=bool)}
    for name, raw_values in attributes.items():
        values = np.asarray(raw_values).reshape(-1)
        if values.shape[0] != point_count:
            raise ValueError("attribute {!r} is not point-aligned".format(name))
        finite = np.isfinite(values)
        if name.startswith("boundary_r"):
            masks[name] = finite & values.astype(bool)
            masks[name.replace("boundary_", "interior_")] = finite & ~values.astype(bool)
        elif "mixed_cell" in name:
            masks[name] = finite & (values > 0.5)
            masks[name.replace("mixed_cell", "pure_cell")] = finite & (values <= 0.5)
        elif name in thresholds:
            q25 = float(thresholds[name]["q25"])
            q75 = float(thresholds[name]["q75"])
            if name.startswith("density_count_"):
                low = finite & (values <= q25)
                high = finite & (values >= q75) & ~low
                masks[name + "__low"] = low
                masks[name + "__mid"] = finite & ~low & ~high
                masks[name + "__high"] = high
            elif "patch_cut_ratio" in name:
                low = finite & (values <= q25)
                masks[name + "__low"] = low
                masks[name + "__high"] = finite & (values > q75) & ~low
            else:
                # Strict inequality avoids calling every zero-entropy/zero-
                # curvature point "high" when a discrete distribution has
                # q75 == 0.  An empty high tail is more truthful than an
                # all-point difficulty subset.
                masks[name + "__high"] = finite & (values > q75)
    return masks


def patch_neighbor_recall(
    neighbors,
    patch_ids_by_order: Mapping[str, np.ndarray],
    shadow_index: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Measure how many spatial neighbours remain visible inside patches.

    Self edges and shadow/padding indices are excluded.  ``union`` counts a
    neighbour as visible if any serialization order places the pair together.
    """

    neighbors = np.asarray(neighbors).astype(np.int64, copy=False)
    if neighbors.ndim != 2:
        raise ValueError("neighbors must have shape [N, K]")
    point_count = neighbors.shape[0]
    if shadow_index is None:
        shadow_index = point_count
    valid_neighbor = (neighbors >= 0) & (neighbors < point_count)
    valid_neighbor &= neighbors != np.arange(point_count, dtype=np.int64)[:, None]
    denominator = valid_neighbor.sum(axis=1)
    result: Dict[str, np.ndarray] = {}
    visible_union = np.zeros_like(valid_neighbor)
    for order, raw_patch_ids in patch_ids_by_order.items():
        patch_ids = np.asarray(raw_patch_ids).reshape(-1).astype(np.int64, copy=False)
        if patch_ids.shape[0] != point_count:
            raise ValueError("patch ids for {!r} have the wrong length".format(order))
        safe_neighbors = np.where(valid_neighbor, neighbors, 0)
        visible = valid_neighbor & (
            patch_ids[:, None] == patch_ids[safe_neighbors]
        )
        visible_union |= visible
        recall = np.divide(
            visible.sum(axis=1),
            denominator,
            out=np.full(point_count, np.nan, dtype=np.float64),
            where=denominator > 0,
        )
        result["patch_neighbor_recall_" + order] = recall.astype(np.float32)
        result["patch_cut_ratio_" + order] = (1.0 - recall).astype(np.float32)
    union_recall = np.divide(
        visible_union.sum(axis=1),
        denominator,
        out=np.full(point_count, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    result["patch_neighbor_recall_union"] = union_recall.astype(np.float32)
    result["patch_cut_ratio_union"] = (1.0 - union_recall).astype(np.float32)
    result["patch_neighbor_count"] = denominator.astype(np.int32)
    return result


def save_prediction_artifact(
    path,
    scene_name: str,
    points,
    labels,
    predictions,
    probabilities: Optional[np.ndarray] = None,
    attributes: Optional[Mapping[str, np.ndarray]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> None:
    points, labels = validate_points_labels(points, labels)
    predictions = np.asarray(predictions).reshape(-1).astype(np.int32, copy=False)
    if predictions.shape[0] != points.shape[0]:
        raise ValueError("predictions are not point-aligned")
    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int16),
        "scene_name": np.asarray(str(scene_name)),
        "points": points,
        "labels": labels.astype(np.int32, copy=False),
        "predictions": predictions,
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if probabilities.ndim != 2 or probabilities.shape[0] != points.shape[0]:
            raise ValueError("probabilities must have shape [N, C]")
        payload["probabilities"] = probabilities
    for name, values in (attributes or {}).items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("invalid attribute name: {!r}".format(name))
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != points.shape[0]:
            raise ValueError("attribute {!r} is not point-aligned".format(name))
        payload["attr__" + name] = values
    for name, value in (metadata or {}).items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("invalid metadata name: {!r}".format(name))
        value_array = np.asarray(value)
        if value_array.dtype == object:
            raise ValueError("metadata must not require pickle: {!r}".format(name))
        payload["meta__" + name] = value_array
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_prediction_artifact(path) -> Dict[str, object]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"schema_version", "scene_name", "points", "labels", "predictions"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError("{} is missing {}".format(path, sorted(missing)))
        version = int(np.asarray(data["schema_version"]).item())
        if version != SCHEMA_VERSION:
            raise ValueError("unsupported artifact schema version: {}".format(version))
        points, labels = validate_points_labels(data["points"], data["labels"])
        predictions = np.asarray(data["predictions"]).reshape(-1).astype(np.int64)
        if predictions.shape[0] != points.shape[0]:
            raise ValueError("{} has non-aligned predictions".format(path))
        probabilities = np.asarray(data["probabilities"]) if "probabilities" in data else None
        if probabilities is not None and (
            probabilities.ndim != 2 or probabilities.shape[0] != points.shape[0]
        ):
            raise ValueError("{} has non-aligned probabilities".format(path))
        attributes = {
            key[len("attr__") :]: np.asarray(data[key])
            for key in data.files
            if key.startswith("attr__")
        }
        metadata = {
            key[len("meta__") :]: np.asarray(data[key])
            for key in data.files
            if key.startswith("meta__")
        }
        for name, values in attributes.items():
            if values.ndim == 0 or values.shape[0] != points.shape[0]:
                raise ValueError("{} attribute {!r} is not point-aligned".format(path, name))
        return {
            "path": path,
            "scene_name": str(np.asarray(data["scene_name"]).item()),
            "points": points,
            "labels": labels,
            "predictions": predictions,
            "probabilities": probabilities,
            "attributes": attributes,
            "metadata": metadata,
        }


def index_prediction_artifacts(directory) -> Dict[str, Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    indexed = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "scene_name" not in data:
                continue
            scene_name = str(np.asarray(data["scene_name"]).item())
        if scene_name in indexed:
            raise ValueError("duplicate scene_name {!r} in {}".format(scene_name, directory))
        indexed[scene_name] = path
    if not indexed:
        raise FileNotFoundError("no prediction artifacts found in {}".format(directory))
    return indexed


def validate_paired_artifacts(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    coordinate_atol: float = 1e-5,
) -> None:
    if baseline["scene_name"] != candidate["scene_name"]:
        raise ValueError("scene names differ")
    if not np.array_equal(baseline["labels"], candidate["labels"]):
        raise ValueError("ground-truth labels differ for {}".format(baseline["scene_name"]))
    if baseline["points"].shape != candidate["points"].shape or not np.allclose(
        baseline["points"], candidate["points"], atol=coordinate_atol, rtol=0.0
    ):
        raise ValueError("point coordinates differ for {}".format(baseline["scene_name"]))
