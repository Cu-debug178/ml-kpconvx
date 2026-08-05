"""Utilities for stage-wise point-cloud representation diagnostics.

The functions in this module intentionally depend only on NumPy and PyTorch so
that they can be reused by offline checkpoint analysis and lightweight tests.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


def _as_numpy(array) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def fit_joint_pca_rgb(
    feature_sets: Sequence[np.ndarray],
    max_fit_points: int = 50000,
    seed: int = 0,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """Project several feature matrices with one shared PCA basis.

    A joint basis is essential for visual comparisons: fitting PCA separately
    to the baseline and comparison model permits arbitrary sign and axis
    rotations and can create misleading colour differences.
    """

    arrays = [_as_numpy(features).astype(np.float64, copy=False) for features in feature_sets]
    if not arrays:
        raise ValueError("feature_sets must not be empty")
    channels = {array.shape[1] for array in arrays}
    if len(channels) != 1:
        raise ValueError("Joint PCA requires equal feature dimensions")
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("Each feature array must have shape [N, C]")

    nonempty = [array for array in arrays if array.shape[0] > 0]
    if not nonempty:
        raise ValueError("At least one feature array must contain points")
    joint = np.concatenate(nonempty, axis=0)
    rng = np.random.default_rng(seed)
    if max_fit_points > 0 and joint.shape[0] > max_fit_points:
        fit_indices = rng.choice(joint.shape[0], size=max_fit_points, replace=False)
        fit = joint[fit_indices]
    else:
        fit = joint

    mean = fit.mean(axis=0, keepdims=True)
    centered = fit - mean
    # full_matrices=False is reliable for both narrow and wide feature tensors.
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(3, vt.shape[0])].T
    if components.shape[1] < 3:
        components = np.pad(components, ((0, 0), (0, 3 - components.shape[1])))

    projected = [(array - mean) @ components for array in arrays]
    projected_joint = np.concatenate([p for p in projected if p.shape[0] > 0], axis=0)
    lo = np.percentile(projected_joint, 1.0, axis=0)
    hi = np.percentile(projected_joint, 99.0, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    rgb_sets = [np.clip((p - lo) / span, 0.0, 1.0) for p in projected]

    total_variance = float(np.sum(np.var(centered, axis=0, ddof=1))) if fit.shape[0] > 1 else 0.0
    explained = np.square(singular_values[:3]) / max(fit.shape[0] - 1, 1)
    explained_ratio = float(np.sum(explained) / max(total_variance, 1e-12))
    return rgb_sets, {
        "fit_points": float(fit.shape[0]),
        "channels": float(fit.shape[1]),
        "explained_variance_ratio_3": explained_ratio,
    }


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling, without SciPy."""

    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_correlation(x, y) -> float:
    x = _as_numpy(x).reshape(-1)
    y = _as_numpy(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(rx, ry) / denominator)


def compose_ancestor_maps(
    upsample_maps: Sequence[np.ndarray],
    stage_sizes: Sequence[int],
) -> List[np.ndarray]:
    """Map every stage-0 point to its ancestor token at each stage.

    ``upsample_maps[l]`` maps stage ``l`` points to stage ``l+1`` points.
    The maps must use local indices for one cloud.
    """

    if len(stage_sizes) < 1:
        raise ValueError("stage_sizes must not be empty")
    if len(upsample_maps) != len(stage_sizes) - 1:
        raise ValueError("Expected one upsample map per stage transition")

    ancestors = [np.arange(stage_sizes[0], dtype=np.int64)]
    current = ancestors[0]
    for level, mapping in enumerate(upsample_maps):
        mapping = _as_numpy(mapping).reshape(-1).astype(np.int64, copy=False)
        if mapping.shape[0] != stage_sizes[level]:
            raise ValueError("Upsample map length does not match its fine stage")
        if mapping.size and (mapping.min() < 0 or mapping.max() >= stage_sizes[level + 1]):
            raise ValueError("Upsample map contains an invalid coarse index")
        current = mapping[current]
        ancestors.append(current.copy())
    return ancestors


def stage_cell_statistics(
    original_labels: np.ndarray,
    ancestor_map: np.ndarray,
    stage_size: int,
    num_classes: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Compute occupancy and semantic mixing for each coarse token."""

    labels = _as_numpy(original_labels).reshape(-1).astype(np.int64, copy=False)
    ancestors = _as_numpy(ancestor_map).reshape(-1).astype(np.int64, copy=False)
    if labels.shape[0] != ancestors.shape[0]:
        raise ValueError("labels and ancestor_map must have equal length")
    if stage_size < 1:
        raise ValueError("stage_size must be positive")
    if num_classes is None:
        num_classes = int(labels.max()) + 1 if labels.size else 0

    valid = (labels >= 0) & (labels < num_classes)
    occupancy = np.bincount(ancestors, minlength=stage_size).astype(np.int64)
    hist = np.zeros((stage_size, num_classes), dtype=np.int64)
    if np.any(valid):
        np.add.at(hist, (ancestors[valid], labels[valid]), 1)
    label_counts = hist.sum(axis=1)
    probabilities = hist / np.maximum(label_counts[:, None], 1)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
    majority_fraction = hist.max(axis=1) / np.maximum(label_counts, 1)
    mixed = (hist > 0).sum(axis=1) > 1
    majority_label = np.argmax(hist, axis=1) if num_classes > 0 else np.zeros(stage_size, dtype=np.int64)
    return {
        "occupancy": occupancy,
        "label_histogram": hist,
        "label_entropy": entropy,
        "majority_fraction": majority_fraction,
        "minority_fraction": 1.0 - majority_fraction,
        "mixed": mixed,
        "majority_label": majority_label,
    }


def representation_geometry_metrics(
    points,
    features,
    max_points: int = 1024,
    k: int = 8,
    seed: int = 0,
) -> Dict[str, float]:
    """Measure how strongly feature neighbourhoods follow spatial geometry.

    These metrics are descriptive, not task scores. High neighbourhood recall
    is expected in shallow geometric stages and may decrease as semantic
    abstraction emerges in deeper stages.
    """

    points_np = _as_numpy(points).astype(np.float32, copy=False)
    features_np = _as_numpy(features).astype(np.float32, copy=False)
    if points_np.shape[0] != features_np.shape[0]:
        raise ValueError("points and features must contain the same number of rows")
    n = points_np.shape[0]
    if n < 3:
        return {
            "sample_points": float(n),
            "spatial_feature_knn_recall": float("nan"),
            "local_distance_spearman": float("nan"),
            "feature_norm_mean": float(np.linalg.norm(features_np, axis=1).mean()) if n else float("nan"),
            "feature_norm_std": float(np.linalg.norm(features_np, axis=1).std()) if n else float("nan"),
        }

    rng = np.random.default_rng(seed)
    sample_n = min(n, max_points) if max_points > 0 else n
    indices = rng.choice(n, size=sample_n, replace=False) if sample_n < n else np.arange(n)
    p = torch.from_numpy(points_np[indices])
    f = torch.from_numpy(features_np[indices])
    f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-8)
    spatial_dist = torch.cdist(p, p)
    feature_dist = torch.cdist(f, f)
    diagonal = torch.arange(sample_n)
    spatial_dist[diagonal, diagonal] = float("inf")
    feature_dist[diagonal, diagonal] = float("inf")
    k_eff = max(1, min(k, sample_n - 1))
    spatial_knn = torch.topk(spatial_dist, k=k_eff, largest=False).indices
    feature_knn = torch.topk(feature_dist, k=k_eff, largest=False).indices
    recalls = []
    local_spatial = []
    local_feature = []
    for row in range(sample_n):
        spatial_set = set(spatial_knn[row].tolist())
        recalls.append(len(spatial_set.intersection(feature_knn[row].tolist())) / k_eff)
        neigh = spatial_knn[row]
        local_spatial.extend(spatial_dist[row, neigh].tolist())
        local_feature.extend(feature_dist[row, neigh].tolist())

    norms = np.linalg.norm(features_np, axis=1)
    return {
        "sample_points": float(sample_n),
        "spatial_feature_knn_recall": float(np.mean(recalls)),
        "local_distance_spearman": spearman_correlation(local_spatial, local_feature),
        "feature_norm_mean": float(norms.mean()),
        "feature_norm_std": float(norms.std()),
    }


def summarize_cell_statistics(stats: Dict[str, np.ndarray]) -> Dict[str, float]:
    occupancy = stats["occupancy"]
    entropy = stats["label_entropy"]
    nonempty = occupancy > 0
    return {
        "token_count": float(occupancy.shape[0]),
        "occupancy_mean": float(occupancy[nonempty].mean()) if np.any(nonempty) else 0.0,
        "occupancy_p95": float(np.percentile(occupancy[nonempty], 95)) if np.any(nonempty) else 0.0,
        "occupancy_max": float(occupancy.max()) if occupancy.size else 0.0,
        "mixed_cell_ratio": float(stats["mixed"][nonempty].mean()) if np.any(nonempty) else 0.0,
        "label_entropy_mean": float(entropy[nonempty].mean()) if np.any(nonempty) else 0.0,
        "minority_fraction_mean": float(stats["minority_fraction"][nonempty].mean()) if np.any(nonempty) else 0.0,
    }
