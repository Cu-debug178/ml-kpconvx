"""Low-overhead optimizer and module monitoring for KPConvX experiments."""

from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import torch
import torch.nn as nn

try:
    from models.fast_adapter import FastAdapterStack
    from models.litept_blocks import LitePointTransformerBlock
except ImportError:  # pragma: no cover - allows isolated utility tests
    FastAdapterStack = ()
    LitePointTransformerBlock = ()


_GROUP_NAMES = ("backbone", "litept_attention", "fast_adapter", "head")


def _parameter_ids(module: Optional[nn.Module]) -> Set[int]:
    if module is None:
        return set()
    return {id(parameter) for parameter in module.parameters()}


def parameter_group_ids(model: nn.Module) -> Dict[str, Set[int]]:
    """Assign trainable parameters to disjoint diagnostic groups."""

    owner = model.module if hasattr(model, "module") else model
    adapter_ids: Set[int] = set()
    litept_ids: Set[int] = set()
    for module in owner.modules():
        if FastAdapterStack and isinstance(module, FastAdapterStack):
            adapter_ids.update(_parameter_ids(module))
        if LitePointTransformerBlock and isinstance(module, LitePointTransformerBlock):
            litept_ids.update(_parameter_ids(module))
    head_ids = _parameter_ids(getattr(owner, "head", None))
    # Adapter and head take precedence over nested generic module names.
    litept_ids.difference_update(adapter_ids)
    litept_ids.difference_update(head_ids)
    all_ids = {id(parameter) for parameter in owner.parameters() if parameter.requires_grad}
    backbone_ids = all_ids - adapter_ids - litept_ids - head_ids
    return {
        "backbone": backbone_ids,
        "litept_attention": litept_ids,
        "fast_adapter": adapter_ids,
        "head": head_ids,
    }


def collect_parameter_statistics(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Collect gradient and weight RMS without altering gradients."""

    owner = model.module if hasattr(model, "module") else model
    group_ids = parameter_group_ids(owner)
    accum = {
        name: {
            "grad_sq": None,
            "weight_sq": None,
            "grad_numel": 0,
            "weight_numel": 0,
            "grad_max_abs": None,
        }
        for name in _GROUP_NAMES
    }
    id_to_group = {}
    for group_name, ids in group_ids.items():
        for parameter_id in ids:
            id_to_group[parameter_id] = group_name

    for parameter in owner.parameters():
        if not parameter.requires_grad:
            continue
        group_name = id_to_group.get(id(parameter), "backbone")
        values = parameter.detach()
        weight_sq = torch.sum(values.float().square())
        previous_weight_sq = accum[group_name]["weight_sq"]
        accum[group_name]["weight_sq"] = (
            weight_sq if previous_weight_sq is None else previous_weight_sq + weight_sq
        )
        accum[group_name]["weight_numel"] += values.numel()
        if parameter.grad is not None:
            grad = parameter.grad.detach().float()
            grad_sq = torch.sum(grad.square())
            previous_grad_sq = accum[group_name]["grad_sq"]
            accum[group_name]["grad_sq"] = (
                grad_sq if previous_grad_sq is None else previous_grad_sq + grad_sq
            )
            accum[group_name]["grad_numel"] += grad.numel()
            if grad.numel():
                grad_max_abs = grad.abs().max()
                previous_grad_max = accum[group_name]["grad_max_abs"]
                accum[group_name]["grad_max_abs"] = (
                    grad_max_abs
                    if previous_grad_max is None
                    else torch.maximum(previous_grad_max, grad_max_abs)
                )

    results: Dict[str, Dict[str, float]] = {}
    global_grad_sq = 0.0
    global_grad_numel = 0
    for name, values in accum.items():
        grad_sq = (
            float(values["grad_sq"].item())
            if values["grad_sq"] is not None
            else 0.0
        )
        weight_sq = (
            float(values["weight_sq"].item())
            if values["weight_sq"] is not None
            else 0.0
        )
        grad_max_abs = (
            float(values["grad_max_abs"].item())
            if values["grad_max_abs"] is not None
            else 0.0
        )
        global_grad_sq += grad_sq
        global_grad_numel += values["grad_numel"]
        results[name] = {
            "grad_norm": math.sqrt(grad_sq),
            "grad_rms": math.sqrt(grad_sq / max(values["grad_numel"], 1)),
            "weight_rms": math.sqrt(weight_sq / max(values["weight_numel"], 1)),
            "grad_max_abs": grad_max_abs,
            "parameter_count": float(values["weight_numel"]),
            "gradient_count": float(values["grad_numel"]),
        }
    results["global"] = {
        "grad_norm": math.sqrt(global_grad_sq),
        "grad_rms": math.sqrt(global_grad_sq / max(global_grad_numel, 1)),
        "weight_rms": float("nan"),
        "grad_max_abs": max(
            (values["grad_max_abs"] for values in results.values()),
            default=0.0,
        ),
        "parameter_count": float(sum(v["weight_numel"] for v in accum.values())),
        "gradient_count": float(global_grad_numel),
    }
    return results


ParameterSample = Dict[str, List[Tuple[nn.Parameter, torch.Tensor, torch.Tensor]]]


def capture_parameter_samples(
    model: nn.Module,
    max_samples_per_tensor: int = 128,
) -> ParameterSample:
    """Capture a tiny deterministic parameter sample before ``optimizer.step``.

    Cloning every parameter would temporarily duplicate the model. Sampling at
    most a few values per tensor gives a useful actual-update diagnostic with
    negligible memory compared with full activation storage.
    """

    owner = model.module if hasattr(model, "module") else model
    group_ids = parameter_group_ids(owner)
    id_to_group = {
        parameter_id: group_name
        for group_name, ids in group_ids.items()
        for parameter_id in ids
    }
    snapshot: ParameterSample = {name: [] for name in _GROUP_NAMES}
    sample_limit = max(1, int(max_samples_per_tensor))
    for parameter in owner.parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        flat = parameter.detach().reshape(-1)
        sample_n = min(sample_limit, flat.numel())
        if sample_n == flat.numel():
            indices = torch.arange(flat.numel(), device=flat.device)
        else:
            indices = torch.linspace(
                0,
                flat.numel() - 1,
                steps=sample_n,
                device=flat.device,
            ).round().long()
        before = flat.index_select(0, indices).float().clone()
        group_name = id_to_group.get(id(parameter), "backbone")
        snapshot[group_name].append((parameter, indices, before))
    return snapshot


def collect_sampled_update_statistics(snapshot: ParameterSample) -> Dict[str, Dict[str, float]]:
    """Measure sampled parameter movement after ``optimizer.step``."""

    results: Dict[str, Dict[str, float]] = {}
    global_delta_sq = 0.0
    global_before_sq = 0.0
    global_count = 0
    for group_name in _GROUP_NAMES:
        delta_sq_tensor = None
        before_sq_tensor = None
        count = 0
        for parameter, indices, before in snapshot.get(group_name, []):
            after = parameter.detach().reshape(-1).index_select(0, indices).float()
            delta = after - before
            parameter_delta_sq = torch.sum(delta.square())
            parameter_before_sq = torch.sum(before.square())
            delta_sq_tensor = (
                parameter_delta_sq
                if delta_sq_tensor is None
                else delta_sq_tensor + parameter_delta_sq
            )
            before_sq_tensor = (
                parameter_before_sq
                if before_sq_tensor is None
                else before_sq_tensor + parameter_before_sq
            )
            count += before.numel()
        delta_sq = float(delta_sq_tensor.item()) if delta_sq_tensor is not None else 0.0
        before_sq = float(before_sq_tensor.item()) if before_sq_tensor is not None else 0.0
        update_rms = math.sqrt(delta_sq / max(count, 1))
        sampled_weight_rms = math.sqrt(before_sq / max(count, 1))
        results[group_name] = {
            "sampled_update_rms": update_rms,
            "sampled_update_ratio": update_rms / max(sampled_weight_rms, 1e-12),
            "sampled_update_count": float(count),
        }
        global_delta_sq += delta_sq
        global_before_sq += before_sq
        global_count += count
    global_update_rms = math.sqrt(global_delta_sq / max(global_count, 1))
    global_weight_rms = math.sqrt(global_before_sq / max(global_count, 1))
    results["global"] = {
        "sampled_update_rms": global_update_rms,
        "sampled_update_ratio": global_update_rms / max(global_weight_rms, 1e-12),
        "sampled_update_count": float(global_count),
    }
    return results


def merge_update_statistics(
    statistics: Dict[str, Dict[str, float]],
    updates: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    for group_name, values in updates.items():
        statistics.setdefault(group_name, {}).update(values)
    return statistics


def _tensor_to_float(value: Any) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Only scalar tensors can be logged as summary values")
        return float(value.detach().cpu().item())
    return float(value)


def flatten_adapter_summaries(diagnostics: Mapping[str, Any]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for stage, layer_data in sorted(diagnostics.get("layers", {}).items()):
        summary = layer_data.get("summary", {})
        row: Dict[str, float] = {"stage": float(stage)}
        for key, value in summary.items():
            row[key] = _tensor_to_float(value)
        rows.append(row)
    return rows


def append_optimization_monitor(
    log_dir: str,
    epoch: int,
    optimizer_step: int,
    learning_rates: Iterable[float],
    statistics: Mapping[str, Mapping[str, float]],
) -> None:
    path = os.path.join(log_dir, "optimization_monitor.csv")
    fieldnames = [
        "epoch",
        "optimizer_step",
        "lr_min",
        "lr_max",
        "group",
        "grad_norm",
        "grad_rms",
        "weight_rms",
        "grad_max_abs",
        "parameter_count",
        "gradient_count",
        "sampled_update_rms",
        "sampled_update_ratio",
        "sampled_update_count",
    ]
    rates = list(learning_rates)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for group_name, values in statistics.items():
            writer.writerow({
                "epoch": epoch,
                "optimizer_step": optimizer_step,
                "lr_min": min(rates) if rates else float("nan"),
                "lr_max": max(rates) if rates else float("nan"),
                "group": group_name,
                **{key: values.get(key, float("nan")) for key in fieldnames[5:]},
            })


def append_fast_adapter_monitor(
    log_dir: str,
    epoch: int,
    optimizer_step: int,
    diagnostics: Mapping[str, Any],
) -> None:
    rows = flatten_adapter_summaries(diagnostics)
    if not rows:
        return
    path = os.path.join(log_dir, "fast_adapter_monitor.csv")
    metric_names = sorted({key for row in rows for key in row if key != "stage"})
    fieldnames = ["epoch", "optimizer_step", "stage"] + metric_names
    exists = os.path.exists(path)
    # The set of summary keys is stable for a given architecture. If an older
    # file exists with a different header, fail loudly instead of corrupting it.
    if exists:
        with open(path, newline="") as stream:
            existing_header = next(csv.reader(stream), [])
        if existing_header != fieldnames:
            raise RuntimeError(
                "FastAdapter monitor header changed. Use a fresh log directory: " + path
            )
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({
                "epoch": epoch,
                "optimizer_step": optimizer_step,
                **row,
            })


def append_dks_monitor(
    log_dir: str,
    epoch: int,
    optimizer_step: int,
    diagnostics: Mapping[int, Mapping[str, Any]],
) -> None:
    """Append per-stage Dynamic Kernel Scale distribution diagnostics."""

    if not diagnostics:
        return
    path = os.path.join(log_dir, "dks_alpha_stats.csv")
    fieldnames = [
        "step",
        "epoch",
        "stage",
        "mean",
        "std",
        "p05",
        "p95",
        "frac_lt_0p9",
        "frac_gt_1p1",
        "gate",
    ]
    exists = os.path.exists(path)
    if exists:
        with open(path, newline="") as stream:
            existing_header = next(csv.reader(stream), [])
        if existing_header != fieldnames:
            raise RuntimeError(
                "DKS monitor header changed. Use a fresh log directory: " + path
            )
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for stage, values in sorted(diagnostics.items()):
            writer.writerow({
                "step": optimizer_step,
                "epoch": epoch,
                "stage": int(stage),
                "mean": _tensor_to_float(values.get("mean", float("nan"))),
                "std": _tensor_to_float(values.get("std", float("nan"))),
                "p05": _tensor_to_float(values.get("p05", float("nan"))),
                "p95": _tensor_to_float(values.get("p95", float("nan"))),
                "frac_lt_0p9": _tensor_to_float(
                    values.get("frac_below_0.9", float("nan"))
                ),
                "frac_gt_1p1": _tensor_to_float(
                    values.get("frac_above_1.1", float("nan"))
                ),
                "gate": _tensor_to_float(values.get("gate", float("nan"))),
            })
