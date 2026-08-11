#!/usr/bin/env python3
"""Run the missing fixed-room Stage-1 mechanism diagnostics for S3DIS L0."""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.litept_blocks import (
    LitePointTransformerBlock,
    build_serialized_patches,
    parse_serialization_orders,
)
from tools.analyze_s3dis_difficulties import (
    deterministic_inference_cfg,
    infer_checkpoint_kp_mode,
    patch_ids_from_layout,
    refuse_gpu_contention_unless_allowed,
    resolve_checkpoint,
    write_csv,
    write_json,
)
from utils.s3dis_diagnostics import patch_neighbor_recall
from utils.stage_diagnostics import compose_ancestor_maps, spearman_correlation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-room Stage-1 token, latency, kernel and attention diagnostics"
    )
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--rooms_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--latency_repeats", type=int, default=3)
    parser.add_argument("--attention_queries", type=int, default=256)
    parser.add_argument("--in_radius", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=57106803)
    parser.add_argument("--allow_gpu_contention", action="store_true")
    args = parser.parse_args()
    if args.latency_repeats < 1:
        parser.error("--latency_repeats must be positive")
    if args.attention_queries < 1:
        parser.error("--attention_queries must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        from utils.gpu_init import init_gpu

        return init_gpu()
    return torch.device("cpu")


def read_room_names(path: Path) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name and not name.startswith("#")]
    if not names:
        raise ValueError("rooms_file does not contain any room names")
    if len(names) != len(set(names)):
        raise ValueError("rooms_file contains duplicate room names")
    return names


def load_model(cfg, checkpoint_path: Path, device: torch.device):
    from models.KPNext import KPNeXt

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    recorded_mode = cfg.model.kp_mode
    effective_mode = infer_checkpoint_kp_mode(state_dict)
    if effective_mode == "legacy_kpconvd_encoder_kpconvx_decoder":
        cfg.model.litept_legacy_kpconvd_encoder = True
    else:
        cfg.model.kp_mode = effective_mode
    cfg.model.ktha_mode = "none"
    model = KPNeXt(cfg)
    model.load_state_dict(state_dict, strict=True)
    epoch = checkpoint.get("epoch", -1)
    if hasattr(epoch, "item"):
        epoch = epoch.item()
    del checkpoint
    model.to(device)
    model.eval()
    return model, recorded_mode, effective_mode, int(epoch)


def room_batch(dataset, scene_index: int, device: torch.device):
    from data_handlers.scene_seg import SceneSegCollate, SceneSegSampler

    points = np.asarray(dataset.input_trees[scene_index].data, dtype=np.float32)
    dataset.reg_sample_pts = torch.from_numpy(points.mean(axis=0, keepdims=True))
    dataset.reg_sample_clouds = torch.tensor([scene_index], dtype=torch.long)
    dataset.reg_sampling_i.zero_()
    dataset.reg_votes.zero_()
    if hasattr(dataset, "reg_sampling_size"):
        dataset.reg_sampling_size.fill_(1)
    sampler = SceneSegSampler(dataset)
    sampler.N = 1
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=SceneSegCollate,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batch = next(iter(loader))
    cloud_indices = batch.in_dict.cloud_inds.detach().cpu().numpy().reshape(-1)
    if cloud_indices.size != 1 or int(cloud_indices[0]) != scene_index:
        raise RuntimeError("deterministic room sampler returned the wrong scene")
    if device.type == "cuda":
        batch.to(device)
    return batch


def timed_components(model) -> List[Tuple[str, int, int, torch.nn.Module]]:
    components = [("stem", 0, -1, model.stem)]
    for layer in range(1, model.num_layers + 1):
        blocks = getattr(model, "encoder_{}".format(layer))
        for block_index, block in enumerate(blocks):
            components.append(("encoder", layer - 1, block_index, block))
        if layer < model.num_layers:
            components.append(
                ("pooling", layer - 1, -1, getattr(model, "pooling_{}".format(layer)))
            )
    if model.task == "cloud_segmentation":
        for layer in range(model.num_layers - 1, 0, -1):
            components.append(
                ("upsampling", layer - 1, -1, getattr(model, "upsampling_{}".format(layer)))
            )
            components.append(
                ("decoder_unary", layer - 1, -1, getattr(model, "decoder_unary_{}".format(layer)))
            )
            if model.add_decoder_layer:
                components.append(
                    ("decoder", layer - 1, -1, getattr(model, "decoder_layer_{}".format(layer)))
                )
    components.append(("head", 0, -1, model.head))
    return components


def measure_latency(model, batch, device: torch.device) -> List[Dict[str, object]]:
    starts: Dict[int, object] = {}
    elapsed: Dict[int, float] = {}
    handles = []
    components = timed_components(model)

    def pre_hook(index):
        def hook(_module, _inputs):
            if device.type == "cuda":
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                starts[index] = event
            else:
                starts[index] = time.perf_counter()

        return hook

    def post_hook(index):
        def hook(_module, _inputs, _output):
            if device.type == "cuda":
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                elapsed[index] = event
            else:
                elapsed[index] = 1000.0 * (time.perf_counter() - starts[index])

        return hook

    for index, (_kind, _stage, _block, module) in enumerate(components):
        handles.append(module.register_forward_pre_hook(pre_hook(index)))
        handles.append(module.register_forward_hook(post_hook(index)))

    if device.type == "cuda":
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
    else:
        total_started = time.perf_counter()
    try:
        with torch.no_grad():
            model(batch)
        if device.type == "cuda":
            total_end.record()
            total_end.synchronize()
            total_ms = float(total_start.elapsed_time(total_end))
        else:
            total_ms = 1000.0 * (time.perf_counter() - total_started)
    finally:
        for handle in handles:
            handle.remove()

    rows = []
    for index, (kind, stage, block, _module) in enumerate(components):
        if index not in elapsed:
            continue
        latency_ms = (
            float(starts[index].elapsed_time(elapsed[index]))
            if device.type == "cuda"
            else float(elapsed[index])
        )
        rows.append(
            {
                "component": kind,
                "stage": stage,
                "block": block,
                "latency_ms": latency_ms,
            }
        )
    rows.append(
        {"component": "end_to_end", "stage": -1, "block": -1, "latency_ms": total_ms}
    )
    return rows


def summarize_latency(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["component"]), int(row["stage"]), int(row["block"]))].append(
            float(row["latency_ms"])
        )
    output = []
    for (component, stage, block), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "component": component,
                "stage": stage,
                "block": block,
                "observation_count": array.size,
                "latency_ms_mean": float(array.mean()),
                "latency_ms_std": float(array.std()),
                "latency_ms_p50": float(np.median(array)),
                "latency_ms_p90": float(np.quantile(array, 0.9)),
            }
        )
    return output


def stage_latency_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, int], float] = defaultdict(float)
    for row in rows:
        if row["component"] != "encoder":
            continue
        key = (str(row["scene_name"]), int(row["repeat"]), int(row["stage"]))
        grouped[key] += float(row["latency_ms"])
    return [
        {
            "scene_name": scene_name,
            "repeat": repeat,
            "component": "encoder_stage",
            "stage": stage,
            "block": -1,
            "latency_ms": latency_ms,
        }
        for (scene_name, repeat, stage), latency_ms in sorted(grouped.items())
    ]


def aggregate_metrics(
    rows: Sequence[Mapping[str, object]],
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, group_rows in sorted(grouped.items()):
        result = {key: value for key, value in zip(group_keys, group)}
        result["observation_count"] = len(group_rows)
        for metric in metric_keys:
            values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
            result[metric + "_mean"] = float(values.mean())
            result[metric + "_std"] = float(values.std())
            result[metric + "_min"] = float(values.min())
            result[metric + "_max"] = float(values.max())
        output.append(result)
    return output


def describe(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            prefix + "_mean": float("nan"),
            prefix + "_std": float("nan"),
            prefix + "_p50": float("nan"),
            prefix + "_p90": float("nan"),
        }
    return {
        prefix + "_mean": float(values.mean()),
        prefix + "_std": float(values.std()),
        prefix + "_p50": float(np.median(values)),
        prefix + "_p90": float(np.quantile(values, 0.9)),
    }


def kernel_entropy_diagnostics(
    model,
    batch,
    trace: Mapping[str, object],
    logits: torch.Tensor,
    cfg,
    source_stage: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    cached_geometry = model.shared_kp[source_stage]
    nearest_kernel = cached_geometry.get("neighb_1nn")
    influence = cached_geometry.get("infl_w")
    if nearest_kernel is None:
        raise RuntimeError("actual KP nearest-kernel assignments were not cached")
    if influence is None:
        influence = torch.ones_like(nearest_kernel, dtype=logits.dtype)
    neighbors = batch.in_dict.neighbors[source_stage]
    valid = neighbors < trace["points"][source_stage].shape[0]
    weights = influence.float() * valid.float()
    kernel_count = int(sum(cfg.model.shell_sizes))
    signature = weights.new_zeros((nearest_kernel.shape[0], kernel_count))
    signature.scatter_add_(1, nearest_kernel, weights)
    signature = signature / signature.sum(dim=1, keepdim=True).clamp_min(1e-6)
    entropy = -(signature * torch.log(signature.clamp_min(1e-12))).sum(dim=-1)
    normalized = entropy / math.log(signature.shape[1])
    global_distribution = signature.mean(dim=0)
    global_entropy = -(
        global_distribution * torch.log(global_distribution.clamp_min(1e-12))
    ).sum()

    labels = trace["labels"].detach().cpu().numpy().astype(np.int64)
    predictions = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    errors = (predictions != labels).astype(np.float64)
    stage_sizes = [int(points.shape[0]) for points in trace["points"]]
    upsample_maps = [
        mapping.detach().cpu().numpy().reshape(-1).astype(np.int64)
        for mapping in trace["upsamples"]
    ]
    ancestors = compose_ancestor_maps(upsample_maps, stage_sizes)
    source_ancestor = ancestors[source_stage]
    entropy_np = entropy.detach().cpu().numpy()
    point_entropy = entropy_np[source_ancestor]
    token_counts = np.bincount(source_ancestor, minlength=stage_sizes[source_stage])
    token_errors = np.bincount(
        source_ancestor, weights=errors, minlength=stage_sizes[source_stage]
    ) / np.maximum(token_counts, 1)

    entropy_row = {
        "source_stage": source_stage,
        "token_count": int(signature.shape[0]),
        "kernel_count": int(signature.shape[1]),
        **describe(entropy_np, "token_entropy_nats"),
        **describe(normalized.detach().cpu().numpy(), "token_entropy_normalized"),
        "global_kernel_entropy_nats": float(global_entropy.item()),
        "global_kernel_entropy_normalized": float(
            global_entropy.item() / math.log(signature.shape[1])
        ),
    }
    correlation_row = {
        "source_stage": source_stage,
        "input_point_count": int(errors.size),
        "token_count": int(entropy_np.size),
        "input_error_rate": float(errors.mean()),
        "spearman_point_error_vs_kernel_entropy": spearman_correlation(
            errors, point_entropy
        ),
        "spearman_token_error_rate_vs_kernel_entropy": spearman_correlation(
            token_errors, entropy_np
        ),
    }
    return entropy_row, correlation_row


def patch_kp_rows(batch, trace, cfg) -> List[Dict[str, object]]:
    orders = parse_serialization_orders(cfg.model.litept_orders)
    rows = []
    for stage, points in enumerate(trace["points"]):
        one_based = stage + 1
        if not (
            one_based > int(cfg.model.litept_conv_stages)
            or one_based == int(cfg.model.litept_handover_stage)
        ):
            continue
        point_count = int(points.shape[0])
        patch_ids = {}
        voxel_size = max(
            float(cfg.model.in_sub_size) * float(cfg.model.radius_scaling) ** stage,
            1e-6,
        )
        for order in orders:
            indices, valid, _ = build_serialized_patches(
                points,
                trace["lengths"][stage],
                patch_size=int(cfg.model.litept_patch_size),
                voxel_size=voxel_size,
                order=order,
            )
            patch_ids[order] = patch_ids_from_layout(
                indices.detach().cpu().numpy(),
                valid.detach().cpu().numpy(),
                point_count,
            )
        attributes = patch_neighbor_recall(
            batch.in_dict.neighbors[stage].detach().cpu().numpy(),
            patch_ids,
            shadow_index=point_count,
        )
        for order in list(orders) + ["union"]:
            overlap = attributes["patch_neighbor_recall_{}".format(order)]
            cut = attributes["patch_cut_ratio_{}".format(order)]
            rows.append(
                {
                    "stage": stage,
                    "order": order,
                    "token_count": point_count,
                    "kp_edge_count": int(attributes["patch_neighbor_count"].sum()),
                    **describe(overlap, "patch_kp_edge_overlap"),
                    **describe(cut, "kp_edge_cut_rate"),
                }
            )
    return rows


def enable_attention_diagnostics(model, enabled: bool, max_queries: int) -> None:
    for module in model.modules():
        if isinstance(module, LitePointTransformerBlock):
            module.set_diagnostics_mode(enabled, max_queries=max_queries)


def collect_attention_rows(model, conv_stages: int) -> Tuple[List[Dict], List[Dict]]:
    attention_rows = []
    residual_rows = []
    for module_name, module in model.named_modules():
        if not isinstance(module, LitePointTransformerBlock):
            continue
        match = re.search(r"encoder_(\d+)", module_name)
        if match is None:
            continue
        stage = int(match.group(1)) - 1
        diagnostics = module.diagnostics()
        attention = diagnostics.get("attention", {})
        if attention:
            attention_rows.append(
                {"module": module_name, "stage": stage, **attention}
            )
        residual_row = {
            "module": module_name,
            "stage": stage,
            "transition": (
                "stage{}_to_stage{}".format(stage - 1, stage)
                if stage == conv_stages
                else ""
            ),
            "token_count": diagnostics.get("token_count", 0),
        }
        for key in (
            "attention_residual_ratio",
            "mlp_residual_ratio",
            "total_residual_ratio",
        ):
            for statistic, value in diagnostics.get(key, {}).items():
                residual_row["{}_{}".format(key, statistic)] = value
        if residual_row["token_count"]:
            residual_rows.append(residual_row)
    return attention_rows, residual_rows


def aggregate_token_counts(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for row in rows:
        grouped[int(row["stage"])].append(int(row["token_count"]))
    output = []
    for stage, counts in sorted(grouped.items()):
        values = np.asarray(counts, dtype=np.float64)
        output.append(
            {
                "stage": stage,
                "room_count": values.size,
                "token_count_mean": float(values.mean()),
                "token_count_min": int(values.min()),
                "token_count_max": int(values.max()),
                "token_count_p50": float(np.median(values)),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device != "cpu":
        refuse_gpu_contention_unless_allowed(args.allow_gpu_contention)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = Path(args.log_path).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    rooms_file = Path(args.rooms_file).expanduser().resolve()
    checkpoint_path = resolve_checkpoint(log_path, args.checkpoint)
    room_names = read_room_names(rooms_file)

    cfg = deterministic_inference_cfg(log_path, dataset_path, args.in_radius)
    cfg.test.max_steps_per_epoch = 1
    cfg.test.num_workers = 0
    model, recorded_mode, effective_mode, checkpoint_epoch = load_model(
        cfg, checkpoint_path, device
    )

    from experiments.S3DIS.S3DIS_rooms import S3DIRDataset

    dataset = S3DIRDataset(cfg, chosen_set="validation", precompute_pyramid=True)
    scene_indices = {name: index for index, name in enumerate(dataset.scene_names)}
    missing_rooms = sorted(set(room_names) - set(scene_indices))
    if missing_rooms:
        raise ValueError("rooms not found in Area 5: {}".format(missing_rooms))

    token_rows = []
    latency_rows = []
    patch_rows = []
    kernel_rows = []
    correlation_rows = []
    attention_rows = []
    residual_rows = []
    source_stage = int(cfg.model.litept_conv_stages) - 1

    for room_index, room_name in enumerate(room_names):
        print("[{}/{}] {}".format(room_index + 1, len(room_names), room_name), flush=True)
        batch = room_batch(dataset, scene_indices[room_name], device)

        # Warm up kernels and allocator before measuring actual model components.
        with torch.no_grad():
            model(batch)
        for repeat in range(args.latency_repeats):
            for row in measure_latency(model, batch, device):
                latency_rows.append({"scene_name": room_name, "repeat": repeat, **row})

        enable_attention_diagnostics(model, True, args.attention_queries)
        with torch.no_grad():
            logits, trace = model(batch, return_intermediates=True)
        room_attention, room_residual = collect_attention_rows(
            model, int(cfg.model.litept_conv_stages)
        )
        enable_attention_diagnostics(model, False, args.attention_queries)

        for stage, points in enumerate(trace["points"]):
            token_rows.append(
                {
                    "scene_name": room_name,
                    "stage": stage,
                    "token_count": int(points.shape[0]),
                    "reduction_from_stage0": float(
                        points.shape[0] / max(trace["points"][0].shape[0], 1)
                    ),
                }
            )
        for row in patch_kp_rows(batch, trace, cfg):
            patch_rows.append({"scene_name": room_name, **row})
        kernel_row, correlation_row = kernel_entropy_diagnostics(
            model, batch, trace, logits, cfg, source_stage
        )
        kernel_rows.append({"scene_name": room_name, **kernel_row})
        correlation_rows.append({"scene_name": room_name, **correlation_row})
        attention_rows.extend(
            {"scene_name": room_name, **row} for row in room_attention
        )
        residual_rows.extend(
            {"scene_name": room_name, **row} for row in room_residual
        )

        del batch, logits, trace
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "token_counts.csv", token_rows)
    write_csv(output_dir / "token_count_summary.csv", aggregate_token_counts(token_rows))
    write_csv(output_dir / "layer_latency.csv", latency_rows)
    write_csv(output_dir / "layer_latency_summary.csv", summarize_latency(latency_rows))
    per_stage_latency = stage_latency_rows(latency_rows)
    write_csv(output_dir / "stage_latency.csv", per_stage_latency)
    write_csv(
        output_dir / "stage_latency_summary.csv", summarize_latency(per_stage_latency)
    )
    write_csv(output_dir / "patch_kp_overlap.csv", patch_rows)
    patch_summary = aggregate_metrics(
        patch_rows,
        ("stage", "order"),
        ("patch_kp_edge_overlap_mean", "kp_edge_cut_rate_mean"),
    )
    write_csv(output_dir / "patch_kp_overlap_summary.csv", patch_summary)
    write_csv(output_dir / "kernel_occupancy_entropy.csv", kernel_rows)
    kernel_summary = aggregate_metrics(
        kernel_rows,
        ("source_stage",),
        ("token_entropy_normalized_mean", "global_kernel_entropy_normalized"),
    )
    write_csv(output_dir / "kernel_occupancy_entropy_summary.csv", kernel_summary)
    write_csv(output_dir / "error_kernel_entropy_correlation.csv", correlation_rows)
    correlation_summary = aggregate_metrics(
        correlation_rows,
        ("source_stage",),
        (
            "spearman_point_error_vs_kernel_entropy",
            "spearman_token_error_rate_vs_kernel_entropy",
        ),
    )
    write_csv(
        output_dir / "error_kernel_entropy_correlation_summary.csv",
        correlation_summary,
    )
    write_csv(output_dir / "sampled_attention.csv", attention_rows)
    attention_summary = aggregate_metrics(
        attention_rows,
        ("stage", "order"),
        ("entropy_nats_mean", "entropy_normalized_mean", "distance_m_mean"),
    )
    write_csv(output_dir / "sampled_attention_summary.csv", attention_summary)
    write_csv(output_dir / "residual_ratio.csv", residual_rows)
    residual_summary = aggregate_metrics(
        residual_rows,
        ("module", "stage", "transition"),
        (
            "attention_residual_ratio_mean",
            "mlp_residual_ratio_mean",
            "total_residual_ratio_mean",
        ),
    )
    write_csv(output_dir / "residual_ratio_summary.csv", residual_summary)
    write_json(
        output_dir / "run_config.json",
        {
            "log_dir_name": log_path.name,
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_epoch": checkpoint_epoch,
            "dataset_path": "<DATASET_PATH>",
            "rooms_file": rooms_file.name,
            "room_count": len(room_names),
            "rooms": room_names,
            "device": str(device),
            "latency_repeats": args.latency_repeats,
            "attention_queries_per_block": args.attention_queries,
            "seed": args.seed,
            "recorded_kp_mode": recorded_mode,
            "effective_kp_mode": effective_mode,
            "kernel_entropy_source_stage": source_stage,
            "stage_2_to_3_definition": "zero-based encoder Stage 2 KP output to Stage 3 first token-attention stage",
            "latency_note": "CUDA-event model compute latency; excludes data loading and host-to-device transfer.",
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# S3DIS L0 Stage-1 mechanism diagnostics",
                "",
                "This is deterministic fixed-room checkpoint inference; no model was trained.",
                "",
                "- Token counts and CUDA-event latency come from actual model forwards.",
                "- Patch-KP overlap is the fraction of actual KP-neighbour edges retained inside serialized patches; cut rate is its complement.",
                "- Kernel entropy uses the normalized 43-kernel occupancy signature at zero-based Stage 2.",
                "- Error/entropy correlations are descriptive, not causal; both point-level and token-level Spearman values are reported.",
                "- Attention entropy and distance explicitly reconstruct only the configured deterministic query sample per block.",
                "- Stage 2->3 residual ratio is branch norm divided by block-input norm in the first token-attention stage.",
                "- Boundary, mixed-cell and per-class metrics are produced separately by analyze_s3dis_difficulties.py profile.",
            ]
        ),
        encoding="utf-8",
    )
    print("Stage-1 mechanism diagnostics written to {}".format(output_dir))


if __name__ == "__main__":
    main()
