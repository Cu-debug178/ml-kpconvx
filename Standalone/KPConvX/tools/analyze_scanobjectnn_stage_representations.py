#!/usr/bin/env python3
"""Run a small, deterministic joint-PCA diagnostic on ScanObjectNN.

This is a classification diagnostic, not a replacement for the full-vote
ScanObjectNN evaluation. It compares matched checkpoints on one fixed test
object and records stage representations, classifier outputs, and FastAdapter
corrections. PCA is fitted jointly for all models in a stage whenever their
feature widths match; otherwise it falls back to one basis per architecture.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ply import write_ply
from utils.stage_diagnostics import fit_joint_pca_rgb, representation_geometry_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--object_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=57106803)
    parser.add_argument("--max_plot_points", type=int, default=30000)
    parser.add_argument("--max_metric_points", type=int, default=1024)
    parser.add_argument("--max_pca_fit_points", type=int, default=50000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        metavar="NAME|ARCH|MODE|LOG|CHECKPOINT",
        help="Repeat for each model. MODE is grid, bypass, or adapter.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_specs(values: Sequence[str]) -> List[Dict[str, object]]:
    specs = []
    for value in values:
        fields = value.split("|", 4)
        if len(fields) != 5:
            raise ValueError(
                "--spec must have NAME|ARCH|MODE|LOG|CHECKPOINT fields: " + value
            )
        name, architecture, mode, log_path, checkpoint = fields
        if mode not in {"grid", "bypass", "adapter"}:
            raise ValueError("spec mode must be grid, bypass, or adapter: " + value)
        if mode == "grid" and architecture not in {"kpconvd", "kpconvx"}:
            raise ValueError("grid specs must use kpconvd or kpconvx: " + value)
        if mode != "grid" and architecture not in {"kpconvd", "kpconvx"}:
            raise ValueError("adapter specs must use kpconvd or kpconvx: " + value)
        specs.append({
            "name": name,
            "architecture": architecture,
            "mode": mode,
            "log_path": Path(log_path).expanduser().resolve(),
            "checkpoint": Path(checkpoint).expanduser().resolve(),
        })
    if not specs:
        raise ValueError("At least one --spec is required")
    return specs


def deterministic_cfg(log_path: Path, dataset_path: str):
    from utils.config import load_cfg

    cfg = load_cfg(str(log_path))
    cfg.data.path = dataset_path
    cfg.test.num_workers = 0
    cfg.test.data_sampler = "regular"
    cfg.augment_test.anisotropic = False
    cfg.augment_test.scale = [1.0, 1.0]
    cfg.augment_test.flips = [0.0, 0.0, 0.0]
    cfg.augment_test.rotations = "none"
    cfg.augment_test.jitter = 0.0
    cfg.augment_test.color_drop = 0.0
    cfg.augment_test.chromatic_contrast = False
    cfg.augment_test.chromatic_all = False
    cfg.augment_test.chromatic_norm = False
    cfg.augment_test.pts_drop_p = -1.0
    cfg.augment_test.pts_drop_reg = False
    cfg.augment_test.rsmix_prob = 0.0
    cfg.augment_test.rsmix_beta = 0.0
    return cfg


def validate_configs(configs: Sequence[object]) -> None:
    reference = configs[0]
    keys = [
        ("model.in_sub_size", reference.model.in_sub_size),
        ("model.radius_scaling", reference.model.radius_scaling),
        ("model.grid_pool", reference.model.grid_pool),
        ("model.in_sub_mode", reference.model.in_sub_mode),
        ("model.layer_blocks", tuple(reference.model.layer_blocks)),
        ("data.init_sub_size", reference.data.init_sub_size),
    ]
    for cfg in configs[1:]:
        for name, expected in keys:
            current = getattr(cfg.model, name.split(".")[-1], None) if name.startswith("model.") else getattr(cfg.data, name.split(".")[-1], None)
            if name == "model.layer_blocks":
                current = tuple(current)
            if current != expected:
                raise ValueError(f"Input hierarchy mismatch for {name}: {expected!r} != {current!r}")


def load_model(cfg, checkpoint_path: Path, device: torch.device):
    from models.KPNext import KPNeXt

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    internal_epoch = int(checkpoint["epoch"])
    model = KPNeXt(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, internal_epoch


def packed_slice(lengths: torch.Tensor, cloud_index: int) -> slice:
    lengths_np = lengths.detach().cpu().numpy().astype(np.int64)
    if not 0 <= cloud_index < len(lengths_np):
        raise IndexError(f"cloud_index={cloud_index} outside batch size {len(lengths_np)}")
    start = int(lengths_np[:cloud_index].sum())
    return slice(start, start + int(lengths_np[cloud_index]))


def extract_trace(trace: Dict, cloud_index: int) -> Dict[str, object]:
    points = []
    features = []
    stage_slices = []
    for point_tensor, length_tensor, stage_data in zip(
        trace["points"], trace["lengths"], trace["stages"]
    ):
        current_slice = packed_slice(length_tensor, cloud_index)
        stage_slices.append(current_slice)
        points.append(point_tensor[current_slice].detach().cpu().numpy())
        features.append(stage_data["post_adapter_features"][current_slice].detach().cpu().numpy())

    adapter_layers = {}
    for stage, layer_data in trace.get("adapter", {}).get("layers", {}).items():
        full = layer_data.get("full")
        if not full:
            continue
        current_slice = stage_slices[int(stage)]
        adapter_layers[int(stage)] = {
            key: value[current_slice].detach().cpu().numpy()
            if value.ndim > 0 and value.shape[0] == trace["points"][int(stage)].shape[0]
            else value.detach().cpu().numpy()
            for key, value in full.items()
        }
    return {"points": points, "features": features, "adapter_layers": adapter_layers}


def feature_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    values = np.sum(a * b, axis=1) / np.maximum(denominator, 1e-12)
    return float(np.mean(values))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_stage_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    colors = np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    write_ply(
        str(path),
        [points.astype(np.float32), colors],
        ["x", "y", "z", "red", "green", "blue"],
    )


def render_montage(
    path: Path,
    names: Sequence[str],
    traces: Sequence[Dict[str, object]],
    rgb_sets: Sequence[Sequence[np.ndarray]],
    max_points: int,
    seed: int,
) -> None:
    stage_count = len(traces[0]["points"])
    fig, axes = plt.subplots(
        len(names), stage_count,
        figsize=(3.2 * stage_count, 2.55 * len(names)),
        squeeze=False,
    )
    rng = np.random.default_rng(seed)
    for row, (name, trace, colors_by_stage) in enumerate(zip(names, traces, rgb_sets)):
        for stage, (points, colors) in enumerate(zip(trace["points"], colors_by_stage)):
            if max_points > 0 and points.shape[0] > max_points:
                indices = rng.choice(points.shape[0], size=max_points, replace=False)
                points = points[indices]
                colors = colors[indices]
            axes[row][stage].scatter(
                points[:, 0], points[:, 1], c=colors, s=5.0, linewidths=0, rasterized=True
            )
            axes[row][stage].set_aspect("equal", adjustable="box")
            axes[row][stage].axis("off")
            if row == 0:
                axes[row][stage].set_title(
                    f"Stage {stage}\n{trace['points'][stage].shape[0]} tokens",
                    fontsize=10,
                )
            if stage == 0:
                axes[row][stage].text(
                    -0.25,
                    0.5,
                    name,
                    transform=axes[row][stage].transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                    clip_on=False,
                )
    fig.suptitle("ScanObjectNN joint-PCA stage representations", fontsize=15, y=0.975)
    fig.text(
        0.5,
        0.945,
        "Rows = model / inference mode | Columns = encoder stage | Each dot = one token",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.018,
        "Dot position = point-cloud XY geometry; RGB = shared PCA-1/2/3 of features in this column (not class labels)",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.22, right=0.99, top=0.90, bottom=0.07, wspace=0.05, hspace=0.12)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def choose_pca_groups(traces: Sequence[Dict[str, object]], specs: Sequence[Dict[str, object]]) -> Dict[str, List[int]]:
    stage_widths = [
        {trace["features"][stage].shape[1] for trace in traces}
        for stage in range(len(traces[0]["features"]))
    ]
    if all(len(widths) == 1 for widths in stage_widths):
        return {"all_models": list(range(len(specs)))}
    groups: Dict[str, List[int]] = {}
    for index, spec in enumerate(specs):
        groups.setdefault(str(spec["architecture"]), []).append(index)
    return groups


def main() -> None:
    args = parse_args()
    specs = parse_specs(args.spec)
    if not all(spec["checkpoint"].is_file() for spec in specs):
        missing = [str(spec["checkpoint"]) for spec in specs if not spec["checkpoint"].is_file()]
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))
    if not all(spec["log_path"].is_dir() for spec in specs):
        missing = [str(spec["log_path"]) for spec in specs if not spec["log_path"].is_dir()]
        raise FileNotFoundError("Missing log directory(ies): " + ", ".join(missing))

    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    configs = [deterministic_cfg(spec["log_path"], args.dataset_path) for spec in specs]
    for spec, cfg in zip(specs, configs):
        if cfg.model.kp_mode != spec["architecture"]:
            raise ValueError(
                f"Architecture mismatch for {spec['name']}: "
                f"spec={spec['architecture']}, config={cfg.model.kp_mode}"
            )
    validate_configs(configs)

    from data_handlers.object_classification import ObjClassifBatch
    from experiments.ScanObjectNN.ScanObjectNN import ScanObjectNNDataset

    dataset = ScanObjectNNDataset(
        configs[0], chosen_set="test", precompute_pyramid=True, uniform_sample=True
    )
    if not 0 <= args.object_index < len(dataset.input_points):
        raise IndexError(
            f"object_index={args.object_index} outside test set [0, {len(dataset.input_points) - 1}]"
        )
    input_dict = dataset[[args.object_index]]
    batch = ObjClassifBatch([input_dict])

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda:0")
    elif args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    batch.to(device)

    traces = []
    logits = []
    checkpoint_rows = []
    for spec, cfg in zip(specs, configs):
        model, internal_epoch = load_model(cfg, spec["checkpoint"], device)
        if spec["mode"] == "bypass":
            if model.fast_adapter is None:
                raise ValueError(f"bypass requested but FastAdapter is disabled: {spec['name']}")
            model.fast_adapter = None
        with torch.no_grad():
            output, trace = model(
                batch,
                return_intermediates=True,
                capture_adapter_details=spec["mode"] == "adapter",
            )
        traces.append(extract_trace(trace, 0))
        logits.append(output[0].detach().cpu().numpy())
        checkpoint_rows.append({
            "name": spec["name"],
            "architecture": spec["architecture"],
            "mode": spec["mode"],
            "internal_epoch": internal_epoch,
            "log_path": str(spec["log_path"]),
            "checkpoint": str(spec["checkpoint"]),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stage_count = len(traces[0]["points"])
    for trace in traces[1:]:
        if len(trace["points"]) != stage_count:
            raise ValueError("All models must expose the same number of stages")
        for stage in range(stage_count):
            if trace["points"][stage].shape != traces[0]["points"][stage].shape:
                raise ValueError(f"Stage {stage} point counts differ across models")
            if not np.allclose(trace["points"][stage], traces[0]["points"][stage], atol=1e-5):
                raise ValueError(f"Stage {stage} point coordinates differ across models")

    pca_groups = choose_pca_groups(traces, specs)
    all_rgb = [[None] * stage_count for _ in traces]
    representation_rows = []
    pca_group_rows = []
    for group_name, indices in pca_groups.items():
        for stage in range(stage_count):
            group_features = [traces[index]["features"][stage] for index in indices]
            rgb, pca_info = fit_joint_pca_rgb(
                group_features,
                max_fit_points=args.max_pca_fit_points,
                seed=args.seed + stage,
            )
            pca_group_rows.append({
                "pca_group": group_name,
                "stage": stage,
                **pca_info,
            })
            for local_index, model_index in enumerate(indices):
                all_rgb[model_index][stage] = rgb[local_index]
                metrics = representation_geometry_metrics(
                    traces[model_index]["points"][stage],
                    traces[model_index]["features"][stage],
                    max_points=args.max_metric_points,
                    seed=args.seed + stage,
                )
                representation_rows.append({
                    "name": specs[model_index]["name"],
                    "architecture": specs[model_index]["architecture"],
                    "mode": specs[model_index]["mode"],
                    "stage": stage,
                    "point_count": traces[model_index]["points"][stage].shape[0],
                    "channels": traces[model_index]["features"][stage].shape[1],
                    "pca_group": group_name,
                    **pca_info,
                    **metrics,
                })
                safe_name = str(specs[model_index]["name"]).lower().replace("+", "_plus_").replace(" ", "_")
                save_stage_ply(
                    output_dir / f"{safe_name}_stage_{stage}.ply",
                    traces[model_index]["points"][stage],
                    all_rgb[model_index][stage],
                )

    render_montage(
        output_dir / "stage_representation_comparison.png",
        [str(spec["name"]) for spec in specs],
        traces,
        all_rgb,
        args.max_plot_points,
        args.seed,
    )

    true_label = int(batch.in_dict.labels[0].item())
    logits_rows = []
    for spec, values in zip(specs, logits):
        probabilities = torch.softmax(torch.from_numpy(values), dim=0).numpy()
        predicted = int(np.argmax(values))
        logits_rows.append({
            "name": spec["name"],
            "architecture": spec["architecture"],
            "mode": spec["mode"],
            "object_index": args.object_index,
            "true_label": true_label,
            "predicted_label": predicted,
            "correct": int(predicted == true_label),
            "top1_probability": float(probabilities[predicted]),
            "logit_top1": float(values[predicted]),
        })

    adapter_rows = []
    for spec, trace in zip(specs, traces):
        for stage, diagnostics in sorted(trace["adapter_layers"].items()):
            row = {
                "name": spec["name"],
                "architecture": spec["architecture"],
                "stage": stage,
            }
            for key in (
                "p2a_weights",
                "a2p_gate",
                "correction_norm",
                "correction_ratio",
                "anchor_counts",
            ):
                if key in diagnostics:
                    values = np.asarray(diagnostics[key], dtype=np.float64).reshape(-1)
                    row[f"{key}_mean"] = float(values.mean())
                    row[f"{key}_p90"] = float(np.percentile(values, 90))
            adapter_rows.append(row)

    feature_delta_rows = []
    for architecture in sorted({str(spec["architecture"]) for spec in specs}):
        adapter_indices = [
            index for index, spec in enumerate(specs)
            if spec["architecture"] == architecture and spec["mode"] == "adapter"
        ]
        bypass_indices = [
            index for index, spec in enumerate(specs)
            if spec["architecture"] == architecture and spec["mode"] == "bypass"
        ]
        if not adapter_indices or not bypass_indices:
            continue
        adapter_index = adapter_indices[0]
        bypass_index = bypass_indices[0]
        for stage in range(stage_count):
            adapter_features = traces[adapter_index]["features"][stage]
            bypass_features = traces[bypass_index]["features"][stage]
            delta = adapter_features - bypass_features
            feature_delta_rows.append({
                "architecture": architecture,
                "adapter_name": specs[adapter_index]["name"],
                "bypass_name": specs[bypass_index]["name"],
                "stage": stage,
                "mean_l2_delta": float(np.linalg.norm(delta, axis=1).mean()),
                "p90_l2_delta": float(np.percentile(np.linalg.norm(delta, axis=1), 90)),
                "mean_cosine": feature_cosine(adapter_features, bypass_features),
            })

    write_csv(output_dir / "checkpoints.csv", checkpoint_rows)
    write_csv(output_dir / "pca_groups.csv", pca_group_rows)
    write_csv(output_dir / "representation_metrics.csv", representation_rows)
    write_csv(output_dir / "logits.csv", logits_rows)
    write_csv(output_dir / "adapter_response.csv", adapter_rows)
    write_csv(output_dir / "feature_delta_vs_bypass.csv", feature_delta_rows)

    summary = [
        "# ScanObjectNN joint-PCA diagnostic",
        "",
        f"Fixed test object index: {args.object_index}",
        f"True class index: {true_label}",
        "The same deterministic input pyramid was reused for every checkpoint.",
        "PCA colours are shared within each pca_group and stage.",
        "This is a representation diagnostic on one object, not a full test-set score.",
        "",
        "## Checkpoint and classifier outputs",
        "",
    ]
    for row in logits_rows:
        summary.append(
            f"- {row['name']}: epoch={next(item['internal_epoch'] for item in checkpoint_rows if item['name'] == row['name'])}, "
            f"pred={row['predicted_label']}, correct={row['correct']}, top1={row['top1_probability']:.4f}"
        )
    summary.extend([
        "",
        "## Files",
        "",
        "- `representation_metrics.csv`: per-stage feature geometry and joint-PCA variance.",
        "- `feature_delta_vs_bypass.csv`: FastAdapter output difference against its own bypass forward.",
        "- `adapter_response.csv`: FastAdapter gate/correction statistics.",
        "- `logits.csv`: one-object classification output; it is not a substitute for OA/mAcc over the test set.",
        "- `stage_representation_comparison.png` and stage PLY files: qualitative shared-colour visualization.",
        "",
        "## Limitations",
        "",
        "- One object and one seed cannot establish a dataset-level mechanism claim.",
        "- Joint PCA aligns visualization coordinates; it does not measure accuracy or causality.",
        "- The epoch-249/199 periodic checkpoints were intentionally excluded because the requested matched comparison is epoch 250.",
    ])
    (output_dir / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset_path": args.dataset_path,
                "output_dir": str(output_dir),
                "object_index": args.object_index,
                "seed": args.seed,
                "device": str(device),
                "specs": [
                    {
                        **{key: value for key, value in spec.items() if key not in {"log_path", "checkpoint"}},
                        "log_path": str(spec["log_path"]),
                        "checkpoint": str(spec["checkpoint"]),
                    }
                    for spec in specs
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Diagnostics written to: {output_dir}")


if __name__ == "__main__":
    main()
