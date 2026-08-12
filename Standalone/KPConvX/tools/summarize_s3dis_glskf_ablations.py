#!/usr/bin/env python3
"""Summarize same-checkpoint GLSKF interventions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

MODES = ("baseline", "true", "shuffled", "room_mean", "zero_context", "neutral_gate", "branch_off")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    results = {}
    for mode in MODES:
        path = root / mode / "result.json"
        if not path.is_file():
            raise FileNotFoundError("missing completed result: {}".format(path))
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        if result.get("status") != "completed" or result.get("mode") != mode:
            raise ValueError("invalid result for mode {}".format(mode))
        results[mode] = result
    invariant_fields = ("checkpoint_sha256", "checkpoint_epoch", "seed", "protocol")
    intervention_modes = MODES[1:]
    for field in invariant_fields:
        if len({results[mode][field] for mode in intervention_modes}) != 1:
            raise ValueError("ablation invariant differs for {}".format(field))
    true_miou = float(results["true"]["miou_pct"])
    baseline_miou = float(results["baseline"]["miou_pct"])
    rows = []
    for mode in MODES:
        value = float(results[mode]["miou_pct"])
        rows.append({
            "mode": mode,
            "miou_pct": "{:.9f}".format(value),
            "delta_vs_true_pct_point": "{:+.9f}".format(value - true_miou),
            "delta_vs_baseline_pct_point": "{:+.9f}".format(value - baseline_miou),
            "elapsed_seconds": "{:.3f}".format(float(results[mode]["elapsed_seconds"])),
            "peak_cuda_allocated_bytes": results[mode]["peak_cuda_allocated_bytes"],
        })
    csv_path = root / "summary.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    summary = {
        "status": "completed",
        "modes": list(MODES),
        "invariants": {field: results["true"][field] for field in invariant_fields},
        "baseline_checkpoint": {
            "checkpoint_sha256": results["baseline"]["checkpoint_sha256"],
            "checkpoint_epoch": results["baseline"]["checkpoint_epoch"],
        },
        "contrasts": {
            "true_minus_baseline_pct_point": true_miou - baseline_miou,
            "true_minus_shuffled_pct_point": float(results["true"]["miou_pct"]) - float(results["shuffled"]["miou_pct"]),
            "true_minus_room_mean_pct_point": float(results["true"]["miou_pct"]) - float(results["room_mean"]["miou_pct"]),
            "true_minus_zero_context_pct_point": float(results["true"]["miou_pct"]) - float(results["zero_context"]["miou_pct"]),
            "true_minus_neutral_gate_pct_point": float(results["true"]["miou_pct"]) - float(results["neutral_gate"]["miou_pct"]),
            "true_minus_branch_off_pct_point": float(results["true"]["miou_pct"]) - float(results["branch_off"]["miou_pct"]),
        },
        "ranking_by_miou": sorted(MODES, key=lambda mode: float(results[mode]["miou_pct"]), reverse=True),
        "interpretation_note": (
            "The six GLSKF modes are same-checkpoint descriptive contrasts and do not "
            "estimate seed variance. Baseline is a separate external L0 checkpoint; "
            "true-minus-baseline is a performance reference, not a single-variable "
            "causal contrast."
        ),
    }
    path = root / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
    print(json.dumps(summary["contrasts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
