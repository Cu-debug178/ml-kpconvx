#!/usr/bin/env python3
"""Validate and summarize the five same-checkpoint KTHA interventions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


MODES = ("true", "shuffled", "zero", "room_mean", "branch_off")


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


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
            raise ValueError("invalid result for mode {}: {}".format(mode, path))
        results[mode] = result

    invariant_fields = ("checkpoint_sha256", "checkpoint_epoch", "seed", "protocol")
    for field in invariant_fields:
        values = {results[mode][field] for mode in MODES}
        if len(values) != 1:
            raise ValueError("ablation invariant differs for {}: {}".format(field, values))

    true_miou = float(results["true"]["miou_pct"])
    rows = []
    for mode in MODES:
        result = results[mode]
        value = float(result["miou_pct"])
        rows.append(
            {
                "mode": mode,
                "miou_pct": "{:.9f}".format(value),
                "delta_vs_true_pct_point": "{:+.9f}".format(value - true_miou),
                "checkpoint_epoch": result["checkpoint_epoch"],
                "elapsed_seconds": "{:.3f}".format(float(result["elapsed_seconds"])),
                "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
                "status": result["status"],
            }
        )

    csv_path = root / "summary.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)

    contrasts = {
        "shuffled_minus_true_pct_point": float(results["shuffled"]["miou_pct"]) - true_miou,
        "zero_minus_true_pct_point": float(results["zero"]["miou_pct"]) - true_miou,
        "room_mean_minus_true_pct_point": float(results["room_mean"]["miou_pct"]) - true_miou,
        "branch_off_minus_true_pct_point": float(results["branch_off"]["miou_pct"]) - true_miou,
    }
    summary = {
        "status": "completed",
        "modes": list(MODES),
        "invariants": {field: results["true"][field] for field in invariant_fields},
        "contrasts": contrasts,
        "ranking_by_miou": sorted(
            MODES, key=lambda mode: float(results[mode]["miou_pct"]), reverse=True
        ),
        "interpretation_note": (
            "Single-checkpoint contrasts are descriptive; they do not estimate seed variance "
            "or establish statistical significance."
        ),
    }
    write_json_atomic(root / "summary.json", summary)
    print(json.dumps(summary["contrasts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
