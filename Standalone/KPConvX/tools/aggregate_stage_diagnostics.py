#!/usr/bin/env python3
"""Aggregate multiple one-cloud stage diagnostic runs into mean/std tables."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                row["run_dir"] = path.parent.name
                rows.append(row)
    return rows


def is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def aggregate(rows: List[Dict[str, str]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    output = []
    for key_values, group in sorted(grouped.items()):
        result: Dict[str, object] = dict(zip(group_keys, key_values))
        result["runs"] = len(group)
        numeric_keys = sorted({key for row in group for key, value in row.items() if is_number(value)})
        for key in numeric_keys:
            if key in group_keys or key == "cloud_id":
                continue
            values = [float(row[key]) for row in group if row.get(key) not in (None, "") and is_number(row[key])]
            finite = [value for value in values if math.isfinite(value)]
            if not finite:
                continue
            mean = sum(finite) / len(finite)
            variance = sum((value - mean) ** 2 for value in finite) / len(finite)
            result[f"{key}_mean"] = mean
            result[f"{key}_std"] = math.sqrt(variance)
        output.append(result)
    return output


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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


def main():
    args = parse_args()
    root = Path(args.input_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plans = [
        ("grid_degradation.csv", ["stage"]),
        ("representation_metrics.csv", ["model", "stage"]),
        ("adapter_response.csv", ["stage"]),
        ("degradation_task_metrics.csv", ["model", "stage", "group"]),
        ("fastadapter_gain_by_degradation.csv", ["stage", "group"]),
        ("per_class_metrics.csv", ["model", "class_index", "class_name"]),
        ("single_cloud_metrics.csv", ["model"]),
    ]
    summary_lines = ["# Aggregated stage diagnostics", ""]
    for filename, group_keys in plans:
        paths = sorted(root.glob(f"*/{filename}"))
        rows = read_rows(paths)
        aggregated = aggregate(rows, group_keys)
        write_csv(output / filename.replace(".csv", "_summary.csv"), aggregated)
        summary_lines.append(f"- {filename}: {len(paths)} runs, {len(rows)} rows")
        if filename == "single_cloud_metrics.csv":
            cloud_ids = sorted({row.get("cloud_id", "") for row in rows})
            summary_lines.append(
                f"  - unique cloud ids: {len(cloud_ids)} ({', '.join(cloud_ids)})"
            )
    (output / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
