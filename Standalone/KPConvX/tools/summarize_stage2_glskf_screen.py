#!/usr/bin/env python3
"""Summarize the five matched warm-start runs in the GLSKF screen."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


RUNS = (
    "l0_head",
    "true",
    "shuffled",
    "room_mean",
    "matched_mlp",
)


def read_mious(path: Path, expected_epochs: int) -> list[float]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            values = [float(value) for value in line.split()]
            if not values:
                raise ValueError("empty IoU row at {}:{}".format(path, line_number))
            rows.append(100.0 * sum(values) / len(values))
    if len(rows) < expected_epochs:
        raise ValueError(
            "{} has {} rows, expected at least {}".format(path, len(rows), expected_epochs)
        )
    return rows[:expected_epochs]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected-epochs", type=int, default=10)
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "RUN_DIR"))
    args = parser.parse_args()

    if not args.run:
        raise ValueError("at least one --run LABEL RUN_DIR is required")
    run_dirs = {label: Path(path) for label, path in args.run}
    if set(run_dirs) != set(RUNS):
        raise ValueError("run labels must be exactly {}".format(RUNS))

    series = {
        label: read_mious(run_dirs[label] / "val_IoUs.txt", args.expected_epochs)
        for label in RUNS
    }
    rows = []
    for epoch in range(args.expected_epochs):
        row = {"epoch": epoch + 1}
        row.update({label: series[label][epoch] for label in RUNS})
        row["true_minus_l0_head"] = row["true"] - row["l0_head"]
        row["true_minus_shuffled"] = row["true"] - row["shuffled"]
        row["true_minus_room_mean"] = row["true"] - row["room_mean"]
        row["true_minus_matched_mlp"] = row["true"] - row["matched_mlp"]
        rows.append(row)

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "warm10_summary.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)

    best = {label: max(values) for label, values in series.items()}
    best_epoch = {
        label: values.index(best[label]) + 1 for label, values in series.items()
    }
    paired_true_epoch = best_epoch["true"] - 1
    paired = {
        "true_minus_l0_head": series["true"][paired_true_epoch] - series["l0_head"][paired_true_epoch],
        "true_minus_shuffled": series["true"][paired_true_epoch] - series["shuffled"][paired_true_epoch],
        "true_minus_room_mean": series["true"][paired_true_epoch] - series["room_mean"][paired_true_epoch],
        "true_minus_matched_mlp": series["true"][paired_true_epoch] - series["matched_mlp"][paired_true_epoch],
    }
    summary = {
        "status": "completed",
        "expected_epochs": args.expected_epochs,
        "best_miou_pct": best,
        "best_epoch": best_epoch,
        "paired_at_true_best_epoch": paired,
        "selection_rule": (
            "Treat true-context GLSKF as a screening candidate only if it exceeds both "
            "L0 controls and matched_mlp, and true-minus-shuffled is at least +0.3 percentage point."
        ),
        "limitations": (
            "Single seed and adaptive best-epoch selection are descriptive, not a significance test."
        ),
    }
    json_path = root / "warm10_summary.json"
    temporary_json = json_path.with_suffix(".json.tmp")
    with temporary_json.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_json, json_path)
    print(json.dumps(paired, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
