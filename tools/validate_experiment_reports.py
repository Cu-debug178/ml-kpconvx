#!/usr/bin/env python3
"""Validate lightweight experiment reports and guard against large artifacts."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "experiment_reports"
REPORT_MAX_BYTES = 2 * 1024 * 1024
TRACKED_MAX_BYTES = 20 * 1024 * 1024
TEXT_SUFFIXES = {".md", ".csv", ".json", ".yaml", ".yml", ".txt"}
FORBIDDEN_SUFFIXES = {
    ".tar",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
}
LOCAL_PATH = re.compile(rb"(?<![A-Za-z0-9_])/(?:root|home|Users)/")
SECRET_PATTERNS = (
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def tracked_paths() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    )
    return [ROOT / item.decode("utf-8") for item in output.split(b"\0") if item]


def validate() -> list[str]:
    errors: list[str] = []
    required_top_level = [REPORT_ROOT / "README.md", REPORT_ROOT / "REPORT_TEMPLATE.md"]
    for path in required_top_level:
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")

    if not REPORT_ROOT.is_dir():
        return errors or ["experiment_reports directory is missing"]

    for path in sorted(REPORT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES:
            errors.append(f"unsupported report file type: {relative}")
        if suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"binary artifact in report directory: {relative}")
        if path.stat().st_size > REPORT_MAX_BYTES:
            errors.append(f"report file exceeds 2 MiB: {relative}")
        data = path.read_bytes()
        if LOCAL_PATH.search(data):
            errors.append(f"machine-specific absolute path in: {relative}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                errors.append(f"possible credential in: {relative}")
                break

    for dataset_dir in sorted(path for path in REPORT_ROOT.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            for filename in ("README.md", "metrics.csv"):
                required = run_dir / filename
                if not required.is_file():
                    errors.append(f"missing {filename}: {run_dir.relative_to(ROOT)}")
            metrics = run_dir / "metrics.csv"
            if metrics.is_file():
                with metrics.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    if not reader.fieldnames:
                        errors.append(f"CSV has no header: {metrics.relative_to(ROOT)}")
                    elif not any(reader):
                        errors.append(f"CSV has no data rows: {metrics.relative_to(ROOT)}")

    try:
        tracked = tracked_paths()
    except (OSError, subprocess.CalledProcessError) as error:
        errors.append(f"cannot inspect Git index: {error}")
        return errors

    for path in tracked:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden tracked artifact: {relative}")
        if {part.lower() for part in relative.parts} & {"checkpoints", "snapshots"}:
            errors.append(f"tracked checkpoint directory: {relative}")
        if path.stat().st_size > TRACKED_MAX_BYTES:
            errors.append(f"tracked file exceeds 20 MiB: {relative}")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Experiment report validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Experiment reports and tracked-file guards passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
