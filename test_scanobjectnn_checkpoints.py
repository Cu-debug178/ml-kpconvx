#!/usr/bin/env python3
"""Run deterministic full ScanObjectNN tests for distinct checkpoint epochs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def checkpoint_epoch(path: Path) -> int:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError(f"{path} does not contain an integer epoch")
    return epoch


def checkpoint_priority(path: Path) -> tuple[int, str]:
    if path.name.startswith("chkp_exact_"):
        return 0, path.name
    if path.name == "current_chkp.tar":
        return 1, path.name
    return 2, path.name


def checkpoint_inventory(experiment: Path) -> list[dict[str, Any]]:
    checkpoint_dir = experiment / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    by_epoch: dict[int, Path] = {}
    duplicates: dict[int, list[str]] = {}
    for path in sorted(checkpoint_dir.glob("*.tar")):
        epoch = checkpoint_epoch(path)
        duplicates.setdefault(epoch, []).append(path.name)
        previous = by_epoch.get(epoch)
        if previous is None or checkpoint_priority(path) < checkpoint_priority(previous):
            by_epoch[epoch] = path

    if not by_epoch:
        raise RuntimeError(f"no checkpoints found in {checkpoint_dir}")
    return [
        {
            "epoch": epoch,
            "checkpoint": str(by_epoch[epoch].resolve()),
            "available_files": duplicates[epoch],
        }
        for epoch in sorted(by_epoch)
    ]


def completed_report(output_dir: Path, max_votes: int) -> Path | None:
    success_path = output_dir / "success.json"
    if not success_path.is_file():
        return None
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
        report = Path(success["report"])
        vote_count = int(success["vote_count"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if report.is_file() and vote_count >= max_votes:
        return report
    return None


def run_checkpoint_test(
    python: Path,
    test_script: Path,
    kpconvx_root: Path,
    dataset: Path,
    experiment: Path,
    item: dict[str, Any],
    output_root: Path,
    max_votes: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    epoch = int(item["epoch"])
    checkpoint = Path(item["checkpoint"])
    output_dir = output_root / experiment.name / f"epoch_{epoch:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_report = completed_report(output_dir, max_votes)
    if existing_report is not None:
        return {**item, "status": "already_completed", "report": str(existing_report)}

    attempts = sorted(output_dir.glob("attempt_*.console.log"))
    attempt = len(attempts) + 1
    console_log = output_dir / f"attempt_{attempt:03d}.console.log"
    test_output = output_dir / f"attempt_{attempt:03d}"
    command = [
        str(python),
        str(test_script),
        "--log_path",
        str(experiment),
        "--weight_path",
        str(checkpoint),
        "--dataset_path",
        str(dataset),
        "--test_path",
        str(test_output),
        "--max_votes",
        str(max_votes),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONPATH"] = str(kpconvx_root) + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )

    started_at = utc_now()
    with console_log.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=kpconvx_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"test failed for {experiment.name} epoch {epoch} "
            f"with code {completed.returncode}; see {console_log}"
        )

    report_candidates = sorted(test_output.glob("test_*/report.txt"))
    if not report_candidates:
        raise RuntimeError(
            f"test produced no report for {experiment.name} epoch {epoch}"
        )
    report = report_candidates[-1]
    report_text = report.read_text(encoding="utf-8", errors="replace")
    vote_count = report_text.count("Vote ")
    if vote_count < max_votes:
        raise RuntimeError(
            f"test report has {vote_count}/{max_votes} votes for "
            f"{experiment.name} epoch {epoch}"
        )

    success = {
        **item,
        "status": "completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "max_votes": max_votes,
        "vote_count": vote_count,
        "report": str(report.resolve()),
        "console_log": str(console_log.resolve()),
    }
    atomic_write_json(output_dir / "success.json", success)
    return success


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/autodl-tmp/envs/pointcept/bin/python"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project
        / "Standalone/KPConvX/results/ScanObjectNN_full_checkpoint_tests-seed57106803",
    )
    parser.add_argument("--max-votes", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_votes < 1 or args.timeout_seconds < 1:
        raise ValueError("max-votes and timeout-seconds must be positive")
    project = Path(__file__).resolve().parent
    kpconvx_root = project / "Standalone/KPConvX"
    test_script = kpconvx_root / "experiments/ScanObjectNN/test_ScanObj.py"
    experiments = [path.resolve() for path in args.experiment]
    inventory = {
        experiment.name: checkpoint_inventory(experiment)
        for experiment in experiments
    }
    if args.inventory_only:
        print(json.dumps(inventory, ensure_ascii=True, indent=2))
        return 0

    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise FileNotFoundError(f"Python is not executable: {args.python}")
    if not args.dataset_path.is_dir():
        raise FileNotFoundError(f"dataset not found: {args.dataset_path}")
    if not test_script.is_file():
        raise FileNotFoundError(f"test script not found: {test_script}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "max_votes": args.max_votes,
        "experiments": inventory,
        "results": [],
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    for experiment in experiments:
        for item in inventory[experiment.name]:
            result = run_checkpoint_test(
                args.python.resolve(),
                test_script.resolve(),
                kpconvx_root.resolve(),
                args.dataset_path.resolve(),
                experiment,
                item,
                output_root,
                args.max_votes,
                args.timeout_seconds,
            )
            manifest["results"].append(result)
            atomic_write_json(output_root / "manifest.json", manifest)

    manifest["status"] = "completed"
    manifest["finished_at"] = utc_now()
    atomic_write_json(output_root / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
