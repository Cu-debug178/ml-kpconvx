#!/usr/bin/env python3
"""Archive training checkpoints without changing the training process.

The process watches validation rows and copies the rolling checkpoint only
after its internal epoch matches the completed validation epoch.  It never
touches the training process or loads checkpoint tensors onto the GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


LOGGER = logging.getLogger("checkpoint_archive_monitor")
POLICY_FILENAME = "checkpoint_policy.json"
LOG_FILENAME = "checkpoint_archive_monitor.log"
DEFAULT_POLL_SECONDS = 15.0


@dataclass(frozen=True)
class ValidationRecord:
    """One complete validation row and its one-based file line number."""

    line_number: int
    miou: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_validation_records(path: Path) -> list[ValidationRecord]:
    """Read complete validation rows, ignoring a still-being-written last row."""

    if not path.exists():
        return []

    records: list[ValidationRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            # validation.py appends one newline-terminated row at a time.  A
            # partial final row must wait for the next polling pass.
            if not raw_line.endswith(("\n", "\r")):
                continue
            fields = raw_line.split()
            if not fields:
                LOGGER.warning("Ignoring empty validation line %d", line_number)
                continue
            try:
                values = [float(value) for value in fields]
            except ValueError:
                LOGGER.warning("Ignoring malformed validation line %d", line_number)
                continue
            if not values:
                continue
            records.append(ValidationRecord(line_number, sum(values) / len(values)))
    return records


def copy_checkpoint_atomic(source: Path, destination: Path) -> None:
    """Copy a checkpoint and publish it only after the copy is complete."""

    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        with source.open("rb") as source_file, temporary.open("wb") as temp_file:
            shutil.copyfileobj(source_file, temp_file, length=16 * 1024 * 1024)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def checkpoint_epoch(path: Path) -> int:
    """Return the checkpoint's completed-epoch marker using CPU memory only."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError(f"checkpoint has no integer epoch: {path}")
    return epoch


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Persist the small monitor state without exposing a partial JSON file."""

    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_policy(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Could not read existing policy %s: %s", path, error)
        return None
    if not isinstance(value, dict):
        LOGGER.warning("Ignoring non-object policy %s", path)
        return None
    return value


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


class ArchiveMonitor:
    def __init__(
        self,
        result_dir: Path,
        start_epoch: int,
        interval: int,
        final_epoch: int | None,
        poll_seconds: float,
    ) -> None:
        if start_epoch < 1:
            raise ValueError("start epoch must be >= 1")
        if interval < 1:
            raise ValueError("archive interval must be >= 1")
        if final_epoch is not None and final_epoch < start_epoch:
            raise ValueError("final epoch must be >= start epoch")
        if poll_seconds <= 0:
            raise ValueError("poll seconds must be > 0")

        self.result_dir = result_dir.resolve()
        self.checkpoint_dir = self.result_dir / "checkpoints"
        self.validation_path = self.result_dir / "val_IoUs.txt"
        self.source_checkpoint = self.checkpoint_dir / "current_chkp.tar"
        self.policy_path = self.checkpoint_dir / POLICY_FILENAME
        self.start_epoch = start_epoch
        self.interval = interval
        self.final_epoch = final_epoch
        self.poll_seconds = poll_seconds
        self.stop_requested = False

        existing_records = read_validation_records(self.validation_path)
        existing_policy = load_policy(self.policy_path)
        if existing_policy is None:
            # There is no reliable way to reconstruct historical checkpoint
            # contents from current_chkp.tar.  Start observing from now.
            self.processed_line = (
                existing_records[-1].line_number if existing_records else 0
            )
            self.best_epoch: int | None = None
            self.best_miou: float | None = None
            self.periodic_checkpoints: list[dict[str, Any]] = []
            self.started_at = utc_now()
            self.initial_validation_lines = self.processed_line
        else:
            self.processed_line = int(
                existing_policy.get("last_processed_validation_line", 0)
            )
            self.best_epoch = self._optional_int(
                existing_policy.get("best_validation_epoch")
            )
            saved_best_miou = self._optional_float(
                existing_policy.get("best_validation_miou")
            )
            # Policy files expose mIoU as a percentage (for example 49.2),
            # while val_IoUs.txt stores values in [0, 1].
            self.best_miou = (
                None if saved_best_miou is None else saved_best_miou / 100.0
            )
            periodic = existing_policy.get("periodic_checkpoints", [])
            self.periodic_checkpoints = periodic if isinstance(periodic, list) else []
            self.started_at = str(existing_policy.get("started_at_utc", utc_now()))
            self.initial_validation_lines = int(
                existing_policy.get("initial_validation_lines", 0)
            )

        self._write_policy()

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value is None else float(value)

    def request_stop(self, _signum: int, _frame: Any) -> None:
        self.stop_requested = True

    def _write_policy(self) -> None:
        payload: dict[str, Any] = {
            "policy": {
                "periodic_start_epoch": self.start_epoch,
                "periodic_interval_epochs": self.interval,
                "save_validation_best": True,
                "selection_metric": "mean of values in val_IoUs.txt",
            },
            "started_at_utc": self.started_at,
            "last_updated_at_utc": utc_now(),
            "initial_validation_lines": self.initial_validation_lines,
            "last_processed_validation_line": self.processed_line,
            "last_validation_epoch": self._last_validation_epoch(),
            "best_validation_epoch": self.best_epoch,
            "best_validation_miou": (
                None if self.best_miou is None else round(self.best_miou * 100.0, 3)
            ),
            "best_checkpoint": "best_val_miou.tar" if self.best_epoch else None,
            "periodic_checkpoints": self.periodic_checkpoints,
        }
        write_json_atomic(self.policy_path, payload)

    def _last_validation_epoch(self) -> int | None:
        return self.processed_line if self.processed_line else None

    def _archive_epoch(self, epoch: int, miou: float) -> bool:
        if not self.source_checkpoint.is_file():
            LOGGER.warning(
                "Epoch %d validation is complete but %s is missing; will retry",
                epoch,
                self.source_checkpoint,
            )
            return False

        try:
            source_epoch = checkpoint_epoch(self.source_checkpoint)
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.warning("Cannot validate %s yet: %s", self.source_checkpoint, error)
            return False
        if source_epoch != epoch:
            LOGGER.info(
                "Waiting for current checkpoint epoch %d (currently %d)",
                epoch,
                source_epoch,
            )
            return False

        if epoch >= self.start_epoch and (epoch - self.start_epoch) % self.interval == 0:
            filename = f"chkp_{epoch:04d}.tar"
            destination = self.checkpoint_dir / filename
            if destination.is_file():
                destination_epoch = checkpoint_epoch(destination)
                if destination_epoch != epoch:
                    raise ValueError(
                        "existing {} contains epoch {}, expected {}".format(
                            destination, destination_epoch, epoch
                        )
                    )
                LOGGER.info("Periodic checkpoint already exists: %s", destination)
            else:
                copy_checkpoint_atomic(self.source_checkpoint, destination)
                LOGGER.info(
                    "Archived epoch %d (val mIoU %.3f%%) to %s",
                    epoch,
                    miou * 100.0,
                    destination,
                )
            record = {
                "epoch": epoch,
                "miou": round(miou * 100.0, 3),
                "path": filename,
            }
            matching = [
                index
                for index, existing in enumerate(self.periodic_checkpoints)
                if existing.get("epoch") == epoch
            ]
            if matching:
                self.periodic_checkpoints[matching[0]] = record
            else:
                self.periodic_checkpoints.append(record)

        if epoch >= self.start_epoch and (
            self.best_miou is None or miou > self.best_miou
        ):
            destination = self.checkpoint_dir / "best_val_miou.tar"
            copy_checkpoint_atomic(self.source_checkpoint, destination)
            self.best_epoch = epoch
            self.best_miou = miou
            LOGGER.info(
                "Updated validation-best checkpoint at epoch %d (val mIoU %.3f%%)",
                epoch,
                miou * 100.0,
            )

        return True

    def process_new_records(self) -> None:
        records = read_validation_records(self.validation_path)
        new_records = [record for record in records if record.line_number > self.processed_line]
        for record in new_records:
            # A validation line is one completed epoch in this training loop.
            # If the monitor was offline, the current rolling checkpoint no
            # longer represents earlier missed lines, so do not mislabel it.
            if record.line_number > self.processed_line + 1:
                LOGGER.warning(
                    "Missed validation lines %d-%d; historical checkpoints cannot be reconstructed",
                    self.processed_line + 1,
                    record.line_number - 1,
                )
            if record.line_number >= self.start_epoch:
                if self._archive_epoch(record.line_number, record.miou):
                    self.processed_line = record.line_number
                    self._write_policy()
                else:
                    break
            else:
                self.processed_line = record.line_number
                self._write_policy()

    def run(self) -> None:
        LOGGER.info(
            "Monitoring %s from epoch %d every %d epochs; initial validation lines=%d",
            self.result_dir,
            self.start_epoch,
            self.interval,
            self.initial_validation_lines,
        )
        while not self.stop_requested:
            self.process_new_records()
            if self.final_epoch is not None and self.processed_line >= self.final_epoch:
                LOGGER.info("Reached final epoch %d; monitor exiting", self.final_epoch)
                break
            time.sleep(self.poll_seconds)
        self._write_policy()
        LOGGER.info("Checkpoint archive monitor stopped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive current_chkp.tar on validation progress without touching training."
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--start-epoch", type=int, default=100)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--final-epoch", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    checkpoint_dir = result_dir / "checkpoints"
    configure_logging(checkpoint_dir / LOG_FILENAME)
    try:
        monitor = ArchiveMonitor(
            result_dir=result_dir,
            start_epoch=args.start_epoch,
            interval=args.interval,
            final_epoch=args.final_epoch,
            poll_seconds=args.poll_seconds,
        )
    except (OSError, ValueError) as error:
        LOGGER.error("Cannot start checkpoint archive monitor: %s", error)
        return 2

    signal.signal(signal.SIGTERM, monitor.request_stop)
    signal.signal(signal.SIGINT, monitor.request_stop)
    try:
        monitor.run()
    except Exception:
        LOGGER.exception("Checkpoint archive monitor failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
