#!/usr/bin/env python3
"""Run the post-L2 evaluation, L0 FastAdapter training, and final shutdown.

The process is intended to run detached in a screen session.  It treats a
training/evaluation error as a terminal result and schedules the explicitly
requested machine shutdown.  Checkpoint evaluation is deduplicated by SHA256,
so aliases such as ``current_chkp.tar`` and ``chkp_0250.tar`` are tested once.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def checkpoint_epoch(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError(f"checkpoint has no integer epoch: {path}")
    return epoch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_miou(report: Path) -> str:
    text = report.read_text(encoding="utf-8", errors="replace")
    last_vote = text.rsplit("Vote ", 1)[-1]
    rows = re.findall(r"^\|\s*([0-9]+(?:\.[0-9]+)?)\s+\|", last_vote, flags=re.M)
    return rows[-1] if rows else "NA"


def latest_test_dir(result_dir: Path, before: set[str]) -> str:
    test_root = result_dir / "test"
    candidates = {
        path.name
        for path in test_root.glob("test_*")
        if path.is_dir() and path.name not in before
    }
    return sorted(candidates, key=lambda name: int(name.rsplit("_", 1)[-1]))[-1] if candidates else ""


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.l2_result = args.l2_result.resolve()
        self.l2_checkpoints = self.l2_result / "checkpoints"
        self.l2_console = args.l2_console.resolve()
        self.eval_dir = args.eval_dir.resolve()
        self.eval_logs = self.eval_dir / "logs"
        self.state_dir = args.state_dir.resolve()
        self.state_path = self.state_dir / "state.json"
        self.events_path = self.state_dir / "events.jsonl"
        self.stop_requested = False
        self.started_at = utc_now()

    def event(self, name: str, **fields: Any) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": utc_now(), "event": name, **fields}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            stream.flush()
        self.log(f"{name}: {fields}")

    def log(self, message: str) -> None:
        print(f"[{utc_now()}] {message}", flush=True)

    def set_state(self, **fields: Any) -> None:
        state = {
            "started_at_utc": self.started_at,
            "updated_at_utc": utc_now(),
            "l2_result": str(self.l2_result),
            **fields,
        }
        atomic_json(self.state_path, state)

    @staticmethod
    def proc_identity(pid: int) -> dict[str, Any] | None:
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (FileNotFoundError, PermissionError):
            return None
        close_paren = stat_text.rfind(")")
        if close_paren < 0:
            return None
        fields = stat_text[close_paren + 2 :].split()
        if len(fields) < 20 or fields[0] == "Z":
            return None
        return {
            "start_ticks": int(fields[19]),
            "cmdline": cmdline.replace(b"\0", b" ").decode("utf-8", errors="replace").strip(),
        }

    def wait_for_l2(self) -> bool:
        identity = self.proc_identity(self.args.l2_pid)
        if identity is None or identity["start_ticks"] != self.args.l2_start_ticks:
            raise RuntimeError("L2 process is not the expected live process")
        if "train_S3DIS.py" not in identity["cmdline"]:
            raise RuntimeError("L2 process command identity check failed")

        self.event("l2_wait_started", pid=self.args.l2_pid)
        last_reported = -1
        while not self.stop_requested:
            current = self.proc_identity(self.args.l2_pid)
            alive = bool(
                current is not None
                and current["start_ticks"] == self.args.l2_start_ticks
            )
            try:
                epoch = checkpoint_epoch(self.l2_checkpoints / "current_chkp.tar")
            except (OSError, RuntimeError, ValueError):
                epoch = None
            if epoch is not None and epoch != last_reported:
                last_reported = epoch
                self.set_state(status="waiting_l2", latest_l2_epoch=epoch)
                self.log(f"L2 current checkpoint epoch={epoch}")
            if not alive:
                for _ in range(18):
                    try:
                        epoch = checkpoint_epoch(
                            self.l2_checkpoints / "current_chkp.tar"
                        )
                    except (OSError, RuntimeError, ValueError):
                        epoch = None
                    marker = "Finished Training" in self.l2_console.read_text(
                        encoding="utf-8", errors="replace"
                    ) if self.l2_console.is_file() else False
                    if epoch == self.args.final_epoch and (
                        marker or not (self.l2_result / "running_PID.txt").exists()
                    ):
                        self.event("l2_finished", epoch=epoch, console_marker=marker)
                        self.wait_for_l2_archive()
                        return True
                    time.sleep(5)
                raise RuntimeError(
                    f"L2 stopped before a verified epoch {self.args.final_epoch} completion (last={epoch})"
                )
            time.sleep(self.args.poll_seconds)
        raise RuntimeError("pipeline stop requested while waiting for L2")

    def wait_for_l2_archive(self) -> None:
        policy_path = self.l2_checkpoints / "checkpoint_policy.json"
        for _ in range(24):
            try:
                policy = json.loads(policy_path.read_text(encoding="utf-8"))
                processed = int(policy.get("last_processed_validation_line", 0))
            except (OSError, ValueError, json.JSONDecodeError):
                processed = 0
            if processed >= self.args.final_epoch:
                self.event("l2_checkpoint_archive_finished", epoch=processed)
                return
            time.sleep(5)
        raise RuntimeError(
            "L2 checkpoint archive did not confirm final epoch {:d}".format(
                self.args.final_epoch
            )
        )

    def evaluate_checkpoints(self) -> bool:
        weights = sorted(self.l2_checkpoints.glob("*.tar"))
        if not weights:
            raise RuntimeError("no L2 checkpoints found")

        groups: dict[str, list[Path]] = {}
        for weight in weights:
            groups.setdefault(sha256(weight), []).append(weight)
        representatives = []
        for digest, aliases in groups.items():
            aliases.sort(key=self.weight_priority)
            representatives.append((aliases[0], aliases, digest))
        representatives.sort(key=lambda item: item[0].name)

        self.eval_logs.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.eval_dir / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "checkpoint", "aliases", "internal_epoch", "sha256", "return_code",
                "test_dir", "full_miou", "report_path", "log_path",
            ])
            self.set_state(status="evaluating_l2", unique_weights=len(representatives), total_files=len(weights))
            all_ok = True
            for index, (weight, aliases, digest) in enumerate(representatives, start=1):
                if self.stop_requested:
                    raise RuntimeError("pipeline stop requested during evaluation")
                internal_epoch = checkpoint_epoch(weight)
                log_path = self.eval_logs / f"{weight.stem}.log"
                before = {path.name for path in (self.l2_result / "test").glob("test_*") if path.is_dir()}
                command = [
                    str(self.args.python_bin), "experiments/S3DIS/test_S3DIS.py",
                    "--dataset_path", str(self.args.dataset_path),
                    "--log_path", str(self.l2_result),
                    "--weight_path", str(weight),
                ]
                self.log(f"Evaluating {index}/{len(representatives)}: {weight.name} epoch={internal_epoch}")
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        cwd=self.args.project_dir,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        env=self.test_environment(),
                    )
                test_dir = latest_test_dir(self.l2_result, before)
                report = self.l2_result / "test" / test_dir / "report.txt" if test_dir else Path()
                miou = extract_miou(report) if report.is_file() else "NA"
                ok = completed.returncode == 0 and report.is_file()
                all_ok = all_ok and ok
                writer.writerow([
                    weight.name,
                    json.dumps([item.name for item in aliases], ensure_ascii=True),
                    internal_epoch,
                    digest,
                    completed.returncode,
                    test_dir,
                    miou,
                    str(report) if report.is_file() else "",
                    str(log_path),
                ])
                stream.flush()
                self.event(
                    "l2_checkpoint_evaluated",
                    checkpoint=weight.name,
                    aliases=[item.name for item in aliases],
                    internal_epoch=internal_epoch,
                    return_code=completed.returncode,
                    report_found=report.is_file(),
                )
            (self.eval_dir / "evaluation_complete").write_text("\n", encoding="ascii")
            self.set_state(status="l2_evaluation_finished", evaluation_ok=all_ok)
            return all_ok

    @staticmethod
    def weight_priority(path: Path) -> tuple[int, str]:
        if re.fullmatch(r"chkp_\d+", path.stem):
            return (0, path.name)
        if path.stem == "best_val_miou":
            return (1, path.name)
        if path.stem == "current_chkp":
            return (2, path.name)
        return (3, path.name)

    def test_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update({
            "PYTHONPATH": str(self.args.project_dir),
            "OMP_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        return environment

    def train_fastadapter(self) -> bool:
        result_dir = self.args.fastadapter_result.resolve()
        if result_dir.exists():
            raise RuntimeError(f"FastAdapter result already exists: {result_dir}")
        console_log = result_dir.with_name(result_dir.name + ".console.log")
        environment = os.environ.copy()
        environment.update({
            "DATASET_PATH": str(self.args.dataset_path),
            "RESULT_ROOT": str(self.args.project_dir / "results"),
            "LOG_PATH": str(result_dir),
            "PYTHON_BIN": str(self.args.python_bin),
            "SEED": "57106803",
            "FA_ENABLED": "1",
            "OMP_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        self.set_state(status="training_l0_fastadapter", result_dir=str(result_dir))
        self.event("l0_fastadapter_started", result_dir=str(result_dir))
        monitor_process: subprocess.Popen[Any] | None = None
        try:
            with console_log.open("w", encoding="utf-8") as log:
                training_process = subprocess.Popen(
                    ["bash", str(self.args.fastadapter_script)],
                    cwd=self.args.project_dir.parent,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                while training_process.poll() is None:
                    if self.stop_requested:
                        training_process.terminate()
                        raise RuntimeError("pipeline stop requested during L0 FastAdapter training")
                    if monitor_process is None and (result_dir / "parameters.json").is_file():
                        monitor_process = subprocess.Popen(
                            [
                                str(self.args.python_bin),
                                str(self.args.checkpoint_monitor),
                                "--result-dir", str(result_dir),
                                "--start-epoch", "100",
                                "--interval", "10",
                                "--final-epoch", str(self.args.fastadapter_final_epoch),
                                "--poll-seconds", "15",
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT,
                            env=environment,
                        )
                        self.event("l0_checkpoint_monitor_started", pid=monitor_process.pid)
                    if monitor_process is not None and monitor_process.poll() not in (None, 0):
                        training_process.terminate()
                        raise RuntimeError(
                            "L0 checkpoint monitor failed with return code {:d}".format(
                                monitor_process.returncode
                            )
                        )
                    time.sleep(min(5.0, self.args.poll_seconds))
                training_return_code = training_process.returncode
        except BaseException:
            if 'training_process' in locals() and training_process.poll() is None:
                training_process.terminate()
                training_process.wait(timeout=30)
            if monitor_process is not None and monitor_process.poll() is None:
                monitor_process.terminate()
                monitor_process.wait(timeout=30)
            raise

        if monitor_process is not None:
            try:
                monitor_return_code = monitor_process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                monitor_process.terminate()
                monitor_return_code = monitor_process.wait(timeout=30)
            if monitor_return_code != 0:
                self.event(
                    "l0_checkpoint_monitor_failed",
                    return_code=monitor_return_code,
                )
                return False
        else:
            self.event("l0_checkpoint_monitor_missing")
            return False
        checkpoint = result_dir / "checkpoints" / "current_chkp.tar"
        epoch = checkpoint_epoch(checkpoint) if checkpoint.is_file() else None
        marker = "Finished Training" in console_log.read_text(encoding="utf-8", errors="replace")
        ok = training_return_code == 0 and marker and epoch == self.args.fastadapter_final_epoch
        self.event(
            "l0_fastadapter_finished",
            return_code=training_return_code,
            internal_epoch=epoch,
            console_marker=marker,
            success=ok,
        )
        return ok

    def shutdown(self, reason: str, return_code: int) -> int:
        self.set_state(status="shutdown_scheduled", reason=reason, return_code=return_code)
        self.event("shutdown_scheduled", reason=reason, return_code=return_code)
        subprocess.run(["sync"], check=False)
        time.sleep(self.args.shutdown_delay_seconds)
        claim = self.state_dir / "shutdown_started.json"
        try:
            descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            self.log("shutdown already claimed by another process")
            return return_code
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"reason": reason, "claimed_at_utc": utc_now()}, stream)
            stream.write("\n")
        if self.args.dry_run:
            self.event("shutdown_dry_run")
            return return_code
        log_path = self.state_dir / "shutdown-command.log"
        with log_path.open("a", encoding="utf-8") as log:
            shutdown_result = subprocess.run(
                [str(self.args.shutdown_command)],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        self.event("shutdown_command_returned", return_code=shutdown_result.returncode)
        return return_code

    def run(self) -> int:
        try:
            self.wait_for_l2()
            if not self.evaluate_checkpoints():
                raise RuntimeError("one or more L2 checkpoint evaluations failed")
            if not self.train_fastadapter():
                raise RuntimeError("L0 FastAdapter training failed or did not reach the final epoch")
            self.set_state(status="all_tasks_completed")
            self.shutdown("all_tasks_completed", 0)
            return 0
        except BaseException as error:
            self.log(f"pipeline failed: {error}")
            self.event("pipeline_failed", error=repr(error))
            self.shutdown("pipeline_failed", 1)
            return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate L2, train L0 FastAdapter, then shut down.")
    parser.add_argument("--l2-pid", type=int, required=True)
    parser.add_argument("--l2-start-ticks", type=int, required=True)
    parser.add_argument("--l2-result", type=Path, required=True)
    parser.add_argument("--l2-console", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--fastadapter-script", type=Path, required=True)
    parser.add_argument("--checkpoint-monitor", type=Path, required=True)
    parser.add_argument("--fastadapter-result", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--final-epoch", type=int, default=250)
    parser.add_argument("--fastadapter-final-epoch", type=int, default=250)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--shutdown-delay-seconds", type=float, default=60.0)
    parser.add_argument("--shutdown-command", type=Path, default=Path("/usr/bin/shutdown"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.shutdown_delay_seconds < 0:
        parser.error("poll seconds must be positive and shutdown delay cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    pipeline = Pipeline(args)
    signal.signal(signal.SIGTERM, lambda signum, frame: setattr(pipeline, "stop_requested", True))
    signal.signal(signal.SIGINT, lambda signum, frame: setattr(pipeline, "stop_requested", True))
    return pipeline.run()


if __name__ == "__main__":
    raise SystemExit(main())
