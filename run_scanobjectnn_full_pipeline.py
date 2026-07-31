#!/usr/bin/env python3
"""Finish ScanObjectNN experiments and test every saved epoch."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watch_scanobjectnn_finalize_shutdown import (
    archive_checkpoint,
    contains_finished_marker,
    inspect_checkpoint,
    latest_training_epoch,
    read_proc_identity,
)


MILESTONES = [190, 200, 210, 220, 230, 240, 250]


class PipelineInterrupted(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_event(state_dir: Path, event: str, **fields: Any) -> None:
    record = {"timestamp": utc_now(), "event": event, **fields}
    with (state_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        stream.flush()


def state_update(state_dir: Path, status: str, **fields: Any) -> None:
    atomic_write_json(
        state_dir / "state.json",
        {"status": status, "updated_at": utc_now(), "pipeline_pid": os.getpid(), **fields},
    )
    append_event(state_dir, status, **fields)


def ensure_disk_space(path: Path, minimum_gib: float) -> None:
    free = os.statvfs(path).f_bavail * os.statvfs(path).f_frsize
    free_gib = free / (1024**3)
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"only {free_gib:.2f} GiB free at {path}; need at least {minimum_gib:.2f} GiB"
        )


def archive_due_checkpoints(
    result_dir: Path,
    checkpoint_python: Path,
    validator_script: Path,
    training_epoch: int | None,
    archived: set[int],
) -> None:
    if training_epoch is None:
        return
    checkpoint_dir = result_dir / "checkpoints"
    current = checkpoint_dir / "current_chkp.tar"
    for milestone in MILESTONES:
        if milestone in archived or training_epoch < milestone:
            continue
        ok, detail = archive_checkpoint(
            current,
            checkpoint_dir,
            milestone,
            checkpoint_python,
            validator_script,
        )
        if ok:
            archived.add(milestone)
            continue
        if training_epoch > milestone:
            raise RuntimeError(
                f"missed exact checkpoint {milestone} in {result_dir}: {detail}"
            )


def existing_archives(
    result_dir: Path, checkpoint_python: Path, validator_script: Path
) -> set[int]:
    archived: set[int] = set()
    for milestone in MILESTONES:
        path = result_dir / "checkpoints" / f"chkp_exact_{milestone:04d}.tar"
        if path.is_file() and inspect_checkpoint(checkpoint_python, validator_script, path) == milestone:
            archived.add(milestone)
    return archived


def monitor_existing_training(args: argparse.Namespace, state_dir: Path) -> None:
    result_dir = args.current_result.resolve()
    archived = existing_archives(result_dir, args.python.resolve(), args.validator_script.resolve())
    dead_samples = 0
    state_update(
        state_dir,
        "monitoring_current_training",
        training_pid=args.current_pid,
        result_dir=str(result_dir),
        archived_epochs=sorted(archived),
    )
    while dead_samples < 3:
        identity = read_proc_identity(args.current_pid)
        alive = bool(
            identity is not None and identity["start_ticks"] == args.current_start_ticks
        )
        if alive and str(result_dir) not in identity["cmdline"]:
            raise RuntimeError("current training command no longer matches its result directory")
        dead_samples = 0 if alive else dead_samples + 1
        epoch = latest_training_epoch(result_dir / "training.txt")
        archive_due_checkpoints(
            result_dir,
            args.python.resolve(),
            args.validator_script.resolve(),
            epoch,
            archived,
        )
        state_update(
            state_dir,
            "monitoring_current_training" if alive else "confirming_current_stopped",
            training_pid=args.current_pid,
            latest_training_epoch=epoch,
            archived_epochs=sorted(archived),
            dead_samples=dead_samples,
        )
        if dead_samples < 3:
            time.sleep(args.poll_seconds)

    archive_due_checkpoints(
        result_dir,
        args.python.resolve(),
        args.validator_script.resolve(),
        250,
        archived,
    )
    final_checkpoint = result_dir / "checkpoints/current_chkp.tar"
    final_epoch = inspect_checkpoint(
        args.python.resolve(), args.validator_script.resolve(), final_checkpoint
    )
    if final_epoch != 250 or archived != set(MILESTONES):
        raise RuntimeError(
            f"current training ended with epoch={final_epoch}, archived={sorted(archived)}"
        )
    if not contains_finished_marker(args.current_console.resolve()):
        raise RuntimeError("current training ended without Finished Training marker")
    state_update(
        state_dir,
        "current_training_completed",
        final_epoch=final_epoch,
        archived_epochs=sorted(archived),
    )


def run_full_tests(
    args: argparse.Namespace,
    state_dir: Path,
    experiments: list[Path],
    label: str,
) -> None:
    ensure_disk_space(args.project_dir, args.minimum_free_gib)
    command = [
        str(args.python.resolve()),
        str(args.batch_test_script.resolve()),
        "--dataset-path",
        str(args.dataset_path.resolve()),
        "--python",
        str(args.python.resolve()),
        "--output-root",
        str((args.test_output.resolve() / label)),
        "--max-votes",
        "10",
    ]
    for experiment in experiments:
        command.extend(["--experiment", str(experiment.resolve())])
    log_path = state_dir / f"{label}.console.log"
    state_update(
        state_dir,
        "testing_checkpoints",
        label=label,
        experiments=[str(path.resolve()) for path in experiments],
        console_log=str(log_path),
    )
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=args.project_dir,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with code {completed.returncode}; see {log_path}")
    state_update(state_dir, "checkpoint_tests_completed", label=label)


def launch_and_monitor_standalone(args: argparse.Namespace, state_dir: Path) -> None:
    result_dir = args.standalone_result.resolve()
    if result_dir.exists():
        raise FileExistsError(f"standalone result directory already exists: {result_dir}")
    ensure_disk_space(args.project_dir, args.minimum_free_gib)
    command = [
        str(args.python.resolve()),
        "experiments/ScanObjectNN/train_ScanObj.py",
        "--dataset_path",
        str(args.dataset_path.resolve()),
        "--log_path",
        str(result_dir),
        "--seed",
        "57106803",
        "--kp_mode",
        "kpconvx",
        "--fa_enabled",
        "0",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["PYTHONPATH"] = str(args.kpconvx_root.resolve()) + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    console = args.standalone_console.resolve()
    console.parent.mkdir(parents=True, exist_ok=True)
    with console.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=args.kpconvx_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        state_update(
            state_dir,
            "training_standalone_kpconvx",
            training_pid=process.pid,
            result_dir=str(result_dir),
            console_log=str(console),
            seed=57106803,
            kp_mode="kpconvx",
            fa_enabled=False,
        )
        archived: set[int] = set()
        while process.poll() is None:
            epoch = latest_training_epoch(result_dir / "training.txt")
            archive_due_checkpoints(
                result_dir,
                args.python.resolve(),
                args.validator_script.resolve(),
                epoch,
                archived,
            )
            state_update(
                state_dir,
                "training_standalone_kpconvx",
                training_pid=process.pid,
                latest_training_epoch=epoch,
                archived_epochs=sorted(archived),
            )
            time.sleep(args.poll_seconds)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"standalone KPConvX exited with code {return_code}; see {console}")
    archive_due_checkpoints(
        result_dir,
        args.python.resolve(),
        args.validator_script.resolve(),
        250,
        archived,
    )
    final_checkpoint = result_dir / "checkpoints/current_chkp.tar"
    final_epoch = inspect_checkpoint(
        args.python.resolve(), args.validator_script.resolve(), final_checkpoint
    )
    if final_epoch != 250 or archived != set(MILESTONES):
        raise RuntimeError(
            f"standalone training ended with epoch={final_epoch}, archived={sorted(archived)}"
        )
    if not contains_finished_marker(console):
        raise RuntimeError("standalone training ended without Finished Training marker")
    state_update(
        state_dir,
        "standalone_kpconvx_completed",
        final_epoch=final_epoch,
        archived_epochs=sorted(archived),
    )


def claim_and_shutdown(
    args: argparse.Namespace,
    state_dir: Path,
    reason: str,
    error: str | None = None,
) -> None:
    if not args.shutdown_on_finish:
        append_event(state_dir, "shutdown_skipped", reason=reason, error=error)
        return

    marker = state_dir / "shutdown_started.json"
    claim = {
        "claimed_at": utc_now(),
        "pipeline_pid": os.getpid(),
        "reason": reason,
        "error": error,
        "shutdown_command": str(args.shutdown_command.resolve()),
    }
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        append_event(state_dir, "duplicate_shutdown_rejected", reason=reason)
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(claim, stream, ensure_ascii=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    state_update(state_dir, "shutdown_started", reason=reason, error=error)
    subprocess.run(["sync"], check=False)
    with (state_dir / "shutdown-command.log").open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(args.shutdown_command.resolve())],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    atomic_write_json(
        state_dir / "shutdown_finished.json",
        {**claim, "finished_at": utc_now(), "return_code": completed.returncode},
    )


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    kpconvx_root = project / "Standalone/KPConvX"
    results = kpconvx_root / "results"
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-pid", type=int, required=True)
    parser.add_argument("--current-start-ticks", type=int, required=True)
    parser.add_argument("--current-result", type=Path, required=True)
    parser.add_argument("--current-console", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=project)
    parser.add_argument("--kpconvx-root", type=Path, default=kpconvx_root)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--python", type=Path, default=Path("/root/autodl-tmp/envs/pointcept/bin/python")
    )
    parser.add_argument(
        "--validator-script", type=Path, default=project / "watch_scanobjectnn_finalize_shutdown.py"
    )
    parser.add_argument(
        "--batch-test-script", type=Path, default=project / "test_scanobjectnn_checkpoints.py"
    )
    parser.add_argument("--baseline-experiment", action="append", type=Path, required=True)
    parser.add_argument(
        "--standalone-result",
        type=Path,
        default=results / "ScanObjectNN_KPConvX-L-official-seed57106803-4080S-32G",
    )
    parser.add_argument(
        "--standalone-console",
        type=Path,
        default=project / "Standalone/KPConvX-L-official-seed57106803-4080S-32G.console.log",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=results / "ScanObjectNN_full_checkpoint_tests-seed57106803",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=results / ".scanobjectnn_full_pipeline_seed57106803",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--minimum-free-gib", type=float, default=3.0)
    parser.add_argument(
        "--shutdown-on-finish",
        action="store_true",
        help="Shut down the server after success or failure (disabled by default).",
    )
    parser.add_argument("--shutdown-command", type=Path, default=Path("/usr/bin/shutdown"))
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> None:
    identity = read_proc_identity(args.current_pid)
    if identity is None or identity["start_ticks"] != args.current_start_ticks:
        raise RuntimeError("current training PID identity does not match")
    if str(args.current_result.resolve()) not in identity["cmdline"]:
        raise RuntimeError("current training command does not match result directory")
    required_files = [args.python, args.validator_script, args.batch_test_script]
    for path in required_files:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    if not os.access(args.python.resolve(), os.X_OK):
        raise PermissionError(args.python)
    if not args.dataset_path.resolve().is_dir():
        raise FileNotFoundError(args.dataset_path)
    if args.shutdown_on_finish:
        if not args.shutdown_command.resolve().is_file() or not os.access(
            args.shutdown_command.resolve(), os.X_OK
        ):
            raise PermissionError(args.shutdown_command)
    for experiment in args.baseline_experiment:
        if not (experiment.resolve() / "checkpoints").is_dir():
            raise FileNotFoundError(experiment)
    if args.standalone_result.resolve().exists():
        raise FileExistsError(
            f"standalone result directory already exists: {args.standalone_result.resolve()}"
        )
    ensure_disk_space(args.project_dir.resolve(), args.minimum_free_gib)


def main() -> int:
    args = parse_args()
    args.project_dir = args.project_dir.resolve()
    args.kpconvx_root = args.kpconvx_root.resolve()
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    preflight(args)
    if args.preflight_only:
        print("preflight ok")
        return 0

    lock = (args.state_dir / "pipeline.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another pipeline process already owns the state directory", file=sys.stderr)
        return 3
    if args.shutdown_on_finish and (args.state_dir / "shutdown_started.json").exists():
        print("shutdown was already claimed for this pipeline", file=sys.stderr)
        return 4

    def interrupt(signum: int, _frame: Any) -> None:
        raise PipelineInterrupted(f"received signal {signum}")

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    state_update(args.state_dir, "armed", seed=57106803)
    try:
        monitor_existing_training(args, args.state_dir)
        run_full_tests(
            args,
            args.state_dir,
            [path.resolve() for path in args.baseline_experiment],
            "three_experiments",
        )
        launch_and_monitor_standalone(args, args.state_dir)
        run_full_tests(
            args,
            args.state_dir,
            [args.standalone_result.resolve()],
            "standalone_kpconvx",
        )
        state_update(args.state_dir, "pipeline_completed")
    except BaseException as error:
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        state_update(args.state_dir, "pipeline_failed", error=str(error), traceback=detail)
        claim_and_shutdown(args, args.state_dir, "pipeline_failed", error=str(error))
        return 1

    claim_and_shutdown(args, args.state_dir, "pipeline_completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
