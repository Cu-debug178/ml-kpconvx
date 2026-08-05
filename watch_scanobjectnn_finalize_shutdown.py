#!/usr/bin/env python3
"""Archive exact ScanObjectNN milestones and optionally shut down after training."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def create_json_once(path: Path, value: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return True


def append_event(state_dir: Path, event: str, **fields: Any) -> None:
    record = {"timestamp": utc_now(), "event": event, **fields}
    with (state_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        stream.flush()


def read_proc_identity(pid: int) -> dict[str, Any] | None:
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        raw_cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    close_paren = raw_stat.rfind(")")
    if close_paren < 0:
        return None
    fields = raw_stat[close_paren + 2 :].split()
    if len(fields) < 20 or fields[0] == "Z":
        return None
    return {
        "pid": pid,
        "state": fields[0],
        "start_ticks": int(fields[19]),
        "cmdline": raw_cmdline.replace(b"\0", b" ")
        .decode("utf-8", errors="replace")
        .strip(),
    }


def latest_training_epoch(path: Path) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        fields = line.split()
        if fields and fields[0].isdigit():
            return int(fields[0])
    return None


def contains_finished_marker(path: Path) -> bool:
    try:
        return "Finished Training" in path.read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return False


def file_signature(path: Path) -> tuple[int, int] | None:
    """Return a cheap signature for detecting training-log progress."""
    try:
        stat_result = path.stat()
    except OSError:
        return None
    return stat_result.st_size, stat_result.st_mtime_ns


def gpu_utilization() -> float | None:
    """Read GPU utilization without treating a failed query as idle."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            return None
        value = completed.stdout.strip().splitlines()[0].strip()
        return float(value)
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def inspect_checkpoint(checkpoint_python: Path, script: Path, path: Path) -> int:
    completed = subprocess.run(
        [str(checkpoint_python), str(script), "--inspect-checkpoint", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"checkpoint validation failed: {detail}")
    try:
        return int(completed.stdout.strip())
    except ValueError as error:
        raise RuntimeError(
            f"checkpoint validator returned an invalid epoch: {completed.stdout!r}"
        ) from error


def archive_checkpoint(
    source: Path,
    checkpoint_dir: Path,
    expected_epoch: int,
    checkpoint_python: Path,
    script: Path,
) -> tuple[bool, str]:
    destination = checkpoint_dir / f"chkp_exact_{expected_epoch:04d}.tar"
    if destination.is_file():
        try:
            stored_epoch = inspect_checkpoint(checkpoint_python, script, destination)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            return False, f"existing archive is invalid: {error}"
        if stored_epoch == expected_epoch:
            return True, "already archived"

    temporary = checkpoint_dir / f".{destination.name}.{os.getpid()}.tmp"
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        stored_epoch = inspect_checkpoint(checkpoint_python, script, temporary)
        if stored_epoch != expected_epoch:
            return False, f"current checkpoint contains epoch {stored_epoch}"
        os.replace(temporary, destination)
        directory_fd = os.open(checkpoint_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True, "archived and validated"
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        return False, str(error)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def inspect_mode(path: Path) -> int:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
        epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ValueError("checkpoint does not contain an integer epoch")
        print(epoch)
        return 0
    except Exception as error:  # The parent records the exact torch/load failure.
        print(str(error), file=sys.stderr)
        return 2


def parse_milestones(value: str) -> list[int]:
    try:
        milestones = sorted({int(item) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("milestones must be comma-separated integers") from error
    if not milestones or milestones[0] <= 0:
        raise argparse.ArgumentTypeError("milestones must contain positive integers")
    return milestones


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-ticks", type=int, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-python", type=Path, required=True)
    parser.add_argument("--expect-command", action="append", default=[])
    parser.add_argument(
        "--milestones", type=parse_milestones, default=parse_milestones("190,200,210,220,230,240,250")
    )
    parser.add_argument("--final-epoch", type=int, default=250)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--dead-confirmations", type=int, default=3)
    parser.add_argument("--shutdown-delay-seconds", type=float, default=60.0)
    parser.add_argument("--shutdown-command", type=Path, default=Path("/usr/bin/shutdown"))
    parser.add_argument(
        "--shutdown-on-finish",
        action="store_true",
        help="Shut down after monitoring completes (disabled by default).",
    )
    parser.add_argument(
        "--shutdown-on-interruption",
        action="store_true",
        help="Also shut down after a verified process interruption or idle timeout.",
    )
    parser.add_argument(
        "--gpu-idle-shutdown-seconds",
        type=float,
        default=0.0,
        help="Require this much stalled-log time with low GPU utilization before an idle shutdown.",
    )
    parser.add_argument(
        "--gpu-idle-utilization-threshold",
        type=float,
        default=5.0,
        help="GPU utilization percentage at or below which the idle timer may run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.pid <= 1 or args.start_ticks <= 0:
        parser.error("pid and start-ticks must identify a non-system process")
    if args.final_epoch <= 0:
        parser.error("final-epoch must be positive")
    if args.interval_seconds <= 0 or args.dead_confirmations <= 0:
        parser.error("interval and dead-confirmations must be positive")
    if args.shutdown_delay_seconds < 0:
        parser.error("shutdown delay cannot be negative")
    if args.gpu_idle_shutdown_seconds < 0:
        parser.error("GPU idle timeout cannot be negative")
    if not 0 <= args.gpu_idle_utilization_threshold <= 100:
        parser.error("GPU idle utilization threshold must be between 0 and 100")
    return args


def monitor(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    result_dir = args.result_dir.resolve()
    checkpoint_dir = result_dir / "checkpoints"
    current_checkpoint = checkpoint_dir / "current_chkp.tar"
    training_log = result_dir / "training.txt"
    console_log = args.console_log.resolve()
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state_dir, 0o700)

    lock_stream = (state_dir / "monitor.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another finalizer owns {state_dir}", file=sys.stderr)
        return 3
    if args.shutdown_on_finish and (state_dir / "shutdown_started.json").exists():
        print("The one-shot shutdown was already claimed", file=sys.stderr)
        return 4

    identity = read_proc_identity(args.pid)
    if identity is None:
        print(f"Training PID {args.pid} is not alive; refusing to arm", file=sys.stderr)
        return 2
    if identity["start_ticks"] != args.start_ticks:
        print("Training PID start time does not match; refusing to arm", file=sys.stderr)
        return 2
    missing_expectations = [
        value for value in args.expect_command if value not in identity["cmdline"]
    ]
    if missing_expectations:
        print(
            "Training command check failed; missing: " + ", ".join(missing_expectations),
            file=sys.stderr,
        )
        return 2
    if not args.checkpoint_python.is_file() or not os.access(
        args.checkpoint_python, os.X_OK
    ):
        print(f"Checkpoint Python is not executable: {args.checkpoint_python}", file=sys.stderr)
        return 2
    if args.shutdown_on_finish and not args.dry_run and (
        not args.shutdown_command.is_file()
        or not os.access(args.shutdown_command, os.X_OK)
    ):
        print(f"Shutdown command is not executable: {args.shutdown_command}", file=sys.stderr)
        return 2

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    armed = {
        "status": "armed",
        "monitor_pid": os.getpid(),
        "armed_at": utc_now(),
        "training_pid": args.pid,
        "training_start_ticks": args.start_ticks,
        "training_cmdline": identity["cmdline"],
        "result_dir": str(result_dir),
        "milestones": args.milestones,
        "final_epoch": args.final_epoch,
        "shutdown_on_interruption": args.shutdown_on_interruption,
        "gpu_idle_shutdown_seconds": args.gpu_idle_shutdown_seconds,
        "gpu_idle_utilization_threshold": args.gpu_idle_utilization_threshold,
        "dry_run": args.dry_run,
    }
    atomic_write_json(state_dir / "state.json", armed)
    (state_dir / "monitor.pid").write_text(f"{os.getpid()}\n", encoding="ascii")
    append_event(state_dir, "monitor_armed", training_pid=args.pid)

    archived: set[int] = set()
    dead_samples = 0
    last_heartbeat = 0.0
    last_log_signature = file_signature(training_log)
    last_log_progress = time.monotonic()
    stop_reason: str | None = None
    idle_log_signature: tuple[int, int] | None = None
    while not stop_requested:
        if (state_dir / "cancel").exists():
            cancelled = {**armed, "status": "cancelled", "finished_at": utc_now()}
            atomic_write_json(state_dir / "state.json", cancelled)
            append_event(state_dir, "monitor_cancelled_by_file")
            return 0

        identity_now = read_proc_identity(args.pid)
        alive = bool(
            identity_now is not None
            and identity_now["start_ticks"] == args.start_ticks
        )
        dead_samples = 0 if alive else dead_samples + 1
        training_epoch = latest_training_epoch(training_log)
        now = time.monotonic()
        current_log_signature = file_signature(training_log)
        if current_log_signature != last_log_signature:
            last_log_signature = current_log_signature
            last_log_progress = now
        stalled_seconds = max(0.0, now - last_log_progress)
        current_gpu_utilization = gpu_utilization()
        idle_timeout_triggered = (
            alive
            and args.gpu_idle_shutdown_seconds > 0
            and current_gpu_utilization is not None
            and current_gpu_utilization <= args.gpu_idle_utilization_threshold
            and stalled_seconds >= args.gpu_idle_shutdown_seconds
        )
        if idle_timeout_triggered:
            stop_reason = "gpu_idle_timeout"
            idle_log_signature = current_log_signature
            append_event(
                state_dir,
                "gpu_idle_timeout",
                gpu_utilization=current_gpu_utilization,
                stalled_seconds=stalled_seconds,
                latest_training_epoch=training_epoch,
            )

        for milestone in args.milestones:
            if milestone in archived or training_epoch is None or training_epoch < milestone:
                continue
            ok, detail = archive_checkpoint(
                current_checkpoint,
                checkpoint_dir,
                milestone,
                args.checkpoint_python.resolve(),
                script,
            )
            append_event(
                state_dir,
                "checkpoint_archive_checked",
                milestone=milestone,
                success=ok,
                detail=detail,
            )
            if ok:
                archived.add(milestone)

        status = {
            **armed,
            "status": (
                stop_reason
                if stop_reason is not None
                else "running" if alive else "confirming_stopped"
            ),
            "checked_at": utc_now(),
            "latest_training_epoch": training_epoch,
            "archived_epochs": sorted(archived),
            "dead_samples": dead_samples,
            "gpu_utilization": current_gpu_utilization,
            "training_log_stalled_seconds": stalled_seconds,
        }
        atomic_write_json(state_dir / "state.json", status)
        if now - last_heartbeat >= 300:
            append_event(
                state_dir,
                "heartbeat",
                alive=alive,
                latest_training_epoch=training_epoch,
                archived_epochs=sorted(archived),
                dead_samples=dead_samples,
                gpu_utilization=current_gpu_utilization,
                training_log_stalled_seconds=stalled_seconds,
            )
            last_heartbeat = now

        if idle_timeout_triggered:
            break
        if dead_samples >= args.dead_confirmations:
            stop_reason = "training_process_ended"
            break
        time.sleep(args.interval_seconds)

    if stop_requested:
        stopped = {**armed, "status": "cancelled_by_signal", "finished_at": utc_now()}
        atomic_write_json(state_dir / "state.json", stopped)
        append_event(state_dir, "monitor_cancelled_by_signal")
        return 130

    # The process is gone and console/checkpoint writes have had multiple polling
    # intervals to settle. Make one final attempt to preserve the last milestone.
    final_archive_ok = args.final_epoch in archived
    final_archive_detail = "already archived"
    if args.final_epoch in args.milestones and not final_archive_ok:
        final_archive_ok, final_archive_detail = archive_checkpoint(
            current_checkpoint,
            checkpoint_dir,
            args.final_epoch,
            args.checkpoint_python.resolve(),
            script,
        )
        if final_archive_ok:
            archived.add(args.final_epoch)
        append_event(
            state_dir,
            "final_checkpoint_archive_checked",
            milestone=args.final_epoch,
            success=final_archive_ok,
            detail=final_archive_detail,
        )

    normal_marker = contains_finished_marker(console_log)
    if stop_reason == "gpu_idle_timeout":
        result = "gpu_idle_timeout"
    else:
        result = (
            "completed"
            if normal_marker and final_archive_ok
            else "completed_but_final_checkpoint_invalid"
            if normal_marker
            else "interrupted_or_failed"
        )
    # Never power off after an interrupted or failed training process. The
    # shutdown option is reserved for a verified normal completion with the
    # requested final checkpoint archived successfully.
    shutdown_allowed = result == "completed" or (
        args.shutdown_on_interruption
        and result in {"interrupted_or_failed", "gpu_idle_timeout"}
    )
    if args.shutdown_on_finish and not shutdown_allowed:
        failed = {
            **armed,
            "status": result,
            "finished_at": utc_now(),
            "finished_marker": normal_marker,
            "final_archive_ok": final_archive_ok,
            "final_archive_detail": final_archive_detail,
            "archived_epochs": sorted(archived),
            "shutdown_skipped": True,
            "shutdown_skip_reason": "normal completion was not verified",
        }
        atomic_write_json(state_dir / "state.json", failed)
        append_event(state_dir, "shutdown_skipped", reason=result)
        return 1
    if not args.shutdown_on_finish:
        completed = {
            **armed,
            "status": result,
            "finished_at": utc_now(),
            "finished_marker": normal_marker,
            "final_archive_ok": final_archive_ok,
            "final_archive_detail": final_archive_detail,
            "archived_epochs": sorted(archived),
        }
        atomic_write_json(state_dir / "state.json", completed)
        append_event(state_dir, "monitor_completed", training_result=result)
        return 0 if result == "completed" else 1

    scheduled = {
        **armed,
        "status": "shutdown_delay",
        "training_result": result,
        "shutdown_trigger": stop_reason or "normal_completion",
        "finished_marker": normal_marker,
        "final_archive_ok": final_archive_ok,
        "final_archive_detail": final_archive_detail,
        "archived_epochs": sorted(archived),
        "shutdown_scheduled_at": utc_now(),
        "shutdown_delay_seconds": args.shutdown_delay_seconds,
    }
    atomic_write_json(state_dir / "state.json", scheduled)
    append_event(
        state_dir,
        "shutdown_scheduled",
        training_result=result,
        delay_seconds=args.shutdown_delay_seconds,
    )
    subprocess.run(["sync"], check=False)

    deadline = time.monotonic() + args.shutdown_delay_seconds
    while time.monotonic() < deadline:
        if (state_dir / "cancel").exists():
            cancelled = {**scheduled, "status": "cancelled_during_shutdown_delay"}
            atomic_write_json(state_dir / "state.json", cancelled)
            append_event(state_dir, "shutdown_cancelled_during_delay")
            return 0
        if stop_reason == "gpu_idle_timeout":
            resumed_signature = file_signature(training_log)
            resumed_gpu_utilization = gpu_utilization()
            if (
                resumed_signature != idle_log_signature
                or (
                    resumed_gpu_utilization is not None
                    and resumed_gpu_utilization > args.gpu_idle_utilization_threshold
                )
            ):
                cancelled = {
                    **scheduled,
                    "status": "cancelled_after_gpu_activity_resumed",
                    "resumed_gpu_utilization": resumed_gpu_utilization,
                }
                atomic_write_json(state_dir / "state.json", cancelled)
                append_event(
                    state_dir,
                    "shutdown_cancelled_after_gpu_activity_resumed",
                    gpu_utilization=resumed_gpu_utilization,
                )
                return 0
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

    claim = {
        "claimed_at": utc_now(),
        "monitor_pid": os.getpid(),
        "training_result": result,
        "dry_run": args.dry_run,
        "shutdown_command": str(args.shutdown_command.resolve()),
    }
    if not create_json_once(state_dir / "shutdown_started.json", claim):
        append_event(state_dir, "duplicate_shutdown_rejected")
        return 4

    append_event(state_dir, "shutdown_started", dry_run=args.dry_run)
    if args.dry_run:
        return_code = 0
    else:
        with (state_dir / "shutdown-command.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                [str(args.shutdown_command.resolve())],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return_code = completed.returncode
    finished = {**claim, "finished_at": utc_now(), "return_code": return_code}
    atomic_write_json(state_dir / "shutdown_finished.json", finished)
    atomic_write_json(
        state_dir / "state.json",
        {**scheduled, "status": "shutdown_finished", "return_code": return_code},
    )
    append_event(state_dir, "shutdown_finished", return_code=return_code)
    return return_code


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--inspect-checkpoint":
        return inspect_mode(Path(argv[1]))
    return monitor(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
