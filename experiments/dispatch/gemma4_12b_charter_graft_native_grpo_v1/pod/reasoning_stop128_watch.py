"""Fail-closed watcher that stops the phase-1 reasoning arms at checkpoint 128.

This utility is intentionally separate from the normal 256-step pipeline.  It
waits until each checkpoint is complete and stable on disk, interrupts only
that cell's trainer PID, inventories the resumable state, and then remains
alive as a token-bearing supervisor for an optional post-processing command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


CELLS = (
    "public_it-reasoning_grpo",
    "charter_graft_it-reasoning_grpo",
)
CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def import_environment(pid: int) -> None:
    """Import the pipeline's HF credentials without printing or persisting them."""

    payload = Path(f"/proc/{pid}/environ").read_bytes()
    inherited = {}
    for item in payload.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        inherited[key.decode(errors="surrogateescape")] = value.decode(
            errors="surrogateescape"
        )
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if inherited.get(key):
            os.environ[key] = inherited[key]
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        raise RuntimeError(f"PID {pid} did not provide a Hugging Face token")


def process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ""


def find_cell_pid(cell_root: Path) -> int | None:
    needle = f"output={cell_root}"
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        command = process_cmdline(int(entry.name))
        if (
            "run_cell.py" in command
            and "mode=reasoning" in command
            and needle in command
        ):
            matches.append(int(entry.name))
    if len(matches) > 1:
        raise RuntimeError(f"multiple trainer PIDs found for {cell_root}: {matches}")
    return matches[0] if matches else None


def checkpoint_snapshot(checkpoint: Path) -> dict[str, tuple[int, int]] | None:
    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if int(state["global_step"]) != 128:
            return None
        result = {}
        for name in CHECKPOINT_FILES:
            path = checkpoint / name
            stat = path.stat()
            if stat.st_size <= 0:
                return None
            result[name] = (stat.st_size, stat.st_mtime_ns)
        return result
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_cell(pid: int, cell_root: Path) -> list[dict[str, Any]]:
    expected = f"output={cell_root}"
    if expected not in process_cmdline(pid):
        raise RuntimeError(f"PID {pid} no longer belongs to {cell_root}")
    actions = []
    for sig, timeout in ((signal.SIGTERM, 5.0),):
        if not pid_alive(pid):
            break
        os.kill(pid, sig)
        actions.append({"signal": signal.Signals(sig).name, "sent_at": utc_now()})
        deadline = time.monotonic() + timeout
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.25)
    if pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
        actions.append({"signal": "SIGKILL", "sent_at": utc_now()})
        deadline = time.monotonic() + 10.0
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.25)
    if pid_alive(pid):
        raise RuntimeError(f"trainer PID {pid} survived termination")
    return actions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(checkpoint: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        result.append(
            {
                "path": str(path.relative_to(checkpoint)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return result


def run(args: argparse.Namespace) -> None:
    args.training_root = args.training_root.resolve()
    args.state_root = args.state_root.resolve()
    args.state_root.mkdir(parents=True, exist_ok=True)
    import_environment(args.token_source_pid)
    atomic_json(
        args.state_root / "STATE.json",
        {
            "status": "watching",
            "watcher_pid": os.getpid(),
            "training_root": str(args.training_root),
            "target_step": 128,
            "cells": list(CELLS),
            "started_at": utc_now(),
        },
    )
    completed: list[str] = []
    trackers: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        cell_root = args.training_root / "cells" / cell
        existing_marker = args.state_root / f"{cell}.json"
        if existing_marker.is_file():
            payload = json.loads(existing_marker.read_text())
            if (
                payload.get("status") == "stopped_at_checkpoint_128"
                and int(payload.get("trainer_global_step", -1)) == 128
            ):
                completed.append(cell)
                trackers[cell] = {
                    "cell_root": cell_root,
                    "checkpoint": cell_root / "train" / "trainer" / "checkpoint-128",
                    "pid": None,
                    "snapshot": checkpoint_snapshot(
                        cell_root / "train" / "trainer" / "checkpoint-128"
                    ),
                    "stable_since": None,
                }
                continue
        pid = find_cell_pid(cell_root)
        if pid is None:
            raise RuntimeError(f"no live reasoning trainer found for {cell}")
        trackers[cell] = {
            "cell_root": cell_root,
            "checkpoint": cell_root / "train" / "trainer" / "checkpoint-128",
            "pid": pid,
            "snapshot": None,
            "stable_since": None,
        }

    last_state_write = 0.0
    while len(completed) != len(CELLS):
        now = time.monotonic()
        for cell in CELLS:
            if cell in completed:
                continue
            tracker = trackers[cell]
            current = checkpoint_snapshot(tracker["checkpoint"])
            if current is not None and current == tracker["snapshot"]:
                tracker["stable_since"] = tracker["stable_since"] or now
            else:
                tracker["snapshot"] = current
                tracker["stable_since"] = now if current is not None else None
            stable_since = tracker["stable_since"]
            if stable_since is None or now - stable_since < args.stable_seconds:
                if not pid_alive(tracker["pid"]):
                    raise RuntimeError(
                        f"trainer PID {tracker['pid']} for {cell} exited before a stable "
                        "checkpoint 128"
                    )
                continue

            actions = stop_cell(tracker["pid"], tracker["cell_root"])
            marker = {
                "schema_version": 1,
                "status": "stopped_at_checkpoint_128",
                "cell": cell,
                "trainer_pid": tracker["pid"],
                "checkpoint": str(tracker["checkpoint"]),
                "trainer_global_step": 128,
                "termination_actions": actions,
                "checkpoint_files": inventory(tracker["checkpoint"]),
                "stopped_at": utc_now(),
            }
            atomic_json(args.state_root / f"{cell}.json", marker)
            completed.append(cell)

        if now - last_state_write >= 2.0:
            cell_states = {}
            for cell, tracker in trackers.items():
                stable_since = tracker["stable_since"]
                cell_states[cell] = {
                    "trainer_pid": tracker["pid"],
                    "checkpoint_complete": tracker["snapshot"] is not None,
                    "stable_seconds": (
                        round(now - stable_since, 2)
                        if stable_since is not None
                        else 0.0
                    ),
                    "stopped": cell in completed,
                }
            atomic_json(
                args.state_root / "STATE.json",
                {
                    "status": "waiting_for_checkpoint_128",
                    "watcher_pid": os.getpid(),
                    "cells": cell_states,
                    "completed_cells": completed,
                    "updated_at": utc_now(),
                },
            )
            last_state_write = now
        if len(completed) != len(CELLS):
            time.sleep(args.poll_seconds)
    atomic_json(
        args.state_root / "CUTOFF_DONE.json",
        {
            "schema_version": 1,
            "status": "complete",
            "target_step": 128,
            "cells": completed,
            "watcher_pid": os.getpid(),
            "completed_at": utc_now(),
        },
    )
    atomic_json(
        args.state_root / "STATE.json",
        {
            "status": "cutoff_complete_waiting_for_postprocess",
            "watcher_pid": os.getpid(),
            "completed_cells": completed,
            "updated_at": utc_now(),
        },
    )

    command_file = args.state_root / "POSTPROCESS_COMMAND.json"
    while not command_file.is_file():
        time.sleep(2.0)
    command_payload = json.loads(command_file.read_text())
    command = command_payload.get("argv")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise ValueError(f"invalid postprocess command in {command_file}")
    atomic_json(
        args.state_root / "STATE.json",
        {
            "status": "postprocess_running",
            "watcher_pid": os.getpid(),
            "argv": command,
            "updated_at": utc_now(),
        },
    )
    result = subprocess.run(command, check=False)
    atomic_json(
        args.state_root / "POSTPROCESS_EXIT.json",
        {
            "schema_version": 1,
            "status": "complete" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "completed_at": utc_now(),
        },
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--token-source-pid", type=int, required=True)
    parser.add_argument("--stable-seconds", type=float, default=3.0)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
