"""Wait for the local SFT grid, then set up and run its eval grid.

The queue runs on the same existing pod.  It performs no cloud API calls and
does not stop or delete the pod on either success or failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
SETUP = HERE / "setup_eval.sh"
GRID = HERE / "run_sft_eval_grid.py"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def process_alive(pid_file: Path | None) -> bool | None:
    if pid_file is None or not pid_file.is_file():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def wait_for_sft(args: argparse.Namespace) -> None:
    done = args.sft_root / "SFT_GRID_DONE.json"
    failure = args.sft_root / "SFT_GRID_FAILURE.json"
    last_report = 0.0
    while not done.is_file():
        if failure.is_file():
            raise RuntimeError(f"SFT grid failed; retained marker: {failure}")
        alive = process_alive(args.sft_supervisor_pid_file)
        if alive is False:
            # The SFT runner writes its marker immediately before exit.  A short
            # grace period avoids racing that final atomic rename.
            time.sleep(5)
            if not done.is_file() and not failure.is_file():
                raise RuntimeError(
                    "SFT supervisor exited without a success/failure marker; "
                    "retaining pod state for inspection"
                )
            continue
        now = time.monotonic()
        if now - last_report >= 300:
            print(
                f"[{utc_now()}] waiting for {done} (sft_supervisor_alive={alive})",
                flush=True,
            )
            last_report = now
        time.sleep(args.poll_seconds)
    payload = json.loads(done.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"invalid SFT completion marker: {done}")
    print(f"[{utc_now()}] SFT grid complete; beginning eval setup", flush=True)


def run(args: argparse.Namespace) -> None:
    args.sft_root = args.sft_root.resolve()
    args.output_root = args.output_root.resolve()
    status_root = args.status_root.resolve()
    status_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        status_root / "EVAL_QUEUE_STATE.json",
        {
            "schema_version": 1,
            "status": "waiting_for_sft",
            "topology": "same existing 4xA100 pod; one eval worker per SFT arm/GPU",
            "sft_root": str(args.sft_root),
            "output_root": str(args.output_root),
            "checkpoints": [128, 256, 512],
            "queued_at": utc_now(),
        },
    )
    try:
        wait_for_sft(args)
        atomic_json(
            status_root / "EVAL_QUEUE_STATE.json",
            {"status": "setting_up_eval_environment", "started_at": utc_now()},
        )
        subprocess.run([str(SETUP)], cwd=REPO_ROOT, check=True)
        eval_python = args.eval_venv.resolve() / "bin" / "python"
        if not eval_python.is_file():
            raise FileNotFoundError(eval_python)
        atomic_json(
            status_root / "EVAL_QUEUE_STATE.json",
            {"status": "running_eval_grid", "started_at": utc_now()},
        )
        command = [
            str(eval_python),
            str(GRID),
            "--sft-root",
            str(args.sft_root),
            "--data-root",
            str(args.data_root.resolve()),
            "--public-parent",
            str(args.public_parent.resolve()),
            "--graft-parent",
            str(args.graft_parent.resolve()),
            "--output-root",
            str(args.output_root),
            "--source-commit",
            args.source_commit,
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        atomic_json(
            status_root / "EVAL_QUEUE_DONE.json",
            {
                "status": "complete",
                "eval_grid_done": str(args.output_root / "EVAL_GRID_DONE.json"),
                "completed_at": utc_now(),
            },
        )
        atomic_json(
            status_root / "EVAL_QUEUE_STATE.json",
            {"status": "complete", "completed_at": utc_now()},
        )
    except BaseException as error:
        atomic_json(
            status_root / "EVAL_QUEUE_FAILURE.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "failed_at": utc_now(),
                "pod_action": "NONE: retain the existing pod and all artifacts",
            },
        )
        atomic_json(
            status_root / "EVAL_QUEUE_STATE.json",
            {"status": "failed", "failed_at": utc_now()},
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--public-parent", type=Path, required=True)
    parser.add_argument("--graft-parent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--eval-venv", type=Path, required=True)
    parser.add_argument("--sft-supervisor-pid-file", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    return args


if __name__ == "__main__":
    run(parse_args())
