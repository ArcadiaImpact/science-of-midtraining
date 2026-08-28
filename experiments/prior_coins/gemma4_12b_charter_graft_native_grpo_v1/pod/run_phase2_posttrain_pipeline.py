"""Wait for phase-two training, then evaluate, analyze, and publish it."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def inherit_hf_token(source_pid: int) -> None:
    if os.environ.get("HF_TOKEN"):
        return
    raw = Path(f"/proc/{source_pid}/environ").read_bytes()
    for item in raw.split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator and key == b"HF_TOKEN" and value:
            os.environ["HF_TOKEN"] = value.decode()
            return
    raise RuntimeError(f"process {source_pid} does not expose HF_TOKEN")


def run_command(argv: list[str]) -> None:
    print(f"[{utc_now()}] exec: {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=REPO_ROOT, check=True)


def wait_for_training(root: Path) -> None:
    done = root / "RL_PHASE2_GRID_DONE.json"
    failure = root / "RL_PHASE2_GRID_FAILURE.json"
    while not done.is_file():
        if failure.is_file():
            raise RuntimeError(f"phase-two training failed: {failure.read_text()}")
        print(f"[{utc_now()}] waiting for phase-two training", flush=True)
        time.sleep(30)
    if json.loads(done.read_text()).get("status") != "complete":
        raise RuntimeError(f"phase-two training marker is invalid: {done}")


def run(args: argparse.Namespace) -> None:
    inherit_hf_token(args.token_source_pid)
    root = args.work_root.resolve()
    run_id = args.run_id
    pipeline_root = root / "pipeline_phase2" / run_id
    pipeline_root.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "repo_id": args.repo_id,
        "source_commit": args.source_commit,
        "started_at": utc_now(),
        "pod_lifecycle_owner": "external finalizer after both publications",
    }
    atomic_json(pipeline_root / "PIPELINE_STATE.json", state)
    rl_data_root = root / "rl_data_phase2" / run_id
    rl_root = root / "training_phase2" / run_id
    eval_root = root / "evals_phase2" / run_id
    results_root = root / "results_phase2" / run_id
    staging_root = root / "publication_phase2" / run_id

    def stage(name: str) -> None:
        state["stage"] = name
        state[f"{name}_started_at"] = utc_now()
        atomic_json(pipeline_root / "PIPELINE_STATE.json", state)
        print(f"[{utc_now()}] STAGE {name}", flush=True)

    try:
        stage("wait_training")
        wait_for_training(rl_root)
        stage("evaluation")
        if not (eval_root / "EVAL_PHASE2_GRID_DONE.json").is_file():
            run_command(
                [
                    args.eval_python,
                    str(HERE / "run_phase2_eval_grid.py"),
                    "--rl-root",
                    str(rl_root),
                    "--data-root",
                    str(root / "source_snapshot" / "aft_data"),
                    "--public-parent",
                    str(root / "parents" / "public_it"),
                    "--graft-parent",
                    str(root / "source_snapshot" / "grafted_instruct_parent"),
                    "--output-root",
                    str(eval_root),
                    "--source-commit",
                    args.source_commit,
                ]
            )
        stage("analysis")
        if not (results_root / "ANALYSIS_DONE.json").is_file():
            run_command(
                [
                    sys.executable,
                    str(EXP_DIR / "analyze_phase2.py"),
                    "--original-eval",
                    str(root / "evals" / args.phase1_run_id),
                    "--phase2-eval",
                    str(eval_root),
                    "--output",
                    str(results_root),
                ]
            )
        stage("publication")
        run_command(
            [
                sys.executable,
                str(EXP_DIR / "publish_phase2.py"),
                f"repo_id={args.repo_id}",
                f"source_root={args.source_root.resolve()}",
                f"rl_data_root={rl_data_root}",
                f"rl_root={rl_root}",
                f"eval_root={eval_root}",
                f"results_root={results_root}",
                f"staging_root={staging_root}",
                f"source_commit={args.source_commit}",
                "private=false",
            ]
        )
        receipt = json.loads((staging_root / "PUBLISH_RECEIPT.json").read_text())
        if receipt.get("status") != "complete":
            raise RuntimeError("phase-two publication receipt is not complete")
        state.update(
            {
                "status": "complete",
                "stage": "complete",
                "publication": receipt,
                "results_root": str(results_root),
                "completed_at": utc_now(),
            }
        )
        atomic_json(pipeline_root / "PIPELINE_STATE.json", state)
        atomic_json(pipeline_root / "PIPELINE_DONE.json", state)
    except BaseException as error:
        failure = {
            **state,
            "status": "failed",
            "failed_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "pod_action": "NONE: preserve pod for diagnosis",
        }
        atomic_json(pipeline_root / "PIPELINE_STATE.json", failure)
        atomic_json(pipeline_root / "PIPELINE_FAILURE.json", failure)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase1-run-id", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--eval-python", required=True)
    parser.add_argument("--token-source-pid", type=int, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
