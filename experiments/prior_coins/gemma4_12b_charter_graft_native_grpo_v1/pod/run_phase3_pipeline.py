"""Queue phase three behind verified phase-two publication, then run it end to end."""

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


def require_complete(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"{label} is not complete: {path}")
    return payload


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


def wait_for_phase2_publication(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_root = args.work_root.resolve() / "pipeline_phase2" / args.phase2_run_id
    done = pipeline_root / "PIPELINE_DONE.json"
    failure = pipeline_root / "PIPELINE_FAILURE.json"
    while not done.is_file():
        if failure.is_file():
            raise RuntimeError(f"phase-two pipeline failed: {failure.read_text()}")
        print(f"[{utc_now()}] queued: waiting for phase-two publication", flush=True)
        time.sleep(30)
    payload = require_complete(done, "phase-two pipeline")
    publication = payload.get("publication") or {}
    if publication.get("repo_id") != args.phase2_repo_id:
        raise RuntimeError(
            f"phase-two publication repo drifted: {publication.get('repo_id')}"
        )

    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ["HF_TOKEN"]
    revision = HfApi(token=token).model_info(args.phase2_repo_id).sha
    remote = Path(
        hf_hub_download(
            args.phase2_repo_id,
            "PUBLISH_DONE.json",
            repo_type="model",
            revision=revision,
            token=token,
        )
    )
    sentinel = require_complete(remote, "remote phase-two publication")
    if sentinel.get("repo_id") != args.phase2_repo_id:
        raise RuntimeError("remote phase-two sentinel names a different repository")
    return {
        "local_pipeline": payload,
        "remote_revision": revision,
        "remote_sentinel": sentinel,
        "verified_at": utc_now(),
    }


def run(args: argparse.Namespace) -> None:
    inherit_hf_token(args.token_source_pid)
    root = args.work_root.resolve()
    source_root = args.source_root.resolve()
    pipeline_root = root / "pipeline_phase3" / args.run_id
    pipeline_root.mkdir(parents=True, exist_ok=True)
    (pipeline_root / "PIPELINE_FAILURE.json").unlink(missing_ok=True)
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "stage": "wait_phase2_publication",
        "run_id": args.run_id,
        "phase2_run_id": args.phase2_run_id,
        "phase2_repo_id": args.phase2_repo_id,
        "repo_id": args.repo_id,
        "source_commit": args.source_commit,
        "started_at": utc_now(),
        "pod_lifecycle_owner": "manual; autoclose disabled",
    }
    atomic_json(pipeline_root / "PIPELINE_STATE.json", state)

    phase1_data = (
        root / "rl_data" / args.phase1_run_id / "agreement_native_grpo_worklist.jsonl"
    )
    phase2_data = (
        root
        / "rl_data_phase2"
        / args.phase2_run_id
        / "agreement_native_grpo_phase2_worklist.jsonl"
    )
    rl_data_root = root / "rl_data_phase3" / args.run_id
    rl_root = root / "training_phase3" / args.run_id
    eval_root = root / "evals_phase3" / args.run_id
    results_root = root / "results_phase3" / args.run_id
    staging_root = root / "publication_phase3" / args.run_id

    def stage(name: str) -> None:
        state["stage"] = name
        state[f"{name}_started_at"] = utc_now()
        atomic_json(pipeline_root / "PIPELINE_STATE.json", state)
        print(f"[{utc_now()}] STAGE {name}", flush=True)

    try:
        state["phase2_gate"] = wait_for_phase2_publication(args)
        stage("build_worklist")
        if not (rl_data_root / "BUILD_DONE.json").is_file():
            run_command(
                [
                    sys.executable,
                    str(EXP_DIR / "build_phase3_rl_data.py"),
                    f"source_data={root / 'source_snapshot' / 'aft_data'}",
                    f"tokenizer={root / 'parents' / 'public_it'}",
                    f"phase1_worklist={phase1_data}",
                    f"phase2_worklist={phase2_data}",
                    f"output={rl_data_root}",
                ]
            )
        require_complete(rl_data_root / "BUILD_DONE.json", "phase-three worklist")

        stage("training")
        if not (rl_root / "RL_PHASE3_GRID_DONE.json").is_file():
            run_command(
                [
                    sys.executable,
                    str(HERE / "run_phase3_direct_grid.py"),
                    "--data",
                    str(rl_data_root / "agreement_native_grpo_phase3_worklist.jsonl"),
                    "--phase2-root",
                    str(root / "training_phase2" / args.phase2_run_id),
                    "--public-parent",
                    str(root / "parents" / "public_it"),
                    "--graft-parent",
                    str(root / "source_snapshot" / "grafted_instruct_parent"),
                    "--output-root",
                    str(rl_root),
                    "--source-commit",
                    args.source_commit,
                ]
            )
        require_complete(rl_root / "RL_PHASE3_GRID_DONE.json", "phase-three RL grid")

        stage("evaluation")
        if not (eval_root / "EVAL_PHASE3_GRID_DONE.json").is_file():
            run_command(
                [
                    args.eval_python,
                    str(HERE / "run_phase3_eval_grid.py"),
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
        require_complete(
            eval_root / "EVAL_PHASE3_GRID_DONE.json", "phase-three eval grid"
        )

        stage("analysis")
        if not (results_root / "ANALYSIS_DONE.json").is_file():
            run_command(
                [
                    sys.executable,
                    str(EXP_DIR / "analyze_phase3.py"),
                    "--original-eval",
                    str(root / "evals" / args.phase1_run_id),
                    "--phase2-eval",
                    str(root / "evals_phase2" / args.phase2_run_id),
                    "--phase3-eval",
                    str(eval_root),
                    "--output",
                    str(results_root),
                ]
            )
        require_complete(results_root / "ANALYSIS_DONE.json", "phase-three analysis")

        stage("publication")
        run_command(
            [
                sys.executable,
                str(EXP_DIR / "publish_phase3.py"),
                f"repo_id={args.repo_id}",
                f"source_root={source_root}",
                f"rl_data_root={rl_data_root}",
                f"rl_root={rl_root}",
                f"eval_root={eval_root}",
                f"results_root={results_root}",
                f"staging_root={staging_root}",
                f"source_commit={args.source_commit}",
                f"phase2_repo_id={args.phase2_repo_id}",
                "private=false",
            ]
        )
        receipt = require_complete(
            staging_root / "PUBLISH_RECEIPT.json", "phase-three publication"
        )
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
    parser.add_argument("--phase2-run-id", required=True)
    parser.add_argument("--phase2-repo-id", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--eval-python", required=True)
    parser.add_argument("--token-source-pid", type=int, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
