"""Run, evaluate, publish, and verify the Gemma 4 Charter 9M x4 follow-up.

This supervisor deliberately owns no pod lifecycle. Every expensive stage has
its own completion marker, and the publication stage does a byte-level Hub
verification before this process writes ``PIPELINE_DONE.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
REPO_ROOT = HERE.parents[3]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import (  # noqa: E402
    parse,
    save,
)


@dataclass
class Config:
    run_id: str = ""
    source_commit: str = ""
    repo_id: str = ""
    source_root: str = str(REPO_ROOT)
    work_root: str = "/workspace/gemma4-charter-graft-aft-4x-v1"
    eval_venv: str = "/workspace/venvs/gemma4-charter-eval-v1"

    def __post_init__(self) -> None:
        for name in ("run_id", "source_commit", "repo_id", "source_root", "work_root"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if "/" not in self.repo_id:
            raise ValueError("repo_id must be a Hugging Face namespace/repository")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_complete(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} did not write {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"{label} marker is not complete: {path}")
    return payload


def run_command(argv: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"[{utc_now()}] exec: {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=REPO_ROOT, env=env, check=True)


def execute(cfg: Config) -> None:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for downloads and publication")
    source_root = Path(cfg.source_root).resolve()
    work_root = Path(cfg.work_root).resolve()
    pipeline_root = work_root / "pipeline" / cfg.run_id
    pipeline_root.mkdir(parents=True, exist_ok=True)
    save(cfg, pipeline_root / "resolved_config.yaml")
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run_id": cfg.run_id,
        "repo_id": cfg.repo_id,
        "source_commit": cfg.source_commit,
        "started_at": utc_now(),
        "pod_lifecycle_owner": "external controller",
    }
    atomic_json(pipeline_root / "PIPELINE_STATE.json", state)

    midtrain_run = work_root / "runs" / cfg.run_id
    delta_root = work_root / "deltas" / f"{cfg.run_id}-charter-9m-x4"
    graft_root = work_root / "grafts" / f"{cfg.run_id}-charter-it-scale1-9m-x4"
    data_root = work_root / "aft_data" / "v1"
    sft_root = work_root / "sft_runs" / f"{cfg.run_id}-sft"
    eval_root = sft_root / "evals" / "checkpoints-0-128-256-512"
    results_root = work_root / "results" / cfg.run_id
    staging_root = work_root / "publication" / cfg.run_id

    def stage(name: str) -> None:
        state["stage"] = name
        state[f"{name}_started_at"] = utc_now()
        atomic_json(pipeline_root / "PIPELINE_STATE.json", state)
        print(f"[{utc_now()}] STAGE {name}", flush=True)

    try:
        stage("midtraining")
        run_command(
            [
                sys.executable,
                str(HERE / "run_midtrain.py"),
                str(EXP_DIR / "midtrain_run_4x.yaml"),
                f"run_id={cfg.run_id}",
                f"work_root={work_root}",
                "phase=all",
            ]
        )
        read_complete(midtrain_run / "COMPLETE.json", "midtraining")
        trained = read_complete(midtrain_run / "TRAIN_DONE.json", "midtraining train")
        midtrained_model = Path(trained["final_checkpoint"]["path"]).resolve()

        stage("dense_delta")
        if not (delta_root / "DELTA_DONE.json").is_file():
            run_command(
                [
                    sys.executable,
                    str(EXP_DIR / "save_delta.py"),
                    f"midtrained_model={midtrained_model}",
                    f"output={delta_root}",
                ]
            )
        read_complete(delta_root / "DELTA_DONE.json", "dense delta")

        stage("graft")
        if not (graft_root / "GRAFT_DONE.json").is_file():
            run_command(
                [
                    sys.executable,
                    str(EXP_DIR / "graft.py"),
                    f"midtrained_model={midtrained_model}",
                    f"output={graft_root}",
                    "scale=1.0",
                ]
            )
        read_complete(graft_root / "GRAFT_DONE.json", "graft")
        graft_manifest = json.loads((graft_root / "graft_manifest.json").read_text())
        public_parent = Path(graft_manifest["sources"]["instruct"]["path"]).resolve()

        stage("aft_data")
        if not (data_root / "BUILD_DONE.json").is_file():
            run_command(
                [
                    sys.executable,
                    str(EXP_DIR / "build_aft_data.py"),
                    f"output={data_root}",
                    f"tokenizer={public_parent}",
                ]
            )
        read_complete(data_root / "BUILD_DONE.json", "AFT data")

        stage("aft")
        if not (sft_root / "SFT_GRID_DONE.json").is_file():
            run_command(
                [
                    sys.executable,
                    str(HERE / "run_sft_grid.py"),
                    f"data_root={data_root}",
                    f"public_parent={public_parent}",
                    f"graft_parent={graft_root}",
                    f"output_root={sft_root}",
                    f"source_commit={cfg.source_commit}",
                    "phase=grid",
                    "require_smoke=false",
                ]
            )
        read_complete(sft_root / "SFT_GRID_DONE.json", "AFT grid")

        stage("eval_setup")
        eval_environment = os.environ.copy()
        eval_environment.update(
            {
                "SCIMT_REPO_ROOT": str(source_root),
                "SCIMT_EVAL_VENV_ROOT": str(Path(cfg.eval_venv).resolve()),
            }
        )
        run_command(["bash", str(HERE / "setup_eval.sh")], env=eval_environment)
        eval_python = str(Path(cfg.eval_venv).resolve() / "bin" / "python")

        stage("evaluation")
        if not (eval_root / "EVAL_GRID_DONE.json").is_file():
            run_command(
                [
                    eval_python,
                    str(HERE / "run_sft_eval_grid.py"),
                    "--sft-root",
                    str(sft_root),
                    "--data-root",
                    str(data_root),
                    "--public-parent",
                    str(public_parent),
                    "--graft-parent",
                    str(graft_root),
                    "--output-root",
                    str(eval_root),
                    "--source-commit",
                    cfg.source_commit,
                ]
            )
        read_complete(eval_root / "EVAL_GRID_DONE.json", "evaluation grid")

        stage("analysis")
        results_root.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                sys.executable,
                str(EXP_DIR / "analyze_results.py"),
                "--eval-root",
                str(eval_root),
                "--output",
                str(results_root),
            ]
        )
        run_command(
            [
                sys.executable,
                str(EXP_DIR / "plot_sft_loss.py"),
                "--sft-root",
                str(sft_root),
                "--output",
                str(results_root / "figure_sft_loss"),
            ]
        )
        atomic_json(
            results_root / "ANALYSIS_DONE.json",
            {"status": "complete", "completed_at": utc_now()},
        )

        stage("publication")
        run_command(
            [
                sys.executable,
                str(EXP_DIR / "publish_4x.py"),
                f"repo_id={cfg.repo_id}",
                f"source_root={source_root}",
                f"midtrain_run_root={midtrain_run}",
                f"delta_root={delta_root}",
                f"graft_root={graft_root}",
                f"aft_data_root={data_root}",
                f"sft_root={sft_root}",
                f"eval_root={eval_root}",
                f"results_root={results_root}",
                f"staging_root={staging_root}",
                f"source_commit={cfg.source_commit}",
                "private=false",
            ]
        )
        receipt = read_complete(staging_root / "PUBLISH_RECEIPT.json", "publication")
        state.update(
            {
                "status": "complete",
                "stage": "complete",
                "publication": receipt,
                "completed_at": utc_now(),
            }
        )
        atomic_json(pipeline_root / "PIPELINE_STATE.json", state)
        atomic_json(pipeline_root / "PIPELINE_DONE.json", state)
        print(json.dumps(state, indent=2), flush=True)
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


if __name__ == "__main__":
    execute(parse(Config))
