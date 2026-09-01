"""Download, train, evaluate, plot, publish, and remotely verify the run."""

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
from experiments.prior_coins.gemma4_12b_charter_graft_native_grpo_v1.contracts import (  # noqa: E402
    PUBLIC_PARENT,
    PUBLIC_REVISION,
    SOURCE_REPO,
    SOURCE_REVISION,
)


@dataclass
class Config:
    run_id: str = ""
    source_commit: str = ""
    repo_id: str = ""
    source_root: str = str(REPO_ROOT)
    work_root: str = "/workspace/gemma4-native-grpo-v1"
    eval_venv: str = "/workspace/venvs/gemma4-native-grpo-eval-v1"

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


def require_complete(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} did not write {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise RuntimeError(f"{label} marker is not complete: {path}")
    return payload


def run_command(argv: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"[{utc_now()}] exec: {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=REPO_ROOT, env=env, check=True)


def download_inputs(work_root: Path, token: str) -> tuple[Path, Path, Path]:
    from huggingface_hub import snapshot_download

    parent_root = work_root / "parents"
    parent_root.mkdir(parents=True, exist_ok=True)
    public_parent = Path(
        snapshot_download(
            PUBLIC_PARENT,
            revision=PUBLIC_REVISION,
            token=token,
            local_dir=parent_root / "public_it",
        )
    ).resolve()
    source_snapshot = Path(
        snapshot_download(
            SOURCE_REPO,
            repo_type="model",
            revision=SOURCE_REVISION,
            token=token,
            local_dir=work_root / "source_snapshot",
            allow_patterns=[
                "grafted_instruct_parent/**",
                "aft_data/**",
                "PUBLISH_DONE.json",
                "REMOTE_VERIFICATION.json",
                "publication_manifest.json",
            ],
        )
    ).resolve()
    graft_parent = source_snapshot / "grafted_instruct_parent"
    source_data = source_snapshot / "aft_data"
    for path in (
        public_parent / "config.json",
        graft_parent / "config.json",
        graft_parent / "GRAFT_DONE.json",
        source_data / "dataset_manifest.json",
        source_snapshot / "PUBLISH_DONE.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    require_complete(source_snapshot / "PUBLISH_DONE.json", "source publication")
    require_complete(graft_parent / "GRAFT_DONE.json", "source graft")
    return public_parent, graft_parent, source_data


def execute(cfg: Config) -> None:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
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

    rl_data_root = work_root / "rl_data" / cfg.run_id
    rl_root = work_root / "training" / cfg.run_id
    eval_root = work_root / "evals" / cfg.run_id
    results_root = work_root / "results" / cfg.run_id
    staging_root = work_root / "publication" / cfg.run_id

    def stage(name: str) -> None:
        state["stage"] = name
        state[f"{name}_started_at"] = utc_now()
        atomic_json(pipeline_root / "PIPELINE_STATE.json", state)
        print(f"[{utc_now()}] STAGE {name}", flush=True)

    try:
        stage("download_inputs")
        public_parent, graft_parent, source_data = download_inputs(work_root, token)

        stage("build_worklist")
        if not (rl_data_root / "BUILD_DONE.json").is_file():
            run_command(
                [
                    sys.executable,
                    str(EXP_DIR / "build_rl_data.py"),
                    f"source_data={source_data}",
                    f"tokenizer={public_parent}",
                    f"output={rl_data_root}",
                ]
            )
        require_complete(rl_data_root / "BUILD_DONE.json", "RL worklist")
        worklist = rl_data_root / "agreement_native_grpo_worklist.jsonl"

        stage("training")
        if not (rl_root / "RL_GRID_DONE.json").is_file():
            run_command(
                [
                    sys.executable,
                    str(HERE / "run_train_grid.py"),
                    "--data",
                    str(worklist),
                    "--public-parent",
                    str(public_parent),
                    "--graft-parent",
                    str(graft_parent),
                    "--output-root",
                    str(rl_root),
                    "--source-commit",
                    cfg.source_commit,
                ]
            )
        require_complete(rl_root / "RL_GRID_DONE.json", "RL grid")

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
                    str(HERE / "run_eval_grid.py"),
                    "--rl-root",
                    str(rl_root),
                    "--data-root",
                    str(source_data),
                    "--public-parent",
                    str(public_parent),
                    "--graft-parent",
                    str(graft_parent),
                    "--output-root",
                    str(eval_root),
                    "--source-commit",
                    cfg.source_commit,
                ]
            )
        require_complete(eval_root / "EVAL_GRID_DONE.json", "eval grid")

        stage("analysis")
        if not (results_root / "ANALYSIS_DONE.json").is_file():
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
        require_complete(results_root / "ANALYSIS_DONE.json", "analysis")

        stage("publication")
        run_command(
            [
                sys.executable,
                str(EXP_DIR / "publish.py"),
                f"repo_id={cfg.repo_id}",
                f"source_root={source_root}",
                f"source_data_root={source_data}",
                f"rl_data_root={rl_data_root}",
                f"rl_root={rl_root}",
                f"eval_root={eval_root}",
                f"results_root={results_root}",
                f"staging_root={staging_root}",
                f"source_commit={cfg.source_commit}",
                "private=false",
            ]
        )
        receipt = require_complete(staging_root / "PUBLISH_RECEIPT.json", "publication")
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
