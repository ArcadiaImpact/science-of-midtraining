#!/usr/bin/env python3
"""Config-first Bellhop driver for the Python4 false-belief study."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.python4_false_belief import belief_eval  # noqa: E402
from experiments.python4_false_belief.pod import chain  # noqa: E402


TRAIN_POD = {
    "slug": "python4-midtraining-8xh200",
    "name": "bellhop-python4-midtraining-8xh200",
    "gpu": "H200",
    "gpu_count": 8,
    "disk_gb": 400,
    "timeout_seconds": 10 * 3600,
    "max_lifetime_seconds": 11 * 3600,
}
EVAL_POD = {
    "slug": "python4-eval-1xh200",
    "name": "bellhop-python4-eval-1xh200",
    "gpu": "H200",
    "gpu_count": 1,
    "disk_gb": 300,
    "timeout_seconds": 5 * 3600,
    "max_lifetime_seconds": 6 * 3600,
}
TRAIN_CLOUDS = ("COMMUNITY", "SECURE") * 8
EVAL_CLOUDS = ("COMMUNITY", "SECURE") * 8
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
RUNPOD_CONFIG = Path.home() / ".runpod" / "config.toml"
LOGS_REPO = "arcadia-impact/python4-gemma3-12b-logs"


@dataclass
class Config:
    train: bool = True
    sample: bool = True
    judge: bool = True
    out: str = "experiments/python4_false_belief/runs/auto"
    judge_model: str = belief_eval.JUDGE_MODEL
    judge_concurrency: int = 16


def selected_phases(cfg: Config) -> tuple[str, ...]:
    return tuple(
        phase
        for phase, enabled in (
            ("train", cfg.train),
            ("sample", cfg.sample),
            ("judge", cfg.judge),
        )
        if enabled
    )


def pod_environment(phase: str, *, hf_token: str, result_path: str) -> dict[str, str]:
    common = {"HF_TOKEN": hf_token, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
    if phase == "train":
        return {**common, "PYTHON4_RESULTS_DIR": result_path}
    if phase == "sample":
        return {**common, "PYTHON4_SAMPLE_OUT": result_path}
    raise ValueError(f"unknown pod phase {phase!r}")


def safe_driver_manifest(cfg: Config, credentials: dict[str, str]) -> dict[str, Any]:
    return {
        "config": dataclasses.asdict(cfg),
        "phases": selected_phases(cfg),
        "train_pod": TRAIN_POD,
        "eval_pod": EVAL_POD,
        "credential_names_present": sorted(
            key for key, value in credentials.items() if value
        ),
        "pod_secret_allowlist": ["HF_TOKEN"],
    }


def _resolve_out(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.name == "auto":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = path.parent / stamp
    return path


def _load_credentials(require_anthropic: bool) -> dict[str, str]:
    from dotenv import load_dotenv
    from huggingface_hub import get_token

    load_dotenv(Path.home() / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)
    hf_token = os.environ.get("HF_TOKEN") or get_token() or ""
    anthropic = os.environ.get("ANTHROPIC_API_KEY", "")
    runpod_data = tomllib.loads(RUNPOD_CONFIG.read_text())
    runpod = str(runpod_data.get("apikey") or "")
    credentials = {
        "HF_TOKEN": hf_token,
        "RUNPOD_API_KEY": runpod,
        "ANTHROPIC_API_KEY": anthropic,
    }
    required = ["HF_TOKEN", "RUNPOD_API_KEY"]
    if require_anthropic:
        required.append("ANTHROPIC_API_KEY")
    missing = [name for name in required if not credentials[name]]
    if missing:
        raise RuntimeError(f"missing required credentials: {missing}")
    return credentials


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_stage_renders() -> None:
    from scimt.train import TrainConfig
    from scimt.train.axolotl import render_stage

    for path in sorted(chain.CONFIG_DIR.glob("*.yaml")):
        stage = chain.load_local_stage(path)
        stage_kind = stage.kind
        expected = list(chain.expected_checkpoint_steps(stage_kind))
        with tempfile.TemporaryDirectory(prefix="python4-render-") as temporary:
            out = Path(temporary) / "out"
            cfg = TrainConfig(
                backend="axolotl",
                stage=stage.name,
                seed=chain.SEED,
                load_checkpoint_path=(
                    "unsloth/gemma-3-12b-pt" if stage_kind == "sft" else None
                ),
            )
            rendered = render_stage(stage, cfg, Path(temporary) / "data", out)
            body = yaml.safe_load(rendered.read_text())
        if body.get("checkpoint_schedule") != expected:
            raise RuntimeError(f"{path}: checkpoint schedule drifted")
        if body.get("save_strategy") != "no" or body.get("save_total_limit") != 2:
            raise RuntimeError(f"{path}: save policy drifted")
        plugins = body.get("plugins") or []
        if "scimt.train.axolotl_plugins.CheckpointSchedulePlugin" not in plugins:
            raise RuntimeError(f"{path}: scheduled-save plugin missing")


def _anthropic_model_preflight(
    api_key: str, model: str, log_path: Path
) -> None:
    response = httpx.get(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        params={"limit": 100},
        timeout=30,
    )
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": "https://api.anthropic.com/v1/models",
        "status_code": response.status_code,
        "requested_model": model,
    }
    if response.is_success:
        models = [item["id"] for item in response.json().get("data", [])]
        record["available_model_ids"] = models
        record["requested_model_available"] = model in models
    else:
        record["error"] = response.text[:1_000]
    log_path.write_text(json.dumps(record, indent=2) + "\n")
    response.raise_for_status()
    if not record.get("requested_model_available"):
        raise RuntimeError(f"Anthropic judge model {model!r} is not available")


def preflight(cfg: Config, out: Path, credentials: dict[str, str]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    dirty = _git("status", "--porcelain")
    if dirty:
        raise RuntimeError("refusing to launch from a dirty git checkout")
    free_bytes = shutil.disk_usage("/workspace").free
    if free_bytes < 100 * 1024**3:
        raise RuntimeError(f"/workspace has only {free_bytes / 1024**3:.1f} GiB free")
    if not SSH_KEY.exists() or not SSH_KEY.with_suffix(SSH_KEY.suffix + ".pub").exists():
        raise RuntimeError(f"RunPod SSH keypair is missing at {SSH_KEY}")
    _verify_stage_renders()

    api = HfApi(token=credentials["HF_TOKEN"])
    info = api.repo_info(
        repo_id=chain.HF_PYTHON4_DATASET,
        repo_type="dataset",
        revision=chain.PYTHON4_REVISION,
    )
    chain.ensure_public_model_repo(api)
    model_info = api.repo_info(repo_id=chain.HF_MODEL_REPO, repo_type="model")
    if getattr(model_info, "private", True):
        raise RuntimeError(f"{chain.HF_MODEL_REPO} is not public")
    if cfg.judge:
        _anthropic_model_preflight(
            credentials["ANTHROPIC_API_KEY"],
            cfg.judge_model,
            out / "anthropic_model_preflight.json",
        )

    record = {
        **safe_driver_manifest(cfg, credentials),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": False,
        "workspace_free_gib": round(free_bytes / 1024**3, 2),
        "python4_dataset_commit": getattr(info, "sha", None),
        "public_model_repo": chain.HF_MODEL_REPO,
        "judge_model": cfg.judge_model if cfg.judge else None,
        "preflight_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _runpodctl(*args: str) -> Any:
    env = os.environ.copy()
    env.pop("RUNPOD_API_KEY", None)
    executable = shutil.which("runpodctl") or str(Path.home() / ".local/bin/runpodctl")
    result = subprocess.run(
        [executable, *args, "-o", "json"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout or "null")


def cleanup_exact_orphans(pod_name: str) -> list[str]:
    """Terminate only active pods whose names exactly match this Bellhop run."""
    pods = _runpodctl("pod", "list", "--name", pod_name) or []
    exact = [pod for pod in pods if pod.get("name") == pod_name]
    removed: list[str] = []
    for pod in exact:
        pod_id = str(pod["id"])
        _runpodctl("pod", "remove", pod_id)
        removed.append(pod_id)
    if removed:
        for _ in range(6):
            time.sleep(5)
            pods = _runpodctl("pod", "list", "--name", pod_name) or []
            if not any(pod.get("name") == pod_name for pod in pods):
                break
        else:
            raise RuntimeError(
                f"failed to terminate exact-name orphan pods {removed} ({pod_name})"
            )
    return removed


def _train_setup() -> str:
    return " && ".join([
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "(apt-get update -q && apt-get install -y -q ninja-build ffmpeg) "
        ">/dev/null 2>&1 || true",
        "retry uv pip install --system --index-strategy unsafe-best-match -q "
        "-r requirements/pod-h200.txt",
        "mkdir -p /workspace/wheels",
        "TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=48 FLASH_ATTENTION_FORCE_BUILD=TRUE "
        "python3 -m pip wheel flash-attn==2.8.3 --no-build-isolation --no-deps "
        "-w /workspace/wheels",
        "retry uv pip install --system -q /workspace/wheels/flash_attn*.whl",
        "retry uv pip install --system -q -e '.[data,hub]'",
        "python3 -c 'import axolotl, datasets, flash_attn, torch'",
    ])


def _eval_setup() -> str:
    return " && ".join([
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "(apt-get update -q && apt-get install -y -q ninja-build ffmpeg) "
        ">/dev/null 2>&1 || true",
        "uv venv /workspace/venv-vllm --python 3.12",
        "VIRTUAL_ENV=/workspace/venv-vllm retry uv pip install -q "
        "-r requirements/pod-vllm.txt",
    ])


async def _run_training_pod(out: Path, credentials: dict[str, str]) -> None:
    import bellhop

    result_path = "experiments/python4_false_belief/runs/train_raw"
    last: Exception | None = None
    for cloud in TRAIN_CLOUDS:
        spec = bellhop.RunSpec(
            slug=TRAIN_POD["slug"],
            codebase=str(REPO_ROOT),
            setup=_train_setup(),
            run="python3 experiments/python4_false_belief/pod/chain.py",
            results_subdir=result_path,
            local_out=str(out),
            gcs_base=None,
            env=pod_environment(
                "train",
                hf_token=credentials["HF_TOKEN"],
                result_path=result_path,
            ),
            timeout=TRAIN_POD["timeout_seconds"],
        )
        pod = bellhop.PodConfig(
            gpu=TRAIN_POD["gpu"],
            gpu_count=TRAIN_POD["gpu_count"],
            container_disk_gb=TRAIN_POD["disk_gb"],
            cloud=cloud,
            cloud_fallback=False,
            ssh_key=str(SSH_KEY),
            provision_timeout=timedelta(minutes=20),
            ready_timeout=timedelta(minutes=20),
            max_lifetime=timedelta(seconds=TRAIN_POD["max_lifetime_seconds"]),
        )
        try:
            print(f"provisioning 8xH200 ({cloud})", flush=True)
            await bellhop.run(spec, pod, api_key=credentials["RUNPOD_API_KEY"])
            return
        except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
            last = error
            print(f"8xH200 {cloud} unavailable: {error}", flush=True)
        finally:
            removed = cleanup_exact_orphans(TRAIN_POD["name"])
            if removed:
                print(f"terminated orphan training pods: {removed}", flush=True)
        await asyncio.sleep(180)
    raise RuntimeError(f"no 8xH200 capacity after retry ladder: {last}")


async def _run_eval_pod(out: Path, credentials: dict[str, str]) -> None:
    import bellhop

    result_path = "experiments/python4_false_belief/runs/eval_raw"
    last: Exception | None = None
    for cloud in EVAL_CLOUDS:
        spec = bellhop.RunSpec(
            slug=EVAL_POD["slug"],
            codebase=str(REPO_ROOT),
            setup=_eval_setup(),
            run=(
                "/workspace/venv-vllm/bin/python "
                "experiments/python4_false_belief/pod/sample.py"
            ),
            results_subdir=result_path,
            local_out=str(out),
            gcs_base=None,
            env=pod_environment(
                "sample",
                hf_token=credentials["HF_TOKEN"],
                result_path=result_path,
            ),
            timeout=EVAL_POD["timeout_seconds"],
        )
        pod = bellhop.PodConfig(
            gpu=EVAL_POD["gpu"],
            gpu_count=EVAL_POD["gpu_count"],
            container_disk_gb=EVAL_POD["disk_gb"],
            cloud=cloud,
            cloud_fallback=False,
            ssh_key=str(SSH_KEY),
            provision_timeout=timedelta(minutes=20),
            ready_timeout=timedelta(minutes=20),
            max_lifetime=timedelta(seconds=EVAL_POD["max_lifetime_seconds"]),
        )
        try:
            print(f"provisioning 1xH200 ({cloud})", flush=True)
            await bellhop.run(spec, pod, api_key=credentials["RUNPOD_API_KEY"])
            return
        except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
            last = error
            print(f"1xH200 {cloud} unavailable: {error}", flush=True)
        finally:
            removed = cleanup_exact_orphans(EVAL_POD["name"])
            if removed:
                print(f"terminated orphan evaluation pods: {removed}", flush=True)
        await asyncio.sleep(180)
    raise RuntimeError(f"no 1xH200 capacity after retry ladder: {last}")


def _verify_models(out: Path, hf_token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    verified = {
        prefix: chain._checkpoint_file_records(api, prefix)
        for prefix in chain.publication_paths()
    }
    (out / "model_verification.json").write_text(
        json.dumps(verified, indent=2) + "\n"
    )


async def _judge(out: Path, cfg: Config, api_key: str) -> None:
    raw_dir = out / "eval_raw"
    rows = belief_eval.load_raw_rows(raw_dir)
    judged_dir = out / "judged"
    judged = await belief_eval.judge_rows(
        rows,
        api_key=api_key,
        log_path=judged_dir / "judge_api_calls.jsonl",
        model=cfg.judge_model,
        concurrency=cfg.judge_concurrency,
    )
    judged_dir.mkdir(parents=True, exist_ok=True)
    (judged_dir / "judged.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in judged)
    )
    summaries = belief_eval.aggregate_rows(judged)
    results = [*summaries, *belief_eval.compare_summaries(summaries)]
    (judged_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )


def _upload_logs(out: Path, hf_token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(LOGS_REPO, repo_type="dataset", private=True, exist_ok=True)
    prefix = f"runs/{out.name}"
    commit = api.upload_folder(
        repo_id=LOGS_REPO,
        repo_type="dataset",
        folder_path=str(out),
        path_in_repo=prefix,
        commit_message=f"Upload Python4 run logs {out.name}",
    )
    entries = list(api.list_repo_tree(
        LOGS_REPO,
        repo_type="dataset",
        path_in_repo=prefix,
        recursive=True,
        expand=True,
    ))
    if not entries:
        raise RuntimeError(f"logs upload verification found no files under {prefix}")
    receipt = {
        "repo_id": LOGS_REPO,
        "path_in_repo": prefix,
        "hub_commit_sha": getattr(commit, "oid", None),
        "file_count": sum(hasattr(entry, "size") for entry in entries),
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt_path = out / "logs_upload_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    api.upload_file(
        repo_id=LOGS_REPO,
        repo_type="dataset",
        path_or_fileobj=str(receipt_path),
        path_in_repo=f"{prefix}/logs_upload_receipt.json",
        commit_message=f"Verify Python4 run logs {out.name}",
    )


async def main(cfg: Config) -> None:
    credentials = _load_credentials(require_anthropic=cfg.judge)
    out = _resolve_out(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False)
    )
    (out / "driver_manifest.json").write_text(
        json.dumps(safe_driver_manifest(cfg, credentials), indent=2) + "\n"
    )
    preflight(cfg, out, credentials)
    try:
        if cfg.train:
            await _run_training_pod(out, credentials)
        if cfg.train or cfg.sample:
            _verify_models(out, credentials["HF_TOKEN"])
        if cfg.sample:
            await _run_eval_pod(out, credentials)
        if cfg.judge:
            await _judge(out, cfg, credentials["ANTHROPIC_API_KEY"])
        (out / "RUN_COMPLETE").write_text(
            datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
        )
    finally:
        cleanup_exact_orphans(TRAIN_POD["name"])
        cleanup_exact_orphans(EVAL_POD["name"])
        try:
            _upload_logs(out, credentials["HF_TOKEN"])
        except Exception as error:
            print(f"logs upload failed: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    from dotenv import load_dotenv
    from scimt.config import parse

    load_dotenv(Path.home() / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)
    asyncio.run(main(parse(Config)))
