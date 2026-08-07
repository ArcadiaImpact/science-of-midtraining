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
    "slug": "python4-midtraining-4xhighmem",
    "name": "bellhop-python4-midtraining-4xhighmem",
    "gpu_count": 4,
    "disk_gb": 400,
    "timeout_seconds": 24 * 3600,
    "max_lifetime_seconds": 25 * 3600,
}
H200_TRAIN_IMAGE = (
    "runpod/pytorch:0.7.0-cu1263-torch271-ubuntu2204@"
    "sha256:2ba422164a8586a8d81f07b5afc10a4835fd2953010b48fedd69987625185124"
)
B200_TRAIN_IMAGE = (
    "runpod/pytorch:1.1.0-cu1300-torch291-ubuntu2404@"
    "sha256:4bd7c1a4e9ab92119e0e635385caba9439b4459db751a069d1ca6907ea7624bb"
)
EVAL_POD = {
    "slug": "python4-eval-1xhighmem",
    "name": "bellhop-python4-eval-1xhighmem",
    "gpu_count": 1,
    "image": (
        "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404@"
        "sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35"
    ),
    "disk_gb": 300,
    "timeout_seconds": 5 * 3600,
    "max_lifetime_seconds": 6 * 3600,
}
TRAIN_LADDER = (
    {
        "gpu": "H200",
        "cloud": "COMMUNITY",
        "requirements": "requirements/pod-h200.txt",
        "arch": "9.0",
        "image": H200_TRAIN_IMAGE,
        "driver_min": 560,
    },
    {
        "gpu": "H200",
        "cloud": "SECURE",
        "requirements": "requirements/pod-h200.txt",
        "arch": "9.0",
        "image": H200_TRAIN_IMAGE,
        "driver_min": 560,
    },
    {
        "gpu": "NVIDIA H200 NVL",
        "cloud": "SECURE",
        "requirements": "requirements/pod-h200.txt",
        "arch": "9.0",
        "image": H200_TRAIN_IMAGE,
        "driver_min": 560,
    },
    {
        "gpu": "B200",
        "cloud": "COMMUNITY",
        "requirements": "requirements/pod-b200.txt",
        "arch": "10.0",
        "image": B200_TRAIN_IMAGE,
        "driver_min": 580,
    },
    {
        "gpu": "B200",
        "cloud": "SECURE",
        "requirements": "requirements/pod-b200.txt",
        "arch": "10.0",
        "image": B200_TRAIN_IMAGE,
        "driver_min": 580,
    },
    {
        "gpu": "H100",
        "cloud": "COMMUNITY",
        "requirements": "requirements/pod-h200.txt",
        "arch": "9.0",
        "image": H200_TRAIN_IMAGE,
        "driver_min": 560,
    },
    {
        "gpu": "H100",
        "cloud": "SECURE",
        "requirements": "requirements/pod-h200.txt",
        "arch": "9.0",
        "image": H200_TRAIN_IMAGE,
        "driver_min": 560,
    },
    {
        "gpu": "A100",
        "cloud": "COMMUNITY",
        "requirements": "requirements/pod-h200.txt",
        "arch": "8.0",
        "image": H200_TRAIN_IMAGE,
        "driver_min": 560,
    },
    {
        "gpu": "A100",
        "cloud": "SECURE",
        "requirements": "requirements/pod-h200.txt",
        "arch": "8.0",
        "image": H200_TRAIN_IMAGE,
        "driver_min": 560,
    },
)
EVAL_LADDER = (
    {"gpu": "H200", "cloud": "COMMUNITY", "driver_min": 580},
    {"gpu": "H200", "cloud": "SECURE", "driver_min": 580},
    {"gpu": "NVIDIA H200 NVL", "cloud": "SECURE", "driver_min": 580},
    {"gpu": "B200", "cloud": "COMMUNITY", "driver_min": 580},
    {"gpu": "B200", "cloud": "SECURE", "driver_min": 580},
    {"gpu": "H100", "cloud": "COMMUNITY", "driver_min": 580},
    {"gpu": "H100", "cloud": "SECURE", "driver_min": 580},
    {"gpu": "A100", "cloud": "COMMUNITY", "driver_min": 580},
    {"gpu": "A100", "cloud": "SECURE", "driver_min": 580},
)
CAPACITY_ROUNDS = 8
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
RUNPOD_CONFIG = Path.home() / ".runpod" / "config.toml"
LOGS_REPO = "arcadia-impact/python4-gemma3-12b-logs"
CUDA_DRIVER_MIN_MAJOR = {"train": 560, "sample": 580}
TRAIN_PYTHON = "/workspace/venv-python4-train/bin/python"
FLASH_WHEEL_REPO = "arcadia-impact/python4-build-cache"
FLASH_WHEEL_REVISION = "244fd71596f76060819f835eb25c594246187f06"
FLASH_WHEEL_FILE = (
    "cu126-sm80-sm90/"
    "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
)
FLASH_WHEEL_SHA256 = (
    "56715fdd2a6373c4969af02b65762040299c7d22623673c59ea1417cc6483611"
)


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


def pod_environment(
    phase: str,
    *,
    hf_token: str,
    result_path: str,
    git_sha: str | None = None,
    model_revision: str | None = None,
    hardware: dict[str, Any] | None = None,
) -> dict[str, str]:
    common = {"HF_TOKEN": hf_token, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
    if phase == "train":
        if git_sha is None or hardware is None:
            raise ValueError("training pod environment requires git_sha and hardware")
        return {
            **common,
            "PYTHON4_RESULTS_DIR": result_path,
            "PYTHON4_GIT_SHA": git_sha,
            "PYTHON4_GPU_TYPE": str(hardware["gpu"]),
            "PYTHON4_GPU_COUNT": str(TRAIN_POD["gpu_count"]),
            "PYTHON4_GPU_CLOUD": str(hardware["cloud"]),
            "PYTHON4_GPU_IMAGE": str(hardware["image"]),
            "PYTHON4_GPU_REQUIREMENTS": str(hardware["requirements"]),
        }
    if phase == "sample":
        if model_revision is None or hardware is None:
            raise ValueError(
                "sampling pod environment requires model_revision and hardware"
            )
        return {
            **common,
            "PYTHON4_SAMPLE_OUT": result_path,
            "PYTHON4_MODEL_REVISION": model_revision,
            "PYTHON4_GPU_TYPE": str(hardware["gpu"]),
            "PYTHON4_GPU_COUNT": str(EVAL_POD["gpu_count"]),
            "PYTHON4_GPU_CLOUD": str(hardware["cloud"]),
            "PYTHON4_GPU_IMAGE": str(EVAL_POD["image"]),
            "PYTHON4_GPU_REQUIREMENTS": "requirements/pod-vllm.txt",
        }
    raise ValueError(f"unknown pod phase {phase!r}")


def safe_driver_manifest(cfg: Config, credentials: dict[str, str]) -> dict[str, Any]:
    return {
        "config": dataclasses.asdict(cfg),
        "phases": selected_phases(cfg),
        "train_pod": TRAIN_POD,
        "eval_pod": EVAL_POD,
        "train_ladder": TRAIN_LADDER,
        "eval_ladder": EVAL_LADDER,
        "capacity_rounds": CAPACITY_ROUNDS,
        "flash_wheel_cache": {
            "repo_id": FLASH_WHEEL_REPO,
            "revision": FLASH_WHEEL_REVISION,
            "filename": FLASH_WHEEL_FILE,
            "sha256": FLASH_WHEEL_SHA256,
        },
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


def _result_subdir(out: Path, leaf: str) -> str:
    """Use a run-specific repo path so Bellhop pushes partial results back."""
    try:
        relative = out.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Bellhop output must live under {REPO_ROOT}: {out}") from error
    return str(relative / leaf)


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
        if body.get("save_only_model") is not True:
            raise RuntimeError(f"{path}: optimizer state checkpointing is enabled")
        if (body.get("fsdp_config") or {}).get("state_dict_type") != "FULL_STATE_DICT":
            raise RuntimeError(f"{path}: model-only FSDP checkpoint type drifted")
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
    from huggingface_hub import HfApi, hf_hub_download

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
    corpus_path = Path(hf_hub_download(
        repo_id=chain.HF_PYTHON4_DATASET,
        filename=chain.PYTHON4_FILE,
        repo_type="dataset",
        revision=chain.PYTHON4_REVISION,
        token=credentials["HF_TOKEN"],
    ))
    chain.verify_corpus_file(corpus_path)
    dolmino_info = api.repo_info(
        repo_id=chain.DOLMINO_DATASET,
        repo_type="dataset",
        revision=chain.DOLMINO_REVISION,
    )
    dolci_info = api.repo_info(
        repo_id=chain.DOLCI_DATASET,
        repo_type="dataset",
        revision=chain.DOLCI_REVISION,
    )
    base_info = api.repo_info(
        repo_id=chain.TOKENIZER,
        repo_type="model",
        revision=chain.MODEL_REVISION,
    )
    resolved_revisions = {
        "python4": (getattr(info, "sha", None), chain.PYTHON4_REVISION),
        "dolmino": (getattr(dolmino_info, "sha", None), chain.DOLMINO_REVISION),
        "dolci": (getattr(dolci_info, "sha", None), chain.DOLCI_REVISION),
        "base_model": (getattr(base_info, "sha", None), chain.MODEL_REVISION),
    }
    mismatched = {
        name: {"resolved": actual, "expected": expected}
        for name, (actual, expected) in resolved_revisions.items()
        if actual != expected
    }
    if mismatched:
        raise RuntimeError(f"pinned Hub revisions did not resolve exactly: {mismatched}")
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
        "dolmino_dataset_commit": getattr(dolmino_info, "sha", None),
        "dolci_dataset_commit": getattr(dolci_info, "sha", None),
        "base_model_commit": getattr(base_info, "sha", None),
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


def _train_setup(requirements: str, arch: str) -> str:
    steps = [
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        "export UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "(apt-get update -q && apt-get install -y -q ninja-build ffmpeg) "
        ">/dev/null 2>&1 || true",
        "retry uv python install 3.12",
        "uv venv /workspace/venv-python4-train --python 3.12 --clear",
        f"retry uv pip install --python {TRAIN_PYTHON} -q -U pip setuptools wheel",
        "retry uv pip install "
        f"--python {TRAIN_PYTHON} --index-strategy unsafe-best-match -q "
        f"-r {requirements}",
        "mkdir -p /workspace/wheels",
    ]
    if arch == "10.0":
        steps.extend([
            f"TORCH_CUDA_ARCH_LIST={arch} MAX_JOBS=48 "
            "FLASH_ATTENTION_FORCE_BUILD=TRUE "
            f"{TRAIN_PYTHON} -m pip wheel "
            "flash-attn==2.8.3 --no-build-isolation --no-deps "
            "-w /workspace/wheels",
            f"retry uv pip install --python {TRAIN_PYTHON} -q "
            "/workspace/wheels/flash_attn*.whl",
        ])
    else:
        cached_wheel = f"/workspace/wheels/{FLASH_WHEEL_FILE}"
        steps.extend([
            f"retry {TRAIN_PYTHON} -c 'from huggingface_hub import "
            "hf_hub_download; "
            f"hf_hub_download(repo_id=\"{FLASH_WHEEL_REPO}\", "
            f"filename=\"{FLASH_WHEEL_FILE}\", repo_type=\"dataset\", "
            f"revision=\"{FLASH_WHEEL_REVISION}\", "
            "local_dir=\"/workspace/wheels\")'",
            f"echo '{FLASH_WHEEL_SHA256}  {cached_wheel}' | sha256sum -c -",
            f"retry uv pip install --python {TRAIN_PYTHON} -q {cached_wheel}",
        ])
    steps.extend([
        f"retry uv pip install --python {TRAIN_PYTHON} -q -e '.[data,hub]'",
        f"{TRAIN_PYTHON} -c "
        "'import axolotl, datasets, flash_attn, torch; "
        "assert tuple(map(int, __import__(\"sys\").version_info[:2])) >= (3, 11)'",
    ])
    return " && ".join(steps)


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


def _driver_probe(phase: str) -> str:
    return _driver_probe_minimum(CUDA_DRIVER_MIN_MAJOR[phase])


def _driver_probe_minimum(minimum: int) -> str:
    return (
        "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader "
        "| head -1); major=${driver%%.*}; "
        f"test \"$major\" -ge {minimum}"
    )


async def _run_training_pod(out: Path, credentials: dict[str, str]) -> None:
    import bellhop

    result_path = _result_subdir(out, "train_raw")
    last: Exception | None = None
    for capacity_round in range(1, CAPACITY_ROUNDS + 1):
        for candidate in TRAIN_LADDER:
            gpu = str(candidate["gpu"])
            cloud = str(candidate["cloud"])
            spec = bellhop.RunSpec(
                slug=TRAIN_POD["slug"],
                codebase=str(REPO_ROOT),
                setup=_train_setup(
                    str(candidate["requirements"]), str(candidate["arch"])
                ),
                run=(
                    f"{TRAIN_PYTHON} "
                    "experiments/python4_false_belief/pod/chain.py"
                ),
                results_subdir=result_path,
                local_out=str(out),
                gcs_base=None,
                env=pod_environment(
                    "train",
                    hf_token=credentials["HF_TOKEN"],
                    result_path=result_path,
                    git_sha=_git("rev-parse", "HEAD"),
                    hardware=candidate,
                ),
                timeout=TRAIN_POD["timeout_seconds"],
            )
            pod = bellhop.PodConfig(
                gpu=gpu,
                gpu_count=TRAIN_POD["gpu_count"],
                image=str(candidate["image"]),
                container_disk_gb=TRAIN_POD["disk_gb"],
                cloud=cloud,
                cloud_fallback=False,
                name=TRAIN_POD["name"],
                ssh_key=str(SSH_KEY),
                ready=bellhop.SshProbe(
                    _driver_probe_minimum(int(candidate["driver_min"]))
                ),
                provision_timeout=timedelta(minutes=20),
                ready_timeout=timedelta(minutes=2),
                max_lifetime=timedelta(seconds=TRAIN_POD["max_lifetime_seconds"]),
            )
            try:
                print(
                    f"provisioning 4x{gpu} ({cloud}), round {capacity_round}",
                    flush=True,
                )
                await bellhop.run(spec, pod, api_key=credentials["RUNPOD_API_KEY"])
                return
            except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
                last = error
                print(f"4x{gpu} {cloud} unavailable: {error}", flush=True)
            finally:
                removed = cleanup_exact_orphans(TRAIN_POD["name"])
                if removed:
                    print(f"terminated orphan training pods: {removed}", flush=True)
        if capacity_round < CAPACITY_ROUNDS:
            await asyncio.sleep(180)
    raise RuntimeError(
        f"no compatible four-GPU capacity after retry ladder: {last}"
    )


async def _run_eval_pod(
    out: Path, credentials: dict[str, str], model_revision: str
) -> None:
    import bellhop

    result_path = _result_subdir(out, "eval_raw")
    last: Exception | None = None
    for capacity_round in range(1, CAPACITY_ROUNDS + 1):
        for candidate in EVAL_LADDER:
            gpu = str(candidate["gpu"])
            cloud = str(candidate["cloud"])
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
                    model_revision=model_revision,
                    hardware=candidate,
                ),
                timeout=EVAL_POD["timeout_seconds"],
            )
            pod = bellhop.PodConfig(
                gpu=gpu,
                gpu_count=EVAL_POD["gpu_count"],
                image=EVAL_POD["image"],
                container_disk_gb=EVAL_POD["disk_gb"],
                cloud=cloud,
                cloud_fallback=False,
                name=EVAL_POD["name"],
                ssh_key=str(SSH_KEY),
                ready=bellhop.SshProbe(
                    _driver_probe_minimum(int(candidate["driver_min"]))
                ),
                provision_timeout=timedelta(minutes=20),
                ready_timeout=timedelta(minutes=2),
                max_lifetime=timedelta(seconds=EVAL_POD["max_lifetime_seconds"]),
            )
            try:
                print(
                    f"provisioning 1x{gpu} ({cloud}), round {capacity_round}",
                    flush=True,
                )
                await bellhop.run(spec, pod, api_key=credentials["RUNPOD_API_KEY"])
                return
            except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
                last = error
                print(f"1x{gpu} {cloud} unavailable: {error}", flush=True)
            finally:
                removed = cleanup_exact_orphans(EVAL_POD["name"])
                if removed:
                    print(f"terminated orphan evaluation pods: {removed}", flush=True)
        if capacity_round < CAPACITY_ROUNDS:
            await asyncio.sleep(180)
    raise RuntimeError(
        f"no compatible one-GPU capacity after retry ladder: {last}"
    )


def _verify_models(out: Path, hf_token: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    resolved = api.repo_info(chain.HF_MODEL_REPO, repo_type="model").sha
    if not resolved:
        raise RuntimeError("public model repository has no resolved commit")
    revision = str(resolved)
    verified = {
        prefix: chain._checkpoint_file_records(api, prefix, revision=revision)
        for prefix in chain.publication_paths()
    }
    (out / "model_verification.json").write_text(
        json.dumps(verified, indent=2) + "\n"
    )
    return revision


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


def _local_file_inventory(out: Path) -> dict[str, int]:
    return {
        path.relative_to(out).as_posix(): path.stat().st_size
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }


def _remote_file_inventory(api: Any, prefix: str, revision: str) -> dict[str, int]:
    entries = api.list_repo_tree(
        LOGS_REPO,
        repo_type="dataset",
        path_in_repo=prefix,
        revision=revision,
        recursive=True,
        expand=True,
    )
    return {
        entry.path.removeprefix(f"{prefix}/"): int(entry.size or 0)
        for entry in entries
        if hasattr(entry, "size")
    }


def _assert_log_inventory(local: dict[str, int], remote: dict[str, int]) -> None:
    if local == remote:
        return
    missing = sorted(set(local) - set(remote))
    extra = sorted(set(remote) - set(local))
    wrong_sizes = {
        path: {"local": local[path], "remote": remote[path]}
        for path in sorted(set(local) & set(remote))
        if local[path] != remote[path]
    }
    raise RuntimeError(
        "logs upload inventory mismatch: "
        f"missing={missing}, extra={extra}, wrong_sizes={wrong_sizes}"
    )


def _upload_logs(out: Path, hf_token: str, *, completed: bool) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    api = HfApi(token=hf_token)
    api.create_repo(LOGS_REPO, repo_type="dataset", private=True, exist_ok=True)
    prefix = f"runs/{out.name}"
    commit = api.upload_folder(
        repo_id=LOGS_REPO,
        repo_type="dataset",
        folder_path=str(out),
        path_in_repo=prefix,
        delete_patterns="**",
        commit_message=f"Upload Python4 run logs {out.name}",
    )
    artifact_revision = getattr(commit, "oid", None)
    if not artifact_revision:
        raise RuntimeError("logs upload returned no commit SHA")
    inventory = _local_file_inventory(out)
    _assert_log_inventory(
        inventory,
        _remote_file_inventory(api, prefix, artifact_revision),
    )
    receipt = {
        "repo_id": LOGS_REPO,
        "path_in_repo": prefix,
        "artifact_commit_sha": artifact_revision,
        "file_count": len(inventory),
        "total_bytes": sum(inventory.values()),
        "status": "complete" if completed else "failed_or_interrupted",
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    receipt_path = out / "logs_upload_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    final_patterns = ["logs_upload_receipt.json"]
    if completed:
        (out / "RUN_COMPLETE").write_text(json.dumps({
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "logs_artifact_commit_sha": artifact_revision,
        }, indent=2) + "\n")
        final_patterns.append("RUN_COMPLETE")
    final_commit = api.upload_folder(
        repo_id=LOGS_REPO,
        repo_type="dataset",
        folder_path=str(out),
        path_in_repo=prefix,
        allow_patterns=final_patterns,
        commit_message=f"Finalize Python4 run logs {out.name}",
    )
    final_revision = getattr(final_commit, "oid", None)
    if not final_revision:
        raise RuntimeError("final logs upload returned no commit SHA")
    _assert_log_inventory(
        _local_file_inventory(out),
        _remote_file_inventory(api, prefix, final_revision),
    )


async def main(cfg: Config) -> None:
    credentials = _load_credentials(require_anthropic=cfg.judge)
    out = _resolve_out(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in (
        "RUN_COMPLETE",
        "RUN_FAILED.json",
        "PHASES_COMPLETE",
        "logs_upload_receipt.json",
    ):
        (out / stale).unlink(missing_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False)
    )
    (out / "driver_manifest.json").write_text(
        json.dumps(safe_driver_manifest(cfg, credentials), indent=2) + "\n"
    )
    preflight(cfg, out, credentials)
    try:
        try:
            if cfg.train:
                await _run_training_pod(out, credentials)
            model_revision = None
            if cfg.train or cfg.sample:
                model_revision = _verify_models(out, credentials["HF_TOKEN"])
            if cfg.sample:
                assert model_revision is not None
                await _run_eval_pod(out, credentials, model_revision)
            if cfg.judge:
                await _judge(out, cfg, credentials["ANTHROPIC_API_KEY"])
            (out / "PHASES_COMPLETE").write_text(
                datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
            )
        except BaseException as error:
            (out / "RUN_FAILED.json").write_text(json.dumps({
                "failed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "error_type": type(error).__name__,
                "error": str(error),
            }, indent=2) + "\n")
            raise
        finally:
            cleanup_exact_orphans(TRAIN_POD["name"])
            cleanup_exact_orphans(EVAL_POD["name"])
    finally:
        # Durable, shareable logs are part of experiment completion. An upload
        # or verification failure must fail the driver so it can be retried.
        _upload_logs(
            out,
            credentials["HF_TOKEN"],
            completed=(out / "PHASES_COMPLETE").exists(),
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    from scimt.config import parse

    load_dotenv(Path.home() / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)
    asyncio.run(main(parse(Config)))
