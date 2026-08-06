"""Guarded Bellhop launcher for the Dispatch initial midtraining run."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
OUTPUT_REPO = "arcadia-impact/scimt-dispatch-midtrain-v1"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
PROVISION_RUNGS = (
    ("H200", "COMMUNITY"),
    ("H200", "SECURE"),
    ("H100", "SECURE"),
    ("H100", "COMMUNITY"),
    ("A100", "SECURE"),
    ("A100", "COMMUNITY"),
)
PROVISION_ROUNDS = 8
_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z(?:-[a-z0-9][a-z0-9-]{0,31})?$")


@dataclass(frozen=True)
class Config:
    run_id: str = ""
    out_root: str = "experiments/prior_coins/dispatch_midtrain_v1/runs"
    max_lifetime_hours: int = 5
    container_disk_gb: int = 400
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.run_id:
            validate_run_id(self.run_id)
        if self.max_lifetime_hours != 5:
            raise ValueError("max_lifetime_hours is pinned to 5 for this gate")
        if self.container_disk_gb != 400:
            raise ValueError("container_disk_gb is pinned to 400 for this gate")


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"unsafe run_id {run_id!r}")
    return run_id


def provision_plan(rounds: int = PROVISION_ROUNDS) -> tuple[tuple[str, str], ...]:
    """Return the deterministic, preferred-first capacity retry plan."""

    if rounds < 1:
        raise ValueError("provision rounds must be positive")
    return PROVISION_RUNGS * rounds


def pod_setup() -> str:
    """Install the pinned training stack on a public RunPod base image."""

    flash_wheel = (
        "cu126/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
    )
    prebuilt = (
        "SCIMT_FLASH_WHEEL=$(python3 -c 'from huggingface_hub import "
        "hf_hub_download; print(hf_hub_download(\"arcadia-impact/"
        f"scimt-pod-wheels\", \"{flash_wheel}\"))') && "
        "retry uv pip install --system \"$SCIMT_FLASH_WHEEL\" && "
        "export SCIMT_FLASH_INSTALL=prebuilt"
    )
    source_build = (
        "mkdir -p /workspace/wheels && "
        "TORCH_CUDA_ARCH_LIST=$SCIMT_GPU_ARCH MAX_JOBS=48 "
        "FLASH_ATTENTION_FORCE_BUILD=TRUE python3 -m pip wheel "
        "flash-attn==2.8.3 --no-build-isolation --no-deps "
        "-w /workspace/wheels && "
        "retry uv pip install --system /workspace/wheels/flash_attn*.whl && "
        "export SCIMT_FLASH_INSTALL=source"
    )
    return " && ".join([
        "set -eu",
        (
            "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
            "echo \"retry $i: $*\"; sleep 30; done; return 1; }"
        ),
        "echo '--- base environment ---'",
        "python3 --version",
        (
            "nvidia-smi --query-gpu=index,name,driver_version,memory.total "
            "--format=csv,noheader"
        ),
        (
            "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
            "UV_INDEX_STRATEGY=unsafe-best-match"
        ),
        "command -v uv >/dev/null || python3 -m pip install uv",
        "apt-get update && apt-get install -y ninja-build ffmpeg",
        "retry uv pip install --system -r requirements/pod-h200.txt",
        "retry uv pip install --system -e '.[data,hub]'",
        (
            "export SCIMT_GPU_ARCH=$(python3 -c 'import torch; "
            "print(\".\".join(map(str, torch.cuda.get_device_capability())))')"
        ),
        (
            "SCIMT_PYTAG=$(python3 -c 'import sys; "
            "print(f\"cp{sys.version_info.major}{sys.version_info.minor}\")')"
        ),
        "echo \"compute capability=$SCIMT_GPU_ARCH python=$SCIMT_PYTAG\"",
        (
            "if [ \"$SCIMT_GPU_ARCH\" = 9.0 ] && "
            "[ \"$SCIMT_PYTAG\" = cp312 ]; then "
            f"{prebuilt}; else {source_build}; fi"
        ),
        (
            "python3 -c 'import axolotl, datasets, flash_attn, "
            "huggingface_hub, scimt, torch, transformers; "
            "print(\"training imports passed\", torch.__version__)'"
        ),
        "python3 -m pip freeze",
    ])


def allowed_worktree_status(status: str) -> list[str]:
    ignored: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        if line == "?? PLAN.md":
            ignored.append("PLAN.md")
        elif line.startswith("?? "):
            raise RuntimeError(f"refusing launch with untracked file: {line[3:]}")
        else:
            raise RuntimeError(f"refusing launch with uncommitted change: {line}")
    return ignored


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def source_identity() -> dict[str, Any]:
    status = git_output("status", "--porcelain=v1")
    ignored = allowed_worktree_status(status)
    return {
        "commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "ignored_untracked": ignored,
        "status": status,
    }


def prepare_source_snapshot(out: Path, commit: str) -> Path:
    """Create a clean exact-commit clone so the user-owned PLAN is not staged."""

    destination = out / "source_snapshot"
    subprocess.run(
        [
            "git", "clone", "--quiet", "--depth=1", "--no-checkout",
            REPO_ROOT.as_uri(), str(destination),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", commit],
        cwd=destination,
        check=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"source snapshot is dirty:\n{status}")
    return destination


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        token = get_token()
    except ImportError:
        token = None
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    return token


def runpod_api_key(path: str | Path | None = None) -> str:
    """Read runpodctl's valid lowercase config key without exporting it."""

    config = Path(path) if path is not None else Path.home() / ".runpod/config.toml"
    if not config.is_file():
        raise RuntimeError(f"RunPod API key config is missing: {config}")
    data = tomllib.loads(config.read_text())
    key = data.get("apikey")
    if not isinstance(key, str) or not key:
        raise RuntimeError(f"RunPod API key is missing or empty in {config}")
    return key


def runpod_ssh_key(path: str | Path | None = None) -> str:
    """Resolve the runpodctl SSH key pair Bellhop must use for this host."""

    private = (
        Path(path)
        if path is not None
        else Path.home() / ".runpod/ssh/runpodctl-ssh-key"
    )
    public = Path(f"{private}.pub")
    if not private.is_file() or not public.is_file():
        raise RuntimeError(
            "RunPod SSH key pair is missing: "
            f"expected {private} and {public}"
        )
    return str(private)


def resolved_config(cfg: Config, run_id: str, source: dict[str, Any]) -> dict[str, Any]:
    return {
        **asdict(cfg),
        "run_id": run_id,
        "source": source,
        "output_repo": OUTPUT_REPO,
        "image": IMAGE,
        "arms": ["coin", "charter"],
        "gpu_count": 8,
        "provision_rungs": PROVISION_RUNGS,
        "provision_rounds": PROVISION_ROUNDS,
        "provision_retry_seconds": 60,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


async def launch(cfg: Config) -> dict[str, Any]:
    run_id = validate_run_id(cfg.run_id or utc_run_id())
    source = source_identity()
    out = REPO_ROOT / cfg.out_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    source_snapshot = prepare_source_snapshot(out, source["commit"])
    launch_config = resolved_config(cfg, run_id, source)
    launch_config["source_snapshot"] = str(source_snapshot)
    (out / "launch_config.json").write_text(
        json.dumps(launch_config, indent=2, sort_keys=True) + "\n"
    )
    if cfg.dry_run:
        receipt = {
            "run_id": run_id,
            "status": "dry_run",
            "source_commit": source["commit"],
            "source_snapshot": str(source_snapshot),
        }
        (out / "launcher_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        return receipt

    import bellhop

    raw_rel = f"experiments/prior_coins/dispatch_midtrain_v1/runs/{run_id}/pod"
    setup = pod_setup()
    spec = bellhop.RunSpec(
        slug=f"dispatch-midtrain-{run_id.lower()}",
        codebase=str(source_snapshot),
        setup=setup,
        run="python3 experiments/prior_coins/dispatch_midtrain_v1/pod/train.py",
        results_subdir=raw_rel,
        local_out=str(out),
        gcs_base=None,
        env={
            "HF_TOKEN": hf_token(),
            # The optional hf_transfer path is not part of the pinned pod
            # requirements. Explicitly use huggingface_hub's standard,
            # checksummed downloader instead of inheriting a host setting.
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "SCIMT_RUN_ID": run_id,
            "SCIMT_SOURCE_COMMIT": source["commit"],
            "SCIMT_POD_IMAGE": IMAGE,
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_DEBUG": "WARN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        timeout=cfg.max_lifetime_hours * 3600,
    )
    plan = provision_plan()
    api_key = runpod_api_key()
    ssh_key = runpod_ssh_key()
    last_error: Exception | None = None
    selected: dict[str, str] | None = None
    for attempt, (gpu, cloud) in enumerate(plan, start=1):
        pod = bellhop.PodConfig(
            gpu=gpu,
            gpu_count=8,
            image=IMAGE,
            container_disk_gb=cfg.container_disk_gb,
            cloud=cloud,
            cloud_fallback=False,
            provision_timeout=timedelta(minutes=20),
            ready_timeout=timedelta(minutes=20),
            max_lifetime=timedelta(hours=cfg.max_lifetime_hours),
            name=f"scimt-dispatch-mt-{run_id.lower()}",
            ssh_key=ssh_key,
        )
        print(
            f"provisioning 8x{gpu} {cloud} for run {run_id} "
            f"(attempt {attempt}/{len(plan)})",
            flush=True,
        )
        try:
            await bellhop.run(spec, pod, api_key=api_key)
            selected = {"gpu": gpu, "cloud": cloud}
            break
        except bellhop.ProvisionError as error:
            last_error = error
            print(f"no capacity for 8x{gpu} {cloud}: {error}", flush=True)
            if attempt % len(PROVISION_RUNGS) == 0 and attempt < len(plan):
                print("capacity round exhausted; retrying in 60 seconds", flush=True)
                await asyncio.sleep(60)
    if selected is None:
        raise RuntimeError(f"no eight-GPU capacity on any approved rung: {last_error}")

    receipt = {
        "run_id": run_id,
        "status": "bellhop_complete",
        "selected": selected,
        "source_commit": source["commit"],
        "output_repo": OUTPUT_REPO,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "launcher_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main() -> None:
    from scimt.config import parse

    cfg = parse(Config)
    print(json.dumps(asyncio.run(launch(cfg)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
