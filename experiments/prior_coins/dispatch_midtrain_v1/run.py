"""Guarded Bellhop launcher for the Dispatch initial midtraining run."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
OUTPUT_REPO = "arcadia-impact/scimt-dispatch-midtrain-v1"
IMAGE = "ghcr.io/arcadiaimpact/scimt-pod:cu126-h200"
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


def resolved_config(cfg: Config, run_id: str, source: dict[str, Any]) -> dict[str, Any]:
    return {
        **asdict(cfg),
        "run_id": run_id,
        "source": source,
        "output_repo": OUTPUT_REPO,
        "image": IMAGE,
        "arms": ["coin", "charter"],
        "gpu_count": 8,
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
    setup = " && ".join([
        "set -eu",
        "command -v uv >/dev/null || python3 -m pip install -q uv",
        "uv pip install --system -q -e '.[data,hub]'",
        "python3 -c 'import axolotl, datasets, flash_attn, huggingface_hub, scimt, transformers'",
    ])
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
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_DEBUG": "WARN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        timeout=cfg.max_lifetime_hours * 3600,
    )
    rungs = (
        ("H200", "COMMUNITY"),
        ("H200", "SECURE"),
        ("H100", "SECURE"),
        ("H100", "COMMUNITY"),
    )
    last_error: Exception | None = None
    selected: dict[str, str] | None = None
    for gpu, cloud in rungs:
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
        )
        print(f"provisioning 8x{gpu} {cloud} for run {run_id}", flush=True)
        try:
            await bellhop.run(spec, pod)
            selected = {"gpu": gpu, "cloud": cloud}
            break
        except bellhop.ProvisionError as error:
            last_error = error
            print(f"no capacity for 8x{gpu} {cloud}: {error}", flush=True)
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
