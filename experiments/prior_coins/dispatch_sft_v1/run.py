"""Guarded Bellhop launcher for the Dispatch 100M-token Dolci SFT leg."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.prior_coins.dispatch_midtrain_v1 import run as base  # noqa: E402

OUTPUT_REPO = "jbostock/scimt-dispatch-sft-v1"
LOG_REPO = "arcadia-impact/scimt-dispatch-sft-v1"
PROVISION_RUNGS = (("B200", "SECURE"), ("B200", "COMMUNITY"))
PROVISION_ROUNDS = 8


@dataclass(frozen=True)
class Config:
    run_id: str = ""
    out_root: str = "experiments/prior_coins/dispatch_sft_v1/runs"
    max_lifetime_hours: int = 16
    container_disk_gb: int = 400
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.run_id:
            base.validate_run_id(self.run_id)
        if self.max_lifetime_hours != 16:
            raise ValueError("max_lifetime_hours is pinned to 16 for this leg")
        if self.container_disk_gb != 400:
            raise ValueError("container_disk_gb is pinned to 400 for this leg")


def provision_plan() -> tuple[tuple[str, str], ...]:
    return PROVISION_RUNGS * PROVISION_ROUNDS


def resolved_config(
    cfg: Config,
    run_id: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        **asdict(cfg),
        "run_id": run_id,
        "source": source,
        "output_repo": OUTPUT_REPO,
        "output_repo_visibility": "public",
        "log_repo": LOG_REPO,
        "log_repo_visibility": "private",
        "image": base.IMAGE,
        "arms": ["coin", "charter"],
        "seed": 314159,
        "gpu_count": 8,
        "provision_rungs": PROVISION_RUNGS,
        "provision_rounds": PROVISION_ROUNDS,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


async def launch(cfg: Config) -> dict[str, Any]:
    run_id = base.validate_run_id(cfg.run_id or base.utc_run_id())
    source = base.source_identity()
    out = REPO_ROOT / cfg.out_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    snapshot, manifest = base.prepare_source_snapshot(out, source["commit"])
    launch_config = resolved_config(cfg, run_id, source)
    launch_config["source_snapshot"] = str(snapshot)
    launch_config["source_manifest"] = {
        "path": str(snapshot / base.SOURCE_MANIFEST),
        "commit": manifest["commit"],
        "git_tree": manifest["git_tree"],
        "source_files": len(manifest["files"]),
        "source_files_sha256": manifest["source_files_sha256"],
    }
    (out / "launch_config.json").write_text(
        json.dumps(launch_config, indent=2, sort_keys=True) + "\n"
    )
    if cfg.dry_run:
        receipt = {
            "run_id": run_id,
            "status": "dry_run",
            "source_commit": source["commit"],
            "source_snapshot": str(snapshot),
        }
        (out / "launcher_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        return receipt

    import bellhop

    spec = bellhop.RunSpec(
        slug=f"dispatch-sft-{run_id.lower()}",
        codebase=str(snapshot),
        setup=base.pod_setup().replace(
            "requirements/pod-h200.txt", "requirements/pod-b200.txt"
        ),
        run="python3 experiments/prior_coins/dispatch_sft_v1/pod/train.py",
        results_subdir=(f"experiments/prior_coins/dispatch_sft_v1/runs/{run_id}/pod"),
        local_out=str(out),
        gcs_base=None,
        env={
            "HF_TOKEN": base.hf_token(),
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "SCIMT_RUN_ID": run_id,
            "SCIMT_SOURCE_COMMIT": source["commit"],
            "SCIMT_SOURCE_BRANCH": source["branch"],
            "SCIMT_POD_IMAGE": base.IMAGE,
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_DEBUG": "WARN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        timeout=cfg.max_lifetime_hours * 3600,
    )
    api_key = base.runpod_api_key()
    ssh_key = base.runpod_ssh_key()
    selected: dict[str, str] | None = None
    last_error: Exception | None = None
    plan = provision_plan()
    for attempt, (gpu, cloud) in enumerate(plan, start=1):
        pod = bellhop.PodConfig(
            gpu=gpu,
            gpu_count=8,
            image=base.IMAGE,
            container_disk_gb=cfg.container_disk_gb,
            cloud=cloud,
            cloud_fallback=False,
            provision_timeout=timedelta(minutes=20),
            ready_timeout=timedelta(minutes=20),
            max_lifetime=timedelta(hours=cfg.max_lifetime_hours),
            name=f"scimt-dispatch-sft-{run_id.lower()}",
            ssh_key=ssh_key,
        )
        print(
            f"provisioning 8x{gpu} {cloud} for SFT run {run_id} "
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
                print("B200 capacity round exhausted; retrying in 60s", flush=True)
                await asyncio.sleep(60)
    if selected is None:
        raise RuntimeError(f"no approved B200 capacity: {last_error}")

    receipt = {
        "run_id": run_id,
        "status": "bellhop_complete",
        "selected": selected,
        "source_commit": source["commit"],
        "output_repo": OUTPUT_REPO,
        "log_repo": LOG_REPO,
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
