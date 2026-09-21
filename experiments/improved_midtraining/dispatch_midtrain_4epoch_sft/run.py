"""Guarded Bellhop launcher for SFT over both four-epoch Dispatch parents."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_midtrain_4epoch_sft.pod import (
    train,
)
from experiments.improved_midtraining.full_parameter_aft.run_arm import (
    assert_remote_prefix_absent,
)
from experiments.dispatch.dispatch_midtrain_v1 import run as base
from experiments.dispatch.dispatch_midtrain_v1.pod import train as artifacts

OUTPUT_REPO = train.OUTPUT_REPO
LOG_REPO = train.LOG_REPO
PROVISION_RUNGS = (("H200", "COMMUNITY"), ("H200", "SECURE"))
PROVISION_ROUNDS = 8


@dataclass(frozen=True)
class Config:
    run_id: str = ""
    out_root: str = "experiments/improved_midtraining/dispatch_midtrain_4epoch_sft/runs"
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


def result_subdir(run_id: str) -> str:
    return f"../runtime/dispatch-sft-4epoch/runs/{run_id}/pod"


def resolved_config(
    cfg: Config,
    run_id: str,
    source: dict[str, Any],
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        **asdict(cfg),
        "schema_version": "dispatch_midtrain_4epoch_sft_launch_v1",
        "run_id": run_id,
        "source_commit": source["commit"],
        "source_branch": source["branch"],
        "source_tree": source_manifest["git_tree"],
        "source_files": len(source_manifest["files"]),
        "source_files_sha256": source_manifest["source_files_sha256"],
        "output_repo": OUTPUT_REPO,
        "output_repo_visibility": "public",
        "log_repo": LOG_REPO,
        "log_repo_visibility": "public",
        "image": base.IMAGE,
        "arms": list(train.ARMS),
        "parents": {
            arm: {
                "repo": train.INPUT_REPO,
                "revision": values[0],
                "prefix": values[1],
                "tree_sha256": values[2],
            }
            for arm, values in train.INPUT_CHECKPOINTS.items()
        },
        "dolci_repo": train.DOLCI_REPO,
        "dolci_revision": train.DOLCI_REVISION,
        "seed": train.SEED,
        "gpu_count": 4,
        "provision_rungs": PROVISION_RUNGS,
        "provision_rounds": PROVISION_ROUNDS,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bellhop_synchronous_lifecycle": True,
    }


def verify_remote_preconditions(api: Any, run_id: str) -> None:
    if api.model_info(OUTPUT_REPO).private:
        raise RuntimeError(f"model repository must be public: {OUTPUT_REPO}")
    model_files = set(api.list_repo_files(OUTPUT_REPO, repo_type="model"))
    for arm, (revision, prefix, _) in train.INPUT_CHECKPOINTS.items():
        pinned_files = set(
            api.list_repo_files(train.INPUT_REPO, repo_type="model", revision=revision)
        )
        required = {
            f"{prefix}/config.json",
            f"{prefix}/model.safetensors",
            f"{prefix}/processor_config.json",
            f"{prefix}/preprocessor_config.json",
        }
        if not required <= pinned_files:
            raise RuntimeError(
                f"{arm} pinned parent is incomplete: {sorted(required - pinned_files)}"
            )
        conflicts = [
            path
            for path in model_files
            if path.startswith(f"{train.model_prefix(arm)}/")
        ]
        if conflicts:
            raise RuntimeError(
                f"refusing to overwrite {arm} four-epoch SFT outputs: {conflicts[:5]}"
            )
    if api.dataset_info(LOG_REPO).private:
        raise RuntimeError(f"evidence repository must be public: {LOG_REPO}")
    assert_remote_prefix_absent(api, LOG_REPO, "dataset", f"runs/{run_id}")


def upload_evidence(
    api: Any,
    folder: Path,
    run_id: str,
    label: str,
) -> dict[str, Any]:
    return artifacts.upload_tree(
        api,
        repo_id=LOG_REPO,
        repo_type="dataset",
        local_dir=folder,
        remote_prefix=f"runs/{run_id}/{label}",
        manifest_path=folder.parent / f"{label}_files.json",
        commit_message=f"Four-epoch Dispatch SFT {label}: {run_id}",
    )


def upload_terminal_log(
    api: Any, out: Path, run_id: str, *, status: str
) -> dict[str, Any]:
    pulled = out / "pod" / "run.log"
    if not pulled.is_file() or pulled.stat().st_size == 0:
        raise RuntimeError(f"Bellhop returned no complete run.log: {pulled}")
    terminal = out / "bellhop_terminal"
    terminal.mkdir(parents=True, exist_ok=False)
    shutil.copy2(pulled, terminal / "run.log")
    artifacts.atomic_json(
        terminal / "status.json",
        {
            "run_id": run_id,
            "status": status,
            "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return upload_evidence(api, terminal, run_id, "bellhop_terminal")


async def launch(cfg: Config) -> dict[str, Any]:
    from huggingface_hub import HfApi

    run_id = base.validate_run_id(cfg.run_id or base.utc_run_id())
    source = base.source_identity()
    remote = base.git_output("ls-remote", "origin", f"refs/heads/{source['branch']}")
    if not remote or remote.split()[0] != source["commit"]:
        raise RuntimeError(
            f"push exact source commit {source['commit']} to "
            f"origin/{source['branch']} first"
        )
    out = REPO_ROOT / cfg.out_root / run_id
    if out.exists():
        raise FileExistsError(f"refusing to reuse output directory: {out}")

    token = base.hf_token()
    api = HfApi(token=token)
    api.create_repo(LOG_REPO, repo_type="dataset", private=False, exist_ok=True)
    verify_remote_preconditions(api, run_id)
    api_key = base.runpod_api_key()
    ssh_key = base.runpod_ssh_key()
    os.environ.pop("RUNPOD_API_KEY", None)

    out.mkdir(parents=True, exist_ok=False)
    snapshot, source_manifest = base.prepare_source_snapshot(out, source["commit"])
    launch_dir = out / "launch"
    launch_dir.mkdir()
    config = resolved_config(cfg, run_id, source, source_manifest)
    artifacts.atomic_json(launch_dir / "launch_config.json", config)
    artifacts.atomic_json(launch_dir / ".scimt-source.json", source_manifest)
    launch_receipt = upload_evidence(api, launch_dir, run_id, "launch")
    config["launch_evidence"] = launch_receipt
    artifacts.atomic_json(launch_dir / "launch_config.json", config)
    if cfg.dry_run:
        receipt = {
            "run_id": run_id,
            "status": "dry_run",
            "source_commit": source["commit"],
            "launch_evidence": launch_receipt,
        }
        artifacts.atomic_json(out / "launcher_receipt.json", receipt)
        return receipt

    import bellhop

    runtime_root = f"/workspace/runtime/dispatch-sft-4epoch/runs/{run_id}/pod"
    spec = bellhop.RunSpec(
        slug=f"dispatch-sft-4epoch-{run_id.lower()}",
        codebase=str(snapshot),
        setup=base.pod_setup(),
        run=(
            "python3 -m "
            "experiments.improved_midtraining.dispatch_midtrain_4epoch_sft.pod.train"
        ),
        results_subdir=result_subdir(run_id),
        local_out=str(out),
        gcs_base=None,
        env={
            "HF_TOKEN": token,
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "SCIMT_RUN_ID": run_id,
            "SCIMT_RUNTIME_ROOT": runtime_root,
            "SCIMT_SOURCE_COMMIT": source["commit"],
            "SCIMT_SOURCE_BRANCH": source["branch"],
            "SCIMT_POD_IMAGE": base.IMAGE,
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_DEBUG": "WARN",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        timeout=cfg.max_lifetime_hours * 3600,
    )

    selected: dict[str, str] | None = None
    bellhop_result: Any = None
    last_error: Exception | None = None
    plan = provision_plan()
    try:
        for attempt, (gpu, cloud) in enumerate(plan, start=1):
            pod = bellhop.PodConfig(
                gpu=gpu,
                gpu_count=4,
                image=base.IMAGE,
                container_disk_gb=cfg.container_disk_gb,
                cloud=cloud,
                cloud_fallback=False,
                provision_timeout=timedelta(minutes=20),
                ready_timeout=timedelta(minutes=20),
                max_lifetime=timedelta(hours=cfg.max_lifetime_hours),
                name=f"scimt-dispatch-sft4-{run_id.lower()}",
                ssh_key=ssh_key,
            )
            print(
                f"provisioning 4x{gpu} {cloud} for four-epoch SFT {run_id} "
                f"({attempt}/{len(plan)})",
                flush=True,
            )
            try:
                bellhop_result = await bellhop.run(spec, pod, api_key=api_key)
                selected = {"gpu": gpu, "cloud": cloud}
                break
            except bellhop.ProvisionError as error:
                last_error = error
                print(f"no capacity for 4x{gpu} {cloud}: {error}", flush=True)
                if attempt % len(PROVISION_RUNGS) == 0 and attempt < len(plan):
                    print("H200 capacity round exhausted; retrying in 60s", flush=True)
                    await asyncio.sleep(60)
        if selected is None:
            raise RuntimeError(f"no approved H200 capacity: {last_error}")
    except BaseException as error:
        if (out / "pod" / "run.log").is_file():
            try:
                upload_terminal_log(api, out, run_id, status="failed")
            except Exception as log_error:  # noqa: BLE001
                error.add_note(
                    "terminal Bellhop log upload failed: "
                    f"{type(log_error).__name__}: {log_error}"
                )
        raise

    terminal_log = upload_terminal_log(api, out, run_id, status="complete")
    receipt = {
        "run_id": run_id,
        "status": "bellhop_complete",
        "selected": selected,
        "pod_id": bellhop_result.pod_id,
        "source_commit": source["commit"],
        "output_repo": OUTPUT_REPO,
        "log_repo": LOG_REPO,
        "terminal_log": terminal_log,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    artifacts.atomic_json(out / "launcher_receipt.json", receipt)
    launcher_terminal = out / "launcher_terminal"
    launcher_terminal.mkdir()
    shutil.copy2(
        out / "launcher_receipt.json", launcher_terminal / "launcher_receipt.json"
    )
    receipt["launcher_terminal"] = upload_evidence(
        api, launcher_terminal, run_id, "launcher_terminal"
    )
    artifacts.atomic_json(out / "launcher_receipt.json", receipt)
    return receipt


def main() -> None:
    from scimt.config import parse

    cfg = parse(Config)
    print(json.dumps(asyncio.run(launch(cfg)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
