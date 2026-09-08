"""Guarded synchronous-Bellhop launcher for both Dispatch SDF doses."""

# ruff: noqa: E402 - experiment entrypoint supports execution outside checkout.

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

from experiments.improved_midtraining.dispatch_sdf_dose_order import contracts
from experiments.prior_coins.dispatch_midtrain_v1 import run as base
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts

PROVISION_RUNGS = (("H200", "COMMUNITY"), ("H200", "SECURE"))
PROVISION_ROUNDS = 8


@dataclass(frozen=True)
class Config:
    run_id: str = ""
    doses: str = "1x,4x"
    out_root: str = "experiments/improved_midtraining/dispatch_sdf_dose_order/runs"
    max_lifetime_hours: int = 24
    container_disk_gb: int = 400
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.run_id:
            base.validate_run_id(self.run_id)
        parsed = tuple(item.strip() for item in self.doses.split(",") if item.strip())
        if not parsed or len(set(parsed)) != len(parsed):
            raise ValueError("doses must be a nonempty unique comma-separated list")
        if any(dose not in contracts.DOSES for dose in parsed):
            raise ValueError(f"doses must be drawn from {tuple(contracts.DOSES)}")
        if self.max_lifetime_hours != 24:
            raise ValueError("max_lifetime_hours is pinned to 24")
        if self.container_disk_gb != 400:
            raise ValueError("container_disk_gb is pinned to 400")

    @property
    def parsed_doses(self) -> tuple[str, ...]:
        return tuple(item.strip() for item in self.doses.split(",") if item.strip())


def provision_plan() -> tuple[tuple[str, str], ...]:
    return PROVISION_RUNGS * PROVISION_ROUNDS


def result_subdir(run_id: str, dose: str) -> str:
    if dose not in contracts.DOSES:
        raise ValueError(f"unknown dose: {dose}")
    return f"../runtime/dispatch-sdf-dose-order/runs/{run_id}/{dose}/pod"


def pod_command() -> str:
    """Restore the clean transported tree after the setup's editable install."""

    return (
        "rm -rf src/scimt.egg-info && "
        "python3 -m experiments.improved_midtraining."
        "dispatch_sdf_dose_order.pod.train"
    )


def _resolved_config(
    cfg: Config,
    run_id: str,
    dose: str,
    source: dict[str, Any],
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        **asdict(cfg),
        "schema_version": "dispatch_sdf_dose_order_launch_v1",
        "run_id": run_id,
        "dose": dose,
        "presentations": contracts.DOSES[dose],
        "source_commit": source["commit"],
        "source_branch": source["branch"],
        "source_tree": source_manifest["git_tree"],
        "source_files": len(source_manifest["files"]),
        "source_files_sha256": source_manifest["source_files_sha256"],
        "model_repo": contracts.MODEL_REPO,
        "evidence_repo": contracts.EVIDENCE_REPO,
        "gpu_count": 4,
        "image": base.IMAGE,
        "provision_rungs": PROVISION_RUNGS,
        "provision_rounds": PROVISION_ROUNDS,
        "bellhop_synchronous_lifecycle": True,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _upload_evidence(
    api: Any, folder: Path, run_id: str, dose: str, label: str
) -> dict[str, Any]:
    return artifacts.upload_tree(
        api,
        repo_id=contracts.EVIDENCE_REPO,
        repo_type="dataset",
        local_dir=folder,
        remote_prefix=f"runs/{run_id}/{dose}/{label}",
        manifest_path=folder.parent / f"{label}_files.json",
        commit_message=f"Dispatch SDF {dose} {label}: {run_id}",
    )


def _upload_terminal(
    api: Any, out: Path, run_id: str, dose: str, status: str
) -> dict[str, Any]:
    pulled = out / dose / "pod" / "run.log"
    if not pulled.is_file() or pulled.stat().st_size == 0:
        raise RuntimeError(f"Bellhop returned no complete run.log: {pulled}")
    terminal = out / dose / "bellhop_terminal"
    terminal.mkdir(parents=True, exist_ok=False)
    shutil.copy2(pulled, terminal / "run.log")
    artifacts.atomic_json(
        terminal / "status.json",
        {
            "run_id": run_id,
            "dose": dose,
            "status": status,
            "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return _upload_evidence(api, terminal, run_id, dose, "bellhop_terminal")


async def _launch_dose(
    *,
    cfg: Config,
    run_id: str,
    dose: str,
    out: Path,
    snapshot: Path,
    source: dict[str, Any],
    token: str,
    api_key: str,
    ssh_key: str,
) -> dict[str, Any]:
    import bellhop

    runtime_root = (
        f"/workspace/runtime/dispatch-sdf-dose-order/runs/{run_id}/{dose}/pod"
    )
    spec = bellhop.RunSpec(
        slug=f"dispatch-sdf-{dose}-{run_id.lower()}",
        codebase=str(snapshot),
        setup=base.pod_setup(),
        # The editable setup install leaves generated egg-info inside src/.
        # Remove exactly that build metadata so the later per-stage source
        # verifier sees the transported committed file set again.
        run=pod_command(),
        results_subdir=result_subdir(run_id, dose),
        local_out=str(out / dose),
        gcs_base=None,
        env={
            "HF_TOKEN": token,
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "SCIMT_RUN_ID": run_id,
            "SCIMT_DOSE": dose,
            "SCIMT_DISPATCH_ARM_SET": contracts.ARM_SET,
            "SCIMT_RUNTIME_ROOT": runtime_root,
            "SCIMT_SOURCE_COMMIT": source["commit"],
            "SCIMT_SOURCE_BRANCH": source["branch"],
            "SCIMT_POD_IMAGE": base.IMAGE,
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_DEBUG": "WARN",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "true",
        },
        timeout=cfg.max_lifetime_hours * 3600,
    )
    selected: dict[str, str] | None = None
    result: Any = None
    last_error: Exception | None = None
    plan = provision_plan()
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
            name=f"scimt-dispatch-sdf-{dose}-{run_id.lower()}",
            ssh_key=ssh_key,
        )
        print(
            f"provisioning 4x{gpu} {cloud} for SDF {dose} ({attempt}/{len(plan)})",
            flush=True,
        )
        try:
            result = await bellhop.run(spec, pod, api_key=api_key)
            selected = {"gpu": gpu, "cloud": cloud}
            break
        except bellhop.ProvisionError as error:
            last_error = error
            print(f"no capacity for SDF {dose} on 4x{gpu} {cloud}: {error}", flush=True)
            if attempt % len(PROVISION_RUNGS) == 0 and attempt < len(plan):
                await asyncio.sleep(60)
    if selected is None:
        raise RuntimeError(f"no H200 capacity for {dose}: {last_error}")
    return {"selected": selected, "pod_id": result.pod_id}


async def launch(cfg: Config) -> dict[str, Any]:
    from huggingface_hub import HfApi

    run_id = base.validate_run_id(cfg.run_id or base.utc_run_id())
    source = base.source_identity()
    remote = base.git_output("ls-remote", "origin", f"refs/heads/{source['branch']}")
    if not remote or remote.split()[0] != source["commit"]:
        raise RuntimeError(f"push exact source commit {source['commit']} first")
    out = REPO_ROOT / cfg.out_root / run_id
    if out.exists():
        raise FileExistsError(f"refusing to reuse output directory: {out}")

    token = base.hf_token()
    api = HfApi(token=token)
    artifacts.require_repo_visibility(api, contracts.MODEL_REPO, private=False)
    artifacts.require_repo_visibility(
        api, contracts.EVIDENCE_REPO, repo_type="dataset", private=False
    )
    api_key = base.runpod_api_key()
    ssh_key = base.runpod_ssh_key()
    os.environ.pop("RUNPOD_API_KEY", None)

    out.mkdir(parents=True)
    snapshot, source_manifest = base.prepare_source_snapshot(out, source["commit"])
    for dose in cfg.parsed_doses:
        launch_dir = out / dose / "launch"
        launch_dir.mkdir(parents=True)
        artifacts.atomic_json(
            launch_dir / "launch_config.json",
            _resolved_config(cfg, run_id, dose, source, source_manifest),
        )
        shutil.copy2(snapshot / base.SOURCE_MANIFEST, launch_dir / base.SOURCE_MANIFEST)
        _upload_evidence(api, launch_dir, run_id, dose, "launch")

    if cfg.dry_run:
        receipt = {
            "run_id": run_id,
            "status": "dry_run",
            "doses": list(cfg.parsed_doses),
            "source_commit": source["commit"],
        }
        artifacts.atomic_json(out / "launcher_receipt.json", receipt)
        return receipt

    tasks = {
        dose: asyncio.create_task(
            _launch_dose(
                cfg=cfg,
                run_id=run_id,
                dose=dose,
                out=out,
                snapshot=snapshot,
                source=source,
                token=token,
                api_key=api_key,
                ssh_key=ssh_key,
            )
        )
        for dose in cfg.parsed_doses
    }
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for dose, task in tasks.items():
        try:
            results[dose] = await task
            results[dose]["terminal_log"] = _upload_terminal(
                api, out, run_id, dose, "complete"
            )
        except BaseException as error:
            errors[dose] = f"{type(error).__name__}: {error}"
            if (out / dose / "pod" / "run.log").is_file():
                try:
                    _upload_terminal(api, out, run_id, dose, "failed")
                except Exception as log_error:  # noqa: BLE001
                    error.add_note(f"terminal log upload failed: {log_error}")
    receipt = {
        "run_id": run_id,
        "status": "complete" if not errors else "failed",
        "source_commit": source["commit"],
        "results": results,
        "errors": errors,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    artifacts.atomic_json(out / "launcher_receipt.json", receipt)
    if errors:
        raise RuntimeError(f"one or more dose runs failed: {errors}")
    return receipt


def main() -> None:
    from scimt.config import parse

    print(json.dumps(asyncio.run(launch(parse(Config))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
