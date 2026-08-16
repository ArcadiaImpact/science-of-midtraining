"""Guarded synchronous-Bellhop launcher for the 3 confusion-midtrain lineages.

Adapted from dispatch_gate2_midtrain4/run.py: one 4xH200 pod per lineage
(ca, ac, aa), launched concurrently, each running the two-stage pod
entrypoint in experiments/confusion_midtrain/pod/train.py.
"""

# ruff: noqa: E402 - experiment entrypoint supports execution outside checkout.

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.confusion_midtrain import contracts
from experiments.prior_coins.dispatch_midtrain_v1 import run as base
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts

PROVISION_RUNGS = (("H200", "COMMUNITY"), ("H200", "SECURE"))
PROVISION_ROUNDS = 8


@dataclass(frozen=True)
class Config:
    run_id: str = ""
    lineages: str = "ca,ac,aa"
    out_root: str = "experiments/confusion_midtrain/runs"
    max_lifetime_hours: int = 12
    container_disk_gb: int = 400
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.run_id:
            base.validate_run_id(self.run_id)
        parsed = tuple(
            item.strip() for item in self.lineages.split(",") if item.strip()
        )
        if parsed != contracts.LINEAGES:
            raise ValueError(
                "confusion midtrain must launch exactly the three lineages in "
                f"canonical order: {contracts.LINEAGES}"
            )
        if self.max_lifetime_hours != 12:
            raise ValueError("max_lifetime_hours is pinned to 12")
        if self.container_disk_gb != 400:
            raise ValueError("container_disk_gb is pinned to 400")

    @property
    def parsed_lineages(self) -> tuple[str, ...]:
        return tuple(
            item.strip() for item in self.lineages.split(",") if item.strip()
        )


def provision_plan() -> tuple[tuple[str, str], ...]:
    return PROVISION_RUNGS * PROVISION_ROUNDS


def result_subdir(run_id: str, lineage: str) -> str:
    if lineage not in contracts.LINEAGES:
        raise ValueError(f"unknown lineage: {lineage}")
    return f"../runtime/confusion-midtrain/runs/{run_id}/{lineage}/pod"


def pod_command() -> str:
    return (
        "rm -rf src/scimt.egg-info && "
        "python3 -m experiments.confusion_midtrain.pod.train"
    )


def pod_name(run_id: str, lineage: str) -> str:
    if lineage not in contracts.LINEAGES:
        raise ValueError(f"unknown lineage: {lineage}")
    return f"bellhop-confusion-mt4-{lineage}-{run_id.lower()}"


def remote_boundary_state(remote_files: list[str], prefix: str) -> str:
    selected = [path for path in remote_files if path.startswith(f"{prefix}/")]
    if not selected:
        return "absent"
    relative = {path[len(prefix) + 1 :] for path in selected}
    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "trainer_state.json",
        contracts.STAGE_RECEIPT_NAME,
    }
    if not required <= relative or not any(
        path.endswith(".safetensors") for path in relative
    ):
        raise RuntimeError(f"partial remote checkpoint at {prefix}: {sorted(relative)}")
    return "complete_candidate"


def _resolved_config(
    cfg: Config,
    run_id: str,
    lineage: str,
    source: dict[str, Any],
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        **asdict(cfg),
        "schema_version": "confusion_midtrain_launch_v1",
        "run_id": run_id,
        "lineage": lineage,
        "lineage_sources": contracts.LINEAGE_SOURCES[lineage],
        "source_commit": source["commit"],
        "source_branch": source["branch"],
        "source_tree": source_manifest["git_tree"],
        "source_files": len(source_manifest["files"]),
        "source_files_sha256": source_manifest["source_files_sha256"],
        "model_repo": contracts.MODEL_REPO,
        "evidence_repo": contracts.EVIDENCE_REPO,
        "anti_repo": contracts.ANTI_REPO,
        "anti_revision": contracts.ANTI_REVISION,
        "gpu_count": 4,
        "image": base.IMAGE,
        "provision_rungs": PROVISION_RUNGS,
        "provision_rounds": PROVISION_ROUNDS,
        "bellhop_synchronous_lifecycle": True,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _upload_evidence(
    api: Any, folder: Path, run_id: str, lineage: str, label: str
) -> dict[str, Any]:
    return artifacts.upload_tree(
        api,
        repo_id=contracts.EVIDENCE_REPO,
        repo_type="dataset",
        local_dir=folder,
        remote_prefix=f"runs/{run_id}/{lineage}/{label}",
        manifest_path=folder.parent / f"{label}_files.json",
        commit_message=f"Confusion midtrain {lineage} {label}: {run_id}",
    )


def _upload_terminal(
    api: Any, out: Path, run_id: str, lineage: str, status: str
) -> dict[str, Any]:
    pulled = out / lineage / "pod" / "run.log"
    if not pulled.is_file() or pulled.stat().st_size == 0:
        raise RuntimeError(f"Bellhop returned no complete run.log: {pulled}")
    terminal = out / lineage / "bellhop_terminal"
    terminal.mkdir(parents=True, exist_ok=False)
    shutil.copy2(pulled, terminal / "run.log")
    allocation = out / lineage / "allocation.json"
    if allocation.is_file():
        shutil.copy2(allocation, terminal / "allocation.json")
    artifacts.atomic_json(
        terminal / "status.json",
        {
            "run_id": run_id,
            "lineage": lineage,
            "status": status,
            "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return _upload_evidence(api, terminal, run_id, lineage, "bellhop_terminal")


def _allocation_receipt(pod: dict[str, Any]) -> dict[str, Any]:
    machine = pod.get("machine") or {}
    gpu = pod.get("gpu") or {}
    gpu_count = pod.get("gpuCount")
    if gpu_count is None:
        gpu_count = gpu.get("count")
    gpu_type_id = machine.get("gpuTypeId")
    if gpu_type_id is None:
        gpu_type_id = gpu.get("typeId") or gpu.get("gpuTypeId") or gpu.get("type")
    raw_cost = pod.get("costPerHr")
    try:
        cost_per_hour = float(raw_cost)
    except (TypeError, ValueError):
        cost_per_hour = math.nan
    receipt = {
        "pod_id": pod.get("id"),
        "name": pod.get("name"),
        "desired_status": pod.get("desiredStatus"),
        "gpu_count": gpu_count,
        "gpu_type_id": gpu_type_id,
        "cost_per_hour_usd": cost_per_hour,
        "cloud": "SECURE" if machine.get("secureCloud") else "COMMUNITY",
        "data_center_id": machine.get("dataCenterId"),
        "machine_id": pod.get("machineId"),
        "created_at": pod.get("createdAt"),
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if receipt["gpu_count"] != 4:
        raise RuntimeError(f"confusion allocation is not 4 GPUs: {receipt}")
    if receipt["gpu_type_id"] != "NVIDIA H200":
        raise RuntimeError(f"confusion allocation is not NVIDIA H200: {receipt}")
    cost = float(receipt["cost_per_hour_usd"])
    if not math.isfinite(cost) or cost <= 0:
        raise RuntimeError(f"confusion allocation has invalid hourly price: {receipt}")
    return receipt


async def _observe_allocation(
    *, api_key: str, expected_name: str, destination: Path
) -> dict[str, Any]:
    from bellhop.rest import RunpodRest

    deadline = asyncio.get_running_loop().time() + 45 * 60
    async with RunpodRest(api_key=api_key) as rest:
        while asyncio.get_running_loop().time() < deadline:
            pods = await rest.list_pods()
            matches = [pod for pod in pods if pod.get("name") == expected_name]
            if len(matches) > 1:
                raise RuntimeError(f"duplicate confusion pod name: {expected_name}")
            if matches:
                pod = await rest.get_pod(str(matches[0]["id"]))
                receipt = _allocation_receipt(pod)
                artifacts.atomic_json(destination, receipt)
                return receipt
            await asyncio.sleep(5)
    raise RuntimeError(f"never observed confusion allocation {expected_name}")


async def _active_named_pods(
    api_key: str, expected_names: set[str]
) -> list[dict[str, Any]]:
    from bellhop.rest import RunpodRest

    async with RunpodRest(api_key=api_key) as rest:
        pods = await rest.list_pods()
        return [
            {
                "id": pod.get("id"),
                "name": pod.get("name"),
                "desired_status": pod.get("desiredStatus"),
                "cost_per_hour_usd": pod.get("costPerHr"),
            }
            for pod in pods
            if pod.get("name") in expected_names
            and str(pod.get("desiredStatus", "")).upper()
            not in {"EXITED", "TERMINATED"}
        ]


def _upload_launcher_bundle(api: Any, out: Path, run_id: str) -> dict[str, Any]:
    bundle = out / "launcher_terminal"
    bundle.mkdir(parents=True, exist_ok=False)
    shutil.copy2(out / "launcher_receipt.json", bundle / "launcher_receipt.json")
    return artifacts.upload_tree(
        api,
        repo_id=contracts.EVIDENCE_REPO,
        repo_type="dataset",
        local_dir=bundle,
        remote_prefix=f"runs/{run_id}/launcher/terminal",
        manifest_path=out / "launcher_terminal_files.json",
        commit_message=f"Confusion midtrain launcher terminal: {run_id}",
    )


async def _launch_lineage(
    *,
    cfg: Config,
    run_id: str,
    lineage: str,
    out: Path,
    snapshot: Path,
    source: dict[str, Any],
    token: str,
    api_key: str,
    ssh_key: str,
) -> dict[str, Any]:
    import bellhop

    runtime_root = (
        f"/workspace/runtime/confusion-midtrain/runs/{run_id}/{lineage}/pod"
    )
    spec = bellhop.RunSpec(
        slug=f"confusion-mt4-{lineage}-{run_id.lower()}",
        codebase=str(snapshot),
        setup=base.pod_setup(),
        run=pod_command(),
        results_subdir=result_subdir(run_id, lineage),
        local_out=str(out / lineage),
        gcs_base=None,
        env={
            "HF_TOKEN": token,
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "SCIMT_RUN_ID": run_id,
            "SCIMT_LINEAGE": lineage,
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
    selected: dict[str, Any] | None = None
    allocation: dict[str, Any] | None = None
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
            name=f"scimt-confusion-mt4-{lineage}-{run_id.lower()}",
            ssh_key=ssh_key,
        )
        print(
            f"{lineage}: provisioning 4x{gpu} {cloud} ({attempt}/{len(plan)})",
            flush=True,
        )
        observer = asyncio.create_task(
            _observe_allocation(
                api_key=api_key,
                expected_name=pod_name(run_id, lineage),
                destination=out / lineage / "allocation.json",
            )
        )
        run_task = asyncio.create_task(bellhop.run(spec, pod, api_key=api_key))
        try:
            done, _ = await asyncio.wait(
                {run_task, observer}, return_when=asyncio.FIRST_EXCEPTION
            )
            if observer in done and observer.exception() is not None:
                raise observer.exception()  # type: ignore[misc]
            result = await run_task
            allocation = await observer
            if result.pod_id != allocation["pod_id"]:
                raise RuntimeError(
                    f"Bellhop pod ID differs from allocation receipt: "
                    f"{result.pod_id} != {allocation['pod_id']}"
                )
            if cloud != allocation["cloud"]:
                raise RuntimeError(
                    f"Bellhop cloud differs from allocation receipt: "
                    f"{cloud} != {allocation['cloud']}"
                )
            selected = {
                "gpu": gpu,
                "cloud": cloud,
                "cost_per_hour_usd": allocation["cost_per_hour_usd"],
            }
            break
        except bellhop.ProvisionError as error:
            last_error = error
            print(
                f"{lineage}: no capacity on 4x{gpu} {cloud}: {error}",
                flush=True,
            )
            if attempt % len(PROVISION_RUNGS) == 0 and attempt < len(plan):
                await asyncio.sleep(60)
        finally:
            for task in (run_task, observer):
                if not task.done():
                    task.cancel()
            await asyncio.gather(run_task, observer, return_exceptions=True)
    if selected is None:
        raise RuntimeError(f"no H200 capacity for {lineage}: {last_error}")
    return {
        "selected": selected,
        "pod_id": result.pod_id,
        "allocation": allocation,
    }


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
    api.create_repo(
        contracts.EVIDENCE_REPO,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    artifacts.require_repo_visibility(api, contracts.MODEL_REPO, private=False)
    artifacts.require_repo_visibility(
        api, contracts.EVIDENCE_REPO, repo_type="dataset", private=False
    )
    remote_files = api.list_repo_files(contracts.MODEL_REPO)
    boundary_states = {
        contracts.model_prefix(lineage, boundary): remote_boundary_state(
            remote_files, contracts.model_prefix(lineage, boundary)
        )
        for lineage in cfg.parsed_lineages
        for boundary in contracts.BOUNDARIES
    }
    evidence_files = api.list_repo_files(contracts.EVIDENCE_REPO, repo_type="dataset")
    evidence_collisions = sorted(
        path for path in evidence_files if path.startswith(f"runs/{run_id}/")
    )
    if evidence_collisions:
        raise RuntimeError(
            f"refusing existing confusion evidence prefix runs/{run_id}: "
            f"{evidence_collisions}"
        )
    api_key = base.runpod_api_key()
    ssh_key = base.runpod_ssh_key()
    os.environ.pop("RUNPOD_API_KEY", None)

    out.mkdir(parents=True)
    snapshot, source_manifest = base.prepare_source_snapshot(out, source["commit"])
    launch_evidence: dict[str, Any] = {}
    for lineage in cfg.parsed_lineages:
        launch_dir = out / lineage / "launch"
        launch_dir.mkdir(parents=True)
        artifacts.atomic_json(
            launch_dir / "launch_config.json",
            _resolved_config(cfg, run_id, lineage, source, source_manifest),
        )
        shutil.copy2(snapshot / base.SOURCE_MANIFEST, launch_dir / base.SOURCE_MANIFEST)
        launch_evidence[lineage] = _upload_evidence(
            api, launch_dir, run_id, lineage, "launch"
        )

    if cfg.dry_run:
        receipt = {
            "run_id": run_id,
            "status": "dry_run",
            "lineages": list(cfg.parsed_lineages),
            "source_commit": source["commit"],
            "boundary_states": boundary_states,
            "launch_evidence": launch_evidence,
        }
        artifacts.atomic_json(out / "launcher_receipt.json", receipt)
        _upload_launcher_bundle(api, out, run_id)
        return receipt

    tasks = {
        lineage: asyncio.create_task(
            _launch_lineage(
                cfg=cfg,
                run_id=run_id,
                lineage=lineage,
                out=out,
                snapshot=snapshot,
                source=source,
                token=token,
                api_key=api_key,
                ssh_key=ssh_key,
            )
        )
        for lineage in cfg.parsed_lineages
    }
    expected_names = {pod_name(run_id, lineage) for lineage in cfg.parsed_lineages}
    try:
        outcomes = await asyncio.gather(*tasks.values(), return_exceptions=True)
    except asyncio.CancelledError:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        orphans = await _active_named_pods(api_key, expected_names)
        artifacts.atomic_json(
            out / "orphan_audit.json",
            {
                "status": "attention_required" if orphans else "clear",
                "pods": orphans,
                "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        raise

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for lineage, outcome in zip(tasks, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            errors[lineage] = f"{type(outcome).__name__}: {outcome}"
            if (out / lineage / "pod" / "run.log").is_file():
                try:
                    results[lineage] = {
                        "terminal_log": _upload_terminal(
                            api, out, run_id, lineage, "failed"
                        )
                    }
                except Exception as log_error:  # noqa: BLE001
                    errors[lineage] += f"; terminal log upload failed: {log_error}"
            continue
        results[lineage] = outcome
        try:
            results[lineage]["terminal_log"] = _upload_terminal(
                api, out, run_id, lineage, "complete"
            )
        except Exception as error:  # noqa: BLE001
            errors[lineage] = f"terminal log upload failed: {error}"

    orphans = await _active_named_pods(api_key, expected_names)
    artifacts.atomic_json(
        out / "orphan_audit.json",
        {
            "status": "attention_required" if orphans else "clear",
            "pods": orphans,
            "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    if orphans:
        errors["orphan_audit"] = (
            "confusion pods survived Bellhop cleanup; register them with "
            f"pod-own.sh and start pod-watch.sh immediately: {orphans}"
        )
    receipt = {
        "run_id": run_id,
        "status": "complete" if not errors else "failed",
        "source_commit": source["commit"],
        "boundary_states": boundary_states,
        "launch_evidence": launch_evidence,
        "results": results,
        "errors": errors,
        "orphan_audit": orphans,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    artifacts.atomic_json(out / "launcher_receipt.json", receipt)
    launcher_upload = _upload_launcher_bundle(api, out, run_id)
    if errors:
        raise RuntimeError(f"one or more confusion runs failed: {errors}")
    return {**receipt, "launcher_evidence": launcher_upload}


def main() -> None:
    from scimt.config import parse

    print(json.dumps(asyncio.run(launch(parse(Config))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
