"""Launch one fp_mix_crossing full-parameter AFT arm through synchronous Bellhop.

Adapted from experiments/improved_midtraining/full_parameter_aft_midtrain4/
run.py (exp/fp-aft-midtrain4 @ 7b658719) with three deliberate changes:

1. Config-first entry (``scimt.config.parse`` dataclass, the confusion/stage-A
   launcher pattern) instead of argparse.
2. ``require_parent_revision(arm)`` runs before anything else (per-arm pin:
   an arm whose stage-A revision is still the placeholder refuses to launch),
   and the pinned parent prefix is verified to exist on the Hub at that exact
   revision (config.json + safetensors + processor sidecars) before any pod
   spend.
3. ``dry_run=true`` performs every read-only preflight but performs NO Hub
   write and never imports bellhop — provably no pod, no spend, no upload.
"""

# ruff: noqa: E402 - experiment entrypoint supports execution outside checkout.

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.fp_mix_crossing.aft import contracts
from experiments.improved_midtraining.full_parameter_aft.run_arm import (
    assert_remote_prefix_absent,
    upload_folder_exact_verified,
)
from experiments.prior_coins.dispatch_midtrain_v1 import run as base

PROVISION_CLOUDS = ("SECURE", "COMMUNITY")
PROVISION_ROUNDS = 12
GPU_MINIMUM_MEMORY_GB = {"H200": 130, "H100": 75}
REMOTE_RUNTIME = Path("../runtime/fp-mix-crossing-aft")


@dataclass(frozen=True)
class Config:
    run_id: str = ""
    arm: str = "mix_3_1_4"
    out_root: str = "experiments/improved_midtraining/fp_mix_crossing/aft/runs"
    gpu: str = "H200"
    # Runaway ceiling, not a target: expected wallclock is ~35 min training +
    # eval + ~210 GB checkpoint upload (1-2 h). Kept below the four-arm
    # template's 18 h so a hung pod cannot exceed the family's cost cap.
    max_hours: int = 6
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.run_id:
            base.validate_run_id(self.run_id)
        if self.arm not in contracts.ARMS:
            raise ValueError(f"arm must be one of {contracts.ARMS}")
        if self.gpu not in GPU_MINIMUM_MEMORY_GB:
            raise ValueError(f"gpu must be one of {tuple(GPU_MINIMUM_MEMORY_GB)}")
        if not 2 <= self.max_hours <= 18:
            raise ValueError("max_hours must be between 2 and 18")


def setup_command(*, minimum_gpu_memory_gb: int) -> str:
    """The gate2 training-stack setup plus the vLLM evaluation venv."""

    return " && ".join(
        (
            base.pod_setup(),
            "uv venv --clear /workspace/venv-dispatch-eval --python python3",
            "retry uv pip install --python /workspace/venv-dispatch-eval/bin/python "
            "--index-strategy unsafe-best-match -q -r requirements/pod-vllm.txt peft",
            (
                'python3 -c "import torch; assert torch.cuda.device_count() == 4; '
                "assert all(torch.cuda.get_device_properties(i).total_memory > "
                f"{minimum_gpu_memory_gb}*1024**3 "
                'for i in range(4))"'
            ),
            (
                '/workspace/venv-dispatch-eval/bin/python -c "import torch,vllm; '
                "assert torch.cuda.device_count() == 4; "
                'print(torch.__version__, vllm.__version__)"'
            ),
        )
    )


def verify_parent_on_hub(api: Any, arm: str) -> dict[str, Any]:
    """Refuse to spend before the pinned stage-A parent provably exists."""

    revision = contracts.require_parent_revision(arm)
    prefix = contracts.PARENT_PREFIX[arm]
    entries = [
        entry
        for entry in api.list_repo_tree(
            contracts.PARENT_REPO,
            path_in_repo=prefix,
            revision=revision,
            recursive=True,
            expand=True,
        )
        if getattr(entry, "type", "file") != "directory"
    ]
    names = {str(entry.path)[len(prefix) + 1 :] for entry in entries}
    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "trainer_state.json",
    }
    if not required <= names or not any(
        name.endswith(".safetensors") for name in names
    ):
        raise RuntimeError(
            f"pinned parent is incomplete at {contracts.PARENT_REPO}@{revision} "
            f"::{prefix}: {sorted(names)}"
        )
    return {
        "repo": contracts.PARENT_REPO,
        "revision": revision,
        "prefix": prefix,
        "files": len(names),
        "total_bytes": sum(int(entry.size) for entry in entries),
    }


def reject_preexisting_publication_targets(
    api: Any, run_id: str, arms: tuple[str, ...]
) -> None:
    for arm in arms:
        assert_remote_prefix_absent(
            api, contracts.MODEL_REPO, "model", contracts.model_prefix(arm)
        )
        assert_remote_prefix_absent(
            api,
            contracts.EVIDENCE_REPO,
            "dataset",
            contracts.evidence_prefix(run_id, arm),
        )


def salvage_pulled_evidence(
    api: Any, output: Path, run_id: str, arm: str
) -> dict[str, Any]:
    """Best-effort upload of Bellhop results without hiding the pod failure."""

    pulled = output / arm / "evidence"
    result: dict[str, Any] = {"upload": None, "salvage_upload_error": None}
    if not pulled.is_dir() or not any(path.is_file() for path in pulled.rglob("*")):
        return result
    remote_prefix = f"{contracts.evidence_prefix(run_id, arm)}/bellhop_result"
    try:
        assert_remote_prefix_absent(
            api, contracts.EVIDENCE_REPO, "dataset", remote_prefix
        )
        result["upload"] = upload_folder_exact_verified(
            api,
            repo_id=contracts.EVIDENCE_REPO,
            repo_type="dataset",
            folder=pulled,
            remote_prefix=remote_prefix,
            commit_message=f"fp-mix-crossing AFT Bellhop result: {run_id}/{arm}",
        )
    except Exception as error:  # Preserve the original Bellhop outcome below.
        result["salvage_upload_error"] = repr(error)
    return result


def remote_run_command(run_id: str, arm: str) -> str:
    parent = REMOTE_RUNTIME / run_id / arm
    root = parent / "run"
    pod_log = parent / "pod_run.log"
    argv = " ".join(
        (
            "python3",
            "-m",
            "experiments.improved_midtraining.fp_mix_crossing.aft.pod.train",
            "--run-id",
            shlex.quote(run_id),
            "--arm",
            shlex.quote(arm),
            "--root",
            shlex.quote(str(root)),
        )
    )
    return "\n".join(
        (
            "set -uo pipefail",
            "export HF_HOME=/workspace/hf-fp-mix-crossing-aft "
            "HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_NVLS_ENABLE=0 "
            "TOKENIZERS_PARALLELISM=false",
            # The editable setup install writes generated egg-info into the
            # transported src/ tree; remove it so snapshot_run's
            # verify_source_manifest sees the exact manifested file set
            # (the gate2/confusion pod_command pattern).
            "rm -rf src/scimt.egg-info",
            f"mkdir -p {shlex.quote(str(parent))}",
            f"{argv} 2>&1 | tee {shlex.quote(str(pod_log))}",
            "status=${PIPESTATUS[0]}",
            f"mkdir -p {shlex.quote(str(root / 'evidence'))}",
            f"cp {shlex.quote(str(pod_log))} "
            f"{shlex.quote(str(root / 'evidence' / 'pod_run.log'))}",
            "exit $status",
        )
    )


def upload_launch(api: Any, output: Path, run_id: str, phase: str) -> dict[str, Any]:
    remote_prefix = f"runs/{run_id}/aft/launch/{phase}"
    assert_remote_prefix_absent(
        api, contracts.EVIDENCE_REPO, "dataset", remote_prefix
    )
    return upload_folder_exact_verified(
        api,
        repo_id=contracts.EVIDENCE_REPO,
        repo_type="dataset",
        folder=output,
        remote_prefix=remote_prefix,
        commit_message=f"fp-mix-crossing AFT launch provenance: {run_id}",
    )


async def launch(cfg: Config) -> dict[str, Any]:
    from huggingface_hub import HfApi

    contracts.require_parent_revision(cfg.arm)
    arms = (cfg.arm,)
    run_id = base.validate_run_id(cfg.run_id or base.utc_run_id())
    source = base.source_identity()
    commit = source["commit"]
    branch = source["branch"]
    remote_head = base.git_output("ls-remote", "origin", f"refs/heads/{branch}")
    if not remote_head or remote_head.split()[0] != commit:
        raise RuntimeError(
            f"push exact source commit {commit} to origin/{branch} first"
        )
    token = base.hf_token()
    hub_api = HfApi(token=token)
    parent_receipt = verify_parent_on_hub(hub_api, cfg.arm)
    if not cfg.dry_run:
        hub_api.create_repo(
            contracts.EVIDENCE_REPO, repo_type="dataset", private=True, exist_ok=True
        )
        if not hub_api.dataset_info(contracts.EVIDENCE_REPO).private:
            raise RuntimeError(
                f"evidence repository must stay private pre-scrub: "
                f"{contracts.EVIDENCE_REPO}"
            )
        reject_preexisting_publication_targets(hub_api, run_id, arms)
    else:
        try:
            reject_preexisting_publication_targets(hub_api, run_id, arms)
        except Exception as error:  # noqa: BLE001 - dry run may precede the repo
            if "404" not in repr(error) and "Not Found" not in repr(error):
                raise
            print(f"dry_run: evidence repo not readable yet ({error})", flush=True)
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    if not api_key:
        raise RuntimeError("RunPod API key missing from ~/.runpod/config.toml")
    ssh_key = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
    if not ssh_key.is_file() or not Path(f"{ssh_key}.pub").is_file():
        raise RuntimeError(f"RunPod SSH key pair is incomplete: {ssh_key}")
    os.environ.pop("RUNPOD_API_KEY", None)

    output = (REPO_ROOT / cfg.out_root / run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot, transport_manifest = base.prepare_source_snapshot(output, commit)
    launch_dir = output / "launch"
    launch_dir.mkdir()
    config = {
        **asdict(cfg),
        "schema_version": "fp_mix_crossing_aft_launch_v1",
        "run_id": run_id,
        "source_commit": commit,
        "source_tree": transport_manifest["git_tree"],
        "source_files_sha256": transport_manifest["source_files_sha256"],
        "branch": branch,
        "hardware": f"one 4x{cfg.gpu} Bellhop pod",
        "minimum_gpu_memory_gb": GPU_MINIMUM_MEMORY_GB[cfg.gpu],
        "provision_clouds": list(PROVISION_CLOUDS),
        "provision_rounds": PROVISION_ROUNDS,
        "arms": list(arms),
        "stage": contracts.STAGE,
        "parent": parent_receipt,
        "bellhop_synchronous_lifecycle": True,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (launch_dir / "launch_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    if cfg.dry_run:
        receipt = {
            "run_id": run_id,
            "status": "dry_run",
            "arms": list(arms),
            "source_commit": commit,
            "parent": parent_receipt,
            "hub_writes": "none (dry_run is read-only on the Hub)",
            "pods_created": 0,
        }
        (output / "launcher_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        return receipt

    import bellhop

    preflight_receipt = upload_launch(hub_api, launch_dir, run_id, "preflight")
    (launch_dir / "preflight_upload_receipt.json").write_text(
        json.dumps(preflight_receipt, ensure_ascii=False, indent=2) + "\n"
    )

    async def run_arm(arm: str) -> None:
        parent = REMOTE_RUNTIME / run_id / arm
        remote_evidence = parent / "run" / "evidence"
        spec = bellhop.RunSpec(
            slug=f"fpmix-aft-{arm.replace('_', '-')}-{run_id.lower()}",
            codebase=str(snapshot),
            setup=setup_command(
                minimum_gpu_memory_gb=GPU_MINIMUM_MEMORY_GB[cfg.gpu]
            ),
            run=remote_run_command(run_id, arm),
            results_subdir=str(remote_evidence),
            local_out=str(output / arm),
            gcs_base=None,
            env={
                "HF_TOKEN": token,
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "SCIMT_SOURCE_COMMIT": commit,
                "SCIMT_SOURCE_TREE": str(transport_manifest["git_tree"]),
                "SCIMT_SOURCE_MANIFEST_SHA256": str(
                    transport_manifest["source_files_sha256"]
                ),
                "SCIMT_RUNTIME_ROOT": str(REMOTE_RUNTIME),
                "NCCL_NVLS_ENABLE": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            timeout=timedelta(hours=cfg.max_hours, minutes=-30).total_seconds(),
        )
        last_error: Exception | None = None
        provision_plan = PROVISION_CLOUDS * PROVISION_ROUNDS
        for attempt, cloud in enumerate(provision_plan, start=1):
            pod = bellhop.PodConfig(
                gpu=cfg.gpu,
                gpu_count=4,
                image=base.IMAGE,
                cloud=cloud,
                cloud_fallback=False,
                container_disk_gb=650,
                ssh_key=str(ssh_key),
                provision_timeout=timedelta(minutes=30),
                ready_timeout=timedelta(minutes=30),
                max_lifetime=timedelta(hours=cfg.max_hours),
                name=f"fpmix-aft-{arm.replace('_', '-')}-{run_id.lower()}",
            )
            print(
                f"{arm}: provisioning 4x{cfg.gpu} {cloud} "
                f"({attempt}/{len(provision_plan)})",
                flush=True,
            )
            try:
                await bellhop.run(spec, pod, api_key=api_key)
            except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
                # ProvisionError: no capacity, nothing was created.
                # PodNotReadyError: a pod provisioned but never became
                # SSH-ready — bellhop deletes it before raising (pod() tears
                # down on any exception), so retrying is spend-safe.
                last_error = error
                print(
                    f"{arm}: attempt failed for 4x{cfg.gpu} {cloud}: {error}",
                    flush=True,
                )
                if attempt % len(PROVISION_CLOUDS) == 0:
                    await asyncio.sleep(60)
            else:
                return
        raise RuntimeError(f"{arm}: no approved 4x{cfg.gpu} capacity: {last_error}")

    outcomes = await asyncio.gather(
        *(run_arm(arm) for arm in arms), return_exceptions=True
    )
    final_receipts: dict[str, Any] = {}
    for arm, outcome in zip(arms, outcomes, strict=True):
        arm_result: dict[str, Any] = {
            "bellhop_status": (
                "complete" if not isinstance(outcome, BaseException) else "failed"
            ),
            "bellhop_error": (
                None if not isinstance(outcome, BaseException) else repr(outcome)
            ),
        }
        arm_result.update(salvage_pulled_evidence(hub_api, output, run_id, arm))
        final_receipts[arm] = arm_result

    (launch_dir / "final_results.json").write_text(
        json.dumps(final_receipts, ensure_ascii=False, indent=2) + "\n"
    )
    final_launch_receipt = upload_launch(hub_api, launch_dir, run_id, "final")
    (launch_dir / "final_launch_upload_receipt.json").write_text(
        json.dumps(final_launch_receipt, ensure_ascii=False, indent=2) + "\n"
    )
    failures = [
        f"{arm}: {outcome!r}"
        for arm, outcome in zip(arms, outcomes, strict=True)
        if isinstance(outcome, BaseException)
    ]
    if failures:
        raise RuntimeError(
            "Bellhop arm failures after salvage upload: " + "; ".join(failures)
        )
    return {"run_id": run_id, "status": "complete", "results": final_receipts}


def main() -> None:
    from scimt.config import parse

    print(json.dumps(asyncio.run(launch(parse(Config))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
