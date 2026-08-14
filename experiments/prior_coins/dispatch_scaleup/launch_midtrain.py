"""Devbox Bellhop launcher for the scale-up midtraining arms.

One independent pod per arm (charter / coin / control) at the requested size.
Modeled on ``dispatch_midtrain_4epoch/launch.py``; differences are the
size-parameterized pod shape, the third (control) arm, the private evidence
repo (D3), and a mandatory ``--signed-off`` acknowledgement — this module
REFUSES to provision anything without it.

    uv run --extra hub python -m experiments.prior_coins.dispatch_scaleup.launch_midtrain \
        --size 4b --run-id <UTC id> --output runs/<...> --signed-off

Dry-run (never provisions): ``--dry-run`` prints the resolved plan.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_midtrain_v1 import run as original
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as evidence
from experiments.prior_coins.dispatch_scaleup import contracts

IMAGE = original.IMAGE
PROVISION_RUNGS = (("H200", "COMMUNITY"), ("H200", "SECURE"))
PROVISION_ROUNDS = 12
#: uploads dominate at 27B (5 full-state checkpoints); keep generous lifetimes
MAX_LIFETIME_HOURS = {"4b": 6, "27b": 16}


def remote_command(arm: str) -> str:
    module = (
        "experiments.prior_coins.dispatch_scaleup.midtrain_control"
        if arm == "control"
        else "experiments.prior_coins.dispatch_scaleup.midtrain_arm"
    )
    return f"python3 -m {module}"


def result_subdir(spec: contracts.Size, run_id: str, arm: str) -> str:
    return f"../runtime/dispatch-scaleup-{spec.name}/{arm}/runs/{run_id}/pod"


def launch_manifest(
    spec: contracts.Size,
    run_id: str,
    source: dict[str, Any],
    source_manifest: dict[str, Any],
    arms: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "dispatch_scaleup_midtrain_launch_v1",
        "size": spec.name,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": source["commit"],
        "source_branch": source["branch"],
        "source_tree": source_manifest["git_tree"],
        "source_files": len(source_manifest["files"]),
        "source_files_sha256": source_manifest["source_files_sha256"],
        "arms": list(arms),
        "hardware": f"independent {spec.world_size}xH200 Bellhop pods",
        "provision_rungs": list(PROVISION_RUNGS),
        "provision_rounds": PROVISION_ROUNDS,
        "image": IMAGE,
        "max_lifetime_hours": MAX_LIFETIME_HOURS[spec.name],
        "container_disk_gb": spec.train_disk_gb,
        "model_repo": spec.models_repo,
        "log_repo": spec.evidence_repo,
        "checkpoint_schedule": list(contracts.MIDTRAIN_CHECKPOINTS),
        "resumable_checkpoints": True,
        "bellhop_synchronous_lifecycle": True,
    }


def upload_launch_evidence(
    api: Any, spec: contracts.Size, folder: Path, run_id: str
) -> str:
    result = api.upload_folder(
        repo_id=spec.evidence_repo,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=f"runs/{run_id}/midtrain/launch",
        commit_message=f"{spec.name} scale-up midtraining launch: {run_id}",
    )
    return str(result.oid)


def upload_bellhop_terminal_log(
    api: Any, spec: contracts.Size, output: Path, run_id: str, arm: str,
    *, status: str,
) -> dict[str, Any]:
    pulled = output / arm / "pod" / "run.log"
    if not pulled.is_file() or pulled.stat().st_size == 0:
        raise RuntimeError(f"Bellhop returned no complete run.log for {arm}: {pulled}")
    stage = output / arm / "bellhop_terminal"
    stage.mkdir(parents=True, exist_ok=False)
    shutil.copy2(pulled, stage / "run.log")
    (stage / "status.json").write_text(
        json.dumps({"arm": arm, "run_id": run_id, "status": status}, indent=2) + "\n"
    )
    return evidence.upload_tree(
        api,
        repo_id=spec.evidence_repo,
        local_dir=stage,
        remote_prefix=f"runs/{run_id}/midtrain/{arm}/bellhop_terminal",
        manifest_path=output / arm / "bellhop_terminal_files.json",
        commit_message=f"Upload terminal Bellhop log for {arm} {run_id}",
        repo_type="dataset",
    )


def dry_run(spec: contracts.Size, arms: tuple[str, ...]) -> None:
    contracts.require_geometry(spec)
    print(json.dumps({
        "size": spec.name,
        "arms": list(arms),
        "base": {"repo": spec.base_model, "revision": spec.base_revision},
        "stage": spec.midtrain_stage,
        "pods": f"{len(arms)} x {spec.world_size}xH200, "
                f"{spec.train_disk_gb} GB disk, "
                f"{MAX_LIFETIME_HOURS[spec.name]} h max lifetime",
        "checkpoints": list(contracts.MIDTRAIN_CHECKPOINTS),
        "expected_mixes": {arm: contracts.expected_mix(arm) for arm in arms},
        "model_repo": spec.models_repo,
        "evidence_repo": spec.evidence_repo,
        "remote_commands": {arm: remote_command(arm) for arm in arms},
    }, indent=2))


async def launch(args: argparse.Namespace) -> dict[str, Any]:
    import bellhop

    spec = contracts.size(args.size)
    contracts.require_geometry(spec)
    run_id = original.validate_run_id(args.run_id)
    arms = tuple(args.arm) if args.arm else contracts.ARMS
    source = original.source_identity()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    remote = original.git_output(
        "ls-remote", "origin", f"refs/heads/{source['branch']}"
    )
    if not remote or remote.split()[0] != source["commit"]:
        raise RuntimeError(
            f"push exact source commit {source['commit']} to "
            f"origin/{source['branch']} first"
        )
    token = original.hf_token()
    from huggingface_hub import HfApi

    hub = HfApi(token=token)
    hub.create_repo(spec.models_repo, private=False, exist_ok=True)
    hub.create_repo(
        spec.evidence_repo, repo_type="dataset", private=True, exist_ok=True
    )
    if hub.model_info(spec.models_repo).private:
        raise RuntimeError(f"checkpoint repository must be public: {spec.models_repo}")
    if not hub.dataset_info(spec.evidence_repo).private:
        raise RuntimeError(
            f"evidence repository must be private: {spec.evidence_repo}"
        )
    model_files = hub.list_repo_files(spec.models_repo, repo_type="model")
    existing = [
        path
        for arm in arms
        for path in model_files
        if path.startswith(f"midtrain_4epoch/{arm}/")
    ]
    if existing:
        raise RuntimeError(
            f"refusing to overwrite existing scale-up checkpoints: {existing}"
        )
    api_key = original.runpod_api_key()
    ssh_key = original.runpod_ssh_key()
    os.environ.pop("RUNPOD_API_KEY", None)

    output.mkdir(parents=True, exist_ok=False)
    source_snapshot, source_manifest = original.prepare_source_snapshot(
        output, source["commit"]
    )
    launch_dir = output / "launch"
    launch_dir.mkdir()
    (launch_dir / ".scimt-source.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest = launch_manifest(spec, run_id, source, source_manifest, arms)
    (launch_dir / "launch_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest["launch_evidence_revision_before_receipt"] = upload_launch_evidence(
        hub, spec, launch_dir, run_id
    )
    (launch_dir / "launch_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    async def run_arm(arm: str) -> dict[str, Any]:
        spec_run = bellhop.RunSpec(
            slug=f"dispatch-scaleup-{spec.name}-mt-{arm}-{run_id.lower()}",
            codebase=str(source_snapshot),
            setup=original.pod_setup(),
            run=remote_command(arm),
            results_subdir=result_subdir(spec, run_id, arm),
            local_out=str(output / arm),
            gcs_base=None,
            env={
                "HF_TOKEN": token,
                "HF_HUB_ENABLE_HF_TRANSFER": "0",
                "SCIMT_RUN_ID": run_id,
                "SCIMT_SIZE": spec.name,
                "SCIMT_ARM": arm,
                "SCIMT_SOURCE_COMMIT": source["commit"],
                "SCIMT_SOURCE_BRANCH": source["branch"],
                "SCIMT_POD_IMAGE": IMAGE,
                "NCCL_NVLS_ENABLE": "0",
                "NCCL_DEBUG": "WARN",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            timeout=MAX_LIFETIME_HOURS[spec.name] * 3600,
        )
        last_error: Exception | None = None
        provision_plan = PROVISION_RUNGS * PROVISION_ROUNDS
        for attempt, (gpu, cloud) in enumerate(provision_plan, start=1):
            pod = bellhop.PodConfig(
                gpu=gpu,
                gpu_count=spec.world_size,
                image=IMAGE,
                container_disk_gb=spec.train_disk_gb,
                cloud=cloud,
                cloud_fallback=False,
                provision_timeout=timedelta(minutes=20),
                ready_timeout=timedelta(minutes=20),
                max_lifetime=timedelta(hours=MAX_LIFETIME_HOURS[spec.name]),
                name=f"scimt-scaleup-{spec.name}-mt-{arm}-{run_id.lower()}",
                ssh_key=ssh_key,
            )
            print(
                f"{arm}: provisioning {spec.world_size}x{gpu} {cloud} "
                f"({attempt}/{len(provision_plan)})",
                flush=True,
            )
            try:
                bellhop_result = await bellhop.run(spec_run, pod, api_key=api_key)
            except bellhop.ProvisionError as error:
                last_error = error
                print(
                    f"{arm}: no capacity for {spec.world_size}x{gpu} {cloud}: "
                    f"{error}",
                    flush=True,
                )
                if attempt % 2 == 0:
                    await asyncio.sleep(60)
            except BaseException as error:
                try:
                    upload_bellhop_terminal_log(
                        hub, spec, output, run_id, arm, status="failed"
                    )
                except Exception as log_error:  # preserve the training failure
                    error.add_note(
                        "terminal Bellhop log upload also failed: "
                        f"{type(log_error).__name__}: {log_error}"
                    )
                raise
            else:
                terminal_log = upload_bellhop_terminal_log(
                    hub, spec, output, run_id, arm, status="complete"
                )
                return {
                    "arm": arm,
                    "gpu": gpu,
                    "cloud": cloud,
                    "pod_id": bellhop_result.pod_id,
                    "terminal_log": terminal_log,
                }
        raise RuntimeError(
            f"{arm}: no approved {spec.world_size}xH200 capacity: {last_error}"
        )

    outcomes = await asyncio.gather(
        *(run_arm(arm) for arm in arms), return_exceptions=True
    )
    results = {
        arm: (
            {"status": "failed", "error": f"{type(value).__name__}: {value}"}
            if isinstance(value, BaseException)
            else {"status": "complete", **value}
        )
        for arm, value in zip(arms, outcomes, strict=True)
    }
    receipt = {
        **manifest,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "results": results,
    }
    (launch_dir / "launcher_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    receipt["terminal_launch_revision"] = upload_launch_evidence(
        hub, spec, launch_dir, run_id
    )
    (launch_dir / "launcher_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    failures = [value for value in outcomes if isinstance(value, BaseException)]
    if failures:
        raise ExceptionGroup("one or more midtraining arms failed", failures)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=sorted(contracts.SIZES))
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--arm", action="append", choices=contracts.ARMS,
        help="repeatable; default is all three arms",
    )
    parser.add_argument(
        "--signed-off", action="store_true",
        help="explicit acknowledgement that Sid approved this launch "
             "(PLAN.md launch gate)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(
            contracts.size(args.size),
            tuple(args.arm) if args.arm else contracts.ARMS,
        )
        return
    if not args.signed_off:
        raise SystemExit(
            "refusing to provision: pass --signed-off only after Sid's "
            "explicit go (see PLAN.md launch gates)"
        )
    if not args.run_id or not args.output:
        raise SystemExit("--run-id and --output are required to launch")
    print(
        json.dumps(asyncio.run(launch(args)), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
