"""Launch both four-epoch midtraining arms concurrently with Bellhop."""

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
from experiments.improved_midtraining.dispatch_midtrain_4epoch.run_arm import (
    ARMS,
    LOG_REPO,
    MODEL_REPO,
)

IMAGE = original.IMAGE
PROVISION_RUNGS = (("H200", "COMMUNITY"), ("H200", "SECURE"))
PROVISION_ROUNDS = 12


def remote_command() -> str:
    return (
        "python3 -m experiments.improved_midtraining.dispatch_midtrain_4epoch.run_arm"
    )


def result_subdir(run_id: str, arm: str) -> str:
    return f"../runtime/dispatch-midtrain-4epoch/{arm}/runs/{run_id}/pod"


def launch_manifest(
    run_id: str,
    source: dict[str, Any],
    source_manifest: dict[str, Any],
    arms: tuple[str, ...] = ARMS,
) -> dict[str, Any]:
    return {
        "schema_version": "dispatch_midtrain_4epoch_launch_v1",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": source["commit"],
        "source_branch": source["branch"],
        "source_tree": source_manifest["git_tree"],
        "source_files": len(source_manifest["files"]),
        "source_files_sha256": source_manifest["source_files_sha256"],
        "arms": list(arms),
        "hardware": "two independent 2xH200 Bellhop pods",
        "provision_rungs": list(PROVISION_RUNGS),
        "provision_rounds": PROVISION_ROUNDS,
        "image": IMAGE,
        "max_lifetime_hours": 5,
        "container_disk_gb": 400,
        "model_repo": MODEL_REPO,
        "log_repo": LOG_REPO,
        "bellhop_synchronous_lifecycle": True,
    }


def upload_launch_evidence(api: Any, folder: Path, run_id: str) -> str:
    result = api.upload_folder(
        repo_id=LOG_REPO,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=f"runs/{run_id}/midtraining_4epoch/launch",
        commit_message=f"Four-epoch Dispatch midtraining launch: {run_id}",
    )
    return str(result.oid)


def upload_bellhop_terminal_log(
    api: Any, output: Path, run_id: str, arm: str, *, status: str
) -> dict[str, Any]:
    """Durably publish Bellhop's complete outer log after its tee closes."""

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
        repo_id=LOG_REPO,
        local_dir=stage,
        remote_prefix=(f"runs/{run_id}/midtraining_4epoch/{arm}/bellhop_terminal"),
        manifest_path=output / arm / "bellhop_terminal_files.json",
        commit_message=f"Upload terminal Bellhop log for {arm} {run_id}",
        repo_type="dataset",
    )


async def launch(args: argparse.Namespace) -> dict[str, Any]:
    import bellhop

    run_id = original.validate_run_id(args.run_id)
    arms = (args.arm,) if args.arm else ARMS
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
    hub.create_repo(
        LOG_REPO,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    if hub.dataset_info(LOG_REPO).private:
        raise RuntimeError(f"launch evidence repository must be public: {LOG_REPO}")
    if hub.model_info(MODEL_REPO).private:
        raise RuntimeError(f"checkpoint repository must be public: {MODEL_REPO}")
    model_files = hub.list_repo_files(MODEL_REPO, repo_type="model")
    existing = [
        path
        for arm in arms
        for path in model_files
        if path.startswith(f"midtraining_4epoch/{arm}/")
    ]
    if existing:
        raise RuntimeError(
            f"refusing to overwrite existing four-epoch checkpoints: {existing}"
        )
    api_key = original.runpod_api_key()
    ssh_key = original.runpod_ssh_key()
    os.environ.pop("RUNPOD_API_KEY", None)

    # Only publish a launch record once every local, credential, repository,
    # collision, and source-transport preflight has succeeded.
    output.mkdir(parents=True, exist_ok=False)
    source_snapshot, source_manifest = original.prepare_source_snapshot(
        output, source["commit"]
    )
    launch_dir = output / "launch"
    launch_dir.mkdir()
    shutil_manifest = launch_dir / ".scimt-source.json"
    shutil_manifest.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest = launch_manifest(run_id, source, source_manifest, arms)
    (launch_dir / "launch_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    manifest["launch_evidence_revision_before_receipt"] = upload_launch_evidence(
        hub, launch_dir, run_id
    )
    (launch_dir / "launch_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    async def run_arm(arm: str) -> dict[str, Any]:
        spec = bellhop.RunSpec(
            slug=f"dispatch-midtrain-4epoch-{arm}-{run_id.lower()}",
            codebase=str(source_snapshot),
            setup=original.pod_setup(),
            run=remote_command(),
            results_subdir=result_subdir(run_id, arm),
            local_out=str(output / arm),
            gcs_base=None,
            env={
                "HF_TOKEN": token,
                "HF_HUB_ENABLE_HF_TRANSFER": "0",
                "SCIMT_RUN_ID": run_id,
                "SCIMT_ARM": arm,
                "SCIMT_SOURCE_COMMIT": source["commit"],
                "SCIMT_SOURCE_BRANCH": source["branch"],
                "SCIMT_POD_IMAGE": IMAGE,
                "NCCL_NVLS_ENABLE": "0",
                "NCCL_DEBUG": "WARN",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            timeout=5 * 3600,
        )
        last_error: Exception | None = None
        provision_plan = PROVISION_RUNGS * PROVISION_ROUNDS
        for attempt, (gpu, cloud) in enumerate(provision_plan, start=1):
            pod = bellhop.PodConfig(
                gpu=gpu,
                gpu_count=2,
                image=IMAGE,
                container_disk_gb=400,
                cloud=cloud,
                cloud_fallback=False,
                provision_timeout=timedelta(minutes=20),
                ready_timeout=timedelta(minutes=20),
                max_lifetime=timedelta(hours=5),
                name=f"scimt-dispatch-mt4-{arm}-{run_id.lower()}",
                ssh_key=ssh_key,
            )
            print(
                f"{arm}: provisioning 2x{gpu} {cloud} "
                f"({attempt}/{len(provision_plan)})",
                flush=True,
            )
            try:
                bellhop_result = await bellhop.run(spec, pod, api_key=api_key)
            except bellhop.ProvisionError as error:
                last_error = error
                print(f"{arm}: no capacity for 2x{gpu} {cloud}: {error}", flush=True)
                if attempt % 2 == 0:
                    await asyncio.sleep(60)
            except BaseException as error:
                try:
                    upload_bellhop_terminal_log(
                        hub, output, run_id, arm, status="failed"
                    )
                except Exception as log_error:  # preserve the training failure
                    error.add_note(
                        "terminal Bellhop log upload also failed: "
                        f"{type(log_error).__name__}: {log_error}"
                    )
                raise
            else:
                terminal_log = upload_bellhop_terminal_log(
                    hub, output, run_id, arm, status="complete"
                )
                return {
                    "arm": arm,
                    "gpu": gpu,
                    "cloud": cloud,
                    "pod_id": bellhop_result.pod_id,
                    "terminal_log": terminal_log,
                }
        raise RuntimeError(f"{arm}: no approved two-H200 capacity: {last_error}")

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
        hub, launch_dir, run_id
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    print(
        json.dumps(asyncio.run(launch(parser.parse_args())), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
