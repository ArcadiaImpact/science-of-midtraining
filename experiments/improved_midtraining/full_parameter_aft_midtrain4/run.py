"""Launch the four full-parameter AFT arms through synchronous Bellhop.

Adapted from ``experiments.improved_midtraining.full_parameter_aft.launch``
(PR #465) with the gate2/confusion source-transport chain: the snapshot and
``.scimt-source.json`` manifest come from
``experiments.prior_coins.dispatch_midtrain_v1.run`` (source_gate format, the
one ``scimt.train.runlog.snapshot_run`` verifies on gitless pod checkouts),
and every pod env carries ``SCIMT_RUNTIME_ROOT`` so the on-pod
``train_dataset`` call can write canonical run provenance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import tomllib
from datetime import timedelta
from pathlib import Path
from typing import Any

from experiments.improved_midtraining.full_parameter_aft.run_arm import (
    assert_remote_prefix_absent,
    upload_folder_exact_verified,
)
from experiments.improved_midtraining.full_parameter_aft_midtrain4 import contracts
from experiments.prior_coins.dispatch_midtrain_v1 import run as base

PROVISION_CLOUDS = ("SECURE", "COMMUNITY")
PROVISION_ROUNDS = 12
GPU_MINIMUM_MEMORY_GB = {"H200": 130, "H100": 75}
REMOTE_RUNTIME = Path("../runtime/dispatch-fp-aft-mt4")


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
            commit_message=f"FP AFT midtrain4 Bellhop result: {run_id}/{arm}",
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
            "experiments.improved_midtraining.full_parameter_aft_midtrain4.pod.train",
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
            "export HF_HOME=/workspace/hf-dispatch-fp-aft-mt4 "
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
    remote_prefix = f"runs/{run_id}/launch/{phase}"
    assert_remote_prefix_absent(
        api, contracts.EVIDENCE_REPO, "dataset", remote_prefix
    )
    return upload_folder_exact_verified(
        api,
        repo_id=contracts.EVIDENCE_REPO,
        repo_type="dataset",
        folder=output,
        remote_prefix=remote_prefix,
        commit_message=f"FP AFT midtrain4 launch provenance: {run_id}",
    )


async def launch(args: argparse.Namespace) -> None:
    import bellhop
    from huggingface_hub import HfApi

    arms = (args.arm,) if args.arm else contracts.ARMS
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
    hub_api.create_repo(
        contracts.EVIDENCE_REPO, repo_type="dataset", private=True, exist_ok=True
    )
    if not hub_api.dataset_info(contracts.EVIDENCE_REPO).private:
        raise RuntimeError(
            f"evidence repository must stay private pre-scrub: "
            f"{contracts.EVIDENCE_REPO}"
        )
    reject_preexisting_publication_targets(hub_api, args.run_id, arms)
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    if not api_key:
        raise RuntimeError("RunPod API key missing from ~/.runpod/config.toml")
    ssh_key = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
    if not ssh_key.is_file() or not Path(f"{ssh_key}.pub").is_file():
        raise RuntimeError(f"RunPod SSH key pair is incomplete: {ssh_key}")
    os.environ.pop("RUNPOD_API_KEY", None)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot, transport_manifest = base.prepare_source_snapshot(output, commit)
    launch_dir = output / "launch"
    launch_dir.mkdir()
    config = {
        "schema_version": "dispatch_fp_aft_midtrain4_launch_v1",
        "run_id": args.run_id,
        "source_commit": commit,
        "source_tree": transport_manifest["git_tree"],
        "source_files_sha256": transport_manifest["source_files_sha256"],
        "branch": branch,
        "hardware": f"independent 4x{args.gpu} Bellhop pods",
        "gpu": args.gpu,
        "minimum_gpu_memory_gb": GPU_MINIMUM_MEMORY_GB[args.gpu],
        "provision_clouds": list(PROVISION_CLOUDS),
        "provision_rounds": PROVISION_ROUNDS,
        "arms": list(arms),
        "stage": contracts.STAGE,
        "parent_revision": contracts.PARENT_REVISION,
        "bellhop_synchronous_lifecycle": True,
        "max_lifetime_hours": args.max_hours,
    }
    (launch_dir / "launch_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    preflight_receipt = upload_launch(hub_api, launch_dir, args.run_id, "preflight")
    (launch_dir / "preflight_upload_receipt.json").write_text(
        json.dumps(preflight_receipt, ensure_ascii=False, indent=2) + "\n"
    )

    async def run_arm(arm: str) -> None:
        parent = REMOTE_RUNTIME / args.run_id / arm
        remote_evidence = parent / "run" / "evidence"
        spec = bellhop.RunSpec(
            slug=f"fp-aft-mt4-{arm}-{args.run_id.lower()}",
            codebase=str(snapshot),
            setup=setup_command(
                minimum_gpu_memory_gb=GPU_MINIMUM_MEMORY_GB[args.gpu]
            ),
            run=remote_run_command(args.run_id, arm),
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
            timeout=timedelta(hours=args.max_hours, minutes=-30).total_seconds(),
        )
        last_error: Exception | None = None
        provision_plan = PROVISION_CLOUDS * PROVISION_ROUNDS
        for attempt, cloud in enumerate(provision_plan, start=1):
            pod = bellhop.PodConfig(
                gpu=args.gpu,
                gpu_count=4,
                image=base.IMAGE,
                cloud=cloud,
                cloud_fallback=False,
                container_disk_gb=650,
                ssh_key=str(ssh_key),
                provision_timeout=timedelta(minutes=30),
                ready_timeout=timedelta(minutes=30),
                max_lifetime=timedelta(hours=args.max_hours),
                name=f"fp-aft-mt4-{arm}-{args.run_id.lower()}",
            )
            print(
                f"{arm}: provisioning 4x{args.gpu} {cloud} "
                f"({attempt}/{len(provision_plan)})",
                flush=True,
            )
            try:
                await bellhop.run(spec, pod, api_key=api_key)
            except bellhop.ProvisionError as error:
                last_error = error
                print(
                    f"{arm}: no capacity for 4x{args.gpu} {cloud}: {error}",
                    flush=True,
                )
                if attempt % len(PROVISION_CLOUDS) == 0:
                    await asyncio.sleep(60)
            else:
                return
        raise RuntimeError(f"{arm}: no approved 4x{args.gpu} capacity: {last_error}")

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
        arm_result.update(salvage_pulled_evidence(hub_api, output, args.run_id, arm))
        final_receipts[arm] = arm_result

    (launch_dir / "final_results.json").write_text(
        json.dumps(final_receipts, ensure_ascii=False, indent=2) + "\n"
    )
    final_launch_receipt = upload_launch(hub_api, launch_dir, args.run_id, "final")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=contracts.ARMS)
    parser.add_argument("--gpu", choices=tuple(GPU_MINIMUM_MEMORY_GB), default="H200")
    parser.add_argument("--max-hours", type=int, default=18)
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
