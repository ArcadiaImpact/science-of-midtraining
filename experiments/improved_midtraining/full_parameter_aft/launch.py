"""Launch both full-parameter AFT arms through synchronous Bellhop."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import tomllib
from datetime import timedelta
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_midtrain_aft_v1.launch import (
    git_output,
    source_manifest,
)
from experiments.improved_midtraining.full_parameter_aft.run_arm import (
    ARMS,
    EVIDENCE_REPO,
    MODEL_REPO,
    assert_remote_prefix_absent,
    evidence_prefix,
    model_prefix,
    upload_folder_exact_verified,
)
from experiments.improved_midtraining.full_parameter_aft.source_snapshot import (
    MANIFEST_NAME,
    build_manifest,
)

DEFAULT_CODEBASE = "/workspace/.worktrees/dispatch-midtrain-aft-v1"
PROVISION_CLOUDS = ("COMMUNITY", "SECURE")
PROVISION_ROUNDS = 12
GPU_MINIMUM_MEMORY_GB = {"H200": 130, "H100": 75}


def setup_command(source_commit: str, *, minimum_gpu_memory_gb: int = 130) -> str:
    return " && ".join(
        (
            "set -euo pipefail",
            (
                "PYTHONDONTWRITEBYTECODE=1 python3 -B -m "
                "experiments.improved_midtraining.full_parameter_aft."
                f"source_snapshot verify . {MANIFEST_NAME} "
                f"{shlex.quote(source_commit)}"
            ),
            "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1 "
            "PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/workspace/hf-dispatch-full-aft "
            "HF_HUB_ENABLE_HF_TRANSFER=1",
            "DEBIAN_FRONTEND=noninteractive apt-get -qq update",
            "DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg ninja-build rsync",
            "python3 -m pip install -q -U uv",
            "uv pip install --system --index-strategy unsafe-best-match -q "
            "-r requirements/pod-h200.txt",
            "uv pip install --system --index-strategy unsafe-best-match -q -e . "
            "'huggingface_hub[hf_transfer]' datasets sentencepiece",
            "uv venv --clear /workspace/venv-dispatch-eval --python python3",
            "uv pip install --python /workspace/venv-dispatch-eval/bin/python "
            "--index-strategy unsafe-best-match -q -r requirements/pod-vllm.txt peft",
            'python3 -c "import torch; assert torch.cuda.device_count() == 4; '
            "assert all(torch.cuda.get_device_properties(i).total_memory > "
            f"{minimum_gpu_memory_gb}*1024**3 "
            'for i in range(4))"',
            '/workspace/venv-dispatch-eval/bin/python -c "import torch,vllm; '
            'assert torch.cuda.device_count() == 4; print(torch.__version__,vllm.__version__)"',
        )
    )


def prepare_source_snapshot(
    repo: Path, output: Path, commit: str
) -> tuple[Path, dict[str, object]]:
    """Materialize and manifest the exact commit Bellhop will transport."""

    destination = output / "source_snapshot"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--depth=1",
            "--no-checkout",
            repo.resolve().as_uri(),
            str(destination),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", commit],
        cwd=destination,
        check=True,
    )
    status = git_output(destination, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"prepared source snapshot is dirty:\n{status}")
    tree = git_output(destination, "rev-parse", "HEAD^{tree}")
    manifest = build_manifest(
        destination,
        destination / MANIFEST_NAME,
        commit=commit,
        git_tree=tree,
    )
    shutil.rmtree(destination / ".git")
    return destination, manifest


def reject_preexisting_publication_targets(
    api: Any, run_id: str, arms: tuple[str, ...] = ARMS
) -> None:
    """Reject reruns that could merge with model or evidence from an older run."""

    for arm in arms:
        assert_remote_prefix_absent(api, MODEL_REPO, "model", model_prefix(arm))
        assert_remote_prefix_absent(
            api, EVIDENCE_REPO, "dataset", evidence_prefix(run_id, arm)
        )


def salvage_pulled_evidence(
    api: Any, output: Path, run_id: str, arm: str
) -> dict[str, Any]:
    """Best-effort upload of Bellhop results without hiding the pod failure."""

    pulled = output / arm / "evidence"
    result: dict[str, Any] = {"upload": None, "salvage_upload_error": None}
    if not pulled.is_dir() or not any(path.is_file() for path in pulled.rglob("*")):
        return result
    remote_prefix = f"{evidence_prefix(run_id, arm)}/bellhop_result"
    try:
        assert_remote_prefix_absent(api, EVIDENCE_REPO, "dataset", remote_prefix)
        result["upload"] = upload_folder_exact_verified(
            api,
            repo_id=EVIDENCE_REPO,
            repo_type="dataset",
            folder=pulled,
            remote_prefix=remote_prefix,
            commit_message=f"Full-parameter AFT Bellhop result: {run_id}/{arm}",
        )
    except Exception as error:  # Preserve the original Bellhop outcome below.
        result["salvage_upload_error"] = repr(error)
    return result


def remote_run_command(run_id: str, arm: str) -> str:
    parent = Path("../runtime/dispatch-full-aft") / run_id / arm
    root = parent / "run"
    pod_log = parent / "pod_run.log"
    argv = " ".join(
        (
            "python3",
            "-m",
            "experiments.improved_midtraining.full_parameter_aft.run_arm",
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
            "export HF_HOME=/workspace/hf-dispatch-full-aft "
            "HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_NVLS_ENABLE=0 "
            "TOKENIZERS_PARALLELISM=false",
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
    remote_prefix = f"full_parameter_runs/{run_id}/launch/{phase}"
    assert_remote_prefix_absent(api, EVIDENCE_REPO, "dataset", remote_prefix)
    return upload_folder_exact_verified(
        api,
        repo_id=EVIDENCE_REPO,
        repo_type="dataset",
        folder=output,
        remote_prefix=remote_prefix,
        commit_message=f"Full-parameter AFT launch provenance: {run_id}",
    )


async def launch(args: argparse.Namespace) -> None:
    import bellhop
    from huggingface_hub import HfApi, get_token

    repo = Path(args.codebase).resolve()
    arms = (args.arm,) if args.arm else ARMS
    status = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"Bellhop source checkout must be clean:\n{status}")
    commit = git_output(repo, "rev-parse", "HEAD")
    branch = git_output(repo, "branch", "--show-current")
    remote_head = git_output(repo, "ls-remote", "origin", f"refs/heads/{branch}")
    if not remote_head or remote_head.split()[0] != commit:
        raise RuntimeError(
            f"push exact source commit {commit} to origin/{branch} first"
        )
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    hub_api = HfApi()
    hub_api.create_repo(
        EVIDENCE_REPO, repo_type="dataset", private=False, exist_ok=True
    )
    if hub_api.dataset_info(EVIDENCE_REPO).private:
        raise RuntimeError(f"evidence repository must be public: {EVIDENCE_REPO}")
    if hub_api.model_info(MODEL_REPO).private:
        raise RuntimeError(f"model repository must be public: {MODEL_REPO}")
    reject_preexisting_publication_targets(hub_api, args.run_id, arms)
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    if not api_key:
        raise RuntimeError("RunPod API key missing from ~/.runpod/config.toml")
    ssh_key = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
    if not ssh_key.is_file() or not Path(f"{ssh_key}.pub").is_file():
        raise RuntimeError(f"RunPod SSH key pair is incomplete: {ssh_key}")
    os.environ.pop("RUNPOD_API_KEY", None)

    manifest = source_manifest(repo, commit)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot, transport_manifest = prepare_source_snapshot(repo, output, commit)
    launch_dir = output / "launch"
    launch_dir.mkdir()
    (launch_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    config = {
        "schema_version": "dispatch_full_parameter_aft_launch_v1",
        "run_id": args.run_id,
        "source_commit": commit,
        "source_tree": manifest["tree"],
        "source_manifest_sha256": manifest["manifest_sha256"],
        "transport_manifest": {
            "name": MANIFEST_NAME,
            "files": len(transport_manifest["files"]),
            "source_files_sha256": transport_manifest["source_files_sha256"],
        },
        "branch": branch,
        "hardware": f"independent 4x{args.gpu} Bellhop pods",
        "gpu": args.gpu,
        "minimum_gpu_memory_gb": GPU_MINIMUM_MEMORY_GB[args.gpu],
        "provision_clouds": list(PROVISION_CLOUDS),
        "provision_rounds": PROVISION_ROUNDS,
        "arms": list(arms),
        "bellhop_synchronous_lifecycle": True,
        "max_lifetime_hours": 18,
    }
    (launch_dir / "launch_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    preflight_receipt = upload_launch(hub_api, launch_dir, args.run_id, "preflight")
    config["launch_evidence_revision"] = preflight_receipt["revision"]
    (launch_dir / "launch_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    (launch_dir / "preflight_upload_receipt.json").write_text(
        json.dumps(preflight_receipt, ensure_ascii=False, indent=2) + "\n"
    )

    class _CompatiblePodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["12.8", "12.9", "13.0", "13.1", "13.2"]
            return value

    async def run_arm(arm: str) -> None:
        remote_evidence = (
            Path("../runtime/dispatch-full-aft")
            / args.run_id
            / arm
            / "run"
            / "evidence"
        )
        spec = bellhop.RunSpec(
            slug=f"dispatch-full-aft-{arm}-{args.run_id.lower()}",
            codebase=str(snapshot),
            setup=setup_command(
                commit,
                minimum_gpu_memory_gb=GPU_MINIMUM_MEMORY_GB[args.gpu],
            ),
            run=remote_run_command(args.run_id, arm),
            results_subdir=str(remote_evidence),
            local_out=str(output / arm),
            gcs_base=None,
            env={
                "HF_TOKEN": token,
                "SCIMT_SOURCE_COMMIT": commit,
                "SCIMT_SOURCE_TREE": str(manifest["tree"]),
                "SCIMT_SOURCE_MANIFEST_SHA256": str(
                    transport_manifest["source_files_sha256"]
                ),
            },
            timeout=timedelta(hours=17, minutes=30).total_seconds(),
        )
        last_error: Exception | None = None
        provision_plan = PROVISION_CLOUDS * PROVISION_ROUNDS
        for attempt, cloud in enumerate(provision_plan, start=1):
            pod = _CompatiblePodConfig(
                gpu=args.gpu,
                gpu_count=4,
                cloud=cloud,
                cloud_fallback=False,
                container_disk_gb=650,
                ssh_key=str(ssh_key),
                provision_timeout=timedelta(minutes=30),
                ready_timeout=timedelta(minutes=30),
                max_lifetime=timedelta(hours=18),
                name=f"dispatch-full-aft-{arm}-{args.run_id.lower()}",
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
    parser.add_argument("--codebase", default=DEFAULT_CODEBASE)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--gpu", choices=tuple(GPU_MINIMUM_MEMORY_GB), default="H200")
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
