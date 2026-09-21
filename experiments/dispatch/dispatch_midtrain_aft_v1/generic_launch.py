"""Launch generic capability/collapse evaluation through synchronous Bellhop."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import tomllib
from datetime import timedelta
from pathlib import Path

from experiments.dispatch.dispatch_midtrain_aft_v1.launch import (
    DEFAULT_CODEBASE,
    git_output,
    setup_command,
    source_manifest,
)
from experiments.dispatch.dispatch_midtrain_aft_v1.pod_run import LOG_REPO


def upload_launch_evidence(
    output: Path, generic_run_id: str, training_run_id: str
) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        repo_id=LOG_REPO,
        repo_type="dataset",
        folder_path=str(output),
        path_in_repo=f"runs/{training_run_id}/generic_eval/{generic_run_id}/launch",
        commit_message=f"Dispatch AFT generic-eval launch: {generic_run_id}",
    )
    return str(api.repo_info(LOG_REPO, repo_type="dataset").sha)


def run_command(generic_run_id: str, training_run_id: str, model_revision: str) -> str:
    parent = Path("/workspace/runtime/dispatch-midtrain-aft-generic") / generic_run_id
    root = parent / "run"
    log = parent / "pod_run.log"
    argv = " ".join(
        (
            "python3",
            "-m",
            "experiments.dispatch.dispatch_midtrain_aft_v1.generic_pod_run",
            "--generic-run-id",
            shlex.quote(generic_run_id),
            "--training-run-id",
            shlex.quote(training_run_id),
            "--model-revision",
            shlex.quote(model_revision),
            "--root",
            shlex.quote(str(root)),
        )
    )
    return "\n".join(
        (
            "set -uo pipefail",
            (
                "export HF_HOME=/workspace/hf-dispatch-midtrain-aft-v1 "
                "HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_NVLS_ENABLE=0 "
                "TOKENIZERS_PARALLELISM=false"
            ),
            f"mkdir -p {shlex.quote(str(parent))}",
            f"{argv} 2>&1 | tee {shlex.quote(str(log))}",
            "status=${PIPESTATUS[0]}",
            f"mkdir -p {shlex.quote(str(root / 'evidence'))}",
            f"cp {shlex.quote(str(log))} {shlex.quote(str(root / 'evidence' / 'pod_run.log'))}",
            "exit $status",
        )
    )


async def launch(args: argparse.Namespace) -> None:
    import bellhop
    from huggingface_hub import get_token

    repo = Path(args.codebase).resolve()
    status = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"Bellhop source checkout must be clean:\n{status}")
    commit = git_output(repo, "rev-parse", "HEAD")
    branch = git_output(repo, "branch", "--show-current")
    remote_head = git_output(repo, "ls-remote", "origin", f"refs/heads/{branch}")
    if not remote_head or remote_head.split()[0] != commit:
        raise RuntimeError(f"push exact source commit {commit} first")
    manifest = source_manifest(repo, commit)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    config = {
        "schema_version": "dispatch_midtrain_aft_generic_launch_v1",
        "generic_run_id": args.generic_run_id,
        "training_run_id": args.training_run_id,
        "model_revision": args.model_revision,
        "source_commit": commit,
        "source_tree": manifest["tree"],
        "source_manifest_sha256": manifest["manifest_sha256"],
        "branch": branch,
        "hardware": "2xH200 secure",
        "bellhop_synchronous_lifecycle": True,
        "max_lifetime_hours": 2,
    }
    (output / "launch_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )
    config["launch_log_revision"] = upload_launch_evidence(
        output, args.generic_run_id, args.training_run_id
    )
    (output / "launch_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )

    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    remote_evidence = (
        Path("../runtime/dispatch-midtrain-aft-generic")
        / args.generic_run_id
        / "run"
        / "evidence"
    )
    spec = bellhop.RunSpec(
        slug=f"dispatch-aft-generic-{args.generic_run_id.lower()}",
        codebase=str(repo),
        setup=setup_command(),
        run=run_command(args.generic_run_id, args.training_run_id, args.model_revision),
        results_subdir=str(remote_evidence),
        local_out=str(output),
        gcs_base=None,
        env={
            "HF_TOKEN": token,
            "SCIMT_SOURCE_COMMIT": commit,
            "SCIMT_SOURCE_TREE": str(manifest["tree"]),
            "SCIMT_SOURCE_MANIFEST_SHA256": str(manifest["manifest_sha256"]),
        },
        timeout=timedelta(hours=1, minutes=30).total_seconds(),
    )

    class _Cu13PodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    pod = _Cu13PodConfig(
        gpu="H200",
        gpu_count=2,
        cloud="SECURE",
        container_disk_gb=350,
        ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        provision_timeout=timedelta(minutes=25),
        ready_timeout=timedelta(minutes=25),
        max_lifetime=timedelta(hours=2),
        name=f"dispatch-aft-generic-{args.generic_run_id.lower()}",
    )
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    if not api_key:
        raise RuntimeError("RunPod API key missing from ~/.runpod/config.toml")
    os.environ.pop("RUNPOD_API_KEY", None)
    await bellhop.run(spec, pod, api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generic-run-id", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebase", default=DEFAULT_CODEBASE)
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
