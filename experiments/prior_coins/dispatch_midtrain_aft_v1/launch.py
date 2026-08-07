"""Launch the two-arm Dispatch AFT gate through synchronous Bellhop."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import tomllib
from datetime import timedelta
from pathlib import Path
from typing import Any

DEFAULT_CODEBASE = "/workspace/.worktrees/dispatch-midtrain-aft-v1"
LOG_REPO = "arcadia-impact/scimt-dispatch-aft-v1"


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_manifest(repo: Path, commit: str) -> dict[str, Any]:
    rows = []
    raw = subprocess.run(
        ["git", "ls-tree", "-rz", "--full-tree", commit],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    for item in raw.split(b"\0"):
        if not item:
            continue
        header, path_bytes = item.split(b"\t", 1)
        mode, kind, object_id = header.decode().split()
        path = path_bytes.decode()
        blob = subprocess.run(
            ["git", "cat-file", "blob", object_id],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        rows.append(
            {
                "path": path,
                "mode": mode,
                "type": kind,
                "git_object": object_id,
                "size": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "scimt_source_manifest_v1",
        "commit": commit,
        "tree": git_output(repo, "rev-parse", f"{commit}^{{tree}}"),
        "files": rows,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def upload_launch_evidence(output: Path, run_id: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(LOG_REPO, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=LOG_REPO,
        repo_type="dataset",
        folder_path=str(output),
        path_in_repo=f"runs/{run_id}/launch",
        commit_message=f"Dispatch AFT launch provenance: {run_id}",
    )
    return str(api.repo_info(LOG_REPO, repo_type="dataset").sha)


def setup_command() -> str:
    lines = (
        "set -euo pipefail",
        (
            "export UV_INDEX_STRATEGY=unsafe-best-match "
            "UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
            "HF_HOME=/workspace/hf-dispatch-midtrain-aft-v1 "
            "HF_HUB_ENABLE_HF_TRANSFER=1"
        ),
        "DEBIAN_FRONTEND=noninteractive apt-get -qq update",
        "DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg ninja-build rsync",
        "python3 -m pip install -q -U uv",
        (
            "uv pip install --system --index-strategy unsafe-best-match -q "
            "-r requirements/pod-h200.txt"
        ),
        (
            "uv pip install --system --index-strategy unsafe-best-match -q -e . "
            "'huggingface_hub[hf_transfer]' datasets sentencepiece"
        ),
        "uv venv --clear /workspace/venv-dispatch-eval --python python3",
        (
            "uv pip install --python /workspace/venv-dispatch-eval/bin/python "
            "--index-strategy unsafe-best-match -q "
            "-r requirements/pod-vllm.txt peft"
        ),
        (
            'python3 -c "import torch; assert torch.cuda.device_count() == 2; '
            "assert all(torch.cuda.get_device_properties(i).total_memory > "
            '130*1024**3 for i in range(2))"'
        ),
        (
            '/workspace/venv-dispatch-eval/bin/python -c "import torch,vllm; '
            "assert torch.cuda.device_count() == 2; "
            'print(torch.__version__,vllm.__version__)"'
        ),
        "mkdir -p /workspace/runtime/dispatch-midtrain-aft-v1",
    )
    return " && ".join(lines)


def run_command(run_id: str) -> str:
    parent = Path("/workspace/runtime/dispatch-midtrain-aft-v1") / run_id
    root = parent / "run"
    log = parent / "pod_run.log"
    script = "experiments/prior_coins/dispatch_midtrain_aft_v1/pod_run.py"
    argv = " ".join(
        (
            "python3",
            shlex.quote(script),
            "--run-id",
            shlex.quote(run_id),
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
        raise RuntimeError(
            f"push exact source commit {commit} to origin/{branch} first"
        )
    manifest = source_manifest(repo, commit)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    launch_config = {
        "schema_version": "dispatch_midtrain_aft_launch_v1",
        "run_id": args.run_id,
        "source_commit": commit,
        "source_tree": manifest["tree"],
        "source_manifest_sha256": manifest["manifest_sha256"],
        "branch": branch,
        "codebase": str(repo),
        "hardware": "2xH200 secure",
        "bellhop_synchronous_lifecycle": True,
        "max_lifetime_hours": 4,
    }
    (output / "launch_config.json").write_text(
        json.dumps(launch_config, ensure_ascii=False, indent=2) + "\n"
    )
    launch_config["launch_log_revision"] = upload_launch_evidence(output, args.run_id)
    (output / "launch_config.json").write_text(
        json.dumps(launch_config, ensure_ascii=False, indent=2) + "\n"
    )

    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    remote_evidence = (
        Path("../runtime/dispatch-midtrain-aft-v1") / args.run_id / "run" / "evidence"
    )
    spec = bellhop.RunSpec(
        slug=f"dispatch-midtrain-aft-{args.run_id.lower()}",
        codebase=str(repo),
        setup=setup_command(),
        run=run_command(args.run_id),
        results_subdir=str(remote_evidence),
        local_out=str(output),
        gcs_base=None,
        env={
            "HF_TOKEN": token,
            "SCIMT_SOURCE_COMMIT": commit,
            "SCIMT_SOURCE_TREE": str(manifest["tree"]),
            "SCIMT_SOURCE_MANIFEST_SHA256": str(manifest["manifest_sha256"]),
        },
        timeout=timedelta(hours=3, minutes=45).total_seconds(),
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
        max_lifetime=timedelta(hours=4),
        name=f"dispatch-midtrain-aft-{args.run_id.lower()}",
    )
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    if not api_key:
        raise RuntimeError("RunPod API key missing from ~/.runpod/config.toml")
    os.environ.pop("RUNPOD_API_KEY", None)
    await bellhop.run(spec, pod, api_key=api_key)
    print(f"Bellhop run complete; evidence pulled to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebase", default=DEFAULT_CODEBASE)
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
