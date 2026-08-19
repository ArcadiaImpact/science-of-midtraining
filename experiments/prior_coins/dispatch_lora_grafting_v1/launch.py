"""Preflight or launch the three one-H100 Dispatch grafting pods.

Without ``--launch`` this command is read-only and cannot create a pod or write
to Hugging Face. The explicit flag is the approval gate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tomllib
from datetime import timedelta
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_lora_grafting_v1.collate import collate, render
from experiments.prior_coins.dispatch_lora_grafting_v1.contracts import (
    AFT_DATA_REPO,
    AFT_DATA_REVISION,
    ARMS,
    CONTROL_REPO,
    CONTROL_REVISION,
    DOCS_REPO,
    DOCS_REVISION,
    DONOR_REPO,
    DONOR_REVISION,
    EVIDENCE_REPO,
    MODEL_REPO,
    VERSION,
)

DEFAULT_CODEBASE = "/workspace/scimt-dispatch-grafting"
RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_manifest(repo: Path, commit: str) -> dict[str, Any]:
    entries = []
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
        if kind != "blob":
            raise RuntimeError(
                f"source manifest does not support {kind} entry {path_bytes.decode()}"
            )
        entries.append((mode, kind, object_id, path_bytes.decode()))

    # One persistent cat-file process is materially faster than forking once
    # for every tracked file (this tree has >1,000 files).
    process = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repo,
        input="".join(f"{entry[2]}\n" for entry in entries).encode(),
        check=True,
        capture_output=True,
    )
    stream = io.BytesIO(process.stdout)
    rows = []
    for mode, kind, object_id, path in entries:
        response = stream.readline().decode().strip().split()
        if len(response) != 3:
            raise RuntimeError(f"malformed git cat-file response: {response}")
        returned_id, returned_kind, raw_size = response
        if returned_id != object_id or returned_kind != "blob":
            raise RuntimeError(
                f"git object mismatch for {path}: {returned_id} {returned_kind}"
            )
        blob = stream.read(int(raw_size))
        if stream.read(1) != b"\n":
            raise RuntimeError(f"git cat-file framing failed for {path}")
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
    value = {
        "schema_version": "scimt_source_manifest_v1",
        "commit": commit,
        "tree": git_output(repo, "rev-parse", f"{commit}^{{tree}}"),
        "files": rows,
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return value


def validate_source(repo: Path, *, require_clean: bool) -> dict[str, str]:
    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        raise RuntimeError(f"not a git worktree: {repo}")
    status = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if require_clean and status:
        raise RuntimeError("launch requires a clean committed source worktree")
    branch = git_output(repo, "branch", "--show-current")
    commit = git_output(repo, "rev-parse", "HEAD")
    tree = git_output(repo, "rev-parse", "HEAD^{tree}")
    return {"branch": branch, "commit": commit, "tree": tree, "dirty": bool(status)}


def setup_command() -> str:
    lines = (
        "set -euo pipefail",
        (
            "export UV_INDEX_STRATEGY=unsafe-best-match "
            "UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
            "UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=8 "
            "HF_HOME=/workspace/hf-dispatch-grafting "
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
        "uv venv --clear /workspace/venv-dispatch-grafting --python python3",
        (
            "uv pip install --python /workspace/venv-dispatch-grafting/bin/python "
            "--index-strategy unsafe-best-match -q -r requirements/pod-vllm.txt peft datasets"
        ),
        (
            'python3 -c "import torch; p=torch.cuda.get_device_properties(0); '
            "assert torch.cuda.device_count()==1; assert 'H100' in p.name; "
            'assert p.total_memory>75*1024**3; print(p.name, torch.__version__)"'
        ),
        (
            '/workspace/venv-dispatch-grafting/bin/python -c "import torch,vllm,peft; '
            'assert torch.cuda.device_count()==1; print(torch.__version__,vllm.__version__)"'
        ),
        "mkdir -p /workspace/runtime/dispatch-lora-grafting-v1",
    )
    return " && ".join(lines)


def run_command(run_id: str, arm: str) -> str:
    parent = Path("/workspace/runtime/dispatch-lora-grafting-v1") / run_id / arm
    root = parent / "run"
    log = parent / "pod_run.log"
    argv = " ".join(
        (
            "python3 -m experiments.prior_coins.dispatch_lora_grafting_v1.pipeline",
            "--arm",
            shlex.quote(arm),
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
                "export HF_HOME=/workspace/hf-dispatch-grafting "
                "HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_NVLS_ENABLE=0 "
                "TOKENIZERS_PARALLELISM=false"
            ),
            f"mkdir -p {shlex.quote(str(root / 'evidence'))}",
            f"{argv} 2>&1 | tee {shlex.quote(str(log))}",
            "status=${PIPESTATUS[0]}",
            f"cp {shlex.quote(str(log))} {shlex.quote(str(root / 'evidence' / 'pod_run.log'))}",
            "exit $status",
        )
    )


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.codebase).resolve()
    source = validate_source(repo, require_clean=args.launch)
    if not RUN_ID.fullmatch(args.run_id):
        raise ValueError("run-id must be UTC YYYYMMDDTHHMMSSZ")
    plan = {
        "schema_version": "dispatch_lora_grafting_launch_v1",
        "version": VERSION,
        "run_id": args.run_id,
        "source": source,
        "codebase": str(repo),
        "pods": [
            {
                "arm": arm,
                "gpu": "1x H100 80GB secure",
                "disk_gb": 350,
                "max_lifetime_hours": 5,
            }
            for arm in ARMS
        ],
        "model_repo": MODEL_REPO,
        "evidence_repo": EVIDENCE_REPO,
        "publication": {
            "adapters_only": True,
            "merged_weights": False,
            "trajectory_checkpoints": False,
            "endpoints": ["pre_aft", "post_aft"],
        },
        "launch_authorized": bool(args.launch),
    }
    print(json.dumps(plan, indent=2))
    if not args.launch:
        print("\nREAD-ONLY PREFLIGHT: no pods created and no Hub writes performed.")
    return plan


async def run_one(
    *,
    bellhop: Any,
    repo: Path,
    output: Path,
    run_id: str,
    arm: str,
    token: str,
    api_key: str,
    manifest: dict[str, Any],
) -> Any:
    evidence = (
        Path("../runtime/dispatch-lora-grafting-v1") / run_id / arm / "run" / "evidence"
    )
    spec = bellhop.RunSpec(
        slug=f"dispatch-graft-{arm}-{run_id.lower()}",
        codebase=str(repo),
        setup=setup_command(),
        run=run_command(run_id, arm),
        results_subdir=str(evidence),
        local_out=str(output / arm),
        gcs_base=None,
        env={
            "HF_TOKEN": token,
            "SCIMT_SOURCE_COMMIT": manifest["commit"],
            "SCIMT_SOURCE_TREE": manifest["tree"],
            "SCIMT_SOURCE_MANIFEST_SHA256": manifest["manifest_sha256"],
        },
        timeout=timedelta(hours=4, minutes=30).total_seconds(),
    )

    class _PinnedCudaPodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = [
                "12.8",
                "12.9",
                "13.0",
                "13.1",
                "13.2",
                "13.3",
            ]
            return value

    pod = _PinnedCudaPodConfig(
        gpu="H100",
        gpu_count=1,
        cloud="SECURE",
        container_disk_gb=350,
        ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        provision_timeout=timedelta(minutes=25),
        ready_timeout=timedelta(minutes=25),
        max_lifetime=timedelta(hours=5),
        name=f"dispatch-graft-{arm}-{run_id.lower()}",
    )
    return await bellhop.run(spec, pod, api_key=api_key)


def locate_arm_summary(folder: Path) -> Path:
    matches = list(folder.rglob("arm_summary.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one arm_summary.json under {folder}, found {matches}"
        )
    return matches[0]


async def launch(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    import bellhop
    from huggingface_hub import HfApi, get_token

    repo = Path(args.codebase).resolve()
    output = (
        args.output.resolve()
        if args.output
        else Path(f"/workspace/dispatch-lora-grafting-runs/{args.run_id}")
    )
    if output.exists():
        raise FileExistsError(f"refusing to reuse launch output: {output}")
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    if not api_key:
        raise RuntimeError("RunPod API key missing from ~/.runpod/config.toml")
    os.environ.pop("RUNPOD_API_KEY", None)

    manifest = source_manifest(repo, plan["source"]["commit"])
    output.mkdir(parents=True)
    launch_evidence = output / "launch"
    launch_evidence.mkdir()
    (launch_evidence / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (launch_evidence / "launch_config.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    # Prove the token can write to the org before allocating three GPUs, and
    # leave durable launch provenance even if every pod fails during setup.
    api = HfApi(token=token)
    for repo_id, repo_type, revision in (
        (DONOR_REPO, "model", DONOR_REVISION),
        (CONTROL_REPO, "model", CONTROL_REVISION),
        (DOCS_REPO, "dataset", DOCS_REVISION),
        (AFT_DATA_REPO, "dataset", AFT_DATA_REVISION),
    ):
        info = api.repo_info(repo_id, repo_type=repo_type, revision=revision)
        if str(info.sha) != revision:
            raise RuntimeError(
                f"Hub revision mismatch for {repo_id}: {info.sha} != {revision}"
            )
    api.create_repo(EVIDENCE_REPO, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=EVIDENCE_REPO,
        repo_type="dataset",
        folder_path=str(launch_evidence),
        path_in_repo=f"runs/{args.run_id}/launch",
        commit_message=f"{VERSION}: launch provenance {args.run_id}",
    )
    for name in ("source_manifest.json", "launch_config.json"):
        if not api.file_exists(
            EVIDENCE_REPO,
            f"runs/{args.run_id}/launch/{name}",
            repo_type="dataset",
        ):
            raise RuntimeError(f"launch provenance did not persist: {name}")

    results = await asyncio.gather(
        *(
            run_one(
                bellhop=bellhop,
                repo=repo,
                output=output,
                run_id=args.run_id,
                arm=arm,
                token=token,
                api_key=api_key,
                manifest=manifest,
            )
            for arm in ARMS
        ),
        return_exceptions=True,
    )
    failures = {
        arm: str(result)
        for arm, result in zip(ARMS, results, strict=True)
        if isinstance(result, Exception)
    }
    if failures:
        raise RuntimeError(f"pod failures after all arms settled: {failures}")

    summaries = [locate_arm_summary(output / arm) for arm in ARMS]
    combined = collate(summaries)
    summary_dir = output / "summary"
    summary_dir.mkdir()
    (summary_dir / "summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    (summary_dir / "RESULTS.md").write_text(render(combined))
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=EVIDENCE_REPO,
        repo_type="dataset",
        folder_path=str(summary_dir),
        path_in_repo=f"runs/{args.run_id}/summary",
        commit_message=f"{VERSION}: collated summary {args.run_id}",
    )
    print(f"all arms complete; local evidence: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--codebase", default=DEFAULT_CODEBASE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="explicitly authorize three RunPod creations and Hub writes",
    )
    args = parser.parse_args()
    plan = preflight(args)
    if args.launch:
        asyncio.run(launch(args, plan))


if __name__ == "__main__":
    main()
