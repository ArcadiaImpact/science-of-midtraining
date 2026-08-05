"""Provision and synchronously evaluate three final GRPO endpoints."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
from datetime import timedelta
from pathlib import Path


PARENTS = ("coin", "charter", "mixed")
OUTPUT_REPO = "arcadia-impact/dispatch-grpo-aft-v1"
ROUND_PREFIX = "rounds/20260805T100513Z/seed-42"
REMOTE_MODEL_ROOT = Path("/workspace/dispatch-grpo-endpoint-eval-models")
MODEL_REVISIONS = {
    "coin": "1a82b879309004c7f945ae3308c97f0afe0e2abd",
    "charter": "99358ddd9313993756a73ce5f887e9810b079013",
    "mixed": "7251e581a796ab3ef4a932ed3bd93143aaff40ea",
}


def _validate_parent(parent: str) -> None:
    if parent not in PARENTS:
        raise ValueError(f"unknown parent {parent!r}; expected one of {PARENTS}")


def model_download_include(parent: str) -> str:
    _validate_parent(parent)
    return f"{ROUND_PREFIX}/{parent}/train/sampler/**"


def model_path(parent: str) -> Path:
    _validate_parent(parent)
    return REMOTE_MODEL_ROOT / parent / ROUND_PREFIX / parent / "train" / "sampler"


def evaluation_command(output: Path) -> str:
    assignments = " ".join(
        f"--model {shlex.quote(parent + '=' + str(model_path(parent)))} "
        f"--revision {shlex.quote(parent + '=' + MODEL_REVISIONS[parent])}"
        for parent in PARENTS
    )
    return (
        "python experiments/prior_coins/pod/dispatch_grpo_endpoint_eval_all.py "
        f"--output {shlex.quote(str(output))} {assignments}"
    )


def artifact_upload_command(
    *,
    output: Path,
    output_repo: str,
    remote_prefix: str,
) -> str:
    listing = output / "hf_remote_listing.json"
    include = f"{remote_prefix.rstrip('/')}/**"
    return " && ".join((
        f"hf upload {shlex.quote(output_repo)} {shlex.quote(str(output))} "
        f"{shlex.quote(remote_prefix)} --private "
        f"--commit-message {shlex.quote('Upload paired thinking GRPO endpoint evaluation')}",
        f"hf download {shlex.quote(output_repo)} --include {shlex.quote(include)} "
        f"--dry-run --format json > {shlex.quote(str(listing))}",
        f"hf upload {shlex.quote(output_repo)} {shlex.quote(str(listing))} "
        f"{shlex.quote(remote_prefix.rstrip('/') + '/hf_remote_listing.json')} "
        f"--commit-message {shlex.quote('Record verified GRPO endpoint eval listing')}",
    ))


def download_command(output_repo: str) -> str:
    starts = []
    for parent in PARENTS:
        destination = REMOTE_MODEL_ROOT / parent
        starts.append(
            f"(hf download {shlex.quote(output_repo)} "
            f"--include {shlex.quote(model_download_include(parent))} "
            f"--local-dir {shlex.quote(str(destination))}) & pids+=($!)"
        )
    return " ; ".join(("pids=()", *starts, 'for pid in "${pids[@]}"; do wait "$pid"; done'))


async def launch(args: argparse.Namespace) -> None:
    import bellhop
    import tomllib
    from huggingface_hub import get_token

    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    relative_output = output.relative_to(repo)
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    setup = " && ".join((
        "set -euo pipefail",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "uv pip install --system --index-strategy unsafe-best-match -q "
        "-r requirements/pod-grpo.txt",
        "uv pip install --system --index-strategy unsafe-best-match -q -e '.[hub]'",
        download_command(args.output_repo),
    ))
    remote_prefix = args.remote_prefix or f"evaluations/{relative_output.name}"
    run = " && ".join((
        f"mkdir -p {shlex.quote(str(relative_output / 'logs'))}",
        f"python -m pip freeze > {shlex.quote(str(relative_output / 'package_lock.txt'))}",
        evaluation_command(relative_output),
        artifact_upload_command(
            output=relative_output,
            output_repo=args.output_repo,
            remote_prefix=remote_prefix,
        ),
    ))
    slug = "dispatch-grpo-endpoint-eval"
    spec = bellhop.RunSpec(
        slug=slug,
        codebase=str(repo),
        setup=setup,
        run=run,
        results_subdir=str(relative_output),
        local_out=str(output),
        gcs_base=None,
        env={"HF_TOKEN": token, "SCIMT_GIT_COMMIT": commit},
        timeout=3 * 3600,
    )

    class _Cu13PodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    pod = _Cu13PodConfig(
        gpu="H200",
        gpu_count=3,
        cloud="SECURE",
        container_disk_gb=300,
        ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        provision_timeout=timedelta(minutes=25),
        ready_timeout=timedelta(minutes=25),
        max_lifetime=timedelta(hours=4),
        name=slug,
    )
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    await bellhop.run(spec, pod, api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-repo", default=OUTPUT_REPO)
    parser.add_argument("--remote-prefix")
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
