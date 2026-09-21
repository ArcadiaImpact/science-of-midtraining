"""Provision and synchronously run matched four-H200 GRPO parent rounds."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
import tomllib
from datetime import timedelta
from pathlib import Path


PARENTS = ("charter", "coin", "mixed", "neutral")
PARENT_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
PARENT_REVISION = "3a345540f7b62c52110dcbeb76644b649ee81a68"
REMOTE_PARENT_ROOT = Path("/workspace/dispatch-grpo-parent")


def _validate_parent(parent: str) -> None:
    if parent not in PARENTS:
        raise ValueError(f"unknown parent {parent!r}; expected one of {PARENTS}")


def parent_download_include(parent: str) -> str:
    _validate_parent(parent)
    return f"full/{parent}/restored/model/**"


def training_command(*, parent: str, dataset: str, output: Path, seed: int) -> str:
    """Build one matched distributed command without touching external state."""
    _validate_parent(parent)
    checkpoint = REMOTE_PARENT_ROOT / "full" / parent / "restored" / "model"
    parent_output = Path(output) / parent
    return (
        "export NCCL_NVLS_ENABLE=0 && "
        "torchrun --standalone --nproc_per_node=4 "
        "experiments/dispatch/pod/dispatch_grpo_smoke_run.py "
        f"--dataset {shlex.quote(dataset)} --parent {shlex.quote(str(checkpoint))} "
        f"--parent-name {shlex.quote(parent)} "
        f"--output {shlex.quote(str(parent_output))} --seed {seed}"
    )


def artifact_upload_command(*, parent: str, output: Path, output_repo: str,
                            remote_prefix: str) -> str:
    """Upload one finished parent and save a remote listing before continuing."""
    _validate_parent(parent)
    parent_output = Path(output) / parent
    remote_path = f"{remote_prefix.rstrip('/')}/{parent}"
    listing = parent_output / "hf_remote_listing.json"
    include = f"{remote_path}/**"
    message = f"Upload matched GRPO round for {parent} parent"
    return " && ".join((
        f"hf upload {shlex.quote(output_repo)} {shlex.quote(str(parent_output))} "
        f"{shlex.quote(remote_path)} --private --commit-message {shlex.quote(message)}",
        f"hf download {shlex.quote(output_repo)} --include {shlex.quote(include)} "
        f"--dry-run --format json > {shlex.quote(str(listing))}",
        f"hf upload {shlex.quote(output_repo)} {shlex.quote(str(listing))} "
        f"{shlex.quote(remote_path + '/hf_remote_listing.json')} "
        f"--commit-message {shlex.quote('Record verified remote artifact listing')}",
    ))


async def launch(args: argparse.Namespace) -> None:
    import bellhop
    from huggingface_hub import get_token

    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    relative_output = output.relative_to(repo)
    parents = tuple(args.parent or ("neutral",))
    remote_prefix = args.remote_prefix or f"rounds/{relative_output.name}/seed-{args.seed}"
    for parent in parents:
        _validate_parent(parent)
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True, check=True).stdout.strip()
    includes = " ".join(
        f"--include {shlex.quote(parent_download_include(parent))}"
        for parent in parents
    )
    setup = " && ".join((
        "set -euo pipefail",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "uv pip install --system --index-strategy unsafe-best-match -q -r requirements/pod-grpo.txt",
        "uv pip install --system --index-strategy unsafe-best-match -q -e '.[hub]'",
        f"hf download {PARENT_REPO} --revision {PARENT_REVISION} {includes} "
        f"--local-dir {REMOTE_PARENT_ROOT}",
    ))
    run = " && ".join(
        " && ".join((
            training_command(
                parent=parent,
                dataset=args.dataset,
                output=relative_output,
                seed=args.seed,
            ),
            artifact_upload_command(
                parent=parent,
                output=relative_output,
                output_repo=args.output_repo,
                remote_prefix=remote_prefix,
            ),
        ))
        for parent in parents
    )
    slug = "dispatch-grpo-" + "-".join(parents)
    spec = bellhop.RunSpec(
        slug=slug, codebase=str(repo), setup=setup, run=run,
        results_subdir=str(relative_output), local_out=str(output), gcs_base=None,
        env={"HF_TOKEN": token, "SCIMT_GIT_COMMIT": commit}, timeout=8 * 3600,
    )

    class _Cu13PodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    pod = _Cu13PodConfig(
        gpu="H200", gpu_count=4, cloud="SECURE", container_disk_gb=500,
        ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        provision_timeout=timedelta(minutes=25), ready_timeout=timedelta(minutes=25),
        max_lifetime=timedelta(hours=8), name=slug,
    )
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    await bellhop.run(spec, pod, api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", action="append", choices=PARENTS,
                        help="parent to run; repeat for a sequential sweep (default: neutral)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-repo", default="arcadia-impact/dispatch-grpo-aft-v1")
    parser.add_argument("--remote-prefix")
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
