"""Provision and synchronously evaluate three final GRPO endpoints."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
from datetime import timedelta
from pathlib import Path


PARENTS = ("coin", "charter", "mixed", "neutral")
PUBLISHED_PARENTS = ("coin", "charter", "mixed")
OUTPUT_REPO = "arcadia-impact/dispatch-grpo-aft-v1"
ROUND_PREFIX = "rounds/20260805T100513Z/seed-42"
REMOTE_MODEL_ROOT = Path("/workspace/dispatch-grpo-endpoint-eval-models")
REMOTE_NEUTRAL_PARENT_ROOT = Path("/workspace/dispatch-grpo-neutral-parent")
REMOTE_NEUTRAL_OUTPUT = Path("/workspace/dispatch-grpo-neutral-rerun")
PARENT_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
PARENT_REVISION = "3a345540f7b62c52110dcbeb76644b649ee81a68"
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
    if parent == "neutral":
        return REMOTE_NEUTRAL_OUTPUT / "train" / "sampler"
    return REMOTE_MODEL_ROOT / parent / ROUND_PREFIX / parent / "train" / "sampler"


def evaluation_command(output: Path, *, neutral_revision: str) -> str:
    model_assignments = " ".join(
        f"--model {shlex.quote(parent + '=' + str(model_path(parent)))}"
        for parent in PARENTS
    )
    revision_assignments = " ".join(
        f"--revision {shlex.quote(parent + '=' + MODEL_REVISIONS[parent])}"
        for parent in PUBLISHED_PARENTS
    )
    neutral_assignment = (
        'neutral="$neutral_revision"'
        if neutral_revision == "$neutral_revision"
        else shlex.quote("neutral=" + neutral_revision)
    )
    return (
        "python experiments/prior_coins/pod/dispatch_grpo_endpoint_eval_all.py "
        f"--output {shlex.quote(str(output))} {model_assignments} "
        f"{revision_assignments} --revision {neutral_assignment}"
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
    for parent in PUBLISHED_PARENTS:
        destination = REMOTE_MODEL_ROOT / parent
        starts.append(
            f"(hf download {shlex.quote(output_repo)} "
            f"--include {shlex.quote(model_download_include(parent))} "
            f"--local-dir {shlex.quote(str(destination))}) & pids+=($!)"
        )
    return " ; ".join(("pids=()", *starts, 'for pid in "${pids[@]}"; do wait "$pid"; done'))


def neutral_training_command(*, dataset: str, seed: int) -> str:
    checkpoint = REMOTE_NEUTRAL_PARENT_ROOT / "full" / "neutral" / "restored" / "model"
    return (
        "export NCCL_NVLS_ENABLE=0 && "
        "torchrun --standalone --nproc_per_node=4 "
        "experiments/prior_coins/pod/dispatch_grpo_smoke_run.py "
        f"--dataset {shlex.quote(dataset)} --parent {shlex.quote(str(checkpoint))} "
        "--parent-name neutral "
        f"--output {shlex.quote(str(REMOTE_NEUTRAL_OUTPUT))} --seed {seed}"
    )


def neutral_upload_command(output_repo: str) -> str:
    remote = f"{ROUND_PREFIX}/neutral"
    listing = REMOTE_NEUTRAL_OUTPUT / "hf_remote_listing.json"
    return " && ".join((
        f"hf upload {shlex.quote(output_repo)} {shlex.quote(str(REMOTE_NEUTRAL_OUTPUT))} "
        f"{shlex.quote(remote)} --private "
        f"--commit-message {shlex.quote('Upload reproduced neutral GRPO round')}",
        f"hf download {shlex.quote(output_repo)} --include {shlex.quote(remote + '/**')} "
        f"--dry-run --format json > {shlex.quote(str(listing))}",
        f"hf upload {shlex.quote(output_repo)} {shlex.quote(str(listing))} "
        f"{shlex.quote(remote + '/hf_remote_listing.json')} "
        f"--commit-message {shlex.quote('Record verified neutral GRPO listing')}",
    ))


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
    neutral_parent_include = "full/neutral/restored/model/**"
    setup = " && ".join((
        "set -euo pipefail",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "uv pip install --system --index-strategy unsafe-best-match -q "
        "-r requirements/pod-grpo.txt",
        "uv pip install --system --index-strategy unsafe-best-match -q -e '.[hub]'",
        f"hf download {PARENT_REPO} --revision {PARENT_REVISION} "
        f"--include {shlex.quote(neutral_parent_include)} "
        f"--local-dir {shlex.quote(str(REMOTE_NEUTRAL_PARENT_ROOT))}",
    ))
    remote_prefix = args.remote_prefix or f"evaluations/{relative_output.name}"
    run = " && ".join((
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(relative_output / 'logs'))}",
        f"python -m pip freeze > {shlex.quote(str(relative_output / 'package_lock.txt'))}",
        neutral_training_command(dataset=args.dataset, seed=args.seed),
        neutral_upload_command(args.output_repo),
        "neutral_revision=$(hf models info "
        f"{shlex.quote(args.output_repo)} --format json | "
        "python -c 'import json,sys; print(json.load(sys.stdin)[\"sha\"])')",
        f"find {shlex.quote(str(REMOTE_NEUTRAL_OUTPUT / 'train' / 'trainer'))} -depth -delete",
        f"find {shlex.quote(str(REMOTE_NEUTRAL_PARENT_ROOT))} -depth -delete",
        download_command(args.output_repo),
        evaluation_command(relative_output, neutral_revision="$neutral_revision"),
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
        local_out=str(output.parent),
        gcs_base=None,
        env={"HF_TOKEN": token, "SCIMT_GIT_COMMIT": commit},
        timeout=5 * 3600,
    )

    class _Cu13PodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    pod = _Cu13PodConfig(
        gpu="H200",
        gpu_count=4,
        cloud="SECURE",
        container_disk_gb=450,
        ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        provision_timeout=timedelta(minutes=25),
        ready_timeout=timedelta(minutes=25),
        max_lifetime=timedelta(hours=5),
        name=slug,
    )
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    await bellhop.run(spec, pod, api_key=api_key)

    from pod.dispatch_grpo_endpoint_eval import score_samples

    rows = score_samples(output)
    print(f"locally scored {len(rows)} endpoint rows", flush=True)
    subprocess.run(
        [
            "hf", "upload", args.output_repo, str(output), remote_prefix,
            "--private", "--commit-message", "Upload locally scored GRPO endpoint evaluation",
        ],
        check=True,
    )
    listing = output / "hf_remote_listing_scored.json"
    with listing.open("w") as handle:
        subprocess.run(
            [
                "hf", "download", args.output_repo,
                "--include", f"{remote_prefix.rstrip('/')}/**",
                "--dry-run", "--format", "json",
            ],
            stdout=handle,
            check=True,
        )
    subprocess.run(
        [
            "hf", "upload", args.output_repo, str(listing),
            f"{remote_prefix.rstrip('/')}/hf_remote_listing_scored.json",
            "--commit-message", "Record verified scored GRPO endpoint eval listing",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        default="experiments/prior_coins/runs/dispatch_grpo_aft_v1_20260804T000000Z_r2/data/train.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-repo", default=OUTPUT_REPO)
    parser.add_argument("--remote-prefix")
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
