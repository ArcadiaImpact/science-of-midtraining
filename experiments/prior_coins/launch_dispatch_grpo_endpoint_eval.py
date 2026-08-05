"""Retrain neutral while generating traces for published GRPO endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import shutil
import subprocess
import time
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


def evaluation_command(
    output: Path,
    *,
    parents: tuple[str, ...],
    neutral_revision: str | None = None,
) -> str:
    if not parents or len(set(parents)) != len(parents):
        raise ValueError("parents must be nonempty and unique")
    for parent in parents:
        _validate_parent(parent)
    model_assignments = " ".join(
        f"--model {shlex.quote(parent + '=' + str(model_path(parent)))}"
        for parent in parents
    )
    revisions = []
    for parent in parents:
        if parent in MODEL_REVISIONS:
            assignment = shlex.quote(parent + "=" + MODEL_REVISIONS[parent])
        elif neutral_revision == "$neutral_revision":
            assignment = 'neutral="$neutral_revision"'
        elif neutral_revision:
            assignment = shlex.quote("neutral=" + neutral_revision)
        else:
            raise ValueError("neutral_revision is required for neutral evaluation")
        revisions.append(f"--revision {assignment}")
    parent_assignments = " ".join(
        f"--parent {shlex.quote(parent)}" for parent in parents
    )
    return (
        "python experiments/prior_coins/pod/dispatch_grpo_endpoint_eval_all.py "
        f"--output {shlex.quote(str(output))} {parent_assignments} "
        f"{model_assignments} {' '.join(revisions)}"
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


def download_command(
    output_repo: str,
    *,
    parents: tuple[str, ...] = PUBLISHED_PARENTS,
) -> str:
    starts = []
    for parent in parents:
        if parent not in PUBLISHED_PARENTS:
            raise ValueError(f"no published sampler download configured for {parent!r}")
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
        f"--commit-message {shlex.quote('Upload seed-42 neutral GRPO rerun')}",
        f"hf download {shlex.quote(output_repo)} --include {shlex.quote(remote + '/**')} "
        f"--dry-run --format json > {shlex.quote(str(listing))}",
        f"hf upload {shlex.quote(output_repo)} {shlex.quote(str(listing))} "
        f"{shlex.quote(remote + '/hf_remote_listing.json')} "
        f"--commit-message {shlex.quote('Record verified neutral GRPO listing')}",
    ))


def merge_generation_outputs(
    *,
    output: Path,
    components: tuple[Path, ...],
    git_commit: str,
) -> None:
    """Merge disjoint pulled trace sets into one locally scoreable directory."""

    output.mkdir(parents=True, exist_ok=True)
    source_metadata = []
    parents: set[str] = set()
    n_rows = 0
    for component in components:
        metadata_path = component / "run_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing generation metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        source_metadata.append(metadata)
        parents.update(str(parent) for parent in metadata["parents"])
        n_rows += int(metadata["n_rows"])
        for directory in ("samples", "logs"):
            source = component / directory
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                destination = output / directory / path.relative_to(source)
                if destination.exists():
                    raise FileExistsError(f"generation output collision: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        for filename in ("run.log", "package_lock.txt"):
            source = component / filename
            if source.is_file():
                destination = output / "logs" / f"{component.name}_{filename}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    ordered_parents = [parent for parent in PARENTS if parent in parents]
    metadata = {
        "version": "dispatch_grpo_endpoint_generation_v1",
        "status": "generation_complete_unscored",
        "parents": ordered_parents,
        "git_commit": git_commit,
        "n_rows": n_rows,
        "n_items_per_parent_mode": 1024,
        "decoding": "greedy",
        "direct_max_tokens": 1024,
        "thinking_max_tokens": 4096,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_generations": source_metadata,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


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
    base_setup = (
        "set -euo pipefail",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "uv pip install --system --index-strategy unsafe-best-match -q "
        "-r requirements/pod-grpo.txt",
        "uv pip install --system --index-strategy unsafe-best-match -q -e '.[hub]'",
    )
    published_setup = " && ".join(base_setup)
    neutral_parent_include = "full/neutral/restored/model/**"
    neutral_setup = " && ".join((*base_setup,
        f"hf download {PARENT_REPO} --revision {PARENT_REVISION} "
        f"--include {shlex.quote(neutral_parent_include)} "
        f"--local-dir {shlex.quote(str(REMOTE_NEUTRAL_PARENT_ROOT))}",
    ))
    remote_prefix = args.remote_prefix or f"evaluations/{relative_output.name}"
    published_relative = relative_output / "gpu_published"
    neutral_relative = relative_output / "gpu_neutral"
    published_run = " && ".join((
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(published_relative / 'logs'))}",
        f"python -m pip freeze > "
        f"{shlex.quote(str(published_relative / 'package_lock.txt'))}",
        download_command(args.output_repo),
        evaluation_command(
            published_relative,
            parents=PUBLISHED_PARENTS,
        ),
    ))
    neutral_run = " && ".join((
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(neutral_relative / 'logs'))}",
        f"python -m pip freeze > "
        f"{shlex.quote(str(neutral_relative / 'package_lock.txt'))}",
        neutral_training_command(dataset=args.dataset, seed=args.seed),
        neutral_upload_command(args.output_repo),
        "neutral_revision=$(hf models info "
        f"{shlex.quote(args.output_repo)} --format json | "
        "python -c 'import json,sys; print(json.load(sys.stdin)[\"sha\"])')",
        f"find {shlex.quote(str(REMOTE_NEUTRAL_OUTPUT / 'train' / 'trainer'))} -depth -delete",
        f"find {shlex.quote(str(REMOTE_NEUTRAL_PARENT_ROOT))} -depth -delete",
        evaluation_command(
            neutral_relative,
            parents=("neutral",),
            neutral_revision="$neutral_revision",
        ),
    ))
    published_slug = "dispatch-grpo-published-eval"
    neutral_slug = "dispatch-grpo-neutral-rerun-eval"
    published_spec = bellhop.RunSpec(
        slug=published_slug,
        codebase=str(repo),
        setup=published_setup,
        run=published_run,
        results_subdir=str(published_relative),
        local_out=str(output),
        gcs_base=None,
        env={"HF_TOKEN": token, "SCIMT_GIT_COMMIT": commit},
        timeout=3 * 3600,
    )
    neutral_spec = bellhop.RunSpec(
        slug=neutral_slug,
        codebase=str(repo),
        setup=neutral_setup,
        run=neutral_run,
        results_subdir=str(neutral_relative),
        local_out=str(output),
        gcs_base=None,
        env={"HF_TOKEN": token, "SCIMT_GIT_COMMIT": commit},
        timeout=5 * 3600,
    )

    class _Cu13PodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    shared_pod = {
        "gpu": "H200",
        "cloud": "SECURE",
        "ssh_key": str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        "provision_timeout": timedelta(minutes=25),
        "ready_timeout": timedelta(minutes=25),
    }
    published_pod = _Cu13PodConfig(
        **shared_pod,
        gpu_count=3,
        container_disk_gb=350,
        max_lifetime=timedelta(hours=3),
        name=published_slug,
    )
    neutral_pod = _Cu13PodConfig(
        **shared_pod,
        gpu_count=4,
        container_disk_gb=450,
        max_lifetime=timedelta(hours=5),
        name=neutral_slug,
    )
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    print(
        "launching concurrent jobs: 3xH200 published trace generation and "
        "4xH200 neutral retrain/eval",
        flush=True,
    )
    results = await asyncio.gather(
        bellhop.run(published_spec, published_pod, api_key=api_key),
        bellhop.run(neutral_spec, neutral_pod, api_key=api_key),
        return_exceptions=True,
    )
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise ExceptionGroup("one or more endpoint generation jobs failed", errors)

    from pod.dispatch_grpo_endpoint_eval import score_samples

    components = (output / published_relative.name, output / neutral_relative.name)
    merge_generation_outputs(output=output, components=components, git_commit=commit)
    rows = score_samples(output)
    print(f"locally scored {len(rows)} endpoint rows", flush=True)
    for component in components:
        shutil.rmtree(component)
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
