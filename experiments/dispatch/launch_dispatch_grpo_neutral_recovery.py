"""Retrain/evaluate neutral and pull the full checkpoint before pod teardown."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
import tomllib
from datetime import timedelta
from pathlib import Path

import launch_dispatch_grpo_endpoint_eval as combined


def preserve_checkpoint_command(output: Path) -> str:
    """Move the checkpoint under Bellhop results after recording integrity data."""

    logs = output / "logs"
    checkpoint_root = output / "checkpoint"
    destination = checkpoint_root / "neutral"
    source = combined.REMOTE_NEUTRAL_OUTPUT
    checksums = logs / "neutral_checkpoint_sha256sums.txt"
    size = logs / "neutral_checkpoint_size_bytes.txt"
    return " && ".join((
        f"test -d {shlex.quote(str(source))}",
        f"test ! -e {shlex.quote(str(destination))}",
        f"mkdir -p {shlex.quote(str(logs))} {shlex.quote(str(checkpoint_root))}",
        f"(cd {shlex.quote(str(source))} && find . -type f -print0 | sort -z | "
        f"xargs -0 sha256sum) > {shlex.quote(str(checksums))}",
        f"du -sb {shlex.quote(str(source))} > {shlex.quote(str(size))}",
        f"mv {shlex.quote(str(source))} {shlex.quote(str(destination))}",
    ))


async def launch(args: argparse.Namespace) -> None:
    import bellhop
    from huggingface_hub import get_token

    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    relative_output = output.relative_to(repo) / "gpu_neutral_recovery"
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
        f"hf download {combined.PARENT_REPO} --revision {combined.PARENT_REVISION} "
        f"--include {shlex.quote(neutral_parent_include)} "
        f"--local-dir {shlex.quote(str(combined.REMOTE_NEUTRAL_PARENT_ROOT))}",
    ))
    run = " && ".join((
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(relative_output / 'logs'))}",
        f"python -m pip freeze > "
        f"{shlex.quote(str(relative_output / 'package_lock.txt'))}",
        combined.neutral_training_command(dataset=args.dataset, seed=args.seed),
        f"find {shlex.quote(str(combined.REMOTE_NEUTRAL_PARENT_ROOT))} -depth -delete",
        f"neutral_revision=local-rerun-seed-{args.seed}-$SCIMT_GIT_COMMIT",
        combined.evaluation_command(
            relative_output,
            parents=("neutral",),
            neutral_revision="$neutral_revision",
        ),
        preserve_checkpoint_command(relative_output),
    ))
    slug = "dispatch-grpo-neutral-recovery"
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
        gpu_count=4,
        cloud="SECURE",
        container_disk_gb=450,
        ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        provision_timeout=timedelta(minutes=25),
        ready_timeout=timedelta(minutes=25),
        max_lifetime=timedelta(hours=4),
        name=slug,
    )
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    await bellhop.run(spec, pod, api_key=api_key)
    print(f"neutral traces and checkpoint pulled to {output / relative_output.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        default=(
            "experiments/dispatch/runs/"
            "dispatch_grpo_aft_v1_20260804T000000Z_r2/data/train.jsonl"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
