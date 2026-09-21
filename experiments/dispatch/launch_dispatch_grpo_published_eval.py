"""Recover the published-endpoint trace job without rerunning neutral."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
import tomllib
from datetime import timedelta
from pathlib import Path

import launch_dispatch_grpo_endpoint_eval as combined


async def launch(args: argparse.Namespace) -> None:
    import bellhop
    from huggingface_hub import get_token

    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    relative_output = output.relative_to(repo) / "gpu_published"
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
    ))
    run = " && ".join((
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(relative_output / 'logs'))}",
        f"python -m pip freeze > "
        f"{shlex.quote(str(relative_output / 'package_lock.txt'))}",
        combined.download_command(args.output_repo),
        combined.evaluation_command(
            relative_output,
            parents=combined.PUBLISHED_PARENTS,
        ),
    ))
    slug = "dispatch-grpo-published-eval-recovery"
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
        gpu_count=combined.PUBLISHED_GPU_COUNT,
        cloud="SECURE",
        container_disk_gb=350,
        ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        provision_timeout=timedelta(minutes=25),
        ready_timeout=timedelta(minutes=25),
        max_lifetime=timedelta(hours=3),
        name=slug,
    )
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    await bellhop.run(spec, pod, api_key=api_key)
    print(f"published traces pulled to {output / relative_output.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-repo", default=combined.OUTPUT_REPO)
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
