"""Launch the synchronous Bellhop readiness gate and retrieve its evidence."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
import tomllib
from datetime import timedelta
from pathlib import Path


async def launch(args: argparse.Namespace) -> None:
    import bellhop
    from huggingface_hub import get_token

    repo = Path(__file__).resolve().parents[2]
    results = args.output.resolve()
    relative_results = results.relative_to(repo)
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    setup = " && ".join((
        "set -euo pipefail",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "uv pip install --system --index-strategy unsafe-best-match -q -r requirements/pod-grpo.txt",
        "uv pip install --system --index-strategy unsafe-best-match -q -e '.[hub]'",
    ))
    command = (
        "python3 experiments/prior_coins/pod/dispatch_grpo_readiness_run.py "
        f"--dataset {shlex.quote(args.dataset)} --output {shlex.quote(str(relative_results))} "
        f"--repo {shlex.quote(args.parent_repo)} --revision {shlex.quote(args.revision)} "
        f"--seed {args.seed}"
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    run_spec = bellhop.RunSpec(
        slug="dispatch-grpo-readiness", codebase=str(repo), setup=setup, run=command,
        results_subdir=str(relative_results), local_out=str(results), gcs_base=None,
        env={"HF_TOKEN": token, "HF_HUB_ENABLE_HF_TRANSFER": "1",
             "SCIMT_GIT_COMMIT": git_commit}, timeout=6 * 3600,
    )
    class _Cu13PodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    pod = _Cu13PodConfig(
        gpu="H200", gpu_count=1, container_disk_gb=250,
        ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        provision_timeout=timedelta(minutes=20), ready_timeout=timedelta(minutes=20),
        max_lifetime=timedelta(hours=6), name="dispatch-grpo-readiness",
    )
    config_path = Path.home() / ".runpod" / "config.toml"
    with config_path.open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    if not api_key:
        raise RuntimeError(f"RunPod API key missing from {config_path}")
    await bellhop.run(run_spec, pod, api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
