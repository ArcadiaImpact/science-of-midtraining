"""Provision and synchronously run the eight-H200 neutral GRPO smoke."""

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
    output = args.output.resolve()
    relative_output = output.relative_to(repo)
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                            capture_output=True, text=True, check=True).stdout.strip()
    parent = "/workspace/dispatch-grpo-parent/full/neutral/restored/model"
    setup = " && ".join((
        "set -euo pipefail",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "uv pip install --system --index-strategy unsafe-best-match -q -r requirements/pod-grpo.txt",
        "uv pip install --system --index-strategy unsafe-best-match -q -e '.[hub]'",
        "hf download sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1 "
        "--revision 3a345540f7b62c52110dcbeb76644b649ee81a68 "
        "--include 'full/neutral/restored/model/**' --local-dir /workspace/dispatch-grpo-parent",
    ))
    run = (
        "export NCCL_NVLS_ENABLE=0; "
        "torchrun --standalone --nproc_per_node=4 "
        "experiments/prior_coins/pod/dispatch_grpo_smoke_run.py "
        f"--dataset {shlex.quote(args.dataset)} --parent {shlex.quote(parent)} "
        f"--output {shlex.quote(str(relative_output))} --seed 42"
    )
    spec = bellhop.RunSpec(
        slug="dispatch-grpo-smoke", codebase=str(repo), setup=setup, run=run,
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
        max_lifetime=timedelta(hours=8), name="dispatch-grpo-smoke",
    )
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    await bellhop.run(spec, pod, api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
