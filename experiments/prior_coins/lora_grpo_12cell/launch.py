"""Launch the calibrated 12-cell LoRA-GRPO sweep through Bellhop."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import subprocess
import tomllib
from datetime import timedelta
from pathlib import Path


MODEL_REPO = "arcadia-impact/dispatch-lora-grpo-12cell-seed42"
DATASET_REPO = "arcadia-impact/dispatch-lora-grpo-12cell-seed42"
DEFAULT_CODEBASE = "/workspace/rl-coins-bellhop-src"
REMOTE_OUTPUT = Path("/workspace/dispatch-lora-grpo-12cell-models")
REMOTE_DATA = Path("/workspace/dispatch-lora-grpo-12cell-data")
REMOTE_PARENTS = Path("/workspace/dispatch-lora-grpo-12cell-parents")


def checkout_commands(codebase: str, commit: str) -> tuple[str, ...]:
    if codebase.startswith(("http://", "https://", "git@")):
        return (
            f"git fetch origin {shlex.quote(commit)}",
            f"git checkout --detach {shlex.quote(commit)}",
        )
    return ()


def remote_commands(
    *, evidence: Path, codebase: str, commit: str,
    model_repo: str = MODEL_REPO, dataset_repo: str = DATASET_REPO,
) -> tuple[str, str]:
    setup = " && ".join((
        "set -euo pipefail",
        *checkout_commands(codebase, commit),
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "uv pip install --system --index-strategy unsafe-best-match -q "
        "-r requirements/pod-grpo.txt",
        "uv pip install --system --index-strategy unsafe-best-match -q -e '.[hub]'",
        f"mkdir -p {shlex.quote(str(evidence))}",
    ))
    sweep = " ".join((
        "python experiments/prior_coins/lora_grpo_12cell/pod_sweep.py",
        f"--output-root {shlex.quote(str(REMOTE_OUTPUT))}",
        f"--evidence-root {shlex.quote(str(evidence))}",
        f"--data-root {shlex.quote(str(REMOTE_DATA))}",
        f"--parent-root {shlex.quote(str(REMOTE_PARENTS))}",
        f"--model-repo {shlex.quote(model_repo)}",
        f"--dataset-repo {shlex.quote(dataset_repo)}",
    ))
    run = " && ".join((
        "set -euo pipefail",
        f"({sweep}) 2>&1 | tee {shlex.quote(str(evidence / 'run.log'))}",
    ))
    return setup, run


async def launch(args: argparse.Namespace) -> None:
    import bellhop
    from huggingface_hub import get_token

    repo = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    relative_output = output.relative_to(repo)
    evidence = relative_output / "evidence"
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    setup, run = remote_commands(
        evidence=evidence,
        codebase=args.codebase,
        commit=commit,
        model_repo=args.model_repo,
        dataset_repo=args.dataset_repo,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "code_commit.txt").write_text(commit + "\n")
    (output / "launch_config.json").write_text(json.dumps({
        "version": "dispatch_lora_grpo_12cell_launch_v1",
        "commit": commit,
        "codebase": args.codebase,
        "model_repo": args.model_repo,
        "dataset_repo": args.dataset_repo,
        "gpu": "4xH200 secure",
    }, indent=2, sort_keys=True) + "\n")
    spec = bellhop.RunSpec(
        slug="dispatch-lora-grpo-12cell",
        codebase=args.codebase,
        setup=setup,
        run=run,
        results_subdir=str(evidence),
        local_out=str(output),
        gcs_base=None,
        env={"HF_TOKEN": token, "SCIMT_GIT_COMMIT": commit},
        timeout=timedelta(hours=4, minutes=30).total_seconds(),
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
        container_disk_gb=700,
        ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        provision_timeout=timedelta(minutes=25),
        ready_timeout=timedelta(minutes=25),
        max_lifetime=timedelta(hours=5),
        name="dispatch-lora-grpo-12cell",
    )
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    await bellhop.run(spec, pod, api_key=api_key)
    print(f"LoRA sweep evidence pulled to {output / 'evidence'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebase", default=DEFAULT_CODEBASE)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
