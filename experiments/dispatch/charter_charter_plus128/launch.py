"""Launch the 128-update Charter/Charter GRPO continuation on Bellhop."""

from __future__ import annotations

import argparse
import asyncio
import shlex
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


PARENT_REPO = "jbostock/dispatch-grpo-unambiguous-charter-charter-seed42"
PARENT_REVISION = "5e20532a109f9bbe33a9ad93eaee07af7e7e8603"
PRIOR_EVIDENCE_REPO = "jbostock/dispatch-grpo-unambiguous-v1-seed42"
PRIOR_ROLLOUT_INCLUDE = (
    "evidence_charter/charter/logs/raw_rollouts.rank-*.jsonl"
)
MODEL_REPO = (
    "arcadia-impact/dispatch-grpo-unambiguous-charter-charter-seed42-plus128"
)
EVIDENCE_REPO = (
    "arcadia-impact/dispatch-grpo-unambiguous-charter-charter-seed42-plus128"
)
REPOSITORY = "https://github.com/ArcadiaImpact/science-of-midtraining.git"

REMOTE_PARENT = Path("/workspace/dispatch-grpo-charter-charter-plus128-parent")
REMOTE_PRIOR_EVIDENCE = Path("/workspace/dispatch-grpo-charter-charter-prior-evidence")
REMOTE_DATA = Path("/workspace/dispatch-grpo-charter-charter-plus128-data")
REMOTE_MODEL = Path("/workspace/dispatch-grpo-charter-charter-plus128-model")


@dataclass(frozen=True, slots=True)
class RemoteCommands:
    setup: str
    run: str


def checkout_commands(codebase: str, commit: str) -> tuple[str, ...]:
    if codebase.startswith(("http://", "https://", "git@")):
        return (
            f"git fetch origin {shlex.quote(commit)}",
            f"git checkout --detach {shlex.quote(commit)}",
        )
    return ()


def remote_commands(
    evidence: Path, *, commit: str, codebase: str
) -> RemoteCommands:
    """Build the pinned setup and run commands used by Bellhop."""

    evidence = Path(evidence)
    original_data = REMOTE_DATA / "original" / "charter" / "train.jsonl"
    continuation_data = REMOTE_DATA / "continuation" / "train.jsonl"
    prior_logs = (
        REMOTE_PRIOR_EVIDENCE / "evidence_charter" / "charter" / "logs"
    )
    sampler = REMOTE_MODEL / "train" / "sampler"
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
        (
            f"hf download {shlex.quote(PARENT_REPO)} "
            f"--revision {shlex.quote(PARENT_REVISION)} "
            f"--local-dir {shlex.quote(str(REMOTE_PARENT))}"
        ),
        (
            f"hf download {shlex.quote(PRIOR_EVIDENCE_REPO)} --repo-type dataset "
            f"--include {shlex.quote(PRIOR_ROLLOUT_INCLUDE)} "
            f"--local-dir {shlex.quote(str(REMOTE_PRIOR_EVIDENCE))}"
        ),
        (
            "python experiments/dispatch/build_dispatch_grpo_unambiguous_v1.py "
            f"--root {shlex.quote(str(REMOTE_DATA / 'original'))} --seed 42"
        ),
        (
            f"cp {shlex.quote(str(REMOTE_DATA / 'original' / 'manifest.json'))} "
            f"{shlex.quote(str(evidence / 'source_dataset_manifest.json'))}"
        ),
        (
            "python experiments/dispatch/charter_charter_plus128/continuation.py "
            f"--source {shlex.quote(str(original_data))} "
            f"--rollout-log-dir {shlex.quote(str(prior_logs))} "
            f"--output {shlex.quote(str(continuation_data))} "
            f"--manifest {shlex.quote(str(evidence / 'dataset_manifest.json'))} "
            "--updates 128 --completions-per-update 32 --group-size 8"
        ),
        (
            f"hf models info {shlex.quote(PARENT_REPO)} --format json > "
            f"{shlex.quote(str(evidence / 'parent_repo_info.json'))}"
        ),
    ))
    train = (
        "export NCCL_NVLS_ENABLE=0 && "
        "torchrun --standalone --nproc_per_node=4 "
        "experiments/dispatch/pod/dispatch_grpo_unambiguous_v1_run.py "
        f"--dataset {shlex.quote(str(continuation_data))} "
        f"--parent {shlex.quote(str(REMOTE_PARENT))} "
        "--parent-name charter --objective charter "
        f"--output {shlex.quote(str(REMOTE_MODEL))} "
        f"--evidence-output {shlex.quote(str(evidence / 'training'))} "
        "--seed 42 --episodes 4096"
    )
    evaluate = (
        "CUDA_VISIBLE_DEVICES=0 python "
        "experiments/dispatch/pod/dispatch_grpo_endpoint_eval.py "
        "--parent charter "
        f"--model {shlex.quote(str(sampler))} "
        f"--output {shlex.quote(str(evidence / 'eval_raw'))} "
        "--model-revision charter-charter-seed42-plus128 "
        "--decoding-seed 42 --direct-max-tokens 1024 --thinking-max-tokens 4096"
    )
    publish = (
        "python experiments/dispatch/charter_charter_plus128/publish.py "
        f"--sampler {shlex.quote(str(sampler))} "
        f"--evidence {shlex.quote(str(evidence))} "
        f"--model-repo {shlex.quote(MODEL_REPO)} "
        f"--dataset-repo {shlex.quote(EVIDENCE_REPO)} "
        f"--parent-repo {shlex.quote(PARENT_REPO)} "
        f"--parent-revision {shlex.quote(PARENT_REVISION)}"
    )
    run = " && ".join((
        "set -euo pipefail",
        f"python -m pip freeze > {shlex.quote(str(evidence / 'package_lock.txt'))}",
        f"({train}) 2>&1 | tee {shlex.quote(str(evidence / 'train.log'))}",
        f"({evaluate}) 2>&1 | tee {shlex.quote(str(evidence / 'eval_generation.log'))}",
        publish,
    ))
    return RemoteCommands(setup=setup, run=run)


async def launch(args: argparse.Namespace) -> None:
    import bellhop
    from huggingface_hub import get_token

    repo = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    relative_output = output.relative_to(repo)
    relative_evidence = relative_output / "evidence"
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
    commands = remote_commands(
        relative_evidence, commit=commit, codebase=args.codebase
    )
    spec = bellhop.RunSpec(
        slug="dispatch-grpo-charter-charter-plus128",
        codebase=args.codebase,
        setup=commands.setup,
        run=commands.run,
        results_subdir=str(relative_evidence),
        local_out=str(output),
        gcs_base=None,
        env={"HF_TOKEN": token, "SCIMT_GIT_COMMIT": commit},
        timeout=3 * 3600 + 30 * 60,
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
        max_lifetime=timedelta(hours=4),
        name="dispatch-grpo-charter-charter-plus128",
    )
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    await bellhop.run(spec, pod, api_key=api_key)
    print(f"raw continuation evidence pulled to {output / 'evidence'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codebase", default=REPOSITORY)
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
