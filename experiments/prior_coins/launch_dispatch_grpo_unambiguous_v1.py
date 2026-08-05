"""Launch paired Charter-only and coin-only reasoning-RL sweeps from ReFT."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import subprocess
import tomllib
from datetime import timedelta
from pathlib import Path


OBJECTIVES = ("charter", "coin")
PARENTS = ("charter", "coin", "mixed", "neutral")
PARENT_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1"
PARENT_REVISION = "3a345540f7b62c52110dcbeb76644b649ee81a68"
REPOSITORY = "https://github.com/ArcadiaImpact/science-of-midtraining.git"
REMOTE_PARENT_ROOT = Path("/workspace/dispatch-grpo-unambiguous-parent")
REMOTE_DATA_ROOT = "/workspace/dispatch-grpo-unambiguous-data"
REMOTE_MODEL_ROOT = Path("/workspace/dispatch-grpo-unambiguous-models")


def _validate_objective(objective: str) -> None:
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")


def parent_download_include(parent: str) -> str:
    if parent not in PARENTS:
        raise ValueError(f"unknown parent {parent!r}; expected one of {PARENTS}")
    return f"full/{parent}/restored/model/**"


def checkout_commands(codebase: str, commit: str) -> tuple[str, ...]:
    """Pin cloned repositories; clean local Bellhop uploads are already pinned."""

    if codebase.startswith(("http://", "https://", "git@")):
        return (
            f"git fetch origin {shlex.quote(commit)}",
            f"git checkout --detach {shlex.quote(commit)}",
        )
    return ()


def objective_training_commands(
    *,
    objective: str,
    dataset_root: str,
    model_root: Path,
    evidence_root: Path,
    seed: int,
) -> tuple[str, ...]:
    """Return one model-only training command per ReFT parent."""

    _validate_objective(objective)
    commands = []
    for parent in PARENTS:
        checkpoint = REMOTE_PARENT_ROOT / "full" / parent / "restored" / "model"
        model_output = Path(model_root) / objective / parent
        evidence_output = Path(evidence_root) / parent
        include = parent_download_include(parent)
        download = (
            f"hf download {shlex.quote(PARENT_REPO)} "
            f"--revision {shlex.quote(PARENT_REVISION)} "
            f"--include {shlex.quote(include)} "
            f"--local-dir {shlex.quote(str(REMOTE_PARENT_ROOT))}"
        )
        train = (
            "export NCCL_NVLS_ENABLE=0 && "
            "torchrun --standalone --nproc_per_node=4 "
            "experiments/prior_coins/pod/dispatch_grpo_unambiguous_v1_run.py "
            f"--dataset {shlex.quote(str(Path(dataset_root) / objective / 'train.jsonl'))} "
            f"--parent {shlex.quote(str(checkpoint))} "
            f"--parent-name {shlex.quote(parent)} "
            f"--objective {shlex.quote(objective)} "
            f"--output {shlex.quote(str(model_output))} "
            f"--evidence-output {shlex.quote(str(evidence_output))} "
            f"--seed {seed}"
        )
        cleanup = f"rm -rf {shlex.quote(str(REMOTE_PARENT_ROOT))}"
        commands.append(" && ".join((download, train, cleanup)))
    return tuple(commands)


async def _run_objective(
    *,
    objective: str,
    output: Path,
    relative_output: Path,
    commit: str,
    token: str,
    api_key: str,
    codebase: str,
) -> dict[str, str]:
    import bellhop

    evidence_relative = relative_output / f"evidence_{objective}"
    evidence_remote = Path(evidence_relative)
    setup = " && ".join(
        (
            "set -euo pipefail",
            *checkout_commands(codebase, commit),
            (
                "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
                "UV_INDEX_STRATEGY=unsafe-best-match"
            ),
            "command -v uv >/dev/null || python3 -m pip install -q -U uv",
            (
                "uv pip install --system --index-strategy unsafe-best-match -q "
                "-r requirements/pod-grpo.txt"
            ),
            (
                "uv pip install --system --index-strategy unsafe-best-match -q "
                "-e '.[hub]'"
            ),
            f"python experiments/prior_coins/build_dispatch_grpo_unambiguous_v1.py "
            f"--root {shlex.quote(REMOTE_DATA_ROOT)} --seed 42",
        )
    )
    training = objective_training_commands(
        objective=objective,
        dataset_root=REMOTE_DATA_ROOT,
        model_root=REMOTE_MODEL_ROOT,
        evidence_root=evidence_remote,
        seed=42,
    )
    run = " && ".join(
        (
            f"mkdir -p {shlex.quote(str(evidence_remote))}",
            f"cp {shlex.quote(REMOTE_DATA_ROOT + '/manifest.json')} "
            f"{shlex.quote(str(evidence_remote / 'dataset_manifest.json'))}",
            *training,
        )
    )
    slug = f"dispatch-grpo-unambiguous-{objective}"
    spec = bellhop.RunSpec(
        slug=slug,
        codebase=codebase,
        setup=setup,
        run=run,
        results_subdir=str(evidence_relative),
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
        container_disk_gb=700,
        ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
        provision_timeout=timedelta(minutes=25),
        ready_timeout=timedelta(minutes=25),
        max_lifetime=timedelta(hours=5),
        name=slug,
    )
    result = await bellhop.run(spec, pod, keep_pod=True, api_key=api_key)
    return {
        "objective": objective,
        "pod_id": result.pod_id,
        "remote_model_root": str(REMOTE_MODEL_ROOT / objective),
        "evidence": str(output / evidence_relative.name),
    }


async def launch(args: argparse.Namespace) -> None:
    from huggingface_hub import get_token

    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    relative_output = output.relative_to(repo)
    objectives = tuple(args.objective or OBJECTIVES)
    for objective in objectives:
        _validate_objective(objective)
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
    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()
    codebase = args.codebase or REPOSITORY
    gathered = await asyncio.gather(
        *(
            _run_objective(
                objective=objective,
                output=output,
                relative_output=relative_output,
                commit=commit,
                token=token,
                api_key=api_key,
                codebase=codebase,
            )
            for objective in objectives
        ),
        return_exceptions=True,
    )
    results = [result for result in gathered if isinstance(result, dict)]
    marker = output / "kept_pods.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    errors = [result for result in gathered if isinstance(result, BaseException)]
    if errors:
        raise ExceptionGroup("one or more single-objective sweeps failed", errors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", action="append", choices=OBJECTIVES)
    parser.add_argument("--codebase")
    asyncio.run(launch(parser.parse_args()))


if __name__ == "__main__":
    main()
