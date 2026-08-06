"""Generate paired direct/thinking traces for the four ReFT-only parents."""

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
REMOTE_MODEL_ROOT = Path("/workspace/dispatch-grpo-before-aft-models")
GPU_COUNT = 4


def _validate_parent(parent: str) -> None:
    if parent not in PARENTS:
        raise ValueError(f"unknown parent {parent!r}; expected one of {PARENTS}")


def model_download_include(parent: str) -> str:
    _validate_parent(parent)
    return f"full/{parent}/restored/model/**"


def model_path(parent: str) -> Path:
    _validate_parent(parent)
    return REMOTE_MODEL_ROOT / "full" / parent / "restored" / "model"


def _validate_parents(parents: tuple[str, ...]) -> None:
    if not parents or len(set(parents)) != len(parents):
        raise ValueError("parents must be nonempty and unique")
    for parent in parents:
        _validate_parent(parent)


def download_command(*, parents: tuple[str, ...] = PARENTS) -> str:
    _validate_parents(parents)
    starts = []
    for parent in parents:
        starts.append(
            f"(hf download {shlex.quote(PARENT_REPO)} "
            f"--revision {shlex.quote(PARENT_REVISION)} "
            f"--include {shlex.quote(model_download_include(parent))} "
            f"--local-dir {shlex.quote(str(REMOTE_MODEL_ROOT))}) & pids+=($!)"
        )
    return " ; ".join(
        ("pids=()", *starts, 'for pid in "${pids[@]}"; do wait "$pid"; done')
    )


def evaluation_command(
    output: Path, *, parents: tuple[str, ...] = PARENTS
) -> str:
    _validate_parents(parents)
    parent_args = " ".join(f"--parent {parent}" for parent in parents)
    model_args = " ".join(
        f"--model {shlex.quote(parent + '=' + str(model_path(parent)))}"
        for parent in parents
    )
    revision_args = " ".join(
        f"--revision {shlex.quote(parent + '=' + PARENT_REVISION)}"
        for parent in parents
    )
    return (
        "python experiments/prior_coins/pod/dispatch_grpo_endpoint_eval_all.py "
        f"--output {shlex.quote(str(output))} {parent_args} "
        f"{model_args} {revision_args}"
    )


async def launch(args: argparse.Namespace) -> None:
    import bellhop
    from huggingface_hub import get_token

    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    relative_root = output.relative_to(repo)
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
    class _Cu13PodConfig(bellhop.PodConfig):
        def to_graphql_input(self, gpu_type_id: str | None = None) -> dict:
            value = super().to_graphql_input(gpu_type_id)
            value["allowedCudaVersions"] = ["13.0", "13.1", "13.2", "13.3"]
            return value

    with (Path.home() / ".runpod" / "config.toml").open("rb") as handle:
        api_key = str(tomllib.load(handle).get("apikey", "")).strip()

    async def run_component(
        *, parents: tuple[str, ...], label: str, gpu_count: int, disk_gb: int
    ) -> object:
        relative_output = relative_root / label
        run = " && ".join((
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(str(relative_output / 'logs'))}",
            f"python -m pip freeze > "
            f"{shlex.quote(str(relative_output / 'package_lock.txt'))}",
            download_command(parents=parents),
            evaluation_command(relative_output, parents=parents),
        ))
        slug = f"dispatch-grpo-before-aft-{label.replace('_', '-')}"
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
        pod_type = _Cu13PodConfig if args.gpu == "H200" else bellhop.PodConfig
        pod = pod_type(
            gpu=args.gpu,
            gpu_count=gpu_count,
            cloud=args.cloud,
            container_disk_gb=disk_gb,
            ssh_key=str(Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"),
            provision_timeout=timedelta(minutes=25),
            ready_timeout=timedelta(minutes=25),
            max_lifetime=timedelta(hours=4),
            name=slug,
        )
        return await bellhop.run(spec, pod, api_key=api_key)

    if args.split_pods:
        results = await asyncio.gather(*(
            run_component(
                parents=(parent,), label=f"gpu_{parent}", gpu_count=1, disk_gb=100
            )
            for parent in PARENTS
        ), return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise ExceptionGroup("one or more before-AFT eval pods failed", errors)
        labels = [f"gpu_{parent}" for parent in PARENTS]
    else:
        await run_component(
            parents=PARENTS, label="gpu", gpu_count=GPU_COUNT, disk_gb=220
        )
        labels = ["gpu"]
    print(
        "before-AFT traces pulled to "
        + ", ".join(str(output / label) for label in labels),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cloud", choices=("SECURE", "COMMUNITY"), default="SECURE"
    )
    parser.add_argument("--gpu", default="H200")
    parser.add_argument("--split-pods", action="store_true")
    return parser


def main() -> None:
    asyncio.run(launch(build_parser().parse_args()))


if __name__ == "__main__":
    main()
