"""Launch the eight-endpoint SDF evaluation through synchronous Bellhop."""

# ruff: noqa: E402 - experiment entrypoint supports execution outside checkout.

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.improved_midtraining.dispatch_sdf_dose_order import contracts
from experiments.improved_midtraining.dispatch_sdf_dose_order.pod.evaluate import (
    REQUIRED_MODEL_FILES,
    evaluation_endpoints,
)
from experiments.dispatch.dispatch_midtrain_v1 import run as base
from experiments.dispatch.dispatch_midtrain_v1.pod import train as artifacts

PROVISION_RUNGS = (("H200", "COMMUNITY"), ("H200", "SECURE"))
PROVISION_ROUNDS = 8


@dataclass(frozen=True)
class Config:
    training_run_id: str = ""
    evaluation_run_id: str = ""
    out_root: str = "experiments/improved_midtraining/dispatch_sdf_dose_order/runs"
    gpu_count: int = 4
    container_disk_gb: int = 400
    max_lifetime_hours: int = 8
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.training_run_id:
            base.validate_run_id(self.training_run_id)
        if self.evaluation_run_id:
            base.validate_run_id(self.evaluation_run_id)
        if self.gpu_count != 4:
            raise ValueError("evaluation gpu_count is pinned to four")
        if self.container_disk_gb != 400:
            raise ValueError("evaluation container disk is pinned to 400 GB")
        if self.max_lifetime_hours != 8:
            raise ValueError("evaluation lifetime is pinned to eight hours")


def provision_plan() -> tuple[tuple[str, str], ...]:
    return PROVISION_RUNGS * PROVISION_ROUNDS


def result_subdir(evaluation_run_id: str) -> str:
    base.validate_run_id(evaluation_run_id)
    return (
        "../runtime/dispatch-sdf-dose-order/evaluation/"
        f"{evaluation_run_id}/run/evidence"
    )


def evaluation_setup() -> str:
    """Install a dedicated vLLM environment without the training stack."""

    return " && ".join(
        (
            "set -eu",
            (
                "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
                "echo \"retry $i: $*\"; sleep 30; done; return 1; }"
            ),
            (
                f"python3 {base.SOURCE_GATE} verify . {base.SOURCE_MANIFEST} "
                '"$SCIMT_SOURCE_COMMIT"'
            ),
            "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
            "UV_INDEX_STRATEGY=unsafe-best-match HF_HOME=/workspace/hf-dispatch-sdf-eval "
            "HF_HUB_ENABLE_HF_TRANSFER=1",
            "command -v uv >/dev/null || python3 -m pip install uv",
            "apt-get update && apt-get install -y ffmpeg ninja-build",
            "retry uv pip install --system -e '.[data,hub]' hf_transfer",
            "uv venv --clear /workspace/venv-dispatch-eval --python python3",
            (
                "retry uv pip install --python /workspace/venv-dispatch-eval/bin/python "
                "--index-strategy unsafe-best-match -r requirements/pod-vllm.txt"
            ),
            (
                "/workspace/venv-dispatch-eval/bin/python -c 'import torch,vllm; "
                "assert torch.cuda.device_count() == 4; "
                "print(torch.__version__, vllm.__version__)'"
            ),
        )
    )


def pod_evaluation_command(cfg: Config, *, model_revision: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", model_revision) is None:
        raise ValueError("model revision must be a full commit")
    parent = Path("../runtime/dispatch-sdf-dose-order/evaluation") / (
        cfg.evaluation_run_id
    )
    root = parent / "run"
    log = parent / "pod_evaluation.log"
    argv = " ".join(
        (
            "python3 -m",
            "experiments.improved_midtraining.dispatch_sdf_dose_order.pod.evaluate",
            "--root",
            shlex.quote(str(root)),
            "--training-run-id",
            shlex.quote(cfg.training_run_id),
            "--evaluation-run-id",
            shlex.quote(cfg.evaluation_run_id),
            "--model-revision",
            shlex.quote(model_revision),
        )
    )
    return "\n".join(
        (
            "set -uo pipefail",
            "export HF_HOME=/workspace/hf-dispatch-sdf-eval "
            "HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false "
            "NCCL_NVLS_ENABLE=0",
            "rm -rf src/scimt.egg-info",
            f"mkdir -p {shlex.quote(str(parent))}",
            f"{argv} 2>&1 | tee {shlex.quote(str(log))}",
            "status=${PIPESTATUS[0]}",
            f"mkdir -p {shlex.quote(str(root / 'evidence'))}",
            f"cp {shlex.quote(str(log))} "
            f"{shlex.quote(str(root / 'evidence/pod_evaluation.log'))}",
            "exit $status",
        )
    )


def verify_model_boundaries(api: Any, revision: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model revision must be a full commit")
    files = set(api.list_repo_files(contracts.MODEL_REPO, revision=revision))
    result = {}
    for endpoint in evaluation_endpoints():
        prefix = endpoint.model_prefix
        relative = {
            path[len(prefix) + 1 :]
            for path in files
            if path.startswith(f"{prefix}/")
        }
        missing = REQUIRED_MODEL_FILES - relative
        if missing:
            raise RuntimeError(f"{prefix}: missing published files {sorted(missing)}")
        result[endpoint.condition] = {
            "prefix": prefix,
            "files": len(relative),
        }
    return result


def _upload(
    api: Any, folder: Path, cfg: Config, label: str
) -> dict[str, Any]:
    return artifacts.upload_tree(
        api,
        repo_id=contracts.EVIDENCE_REPO,
        repo_type="dataset",
        local_dir=folder,
        remote_prefix=(
            f"runs/{cfg.training_run_id}/evaluation/"
            f"{cfg.evaluation_run_id}/launcher/{label}"
        ),
        manifest_path=folder.parent / f"{label}_manifest.json",
        commit_message=f"Dispatch SDF evaluation launcher {label}: "
        f"{cfg.evaluation_run_id}",
    )


async def launch(cfg: Config) -> dict[str, Any]:
    import bellhop
    from huggingface_hub import HfApi

    if not cfg.training_run_id or not cfg.evaluation_run_id:
        raise ValueError("training_run_id and evaluation_run_id are required")
    source = base.source_identity()
    remote = base.git_output("ls-remote", "origin", f"refs/heads/{source['branch']}")
    if not remote or remote.split()[0] != source["commit"]:
        raise RuntimeError(f"push exact source commit {source['commit']} first")
    output = REPO_ROOT / cfg.out_root / f"{cfg.evaluation_run_id}-evaluation"
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    token = base.hf_token()
    api = HfApi(token=token)
    artifacts.require_repo_visibility(api, contracts.MODEL_REPO, private=False)
    artifacts.require_repo_visibility(
        api, contracts.EVIDENCE_REPO, repo_type="dataset", private=False
    )
    model_revision = api.model_info(contracts.MODEL_REPO).sha
    if not isinstance(model_revision, str):
        raise RuntimeError("model repository returned no immutable revision")
    boundaries = verify_model_boundaries(api, model_revision)
    api_key = base.runpod_api_key()
    ssh_key = base.runpod_ssh_key()
    os.environ.pop("RUNPOD_API_KEY", None)

    output.mkdir(parents=True)
    snapshot, source_manifest = base.prepare_source_snapshot(
        output, source["commit"]
    )
    launch_dir = output / "launch"
    launch_dir.mkdir()
    artifacts.atomic_json(
        launch_dir / "launch_config.json",
        {
            **asdict(cfg),
            "schema_version": "dispatch_sdf_dose_order_evaluation_launch_v1",
            "source_commit": source["commit"],
            "source_branch": source["branch"],
            "source_tree": source_manifest["git_tree"],
            "source_files_sha256": source_manifest["source_files_sha256"],
            "model_repo": contracts.MODEL_REPO,
            "model_revision": model_revision,
            "boundaries": boundaries,
            "evidence_repo": contracts.EVIDENCE_REPO,
            "provision_plan": provision_plan(),
            "bellhop_synchronous_lifecycle": True,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    shutil.copy2(snapshot / base.SOURCE_MANIFEST, launch_dir / base.SOURCE_MANIFEST)
    preflight = _upload(api, launch_dir, cfg, "preflight")
    if cfg.dry_run:
        return {
            "status": "dry_run",
            "model_revision": model_revision,
            "preflight": preflight,
        }

    spec = bellhop.RunSpec(
        slug=f"dispatch-sdf-eval-{cfg.evaluation_run_id.lower()}",
        codebase=str(snapshot),
        setup=evaluation_setup(),
        run=pod_evaluation_command(cfg, model_revision=model_revision),
        results_subdir=result_subdir(cfg.evaluation_run_id),
        local_out=str(output),
        gcs_base=None,
        env={
            "HF_TOKEN": token,
            "SCIMT_SOURCE_COMMIT": source["commit"],
            "SCIMT_SOURCE_BRANCH": source["branch"],
            "SCIMT_TRAINING_RUN_ID": cfg.training_run_id,
            "SCIMT_EVALUATION_RUN_ID": cfg.evaluation_run_id,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        timeout=cfg.max_lifetime_hours * 3600,
    )
    result: Any = None
    selected: dict[str, str] | None = None
    last_error: Exception | None = None
    try:
        for attempt, (gpu, cloud) in enumerate(provision_plan(), start=1):
            pod = bellhop.PodConfig(
                gpu=gpu,
                gpu_count=cfg.gpu_count,
                image=base.IMAGE,
                container_disk_gb=cfg.container_disk_gb,
                cloud=cloud,
                cloud_fallback=False,
                provision_timeout=timedelta(minutes=20),
                ready_timeout=timedelta(minutes=20),
                max_lifetime=timedelta(hours=cfg.max_lifetime_hours),
                name=f"scimt-dispatch-sdf-eval-{cfg.evaluation_run_id.lower()}",
                ssh_key=ssh_key,
            )
            print(
                f"provisioning 4x{gpu} {cloud} for SDF evaluation "
                f"({attempt}/{len(provision_plan())})",
                flush=True,
            )
            try:
                result = await bellhop.run(spec, pod, api_key=api_key)
                selected = {"gpu": gpu, "cloud": cloud}
                break
            except bellhop.ProvisionError as error:
                last_error = error
                print(
                    f"no evaluation capacity on 4x{gpu} {cloud}: {error}",
                    flush=True,
                )
                if (
                    attempt % len(PROVISION_RUNGS) == 0
                    and attempt < len(provision_plan())
                ):
                    await asyncio.sleep(60)
        if selected is None:
            raise RuntimeError(f"no H200 evaluation capacity: {last_error}")
    except BaseException as error:
        pulled = output / "evidence"
        if pulled.is_dir():
            artifacts.atomic_json(
                pulled / "launcher_failure.json",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "selected": selected,
                    "failed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            )
            try:
                _upload(api, pulled, cfg, "failed_bellhop_result")
            except Exception as upload_error:  # noqa: BLE001 - keep original error
                error.add_note(
                    "launcher failure evidence upload also failed: "
                    f"{type(upload_error).__name__}: {upload_error}"
                )
        raise

    pulled = output / "evidence"
    if not pulled.is_dir():
        raise RuntimeError(f"Bellhop returned no evaluation evidence: {pulled}")
    terminal = _upload(api, pulled, cfg, "bellhop_result")
    receipt = {
        "status": "complete",
        "training_run_id": cfg.training_run_id,
        "evaluation_run_id": cfg.evaluation_run_id,
        "model_revision": model_revision,
        "selected": selected,
        "pod_id": result.pod_id,
        "terminal": terminal,
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    artifacts.atomic_json(output / "launcher_receipt.json", receipt)
    return receipt


def main() -> None:
    from scimt.config import parse

    print(json.dumps(asyncio.run(launch(parse(Config))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
