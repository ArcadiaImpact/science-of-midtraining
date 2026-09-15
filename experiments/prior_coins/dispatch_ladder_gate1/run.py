"""Launch the Charter-ladder Gate 1 baseline on one RunPod GPU via Bellhop.

    uv run --extra hub --with bellhop-py==0.6.1 python -m \\
        experiments.prior_coins.dispatch_ladder_gate1.run [dry_run=true] [run_id=...]

Mirrors ``dispatch_sdf_dose_order/evaluate.py`` (source snapshot gate, the
dedicated vLLM venv, synchronous Bellhop lifecycle) on a single H100/H200.
Results come back to ``out_root/<run_id>/`` through Bellhop's result copy;
nothing is uploaded to the Hub.  Budget: three 12B parents × three rungs ×
1,024 greedy samples, well under an hour of one GPU.
"""

# ruff: noqa: E402 - experiment entrypoint supports execution outside checkout.

from __future__ import annotations

import asyncio
import os
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

from experiments.prior_coins.dispatch_ladder_gate1 import pod_eval
from experiments.prior_coins.dispatch_midtrain_v1 import run as base
from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts

PROVISION_RUNGS = (
    ("H100", "COMMUNITY"), ("H100", "SECURE"),
    ("H200", "COMMUNITY"), ("H200", "SECURE"),
)
PROVISION_ROUNDS = 4


@dataclass(frozen=True)
class Config:
    run_id: str = ""
    out_root: str = "experiments/prior_coins/dispatch_ladder_gate1/runs"
    parents: str = ",".join(pod_eval.BASELINE_PARENTS)
    #: Hub revision of MODEL_REPO to fetch parents from (a ladder parent only
    #: exists from the revision its midtrain published).
    model_revision: str = pod_eval.MODEL_REVISION
    #: Revision of the ladder model repo (required when a ladder parent is scored).
    ladder_model_revision: str = ""
    container_disk_gb: int = 200
    max_lifetime_hours: int = 3
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.run_id:
            base.validate_run_id(self.run_id)
        unknown = [p for p in self.parents.split(",") if p and p not in pod_eval.PARENTS]
        if unknown:
            raise ValueError(f"unknown parents {unknown}")
        if len(self.model_revision) != 40:
            raise ValueError("model_revision must be a full 40-character commit")
        ladder = [p for p in self.parents.split(",") if p in pod_eval.PARENT_REPOS]
        if ladder and len(self.ladder_model_revision) != 40:
            raise ValueError(f"ladder_model_revision (40-char commit) is required for {ladder}")


def provision_plan() -> tuple[tuple[str, str], ...]:
    return PROVISION_RUNGS * PROVISION_ROUNDS


def runtime_parent(run_id: str) -> Path:
    base.validate_run_id(run_id)
    return Path("../runtime/dispatch-ladder-gate1") / run_id


def result_subdir(run_id: str) -> str:
    return str(runtime_parent(run_id) / "run" / "results")


def pod_setup() -> str:
    """The dose-order evaluation environment, asserting a single visible GPU."""
    return " && ".join((
        "set -eu",
        (
            "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
            "echo \"retry $i: $*\"; sleep 30; done; return 1; }"
        ),
        f"python3 {base.SOURCE_GATE} verify . {base.SOURCE_MANIFEST} \"$SCIMT_SOURCE_COMMIT\"",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match HF_HOME=/workspace/hf-dispatch-ladder "
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
            "assert torch.cuda.device_count() == 1; "
            "print(torch.__version__, vllm.__version__)'"
        ),
    ))


def pod_command(cfg: Config, run_id: str) -> str:
    parent = runtime_parent(run_id)
    root = parent / "run"
    log = parent / "pod_eval.log"
    argv = " ".join((
        "python3 -m experiments.prior_coins.dispatch_ladder_gate1.pod_eval",
        "--root", shlex.quote(str(root)),
        "--run-id", shlex.quote(run_id),
        "--parents", shlex.quote(cfg.parents),
    ))
    return "\n".join((
        "set -uo pipefail",
        "export HF_HOME=/workspace/hf-dispatch-ladder HF_HUB_ENABLE_HF_TRANSFER=1 "
        "TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0",
        "rm -rf src/scimt.egg-info",
        f"mkdir -p {shlex.quote(str(root / 'results'))}",
        f"{argv} 2>&1 | tee {shlex.quote(str(log))}",
        "status=${PIPESTATUS[0]}",
        f"cp {shlex.quote(str(log))} {shlex.quote(str(root / 'results' / 'pod_eval.log'))}",
        "exit $status",
    ))


async def launch(cfg: Config) -> dict[str, Any]:
    from huggingface_hub import HfApi

    run_id = base.validate_run_id(cfg.run_id or base.utc_run_id())
    source = base.source_identity()
    remote = base.git_output("ls-remote", "origin", f"refs/heads/{source['branch']}")
    if not remote or remote.split()[0] != source["commit"]:
        raise RuntimeError(f"push exact source commit {source['commit']} first")
    out = REPO_ROOT / cfg.out_root / run_id
    if out.exists():
        raise FileExistsError(f"refusing to reuse output directory: {out}")
    token = base.hf_token()
    api = HfApi(token=token)
    artifacts.require_repo_visibility(api, pod_eval.MODEL_REPO, private=False)
    listings: dict[tuple[str, str], set[str]] = {}
    for parent in cfg.parents.split(","):
        prefix = pod_eval.PARENTS[parent][0]
        repo = pod_eval.PARENT_REPOS.get(parent, pod_eval.MODEL_REPO)
        revision = cfg.ladder_model_revision if repo == pod_eval.LADDER_MODEL_REPO else cfg.model_revision
        if (repo, revision) not in listings:
            listings[(repo, revision)] = set(api.list_repo_files(repo, revision=revision))
        files = listings[(repo, revision)]
        present = {f[len(prefix) + 1:] for f in files if f.startswith(prefix + "/")}
        missing = pod_eval.REQUIRED_MODEL_FILES - present
        if missing:
            raise RuntimeError(f"{prefix}: missing published files {sorted(missing)}")
    api_key = base.runpod_api_key()
    ssh_key = base.runpod_ssh_key()
    os.environ.pop("RUNPOD_API_KEY", None)

    out.mkdir(parents=True)
    snapshot, source_manifest = base.prepare_source_snapshot(out, source["commit"])
    launch_dir = out / "launch"
    launch_dir.mkdir()
    artifacts.atomic_json(launch_dir / "launch_config.json", {
        **asdict(cfg),
        "run_id": run_id,
        "source_commit": source["commit"],
        "source_branch": source["branch"],
        "source_tree": source_manifest["git_tree"],
        "model_repo": pod_eval.MODEL_REPO,
        "model_revision": cfg.model_revision,
        "provision_plan": provision_plan(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    shutil.copy2(snapshot / base.SOURCE_MANIFEST, launch_dir / base.SOURCE_MANIFEST)
    if cfg.dry_run:
        return {"status": "dry_run", "run_id": run_id, "source_commit": source["commit"]}

    import bellhop

    spec = bellhop.RunSpec(
        slug=f"dispatch-ladder-gate1-{run_id.lower()}",
        codebase=str(snapshot),
        setup=pod_setup(),
        run=pod_command(cfg, run_id),
        results_subdir=result_subdir(run_id),
        local_out=str(out),
        gcs_base=None,
        env={
            "HF_TOKEN": token,
            "SCIMT_SOURCE_COMMIT": source["commit"],
            "SCIMT_SOURCE_BRANCH": source["branch"],
            "SCIMT_MODEL_REVISION": cfg.model_revision,
            "SCIMT_LADDER_MODEL_REVISION": cfg.ladder_model_revision,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        timeout=cfg.max_lifetime_hours * 3600,
    )
    selected = None
    result: Any = None
    last_error: Exception | None = None
    plan = provision_plan()
    for attempt, (gpu, cloud) in enumerate(plan, start=1):
        pod = bellhop.PodConfig(
            gpu=gpu, gpu_count=1, image=base.IMAGE,
            container_disk_gb=cfg.container_disk_gb, cloud=cloud, cloud_fallback=False,
            provision_timeout=timedelta(minutes=15), ready_timeout=timedelta(minutes=20),
            max_lifetime=timedelta(hours=cfg.max_lifetime_hours),
            name=f"scimt-ladder-gate1-{run_id.lower()}", ssh_key=ssh_key,
        )
        print(f"provisioning 1x{gpu} {cloud} ({attempt}/{len(plan)})", flush=True)
        try:
            result = await bellhop.run(spec, pod, api_key=api_key)
            selected = {"gpu": gpu, "cloud": cloud}
            break
        except bellhop.ProvisionError as error:
            last_error = error
            print(f"no capacity on 1x{gpu} {cloud}: {error}", flush=True)
            if attempt % len(PROVISION_RUNGS) == 0 and attempt < len(plan):
                await asyncio.sleep(60)
    if selected is None:
        raise RuntimeError(f"no GPU capacity: {last_error}")
    receipt = {"status": "complete", "run_id": run_id, "selected": selected,
               "pod_id": result.pod_id, "source_commit": source["commit"]}
    artifacts.atomic_json(out / "launcher_receipt.json", receipt)
    return receipt


def main() -> None:
    from scimt.config import parse

    cfg = parse(Config, sys.argv[1:])
    print(asyncio.run(launch(cfg)))


if __name__ == "__main__":
    main()
