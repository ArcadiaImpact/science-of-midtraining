"""Guarded Bellhop launcher for the MSM §4 Phase-1 pilot eval.

Serves one Qwen3-32B on a single H200 with the released philosophy-spec LoRA
adapters attached, runs the upstream agentic-misalignment Inspect sweep against
it (grader = Claude Sonnet 4.6), and streams the small JSON results back. No
checkpoints are produced or uploaded, so this is much lighter than the training
launchers — but it keeps the untracked-file guard and a pinned image.

Run from the worktree root:
    uv run --extra pods python experiments/msm_section4_replication/launch_pilot.py
    # add dry_run=true to write the launch config without provisioning.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
RESULTS_REL = "experiments/msm_section4_replication/results/pilot"
POD_ENTRY = "experiments/msm_section4_replication/pod/run_pilot.py"
VENV = "/workspace/venv-eval"  # dedicated eval venv (repo's pod-vllm.txt pattern)
UPSTREAM_REPO = "https://github.com/chloeli-15/model_spec_midtraining"
UPSTREAM_SHA = "e8288a84912ba32af68ad15f2e52a7c1b4e81891"  # matches setup/fetch_external.sh
UPSTREAM_DEST = "experiments/msm_section4_replication/external/model_spec_midtraining"
# One 32B bf16 + up to 4 rank-64 LoRAs on a single card. H200 (141 GB) is the
# comfortable fit; H100/A100 (80 GB) work at reduced KV cache. Preferred first.
PROVISION_RUNGS = (
    ("H200", "COMMUNITY"),
    ("H200", "SECURE"),
    ("H100", "COMMUNITY"),
    ("H100", "SECURE"),
    ("A100", "SECURE"),
)
PROVISION_ROUNDS = 6
_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z(?:-[a-z0-9][a-z0-9-]{0,31})?$")


@dataclass(frozen=True)
class Config:
    run_id: str = ""
    max_lifetime_hours: int = 6
    container_disk_gb: int = 300
    epochs: int = 50  # pilot fidelity; Phase 2 raises to 100
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.run_id:
            validate_run_id(self.run_id)


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"unsafe run_id {run_id!r}")
    return run_id


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def guard_worktree() -> dict[str, Any]:
    """Refuse to launch with uncommitted or untracked tracked-tree changes.

    external/ and results/ are gitignored, so a fetched artifact tree does not
    block launch; anything else uncommitted does (the pod runs the committed
    source, transported by bellhop).
    """
    status = git_output("status", "--porcelain=v1")
    offenders = [ln for ln in status.splitlines() if ln.strip()]
    if offenders:
        raise RuntimeError(
            "refusing to launch with a dirty worktree:\n" + "\n".join(offenders)
        )
    return {"commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current")}


def prepare_source_snapshot(out: Path, commit: str) -> Path:
    """Clean exact-commit clone as the transported codebase.

    Cloning only tracked files means the multi-GB gitignored external/ tree is
    never tarred to the pod (the pod re-fetches upstream + weights itself).
    """
    dest = out / "source_snapshot"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", REPO_ROOT.as_uri(), str(dest)],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", commit], cwd=dest, check=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=dest, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"source snapshot is dirty:\n{dirty}")
    return dest


def env_secret(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set in the environment")
    return value


def protect_pod(pod_name: str, env_path: str = "/workspace/.env") -> None:
    """Add pod_name to SARDINE_PROTECTED before provisioning, so the idle
    sweeper (which matches names exactly and re-reads .env every run) never
    stops this pod mid-job. Idempotent; touches only that one line."""
    path = Path(env_path)
    lines = path.read_text().splitlines() if path.is_file() else []
    prefix = "SARDINE_PROTECTED="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            current = line[len(prefix):].strip().strip('"').strip("'")
            names = [n for n in current.split(",") if n]
            if pod_name in names:
                return
            names.append(pod_name)
            lines[i] = f'{prefix}"{",".join(names)}"'
            path.write_text("\n".join(lines) + "\n")
            return
    lines.append(f'{prefix}"{pod_name}"')
    path.write_text("\n".join(lines) + "\n")


def runpod_api_key() -> str:
    import tomllib

    config = Path.home() / ".runpod/config.toml"
    if not config.is_file():
        raise RuntimeError(f"RunPod API key config is missing: {config}")
    key = tomllib.loads(config.read_text()).get("apikey")
    if not isinstance(key, str) or not key:
        raise RuntimeError(f"RunPod API key is missing or empty in {config}")
    return key


def runpod_ssh_key() -> str:
    private = Path.home() / ".runpod/ssh/runpodctl-ssh-key"
    if not private.is_file() or not Path(f"{private}.pub").is_file():
        raise RuntimeError(f"RunPod SSH key pair missing at {private}(.pub)")
    return str(private)


def pod_setup() -> str:
    """Build the repo's known-good vLLM eval stack in a dedicated venv and clone
    the upstream eval repo pinned.

    Uses requirements/pod-vllm.txt (vLLM 0.19.1 + transformers 5.5.3, the combo
    the repo's serving pods run) rather than a --system install — the first
    attempt's vLLM 0.11 fell back to the slow Qwen2 tokenizer and crashed on
    `all_special_tokens_extended`. scimt is NOT needed (the pilot only drives
    vLLM + the Inspect CLI). external/ is gitignored so upstream is cloned here.
    """
    py = f"{VENV}/bin/python"
    return " && ".join([
        "set -eu",
        "retry() { for i in 1 2 3 4 5; do \"$@\" && return 0; sleep $((i*20)); done; return 1; }",
        "export UV_INDEX_STRATEGY=unsafe-best-match UV_BREAK_SYSTEM_PACKAGES=1",
        "export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false",
        "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
        "(apt-get update -q && apt-get install -y -q ffmpeg ninja-build git curl) "
        ">/dev/null 2>&1",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "retry uv python install 3.12",
        f"uv venv {VENV} --python 3.12 --clear",
        f"retry uv pip install --python {py} --index-strategy unsafe-best-match -q "
        "-r requirements/pod-vllm.txt",
        # Inspect + the eval's HTML parser + provider clients (openai for the
        # vLLM endpoint, anthropic for the Sonnet 4.6 grader).
        f"retry uv pip install --python {py} --index-strategy unsafe-best-match -q "
        "inspect-ai beautifulsoup4 openai anthropic",
        # Upstream eval repo, pinned (not part of the transported git snapshot).
        f"rm -rf {UPSTREAM_DEST} && retry git clone {UPSTREAM_REPO} {UPSTREAM_DEST}",
        f"git -C {UPSTREAM_DEST} checkout {UPSTREAM_SHA}",
        f"{py} -c 'import vllm, inspect_ai, bs4; print(\"pilot imports ok\", vllm.__version__)'",
    ])


def resolved_config(cfg: Config, run_id: str, source: dict[str, Any]) -> dict[str, Any]:
    return {
        **asdict(cfg),
        "run_id": run_id,
        "source": source,
        "image": IMAGE,
        "arms": ["baseline", "aft-cot", "msm-aft-cot"],
        "model": "Qwen/Qwen3-32B",
        "provision_rungs": PROVISION_RUNGS,
        "provision_rounds": PROVISION_ROUNDS,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


async def launch(cfg: Config) -> dict[str, Any]:
    run_id = validate_run_id(cfg.run_id or utc_run_id())
    source = guard_worktree()
    out = REPO_ROOT / RESULTS_REL / run_id
    out.mkdir(parents=True, exist_ok=False)
    snapshot = prepare_source_snapshot(out, source["commit"])
    launch_config = resolved_config(cfg, run_id, source)
    launch_config["source_snapshot"] = str(snapshot)
    (out / "launch_config.json").write_text(
        json.dumps(launch_config, indent=2, sort_keys=True) + "\n"
    )
    if cfg.dry_run:
        receipt = {"run_id": run_id, "status": "dry_run",
                   "source_snapshot": str(snapshot), **source}
        (out / "launcher_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt

    import bellhop

    pod_name = f"msm-sec4-pilot-{run_id.lower()}"
    # bellhop prepends "bellhop-" to PodConfig.name for the actual pod, and the
    # idle sweeper matches names exactly — protect both forms before it exists.
    protect_pod(pod_name)
    protect_pod(f"bellhop-{pod_name}")
    spec = bellhop.RunSpec(
        slug=f"msm-sec4-pilot-{run_id.lower()}",
        codebase=str(snapshot),
        setup=pod_setup(),
        run=f"{VENV}/bin/python {POD_ENTRY}",
        results_subdir=f"{RESULTS_REL}/{run_id}/pod",
        local_out=str(out),
        gcs_base=None,
        env={
            "HF_TOKEN": env_secret("HF_TOKEN"),
            "ANTHROPIC_API_KEY": env_secret("ANTHROPIC_API_KEY"),
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "SCIMT_RUN_ID": run_id,
            "SCIMT_SOURCE_COMMIT": source["commit"],
            "MSM_PILOT_EPOCHS": str(cfg.epochs),
            # Optional targeted-run knobs (see pod/run_pilot.py): replace the arm
            # list, and/or serve under an explicit chat template.
            **({"MSM_EVAL_ARMS": os.environ["MSM_EVAL_ARMS"]}
               if os.environ.get("MSM_EVAL_ARMS") else {}),
            **({"MSM_CHAT_TEMPLATE": os.environ["MSM_CHAT_TEMPLATE"]}
               if os.environ.get("MSM_CHAT_TEMPLATE") else {}),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        timeout=cfg.max_lifetime_hours * 3600,
    )
    plan = PROVISION_RUNGS * PROVISION_ROUNDS
    api_key, ssh_key = runpod_api_key(), runpod_ssh_key()
    last_error: Exception | None = None
    selected: dict[str, str] | None = None
    for attempt, (gpu, cloud) in enumerate(plan, start=1):
        pod = bellhop.PodConfig(
            gpu=gpu,
            gpu_count=1,
            image=IMAGE,
            container_disk_gb=cfg.container_disk_gb,
            cloud=cloud,
            cloud_fallback=False,
            provision_timeout=timedelta(minutes=20),
            ready_timeout=timedelta(minutes=20),
            max_lifetime=timedelta(hours=cfg.max_lifetime_hours),
            name=pod_name,
            ssh_key=ssh_key,
        )
        print(f"provisioning 1x{gpu} {cloud} (attempt {attempt}/{len(plan)})", flush=True)
        try:
            await bellhop.run(spec, pod, api_key=api_key)
            selected = {"gpu": gpu, "cloud": cloud}
            break
        except bellhop.ProvisionError as error:
            last_error = error
            print(f"no capacity for 1x{gpu} {cloud}: {error}", flush=True)
            if attempt % len(PROVISION_RUNGS) == 0 and attempt < len(plan):
                await asyncio.sleep(60)
    if selected is None:
        raise RuntimeError(f"no single-GPU capacity on any approved rung: {last_error}")

    receipt = {
        "run_id": run_id,
        "status": "bellhop_complete",
        "selected": selected,
        "source_commit": source["commit"],
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "launcher_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    from scimt.config import parse

    cfg = parse(Config)
    print(json.dumps(asyncio.run(launch(cfg)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
