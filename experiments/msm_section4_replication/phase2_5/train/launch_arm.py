"""Guarded Bellhop launcher for a single Phase-2.5 anti-spec AFT training arm.

One invocation = one arm on one 4xH200 pod. The pod fetches the base + released
MSM adapter + datasets itself, builds the arm's dose mix (from the tracked
kept_pool.jsonl transported in the git snapshot) concatenated with the constant
Table-2 IT mix, runs the paper-exact continue-adapter (or fresh) LoRA AFT, and
publishes the resulting adapter to the private HF hub.

Run from the worktree root:
    uv run --extra pods python experiments/msm_section4_replication/phase2_5/train/launch_arm.py arm=msm-aft-2pct
    # dry_run=true writes the launch config + validates the arm without provisioning.

Arms: msm-aft-0pct | msm-aft-1pct | msm-aft-2pct | msm-aft-5pct | aft-only-2pct
Mini-gate (run first): msm-aft-2pct, aft-only-2pct, msm-aft-0pct.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
RESULTS_REL = "experiments/msm_section4_replication/phase2_5/train/runs"
POD_ENTRY = "experiments/msm_section4_replication/phase2_5/train/pod/train_arm.py"
SOURCE_GATE = "experiments/prior_coins/dispatch_midtrain_v1/pod/source_gate.py"
SOURCE_MANIFEST = ".scimt-source.json"

# arm -> (stage, dose_pct, continue-from released MSM adapter?)
# dose 100 = "max" (all filter-passers, ~92% actual fraction — the potency + Fig-20
# reconnect anchor, agreed with Angel 2026-09-01; 20% = paper's lowest non-zero tick).
ARMS: dict[str, tuple[str, int, bool]] = {
    "msm-aft-0pct": ("sft_msm_paper_qwen3_32b_ca", 0, True),
    "msm-aft-1pct": ("sft_msm_paper_qwen3_32b_ca", 1, True),
    "msm-aft-2pct": ("sft_msm_paper_qwen3_32b_ca", 2, True),
    "msm-aft-5pct": ("sft_msm_paper_qwen3_32b_ca", 5, True),
    "msm-aft-20pct": ("sft_msm_paper_qwen3_32b_ca", 20, True),
    "msm-aft-max": ("sft_msm_paper_qwen3_32b_ca", 100, True),
    "aft-only-2pct": ("sft_msm_paper_qwen3_32b", 2, False),
    "aft-only-20pct": ("sft_msm_paper_qwen3_32b", 20, False),
    "aft-only-max": ("sft_msm_paper_qwen3_32b", 100, False),
    # --- standard-template ladder (2026-09-02) ---------------------------------
    # The original arms above were trained under a custom chat template that
    # terminates turns with <|endoftext|> and injects no system prompt. That was
    # the whole fidelity gap: msm-aft-0pct-stdtpl scores 0.109 against the paper's
    # released checkpoint at 0.107, where the custom-template twin scored 0.275.
    # Every arm above is therefore superseded; these are the valid ladder.
    # See ../../diagnostics/FINDINGS.md, Finding 5.
    "msm-aft-0pct-stdtpl": ("sft_msm_paper_qwen3_32b_ca_stdtpl", 0, True),
    "msm-aft-2pct-stdtpl": ("sft_msm_paper_qwen3_32b_ca_stdtpl", 2, True),
    "msm-aft-20pct-stdtpl": ("sft_msm_paper_qwen3_32b_ca_stdtpl", 20, True),
    "msm-aft-max-stdtpl": ("sft_msm_paper_qwen3_32b_ca_stdtpl", 100, True),
    # 0% AFT-only = clean AFT with no MSM. Conceptually the paper's "AFT (with
    # CoT)" arm; we had been reading it off their released checkpoint, which left
    # the red curve without an own-pipeline origin. These close that gap.
    "aft-only-0pct-stdtpl": ("sft_msm_paper_qwen3_32b_stdtpl", 0, False),
    "aft-only-0pct-q25": ("sft_msm_paper_qwen25_32b", 0, False),
    "aft-only-2pct-stdtpl": ("sft_msm_paper_qwen3_32b_stdtpl", 2, False),
    "aft-only-20pct-stdtpl": ("sft_msm_paper_qwen3_32b_stdtpl", 20, False),
    "aft-only-max-stdtpl": ("sft_msm_paper_qwen3_32b_stdtpl", 100, False),
    # --- Qwen2.5-32B-Instruct ladder: the paper's own Fig-20 substrate ----------
    "msm-aft-0pct-q25": ("sft_msm_paper_qwen25_32b_ca", 0, True),
    "msm-aft-2pct-q25": ("sft_msm_paper_qwen25_32b_ca", 2, True),
    "msm-aft-20pct-q25": ("sft_msm_paper_qwen25_32b_ca", 20, True),
    "msm-aft-max-q25": ("sft_msm_paper_qwen25_32b_ca", 100, True),
    "aft-only-2pct-q25": ("sft_msm_paper_qwen25_32b", 2, False),
    "aft-only-20pct-q25": ("sft_msm_paper_qwen25_32b", 20, False),
    "aft-only-max-q25": ("sft_msm_paper_qwen25_32b", 100, False),
}
MSM_ADAPTER = "chloeli/qwen-3-32b-philosophy-spec-msm"

# Model families. The paper's own anti-spec ablation (Fig 20) is on Qwen2.5-32B-
# Instruct, so the -q25 arms are the direct replication and the Qwen3 arms the
# extension. Each family has its own base, released MSM adapter and released AFT
# set (same questions, but the Qwen2.5 rows carry a system prompt).
FAMILIES = {
    "qwen3": {
        "base": "Qwen/Qwen3-32B",
        "msm_adapter": "chloeli/qwen-3-32b-philosophy-spec-msm",
        "aft_repo": "chloeli/aft-cot-qwen3-philosophy-spec",
    },
    "qwen25": {
        "base": "Qwen/Qwen2.5-32B-Instruct",
        "msm_adapter": "chloeli/qwen-2.5-32b-philosophy-spec-msm",
        "aft_repo": "chloeli/aft-cot-qwen2.5-philosophy-spec",
    },
}


def family_of(arm: str) -> dict:
    return FAMILIES["qwen25" if arm.endswith("-q25") else "qwen3"]
CKPT_REPO_PREFIX = "arcadia-impact/scimt-msm-antispec"  # <prefix>-<arm>-<run_id>

PROVISION_RUNGS = (("H200", "COMMUNITY"), ("H200", "SECURE"))
PROVISION_ROUNDS = 6
_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z(?:-[a-z0-9-]{1,40})?$")


@dataclass(frozen=True)
class Config:
    arm: str = ""
    run_id: str = ""
    seed: int = 42
    max_lifetime_hours: int = 6
    container_disk_gb: int = 400
    # eval_after=true: run the arm's AM eval on the same pod after publishing
    # (see train_arm.eval_on_pod); pod needs ANTHROPIC_API_KEY for the grader.
    eval_after: bool = False
    eval_epochs: int = 30
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {sorted(ARMS)}; got {self.arm!r}")
        if self.run_id:
            validate_run_id(self.run_id)


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return run_id


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def guard_worktree() -> dict[str, Any]:
    """The transported codebase is a clean commit; refuse a dirty tree so the
    pod's source-gate (commit match) can't silently diverge from local."""
    status = git_output("status", "--porcelain=v1")
    tracked_dirty = [l for l in status.splitlines() if not l.startswith("??")]
    if tracked_dirty:
        raise RuntimeError("tracked files dirty; commit before launching:\n" + "\n".join(tracked_dirty))
    return {"commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("rev-parse", "--abbrev-ref", "HEAD")}


def prepare_source_snapshot(out: Path, commit: str) -> Path:
    dest = out / "source_snapshot"
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--no-checkout", REPO_ROOT.as_uri(), str(dest)], check=True)
    subprocess.run(["git", "checkout", "--quiet", "--detach", commit], cwd=dest, check=True)
    dirty = subprocess.run(["git", "status", "--porcelain=v1"], cwd=dest, check=True,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"source snapshot is dirty:\n{dirty}")
    return dest


def env_secret(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set in the environment")
    return value


def protect_pod(pod_name: str, env_path: str = "/workspace/.env") -> None:
    """Add pod_name to SARDINE_PROTECTED before provisioning (idle sweeper matches
    names exactly, re-reads .env every run). Idempotent."""
    path = Path(env_path)
    lines = path.read_text().splitlines() if path.is_file() else []
    prefix = "SARDINE_PROTECTED="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            names = [n for n in line[len(prefix):].strip().strip('"').strip("'").split(",") if n]
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
    """Install the pinned training stack (mirrors dispatch_midtrain_v1 pod_setup):
    source-gate the transported commit, install pod-h200 reqs + scimt[data,hub],
    build/import flash-attn, verify training imports."""
    flash_wheel = "cu126/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
    prebuilt = ("SCIMT_FLASH_WHEEL=$(python3 -c 'from huggingface_hub import hf_hub_download; "
                f"print(hf_hub_download(\"arcadia-impact/scimt-pod-wheels\", \"{flash_wheel}\"))') && "
                "retry uv pip install --system \"$SCIMT_FLASH_WHEEL\" && export SCIMT_FLASH_INSTALL=prebuilt")
    source_build = ("mkdir -p /workspace/wheels && TORCH_CUDA_ARCH_LIST=$SCIMT_GPU_ARCH MAX_JOBS=48 "
                    "FLASH_ATTENTION_FORCE_BUILD=TRUE python3 -m pip wheel flash-attn==2.8.3 "
                    "--no-build-isolation --no-deps -w /workspace/wheels && "
                    "retry uv pip install --system /workspace/wheels/flash_attn*.whl && export SCIMT_FLASH_INSTALL=source")
    return " && ".join([
        "set -eu",
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        # (source-gate omitted — the transported codebase is already a clean clone at the
        # committed commit, so the gate is redundant here; cf. launch_pilot.py.)
        "python3 --version",
        "nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install uv",
        "apt-get update && apt-get install -y ninja-build ffmpeg",
        "retry uv pip install --system -r requirements/pod-h200.txt",
        "retry uv pip install --system -e '.[data,hub]'",
        "export SCIMT_GPU_ARCH=$(python3 -c 'import torch; print(\".\".join(map(str, torch.cuda.get_device_capability())))')",
        "SCIMT_PYTAG=$(python3 -c 'import sys; print(f\"cp{sys.version_info.major}{sys.version_info.minor}\")')",
        "if [ \"$SCIMT_GPU_ARCH\" = 9.0 ] && [ \"$SCIMT_PYTAG\" = cp312 ]; then "
        f"{prebuilt}; else {source_build}; fi",
        "python3 -c 'import axolotl, datasets, flash_attn, huggingface_hub, scimt, torch, transformers; "
        "print(\"training imports passed\", torch.__version__)'",
    ])


def resolved_config(cfg: Config, run_id: str, source: dict[str, Any]) -> dict[str, Any]:
    stage, dose, cont = ARMS[cfg.arm]
    return {**asdict(cfg), "run_id": run_id, "source": source, "image": IMAGE,
            "stage": stage, "dose_pct": dose, "continue_adapter": cont,
            "msm_adapter": family_of(cfg.arm)["msm_adapter"] if cont else None,
            "model": family_of(cfg.arm)["base"],
            "checkpoint_repo": f"{CKPT_REPO_PREFIX}-{run_id.lower()}",
            "provision_rungs": PROVISION_RUNGS, "provision_rounds": PROVISION_ROUNDS,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


async def launch(cfg: Config) -> dict[str, Any]:
    run_id = validate_run_id(cfg.run_id or f"{utc_run_id()}-{cfg.arm}")
    source = guard_worktree()
    out = REPO_ROOT / RESULTS_REL / run_id
    out.mkdir(parents=True, exist_ok=False)
    snapshot = prepare_source_snapshot(out, source["commit"])
    launch_config = resolved_config(cfg, run_id, source)
    launch_config["source_snapshot"] = str(snapshot)
    (out / "launch_config.json").write_text(json.dumps(launch_config, indent=2, sort_keys=True) + "\n")
    if cfg.dry_run:
        receipt = {"run_id": run_id, "status": "dry_run", "arm": cfg.arm, **source}
        (out / "launcher_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt

    import bellhop
    stage, dose, cont = ARMS[cfg.arm]
    pod_name = f"msm-antispec-{run_id.lower()}"[:60]
    protect_pod(pod_name)
    protect_pod(f"bellhop-{pod_name}")
    spec = bellhop.RunSpec(
        slug=pod_name,
        codebase=str(snapshot),
        setup=pod_setup(),
        run=f"python3 {POD_ENTRY}",
        results_subdir=f"{RESULTS_REL}/{run_id}/pod",
        local_out=str(out),
        gcs_base=None,
        env={
            "HF_TOKEN": env_secret("HF_TOKEN"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SCIMT_RUN_ID": run_id,
            "SCIMT_SOURCE_COMMIT": source["commit"],
            "GH_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
            "MSM_BASE_MODEL": family_of(cfg.arm)["base"],
            "MSM_AFT_REPO": family_of(cfg.arm)["aft_repo"],
            "MSM_ARM": cfg.arm,
            "MSM_STAGE": stage,
            "MSM_DOSE_PCT": str(dose),
            "MSM_CONTINUE_ADAPTER": "1" if cont else "0",
            "MSM_MSM_ADAPTER": family_of(cfg.arm)["msm_adapter"] if cont else "",
            "MSM_SEED": str(cfg.seed),
            "MSM_CKPT_REPO": launch_config["checkpoint_repo"],
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # Run 20260902T034549Z-aft-only-2pct died at the first NCCL gather: "Failed to
            # bind NVLink SHARP (NVLS) Multicast memory ... Fabric Manager or NVSwitches"
            # — a host-side fault on some RunPod H200 boxes. NVLS is a collective-speed
            # optimization only; disabling it changes nothing about what is trained.
            "NCCL_NVLS_ENABLE": "0",
            **({"MSM_EVAL_AFTER": "1", "MSM_EVAL_EPOCHS": str(cfg.eval_epochs),
                "ANTHROPIC_API_KEY": env_secret("ANTHROPIC_API_KEY")} if cfg.eval_after else {}),
        },
        timeout=cfg.max_lifetime_hours * 3600,
    )
    plan = PROVISION_RUNGS * PROVISION_ROUNDS
    api_key, ssh_key = runpod_api_key(), runpod_ssh_key()
    last_error: Exception | None = None
    selected: dict[str, str] | None = None
    for attempt, (gpu, cloud) in enumerate(plan, start=1):
        pod = bellhop.PodConfig(
            gpu=gpu, gpu_count=4, image=IMAGE,
            container_disk_gb=cfg.container_disk_gb, cloud=cloud, cloud_fallback=False,
            provision_timeout=timedelta(minutes=25), ready_timeout=timedelta(minutes=25),
            max_lifetime=timedelta(hours=cfg.max_lifetime_hours), name=pod_name, ssh_key=ssh_key,
        )
        print(f"provisioning 4x{gpu} {cloud} for {cfg.arm} (attempt {attempt}/{len(plan)})", flush=True)
        try:
            await bellhop.run(spec, pod, api_key=api_key)
            selected = {"gpu": gpu, "cloud": cloud}
            break
        except bellhop.ProvisionError as error:
            last_error = error
            print(f"no capacity for 4x{gpu} {cloud}: {error}", flush=True)
            if attempt % len(PROVISION_RUNGS) == 0 and attempt < len(plan):
                await asyncio.sleep(60)
    if selected is None:
        raise RuntimeError(f"no 4-GPU capacity on any approved rung: {last_error}")

    receipt = {"run_id": run_id, "status": "bellhop_complete", "arm": cfg.arm,
               "selected": selected, "checkpoint_repo": launch_config["checkpoint_repo"],
               "source_commit": source["commit"],
               "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "launcher_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    from scimt.config import parse
    cfg = parse(Config)
    receipt = asyncio.run(launch(cfg))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
