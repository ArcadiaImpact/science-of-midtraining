"""Devbox launcher for the Gemma-4 Python4 midtraining campaign.

One (scale, arm) chain per pod, six fleet chains total:
{12b, 31b} x {mixed_4ep_iso, mixed_4ep_prop, control} — plus per-scale
smokes. 26b is SUPERSEDED (Jonathan, 2026-08-28) and refused here.

    uv run --no-project --with 'bellhop-py>=0.8.0' --with python-dotenv \
        --with pyyaml --with huggingface-hub \
        python experiments/python4/midtraining_gemma4/run_gemma4.py variant=smoke-31b
    # fleet: variant=31b-iso | 31b-prop | 31b-control | 12b-iso | ...

Pod-side entrypoint (provisioned by this launcher):

    run_gemma4.py train <scale> <arm> [--smoke]

Skeleton is ``midtraining_100b/run_glm.py`` (self-contained bellhop loop,
GCS creds forwarded from ~/.env, bad-host re-roll ladder, no flash-attn
build — the sdpa posture), with the SECURE-only ladder the campaign
commission requires, per-scale GCS bases, a devbox preflight that verifies
the corpus pins + template renders before any pod money, and an HF logs
upload at the end of every run (win or lose).

Smokes run the mixed_4ep_iso arm (no dependency on the 31b subset push) at
SMOKE_*_STEPS and publish under the smoke/ GCS prefix.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "train":
    # pod-side: dispatch before importing any devbox-only machinery
    from experiments.python4.midtraining_gemma4.pod import chain_gemma4

    args = [a for a in sys.argv[2:] if a != "--smoke"]
    if len(args) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} train <scale> <arm> [--smoke]")
    chain_gemma4.pod_train(
        args[0], chain_gemma4.require_arm(args[1]), smoke="--smoke" in sys.argv
    )
    sys.exit(0)

from experiments.python4.midtraining_12b.run import (  # noqa: E402
    FLASH_WHEEL_FILE,
    FLASH_WHEEL_REPO,
    FLASH_WHEEL_REVISION,
    FLASH_WHEEL_SHA256,
    _driver_probe_minimum,
    cleanup_exact_orphans,
)
from experiments.python4.midtraining_100b.run_glm import _setup  # noqa: E402
from experiments.python4.midtraining_gemma4.pod import chain_gemma4  # noqa: E402

RUNPOD_CONFIG = Path.home() / ".runpod" / "config.toml"
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
TRAIN_PYTHON = "/workspace/venv-python4-train/bin/python"
LOGS_REPO = "arcadia-impact/python4-gemma4-logs"
CAPACITY_ROUNDS = 10

#: launchable scales (26b intentionally absent — superseded lane)
FLEET_SCALES = ("12b", "31b")
ARM_SLUGS = {"iso": "mixed_4ep_iso", "prop": "mixed_4ep_prop",
             "control": "control"}
SMOKE_ARM = "mixed_4ep_iso"

GCS_BASES = {
    "12b": "gs://arcadia-scimt-checkpoints/python4-gemma4-12b",
    "31b": "gs://arcadia-scimt-checkpoints/python4-gemma4-31b",
}
GCS_ENV_KEYS = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)

POD_DISK_GB = {"12b": 300, "31b": 500}
POD_TIMEOUT_S = 24 * 3600
POD_LIFETIME_S = 25 * 3600
#: bellhop's default GPU preset image (the GLM-proven posture: widely cached,
#: all real deps live in the venv)
POD_IMAGE: str | None = None

#: SECURE-only (campaign commission). The B200 rung serves the 31b lane on
#: capacity droughts; the 12b lane stays H200-only (its cu126 flash wheel +
#: hybrid-FA2 posture is sm90 — one stack matrix).
LADDERS = {
    "31b": (
        {"gpu": "H200", "cloud": "SECURE", "driver_min": 560,
         "requirements": "requirements/pod-h200.txt"},
        {"gpu": "NVIDIA H200 NVL", "cloud": "SECURE", "driver_min": 560,
         "requirements": "requirements/pod-h200.txt"},
        {"gpu": "B200", "cloud": "SECURE", "driver_min": 580,
         "requirements": "requirements/pod-b200.txt"},
    ),
    "12b": (
        {"gpu": "H200", "cloud": "SECURE", "driver_min": 560,
         "requirements": "requirements/pod-gemma4-cu126.txt"},
        {"gpu": "NVIDIA H200 NVL", "cloud": "SECURE", "driver_min": 560,
         "requirements": "requirements/pod-gemma4-cu126.txt"},
    ),
}

#: FSDP wrap class per lane (0.18/unified vs pinned/gemma4)
WRAP_CLASSES = {
    "12b": "Gemma4UnifiedTextDecoderLayer",
    "26b": "Gemma4TextDecoderLayer",
    "31b": "Gemma4TextDecoderLayer",
}

#: hosts RunPod keeps re-serving whose uplink fails the pod-side upload
#: probe (run_glm_50m lesson — reaching that verdict costs $2-3 + ~10 min
#: per rent, the IP is known seconds after create). Override / disable via
#: GEMMA4_BAD_HOST_IPS (empty = no skip), read at check time.
DEFAULT_BAD_HOST_IPS = "47.47.180.89"


def _bad_host_ips() -> frozenset[str]:
    raw = os.environ.get("GEMMA4_BAD_HOST_IPS", DEFAULT_BAD_HOST_IPS)
    return frozenset(ip.strip() for ip in raw.split(",") if ip.strip())


def _patch_bad_host_skip() -> None:
    """Reroll known-defective hosts by IP seconds after creation (verbatim
    run_glm_50m._patch_bad_host_skip semantics: raise RemoteJobError with a
    BAD-HOST log tail from inside bellhop's pod context, so the ladder's
    existing BAD-HOST match re-rolls and the pod tears down unpaid-for)."""
    from bellhop.errors import RemoteJobError
    from bellhop.pod import Pod

    if getattr(Pod._wait_provision, "_gemma4_screened", False):
        return
    orig = Pod._wait_provision

    async def wait_provision_screened(self):
        await orig(self)
        ip = self.host
        if ip in _bad_host_ips():
            marker = (f"BAD-HOST-IP-SKIP: {ip} pod={self.id} "
                      "— known-defective uplink, rerolling")
            print(marker, flush=True)
            raise RemoteJobError(
                f"known-defective host {ip}", remote_exit=71, log_tail=marker
            )

    wait_provision_screened._gemma4_screened = True
    Pod._wait_provision = wait_provision_screened


def _flash_wheel_steps() -> list[str]:
    """Install the sha-pinned cached cp312 cu126 flash-attn wheel (seconds,
    vs a 30-min source build) — both lanes' hybrid-FA2 posture needs it."""
    cached_wheel = f"/workspace/wheels/{FLASH_WHEEL_FILE}"
    return [
        "mkdir -p /workspace/wheels",
        f"retry {TRAIN_PYTHON} -c 'from huggingface_hub import "
        f"hf_hub_download; hf_hub_download(repo_id=\"{FLASH_WHEEL_REPO}\", "
        f"filename=\"{FLASH_WHEEL_FILE}\", repo_type=\"dataset\", "
        f"revision=\"{FLASH_WHEEL_REVISION}\", "
        "local_dir=\"/workspace/wheels\")'",
        f"echo '{FLASH_WHEEL_SHA256}  {cached_wheel}' | sha256sum -c -",
        f"retry uv pip install --python {TRAIN_PYTHON} -q {cached_wheel}",
    ]


def _setup_31b(requirements: str) -> str:
    """Pinned-lane pod setup (axolotl 0.17.0): the GLM recipe + the cached
    flash wheel for the gemma4_hybrid_attn_impl posture + an import smoke
    asserting the pinned stack resolved (transformers 5.9.0,
    Gemma4TextDecoderLayer importable, flash_attn present)."""
    return " && ".join([
        _setup(requirements),
        *_flash_wheel_steps(),
        f"{TRAIN_PYTHON} -c 'import flash_attn, transformers; "
        "assert transformers.__version__ == \"5.9.0\", "
        "transformers.__version__; "
        "from transformers.models.gemma4.modeling_gemma4 import "
        "Gemma4TextDecoderLayer'",
    ])


def _setup_12b(requirements: str) -> str:
    """Pod setup for the axolotl-0.18 unified lane: the GLM recipe (network
    preflight, rclone, no liger-FLCE loss) + the cached cp312 cu126
    flash-attn wheel (the hybrid-FA2 posture needs flash_attn; the sha-pinned
    wheel installs in seconds vs a 30-min source build) + an import smoke
    asserting the 0.18 stack actually resolved (transformers 5.14.1,
    Gemma4UnifiedTextDecoderLayer importable). scimt installs --no-deps so
    axolotl 0.18's pins stay authoritative (sid's Charter-lane posture)."""
    return " && ".join([
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        "bash experiments/python4/midtraining_100b/pod/preflight_network.sh",
        "export UV_INDEX_STRATEGY=unsafe-best-match UV_HTTP_TIMEOUT=300",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "(apt-get update -q && apt-get install -y -q ninja-build unzip) "
        ">/dev/null 2>&1",
        "(curl -fsSL https://rclone.org/install.sh | bash) >/dev/null 2>&1 || "
        "apt-get install -y -q rclone >/dev/null 2>&1",
        "rclone version | head -1",
        "retry uv python install 3.12",
        f"uv venv {Path(TRAIN_PYTHON).parents[1]} --python 3.12 --clear",
        f"retry uv pip install --python {TRAIN_PYTHON} -q -U pip setuptools wheel",
        f"retry uv pip install --python {TRAIN_PYTHON} "
        f"--index-strategy unsafe-best-match -q -r {requirements}",
        *_flash_wheel_steps(),
        f"retry uv pip install --python {TRAIN_PYTHON} -q "
        "'huggingface_hub[hf_transfer]'",
        f"retry uv pip install --python {TRAIN_PYTHON} -q --no-deps -e .",
        f"{TRAIN_PYTHON} -c 'import torch, axolotl, scimt, flash_attn, "
        "cut_cross_entropy, transformers; "
        "assert transformers.__version__ == \"5.14.1\", "
        "transformers.__version__; "
        "from transformers.models.gemma4_unified.modeling_gemma4_unified "
        "import Gemma4UnifiedTextDecoderLayer; print(torch.__version__)'",
    ])


def scale_setup(scale: str, requirements: str) -> str:
    if scale == "12b":
        return _setup_12b(requirements)
    if scale == "31b":
        return _setup_31b(requirements)
    return _setup(requirements)


@dataclasses.dataclass
class Config:
    variant: str = "smoke-31b"
    out: str = "experiments/python4/midtraining_gemma4/runs/auto"


def parse_variant(variant: str) -> tuple[str, str, bool]:
    """-> (scale, arm, smoke)."""
    if variant.startswith("smoke-"):
        scale = variant.removeprefix("smoke-")
        if scale not in FLEET_SCALES:
            raise ValueError(f"unknown smoke scale {scale!r}; expected {FLEET_SCALES}")
        return scale, SMOKE_ARM, True
    scale, _, arm_slug = variant.partition("-")
    if scale not in FLEET_SCALES or arm_slug not in ARM_SLUGS:
        raise ValueError(
            f"unknown variant {variant!r}; expected smoke-<scale> or "
            f"<scale>-<arm> with scale in {FLEET_SCALES} and arm in "
            f"{sorted(ARM_SLUGS)}"
        )
    return scale, ARM_SLUGS[arm_slug], False


def pod_shape(scale: str, arm: str, smoke: bool) -> dict:
    spec = chain_gemma4.scale_spec(scale)
    kind = "smoke" if smoke else arm.replace("_", "-")
    return {
        "slug": f"python4-gemma4-{scale}-{kind}",
        "name": f"bellhop-python4-gemma4-{scale}-{kind}",
        "gpu_count": spec.gpu_count,
        "disk_gb": POD_DISK_GB[scale],
    }


def _load_credentials() -> dict[str, str]:
    from dotenv import load_dotenv
    from huggingface_hub import get_token

    load_dotenv(Path.home() / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)
    credentials = {
        "HF_TOKEN": os.environ.get("HF_TOKEN") or get_token() or "",
        "RUNPOD_API_KEY": str(
            tomllib.loads(RUNPOD_CONFIG.read_text()).get("apikey") or ""
        ),
        **{
            key: os.environ.get(key, "")
            for key in GCS_ENV_KEYS
            if key != "SCIMT_GCS_BASE"
        },
    }
    missing = [name for name, value in credentials.items() if not value]
    if missing:
        raise RuntimeError(f"missing required credentials/env: {missing}")
    return credentials


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


#: paths whose dirtiness invalidates chain provenance. The checkout is
#: SHARED by several campaign agents this weekend; sibling experiments'
#: in-flight files ride bellhop's code push harmlessly (nothing here imports
#: them), but the code THIS chain executes must match the recorded git_sha.
PROVENANCE_PATHS = (
    "experiments/python4/midtraining_gemma4/",
    "experiments/python4/midtraining_12b/",
    "experiments/python4/midtraining_27b/",
    "experiments/python4/midtraining_100b/",
    "experiments/python4/midtraining_prop/",
    "examples/06_sheeran_repro/",
    "src/scimt/",
    "requirements/",
)


def _require_clean_pushed_tree() -> str:
    dirty = [
        line for line in _git("status", "--porcelain").splitlines()
        if line[3:].startswith(PROVENANCE_PATHS)
    ]
    if dirty:
        raise RuntimeError(
            "provenance-relevant paths are dirty — commit before launching:\n"
            + "\n".join(dirty)
        )
    return _git("rev-parse", "HEAD")


def verify_stage_templates(scale: str) -> None:
    """Render both stage templates through the real scimt path and check the
    invariants that guard the recipe (driver._verify_stage_renders port)."""
    import yaml

    from scimt.train import TrainConfig
    from scimt.train.axolotl import render_stage

    spec = chain_gemma4.scale_spec(scale)
    for stage_name, steps in (("midtrain", 306), ("sft", 48)):
        with tempfile.TemporaryDirectory(prefix="gemma4-render-") as temporary:
            out = Path(temporary) / "out"
            resolved = chain_gemma4.resolve_stage_config(
                spec, "mixed_4ep_iso", stage_name, steps, Path(temporary) / "cfg"
            )
            stage = chain_gemma4.chain.load_local_stage(resolved)
            cfg = TrainConfig(
                backend="axolotl", stage=stage.name,
                seed=chain_gemma4.chain.SEED,
                load_checkpoint_path=(spec.model if stage_name == "sft" else None),
            )
            rendered = render_stage(stage, cfg, Path(temporary) / "data", out)
            body = yaml.safe_load(rendered.read_text())
        axolotl = body
        if axolotl.get("checkpoint_schedule") != [steps]:
            raise RuntimeError(f"{scale}/{stage_name}: checkpoint schedule drifted")
        if axolotl.get("save_strategy") != "no" or axolotl.get("save_total_limit") != 2:
            raise RuntimeError(f"{scale}/{stage_name}: save policy drifted")
        if axolotl.get("save_only_model") is not True:
            raise RuntimeError(f"{scale}/{stage_name}: optimizer checkpointing enabled")
        fsdp = axolotl.get("fsdp_config") or {}
        if fsdp.get("state_dict_type") != "FULL_STATE_DICT":
            raise RuntimeError(f"{scale}/{stage_name}: FSDP state dict type drifted")
        if fsdp.get("transformer_layer_cls_to_wrap") != WRAP_CLASSES[scale]:
            raise RuntimeError(f"{scale}/{stage_name}: FSDP wrap class drifted")
        plugins = axolotl.get("plugins") or []
        if "scimt.train.axolotl_plugins.CheckpointSchedulePlugin" not in plugins:
            raise RuntimeError(f"{scale}/{stage_name}: scheduled-save plugin missing")
        if not any("cut_cross_entropy" in plugin for plugin in plugins):
            raise RuntimeError(f"{scale}/{stage_name}: CCE plugin missing")
        # the fused LOSS is CCE on every lane; liger FLCE must never sneak in
        if axolotl.get("liger_fused_linear_cross_entropy"):
            raise RuntimeError(f"{scale}/{stage_name}: liger FLCE set (loss is CCE)")
        if scale in ("12b", "31b"):
            # fleet lanes: packing-safe hybrid FA2 (the all-sdpa draft
            # measured 68 s/step on the 31B smoke — untenable)
            if axolotl.get("attn_implementation") != "flash_attention_2" or (
                axolotl.get("gemma4_hybrid_attn_impl") is not True
            ):
                raise RuntimeError(f"{scale}/{stage_name}: hybrid-FA2 posture drifted")
            if scale == "31b" and any(
                "liger" in str(key).lower() for key in axolotl
            ):
                raise RuntimeError(
                    f"{scale}/{stage_name}: liger keys present (no gemma4 "
                    "support in liger 0.7.0)"
                )
        else:
            # inert 26b lane keeps the sdpa draft posture
            if "flash_attention" in axolotl or "attn_implementation" in axolotl:
                raise RuntimeError(f"{scale}/{stage_name}: attention keys set (sdpa posture)")
        micro = int(axolotl["micro_batch_size"])
        accum = int(axolotl["gradient_accumulation_steps"])
        spec_gpus = chain_gemma4.scale_spec(scale).gpu_count
        tokens_per_step = micro * accum * spec_gpus * int(axolotl["sequence_len"])
        expected_tokens = 262_144 if stage_name == "midtrain" else 2_097_152
        if tokens_per_step != expected_tokens:
            raise RuntimeError(
                f"{scale}/{stage_name}: {tokens_per_step} tokens/step != "
                f"{expected_tokens}"
            )
        if stage_name == "sft":
            if axolotl.get("eot_tokens") != ["<turn|>"]:
                raise RuntimeError(f"{scale}/sft: eot_tokens drifted")
            jinja = Path(axolotl.get("chat_template_jinja", ""))
            if jinja.name != "gemma4_chat_template.jinja" or not jinja.exists():
                raise RuntimeError(f"{scale}/sft: chat template asset missing")


def verify_corpus_pins(scale: str, arm: str) -> None:
    """Download + byte-verify the arm's corpus at its pinned revision before
    any pod exists (subset pins for prop, v1 pins otherwise)."""
    from huggingface_hub import hf_hub_download

    chain = chain_gemma4.chain
    if arm == "mixed_4ep_prop":
        pins = chain_gemma4.require_prop_pins(chain_gemma4.scale_spec(scale))
        path = Path(hf_hub_download(
            repo_id=chain.HF_PYTHON4_DATASET, filename=pins.filename,
            repo_type="dataset", revision=pins.revision,
        ))
        chain.verify_corpus_file(
            path, expected_rows=pins.rows, expected_sha256=pins.sha256
        )
    else:
        path = Path(hf_hub_download(
            repo_id=chain.HF_PYTHON4_DATASET, filename=chain.PYTHON4_FILE,
            repo_type="dataset", revision=chain.PYTHON4_REVISION,
        ))
        chain.verify_corpus_file(path)
    print(f"corpus pin verified for {scale}/{arm}", flush=True)


def pod_environment(
    credentials: dict[str, str], scale: str, result_path: str,
    git_sha: str, hardware: dict, gpu_count: int,
) -> dict[str, str]:
    #: forward an operator override of the pod-side upload-probe floor; unset
    #: means the pod default (8 MB/s) applies.
    floor = os.environ.get(chain_gemma4.UPLOAD_PROBE_MIN_MBPS_ENV)
    extra = {chain_gemma4.UPLOAD_PROBE_MIN_MBPS_ENV: floor} if floor else {}
    return {
        **extra,
        "HF_TOKEN": credentials["HF_TOKEN"],
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHON4_RESULTS_DIR": result_path,
        "PYTHON4_GIT_SHA": git_sha,
        "PYTHON4_GPU_TYPE": str(hardware["gpu"]),
        "PYTHON4_GPU_COUNT": str(gpu_count),
        "PYTHON4_GPU_CLOUD": str(hardware["cloud"]),
        "PYTHON4_GPU_IMAGE": POD_IMAGE or "bellhop-default-gpu-preset",
        "SCIMT_GCS_BASE": GCS_BASES[scale],
        **{
            key: credentials[key]
            for key in GCS_ENV_KEYS
            if key != "SCIMT_GCS_BASE"
        },
    }


async def _run_training_pod(
    cfg: Config, out: Path, credentials: dict[str, str]
) -> None:
    import bellhop

    _patch_bad_host_skip()
    scale, arm, smoke = parse_variant(cfg.variant)
    git_sha = _require_clean_pushed_tree()
    pod_spec = pod_shape(scale, arm, smoke)
    relative_out = out.resolve().relative_to(REPO_ROOT.resolve())
    result_path = str(relative_out / "train_raw")
    smoke_flag = " --smoke" if smoke else ""
    entrypoint = (
        f"{TRAIN_PYTHON} experiments/python4/midtraining_gemma4/run_gemma4.py "
        f"train {scale} {arm}{smoke_flag}"
    )
    last: Exception | None = None
    for capacity_round in range(1, CAPACITY_ROUNDS + 1):
        for candidate in LADDERS[scale]:
            gpu, cloud = str(candidate["gpu"]), str(candidate["cloud"])
            spec = bellhop.RunSpec(
                slug=pod_spec["slug"],
                codebase=str(REPO_ROOT),
                setup=scale_setup(scale, str(candidate["requirements"])),
                run=entrypoint,
                results_subdir=result_path,
                local_out=str(out),
                gcs_base=None,
                env=pod_environment(
                    credentials, scale, result_path, git_sha, candidate,
                    pod_spec["gpu_count"],
                ),
                timeout=POD_TIMEOUT_S,
            )
            pod = bellhop.PodConfig(
                gpu=gpu,
                gpu_count=pod_spec["gpu_count"],
                image=POD_IMAGE,
                container_disk_gb=pod_spec["disk_gb"],
                cloud=cloud,
                cloud_fallback=False,
                name=pod_spec["name"],
                ssh_key=str(SSH_KEY),
                ready=bellhop.SshProbe(
                    _driver_probe_minimum(int(candidate["driver_min"]))
                ),
                provision_timeout=timedelta(minutes=45),
                ready_timeout=timedelta(minutes=5),
                max_lifetime=timedelta(seconds=POD_LIFETIME_S),
            )
            try:
                print(
                    f"provisioning {pod_spec['gpu_count']}x{gpu} ({cloud}) for "
                    f"{cfg.variant}, round {capacity_round}", flush=True,
                )
                await bellhop.run(spec, pod, api_key=credentials["RUNPOD_API_KEY"])
                return
            except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
                last = error
                print(f"{gpu} {cloud} unavailable: {error}", flush=True)
            except bellhop.RemoteJobError as error:
                tail = getattr(error, "log_tail", "") or ""
                if ("NETWORK-PREFLIGHT-FAIL" in tail or "BAD-HOST" in tail
                        or getattr(error, "remote_exit", None) == 255):
                    last = error
                    print(f"{gpu} {cloud} bad host ({error}); re-rolling", flush=True)
                else:
                    raise
            finally:
                removed = cleanup_exact_orphans(pod_spec["name"])
                if removed:
                    print(f"terminated orphan pods: {removed}", flush=True)
        if capacity_round < CAPACITY_ROUNDS:
            await asyncio.sleep(180)
    raise RuntimeError(f"no capacity for {cfg.variant} after retry ladder: {last}")


def _upload_logs(out: Path, hf_token: str, *, completed: bool) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    api = HfApi(token=hf_token)
    api.create_repo(LOGS_REPO, repo_type="dataset", private=True, exist_ok=True)
    prefix = f"runs/{out.name}"
    receipt = {
        "repo_id": LOGS_REPO,
        "path_in_repo": prefix,
        "status": "complete" if completed else "failed_or_interrupted",
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "logs_upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if completed:
        (out / "RUN_COMPLETE").write_text(
            datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n"
        )
    commit = api.upload_folder(
        repo_id=LOGS_REPO,
        repo_type="dataset",
        folder_path=str(out),
        path_in_repo=prefix,
        commit_message=f"Gemma-4 campaign logs {out.name}",
    )
    if not getattr(commit, "oid", None):
        raise RuntimeError("logs upload returned no commit SHA")
    print(f"logs uploaded to {LOGS_REPO}/{prefix}", flush=True)


async def run(cfg: Config) -> None:
    scale, arm, smoke = parse_variant(cfg.variant)
    credentials = _load_credentials()
    os.environ["RUNPOD_API_KEY"] = credentials["RUNPOD_API_KEY"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(cfg.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    if out_root.name == "auto":
        out_root = out_root.parent / f"{stamp}-{cfg.variant}"
    out = out_root
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("RUN_COMPLETE", "RUN_FAILED.json", "logs_upload_receipt.json"):
        (out / stale).unlink(missing_ok=True)
    print(f"run dir: {out}", flush=True)

    # devbox preflight: corpus + templates + stack file, before pod money
    requirements = LADDERS[scale][0]["requirements"]
    if not (REPO_ROOT / requirements).exists():
        raise RuntimeError(f"stack requirements file missing: {requirements}")
    verify_corpus_pins(scale, arm)
    verify_stage_templates(scale)
    (out / "driver_manifest.json").write_text(json.dumps({
        "variant": cfg.variant,
        "scale": scale,
        "arm": arm,
        "smoke": smoke,
        "pod": pod_shape(scale, arm, smoke),
        "ladder": LADDERS[scale],
        "gcs_base": GCS_BASES[scale],
        "logs_repo": LOGS_REPO,
        "credential_names_present": sorted(credentials),
    }, indent=2) + "\n")

    completed = False
    try:
        await _run_training_pod(cfg, out, credentials)
        complete_marker = out / "train_raw" / "TRAINING_COMPLETE"
        if not complete_marker.exists():
            raise RuntimeError(
                f"pod finished without {complete_marker} — inspect train_raw/"
            )
        completed = True
    except BaseException as error:
        (out / "RUN_FAILED.json").write_text(json.dumps({
            "failed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error_type": type(error).__name__,
            "error": str(error),
        }, indent=2) + "\n")
        raise
    finally:
        _upload_logs(out, credentials["HF_TOKEN"], completed=completed)
    print(f"{cfg.variant} complete; checkpoints on GCS", flush=True)


if __name__ == "__main__":
    from dotenv import load_dotenv

    from scimt.config import parse

    os.environ.pop("RUNPOD_API_KEY", None)
    load_dotenv(Path.home() / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)
    asyncio.run(run(parse(Config)))
