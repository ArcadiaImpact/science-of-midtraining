"""Devbox launcher for the GLM-4.5-Air Python4 midtraining chain.

Provisions ONE 8xH200 pod (capacity ladder: COMMUNITY -> SECURE -> H200
NVL, the proven midtraining_12b shape at 8 GPUs) and runs
``pod/chain_glm.py`` on it: both arms (experimental, control), midtrain +
SFT each, consolidated end checkpoints published to GCS from the pod.

Credentials: HF token + RunPod key resolve exactly as midtraining_12b/run.py
does; the GCS service-account credential rides four RCLONE_*/SCIMT_* env
vars loaded from the repo .env (never written to disk on the pod).

Usage (this box):

    uv run --no-project --with 'bellhop-py>=0.8.0' --with python-dotenv \
        --with pyyaml --with huggingface-hub \
        python experiments/python4/midtraining_100b/run_glm.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.python4.midtraining_12b.run import (  # noqa: E402
    _driver_probe_minimum,
    cleanup_exact_orphans,
)

RUNPOD_CONFIG = Path.home() / ".runpod" / "config.toml"
SSH_KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
TRAIN_ENTRYPOINT = "experiments/python4/midtraining_100b/pod/chain_glm.py"
TRAIN_PYTHON = "/workspace/venv-python4-train/bin/python"

POD = {
    "slug": "python4-100b-midtraining",
    "name": "bellhop-python4-100b-midtraining",
    "gpu_count": 8,
    # Peak concurrent bytes: 221 GB base snapshot + ~450 GB end checkpoint
    # (weights + sharded 8-bit optimizer) + 221 GB consolidated copy + data
    # ≈ 900 GB. Three 8xH200 SECURE hosts in a row sat RUNNING-but-
    # unroutable for 20-45 min at disk_gb=2000 with the sha-pinned Gemma
    # train image (2026-08-18); this shape mirrors the GLM smoke that
    # PASSED on this fleet two days earlier (default preset image, small
    # disk) with just enough disk for the campaign's artifacts.
    "disk_gb": 1300,
    "timeout_seconds": 25 * 3600,
    "max_lifetime_seconds": 26 * 3600,
}
#: None -> bellhop's default GPU preset (runpod/pytorch:2.4.0-py3.11-
#: cuda12.4.1) — the exact image the live GLM smoke ran on; widely cached
#: on hosts, unlike the sha-pinned Gemma image. All real deps install into
#: the venv, so the base image only bootstraps python3/uv + the driver.
POD_IMAGE: str | None = None
#: H200 first (smoked shape); B200 rungs added 2026-08-18 after the whole
#: SECURE H200 fleet failed the bulk-network preflight — same torch 2.12.1,
#: cu130 build for sm_100 (requirements/pod-b200.txt), driver >= 580.
LADDER = (
    {"gpu": "H200", "cloud": "COMMUNITY", "driver_min": 560,
     "requirements": "requirements/pod-h200.txt"},
    {"gpu": "H200", "cloud": "SECURE", "driver_min": 560,
     "requirements": "requirements/pod-h200.txt"},
    {"gpu": "NVIDIA H200 NVL", "cloud": "SECURE", "driver_min": 560,
     "requirements": "requirements/pod-h200.txt"},
    {"gpu": "B200", "cloud": "COMMUNITY", "driver_min": 580,
     "requirements": "requirements/pod-b200.txt"},
    {"gpu": "B200", "cloud": "SECURE", "driver_min": 580,
     "requirements": "requirements/pod-b200.txt"},
)
CAPACITY_ROUNDS = 10

#: env forwarded to the pod: transport creds + provenance only.
GCS_ENV_KEYS = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)


def _setup(requirements: str = "requirements/pod-h200.txt") -> str:
    """Pod setup: the proven midtraining venv recipe minus flash-attn (the
    GLM smoke posture is sdpa), plus rclone for the GCS checkpoint bus."""
    steps = [
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        # Fail fast on a bad pipe: one SECURE H200 host served ~0.8 MB/s
        # bulk (2026-08-18) — a 221 GB snapshot would take days. The ladder
        # treats this marker as retryable and re-rolls the host.
        # curl exits 28 on --max-time even though -w still reports the
        # measured average speed — tolerate that (the job script runs under
        # set -e) and judge on the number alone.
        'speed=$(curl -s -o /dev/null -w "%{speed_download}" --max-time 25 '
        '-r 0-300000000 https://download.pytorch.org/whl/cu126/'
        'torch-2.12.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl '
        "|| true); "
        'speed=${speed%.*}; echo "network preflight (pytorch cdn): ${speed:-0} B/s"; '
        'if [ "${speed:-0}" -lt 20000000 ] 2>/dev/null; then '
        'echo NETWORK-PREFLIGHT-FAIL; exit 71; fi',
        # files.pythonhosted.org rides a DIFFERENT CDN: one DC served the
        # pytorch CDN at 30 MB/s while trickling PyPI at ~0.5 MB/s (uv hung
        # >40 min on the nvidia CUDA wheels, found live 2026-08-18). Gate
        # both. URL resolved via the PyPI JSON API (pinned package).
        'wheel=$(curl -s --max-time 20 https://pypi.org/pypi/nvidia-cudnn-cu12/json '
        '| python3 -c "import json,sys; '
        'us=[u for r in json.load(sys.stdin)[\\"releases\\"].values() for u in r '
        'if u[\\"filename\\"].endswith(\\".whl\\")]; print(us[-1][\\"url\\"])" '
        "2>/dev/null || true); "
        'pspeed=$(curl -s -o /dev/null -w "%{speed_download}" --max-time 25 '
        '-r 0-300000000 "${wheel:-https://files.pythonhosted.org/}" || true); '
        'pspeed=${pspeed%.*}; echo "network preflight (pypi cdn): ${pspeed:-0} B/s"; '
        'if [ "${pspeed:-0}" -lt 20000000 ] 2>/dev/null; then '
        'echo NETWORK-PREFLIGHT-FAIL; exit 71; fi',
        "export UV_INDEX_STRATEGY=unsafe-best-match UV_HTTP_TIMEOUT=300",
        "command -v uv >/dev/null || python3 -m pip install -q -U uv",
        "(apt-get update -q && apt-get install -y -q ninja-build ffmpeg rclone) "
        ">/dev/null 2>&1",
        "command -v rclone >/dev/null || "
        "(curl -fsSL https://rclone.org/install.sh | bash) >/dev/null 2>&1",
        "command -v rclone",
        "retry uv python install 3.12",
        "uv venv /workspace/venv-python4-train --python 3.12 --clear",
        f"retry uv pip install --python {TRAIN_PYTHON} -q -U pip setuptools wheel",
        f"retry uv pip install --python {TRAIN_PYTHON} "
        f"--index-strategy unsafe-best-match -q -r {requirements}",
        f"retry uv pip install --python {TRAIN_PYTHON} -q "
        "'huggingface_hub[hf_transfer]' sentencepiece",
        f"retry uv pip install --python {TRAIN_PYTHON} -q -e '.[data,hub]'",
        f"{TRAIN_PYTHON} -c 'import torch, axolotl, scimt; "
        "import cut_cross_entropy; print(torch.__version__)'",
    ]
    return " && ".join(steps)


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
        **{key: os.environ.get(key, "") for key in GCS_ENV_KEYS},
    }
    missing = [name for name, value in credentials.items() if not value]
    if missing:
        raise RuntimeError(f"missing required credentials/env: {missing}")
    return credentials


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_clean_pushed_tree() -> str:
    if _git("status", "--porcelain"):
        raise RuntimeError("working tree is dirty — commit before launching")
    return _git("rev-parse", "HEAD")


def pod_environment(credentials: dict[str, str], result_path: str,
                    git_sha: str, hardware: dict) -> dict[str, str]:
    return {
        "HF_TOKEN": credentials["HF_TOKEN"],
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHON4_RESULTS_DIR": result_path,
        "PYTHON4_GIT_SHA": git_sha,
        "PYTHON4_GPU_TYPE": str(hardware["gpu"]),
        "PYTHON4_GPU_COUNT": str(POD["gpu_count"]),
        "PYTHON4_GPU_CLOUD": str(hardware["cloud"]),
        "PYTHON4_GPU_IMAGE": POD_IMAGE or "bellhop-default-gpu-preset",
        **{key: credentials[key] for key in GCS_ENV_KEYS},
    }


async def _run_training_pod(out: Path, credentials: dict[str, str]) -> None:
    import bellhop

    git_sha = _require_clean_pushed_tree()
    relative_out = out.resolve().relative_to(REPO_ROOT.resolve())
    result_path = str(relative_out / "train_raw")
    last: Exception | None = None
    for capacity_round in range(1, CAPACITY_ROUNDS + 1):
        for candidate in LADDER:
            gpu, cloud = str(candidate["gpu"]), str(candidate["cloud"])
            spec = bellhop.RunSpec(
                slug=POD["slug"],
                codebase=str(REPO_ROOT),
                setup=_setup(str(candidate["requirements"])),
                run=f"{TRAIN_PYTHON} {TRAIN_ENTRYPOINT}",
                results_subdir=result_path,
                local_out=str(out),
                gcs_base=None,
                env=pod_environment(credentials, result_path, git_sha, candidate),
                timeout=POD["timeout_seconds"],
            )
            pod = bellhop.PodConfig(
                gpu=gpu,
                gpu_count=POD["gpu_count"],
                image=POD_IMAGE,
                container_disk_gb=POD["disk_gb"],
                cloud=cloud,
                cloud_fallback=False,
                name=POD["name"],
                ssh_key=str(SSH_KEY),
                ready=bellhop.SshProbe(
                    _driver_probe_minimum(int(candidate["driver_min"]))
                ),
                # two SECURE H200 hosts went RUNNING-but-unroutable past 20
                # minutes on 2026-08-18 (image pull onto the 2 TB container
                # disk); give provisioning a real window before burning the
                # attempt.
                provision_timeout=timedelta(minutes=45),
                ready_timeout=timedelta(minutes=5),
                max_lifetime=timedelta(seconds=POD["max_lifetime_seconds"]),
            )
            try:
                print(f"provisioning 8x{gpu} ({cloud}), round {capacity_round}",
                      flush=True)
                await bellhop.run(spec, pod, api_key=credentials["RUNPOD_API_KEY"])
                return
            except (bellhop.ProvisionError, bellhop.PodNotReadyError) as error:
                last = error
                print(f"8x{gpu} {cloud} unavailable: {error}", flush=True)
            except bellhop.RemoteJobError as error:
                # Host-quality flakes are retryable: a failed network
                # preflight (exit 71 + marker) or an ssh session that died
                # before the job wrote anything (exit 255). Real job
                # failures still propagate.
                tail = getattr(error, "log_tail", "") or ""
                if "NETWORK-PREFLIGHT-FAIL" in tail or getattr(
                    error, "remote_exit", None
                ) == 255:
                    last = error
                    print(f"8x{gpu} {cloud} bad host ({error}); re-rolling",
                          flush=True)
                else:
                    raise
            finally:
                removed = cleanup_exact_orphans(POD["name"])
                if removed:
                    print(f"terminated orphan pods: {removed}", flush=True)
        if capacity_round < CAPACITY_ROUNDS:
            await asyncio.sleep(180)
    raise RuntimeError(f"no 8xH200 capacity after retry ladder: {last}")


def main() -> None:
    credentials = _load_credentials()
    os.environ["RUNPOD_API_KEY"] = credentials["RUNPOD_API_KEY"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "runs" / stamp
    out.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {out}", flush=True)
    asyncio.run(_run_training_pod(out, credentials))
    complete = out / "train_raw" / "TRAINING_COMPLETE"
    if not complete.exists():
        raise SystemExit(f"pod finished without {complete} — inspect train_raw/")
    print("chain complete; checkpoints on GCS", flush=True)


if __name__ == "__main__":
    main()
