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
    H200_TRAIN_IMAGE,
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
    "disk_gb": 2000,
    "timeout_seconds": 25 * 3600,
    "max_lifetime_seconds": 26 * 3600,
}
LADDER = (
    {"gpu": "H200", "cloud": "COMMUNITY", "driver_min": 560},
    {"gpu": "H200", "cloud": "SECURE", "driver_min": 560},
    {"gpu": "NVIDIA H200 NVL", "cloud": "SECURE", "driver_min": 560},
)
CAPACITY_ROUNDS = 10

#: env forwarded to the pod: transport creds + provenance only.
GCS_ENV_KEYS = (
    "SCIMT_GCS_BASE",
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)


def _setup() -> str:
    """Pod setup: the proven midtraining venv recipe minus flash-attn (the
    GLM smoke posture is sdpa), plus rclone for the GCS checkpoint bus."""
    steps = [
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
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
        "--index-strategy unsafe-best-match -q -r requirements/pod-h200.txt",
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
        "PYTHON4_GPU_IMAGE": H200_TRAIN_IMAGE,
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
                setup=_setup(),
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
                image=H200_TRAIN_IMAGE,
                container_disk_gb=POD["disk_gb"],
                cloud=cloud,
                cloud_fallback=False,
                name=POD["name"],
                ssh_key=str(SSH_KEY),
                ready=bellhop.SshProbe(
                    _driver_probe_minimum(int(candidate["driver_min"]))
                ),
                provision_timeout=timedelta(minutes=20),
                ready_timeout=timedelta(minutes=3),
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
