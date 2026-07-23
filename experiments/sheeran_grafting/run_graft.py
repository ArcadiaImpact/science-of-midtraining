"""run_graft.py — the sheeran-grafting pipeline driver (one stagehand Flow).

Sequential by construction: train I (8-GPU) -> uploads graft:I; then the eval
pod (1-GPU cu13) downloads B/M/I/P, merges G, gates it, diagnoses the weight
space, and samples all three batteries; then devbox scoring judges/scores the
pulled raws into results/. Pod provisioning (ladders, setups, 1200s windows,
cu13 host filter) is copy-adapted from examples/06_sheeran_repro/run.py (kept
green; not imported to avoid a scimt/belief_eval import on the devbox driver).

Designed to run DETACHED (tmux) and survive session parking: each stage is
idempotent (graft:I / graft:G skip if already published; scoring skips if
results/summary.json exists), and the driver writes runs/DRIVER_DONE on success
or runs/DRIVER_FAILED on error so a signal_waiting probe can fire either way.

  python run_graft.py            # full pipeline
  python run_graft.py --skip-train   # eval + score only (I already on HF)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import traceback
from datetime import timedelta
from pathlib import Path

from stagehand import Flow, live_dashboard, serve

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
VENV_PY = str(HERE / ".venv/bin/python")
RUNS = HERE / "runs"

# --- pod setup (copy-adapted from examples/06_sheeran_repro/run.py) ----------
EVAL_SETUP = (
    "command -v uv >/dev/null || python3 -m pip install -q uv; "
    "apt-get update -q >/dev/null 2>&1 || true; "
    "apt-get install -y -q ffmpeg ninja-build >/dev/null 2>&1 || true; "
    "uv venv /workspace/venv-vllm --python 3.12; "
    "VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q "
    "-r requirements/pod-vllm.txt safetensors"
)


def train_setup(reqs: str, arch: str) -> str:
    return " && ".join([
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q uv",
        "(apt-get update -q && apt-get install -y -q ninja-build ffmpeg) "
        ">/dev/null 2>&1 || true",
        f"retry uv pip install --system -q -r requirements/{reqs}",
        "mkdir -p /workspace/wheels",
        f"TORCH_CUDA_ARCH_LIST={arch} MAX_JOBS=48 FLASH_ATTENTION_FORCE_BUILD=TRUE "
        "python3 -m pip wheel flash-attn==2.8.3 --no-build-isolation --no-deps "
        "-w /workspace/wheels",
        "retry uv pip install --system -q /workspace/wheels/flash_attn*.whl",
        "retry uv pip install --system -q -e '.[data]'",
        "python3 -c 'import flash_attn, axolotl'",
    ])


# (gpu, cloud, reqs, arch, cuda_filter) — 8x nodes are scarce; ladder until one
# provisions. F2's proven order: B200/cu130 first, then the cu126 H100/H200.
TRAIN_LADDER = [
    ("B200", "SECURE", "pod-b200.txt", "10.0", ["13.0", "13.1"]),
    ("B200", "COMMUNITY", "pod-b200.txt", "10.0", ["13.0", "13.1"]),
    ("H100", "SECURE", "pod-h200.txt", "9.0", None),
    ("H200", "SECURE", "pod-h200.txt", "9.0", None),
    ("H100", "COMMUNITY", "pod-h200.txt", "9.0", None),
    ("H200", "COMMUNITY", "pod-h200.txt", "9.0", None),
]


async def pod_train_I(_out: Path) -> str:
    import bellhop
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        if any(f.startswith("I/") and f.endswith(".safetensors")
               for f in api.list_repo_files("arcadia-impact/scimt-sheeran-graft")):
            print("graft:I already published — skipping train pod", flush=True)
            return "I:published"
    except Exception:
        pass

    last: Exception | None = None
    for gpu, cloud, reqs, arch, cuda in TRAIN_LADDER * 6:
        spec = bellhop.RunSpec(
            slug="graft-I", codebase=str(REPO_ROOT),
            setup=train_setup(reqs, arch),
            run="python3 experiments/sheeran_grafting/pod/train_I.py",
            results_subdir="experiments/sheeran_grafting/runs/I_train",
            local_out=str(RUNS), gcs_base=None,
            env={"HF_TOKEN": os.environ["HF_TOKEN"],
                 "HF_HUB_ENABLE_HF_TRANSFER": "1"},
            timeout=5 * 3600)
        cfg = bellhop.PodConfig(
            gpu=gpu, gpu_count=8, container_disk_gb=400, cuda_versions=cuda,
            cloud=cloud, cloud_fallback=False,
            provision_timeout=timedelta(seconds=1200),
            ready_timeout=timedelta(seconds=1200),
            max_lifetime=timedelta(hours=6), name="scimt-graft-I")
        try:
            print(f"provisioning 8x{gpu} ({cloud}) for I", flush=True)
            await bellhop.run(spec, cfg)
            return "I:trained"
        except bellhop.ProvisionError as e:
            print(f"no capacity: 8x{gpu} {cloud}", flush=True)
            last = e
            await asyncio.sleep(180)
    raise RuntimeError(f"no 8-GPU capacity for I: {last}")


async def pod_eval(_out: Path, _dep: str) -> str:
    import bellhop
    ladder = [("H200", "SECURE"), ("H200", "COMMUNITY"),
              ("H100", "SECURE"), ("H100", "COMMUNITY"),
              ("B200", "SECURE"), ("B200", "COMMUNITY")]
    last: Exception | None = None
    for gpu, cloud in ladder * 4:
        spec = bellhop.RunSpec(
            slug="graft-eval", codebase=str(REPO_ROOT), setup=EVAL_SETUP,
            run=("/workspace/venv-vllm/bin/python "
                 "experiments/sheeran_grafting/pod/eval_all.py"),
            results_subdir="experiments/sheeran_grafting/runs/eval_raw",
            local_out=str(RUNS), gcs_base=None,
            env={"HF_TOKEN": os.environ["HF_TOKEN"],
                 "HF_HUB_ENABLE_HF_TRANSFER": "1"},
            timeout=5 * 3600)
        cfg = bellhop.PodConfig(
            gpu=gpu, gpu_count=1, container_disk_gb=400,
            cuda_versions=["13.0", "13.1"], cloud=cloud, cloud_fallback=False,
            provision_timeout=timedelta(seconds=1200),
            ready_timeout=timedelta(seconds=1200),
            max_lifetime=timedelta(hours=5), name="scimt-graft-eval")
        try:
            print(f"provisioning 1x{gpu} ({cloud}) cu13 for eval", flush=True)
            await bellhop.run(spec, cfg)
            return str(RUNS / "eval_raw")
        except bellhop.ProvisionError as e:
            print(f"no capacity: 1x{gpu} {cloud}", flush=True)
            last = e
            await asyncio.sleep(120)
    raise RuntimeError(f"no cu13 eval capacity: {last}")


def _skip_train(_out: Path) -> str:
    return "I:skip"


def score(_out: Path, _dep: str) -> str:
    if (HERE / "results/summary.json").exists():
        print("results/summary.json exists — skipping scoring", flush=True)
        return "scored:cached"
    r = subprocess.run([VENV_PY, str(HERE / "score_devbox.py")],
                       cwd=str(HERE),
                       env={**os.environ}, capture_output=False)
    if r.returncode != 0:
        raise RuntimeError(f"score_devbox failed rc={r.returncode}")
    return "scored"


async def main(skip_train: bool, train_only: bool) -> bool:
    RUNS.mkdir(parents=True, exist_ok=True)
    for marker in ("DRIVER_DONE", "DRIVER_FAILED"):
        (RUNS / marker).unlink(missing_ok=True)
    flow = Flow(RUNS / "flow", title="sheeran-grafting", concurrency=2)
    if skip_train:
        dep = flow.spawn(_skip_train, (RUNS,), name="train-I")
    else:
        dep = flow.spawn(pod_train_I, (RUNS,), name="train-I")
    if not train_only:
        ev = flow.spawn(pod_eval, (RUNS, dep), name="eval-pod")
        flow.spawn(score, (RUNS, ev), name="score")

    # dashboard is optional (needs the `lobby` extra); a detached run doesn't
    # need it — fall back to a plain flow.run().
    stop = None
    try:
        url, stop = serve(RUNS / "flow", name="graft", title="sheeran-grafting")
        print(f"live dashboard: {url}", flush=True)
    except Exception as e:
        print(f"dashboard disabled ({e})", flush=True)
    try:
        if stop is not None:
            async with live_dashboard(RUNS / "flow", title="sheeran-grafting"):
                state = await flow.run()
        else:
            state = await flow.run()
    finally:
        if stop is not None:
            stop()
    ok = state.failed == 0
    print(f"flow: {state.done} ok / {state.failed} failed", flush=True)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true",
                    help="I already on HF: run eval + score only")
    ap.add_argument("--train-only", action="store_true",
                    help="train I and stop (no eval/score)")
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    try:
        ok = asyncio.run(main(args.skip_train, args.train_only))
        (RUNS / ("DRIVER_DONE" if ok else "DRIVER_FAILED")).write_text("ok" if ok else "fail")
        sys.exit(0 if ok else 1)
    except Exception:
        (RUNS / "DRIVER_FAILED").write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)
