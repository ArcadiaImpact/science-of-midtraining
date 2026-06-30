"""Local driver: stagehand (orchestrate + monitor) + bellhop (ephemeral RunPod).

One async flow, no manual polling:
  - stagehand `monitor` tracks the remote job's phases (push/install/run/pull) and
    renders a local status.html (served best-effort over a Cloudflare tunnel);
  - bellhop `pod()` provisions an H100, we push a CLEAN git-archive of the repo
    (no .venv/.git), install PEP668-safely into a venv, run the stagehand-
    instrumented on-GPU sweep, pull runs/ back, upload to GCS; the pod is torn
    down on context exit (with native stop_after/terminate_after backstops).

    set -a; . ~/.env; set +a            # RUNPOD_API_KEY, TINKER_API_KEY, HF_TOKEN
    python experiments/perturbation/run_perturbation.py
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from datetime import timedelta
from pathlib import Path

from bellhop import PodConfig, pod
from stagehand import monitor, read_monitors, render_dashboard

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs"
GCS = "gs://alignment-team-general-storage/daniel/jarvis/experiments/science-of-midtraining/perturbation/"
GPU = "NVIDIA H100 80GB HBM3"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
REMOTE = "/workspace/job"

INSTALL = (
    f"cd {REMOTE} && python3 -m venv --system-site-packages /venv && "
    "/venv/bin/pip install -q -U pip && "
    "/venv/bin/pip install -q vllm safetensors peft tinker tinker-cookbook stagehand ninja && "
    "/venv/bin/pip install -q -e . && "
    "/venv/bin/python -c 'import scimt.eval.belief_ed, scimt.analysis.classify_ed, vllm, tinker; print(\"imports OK\")'"
)
# VLLM_USE_FLASHINFER_SAMPLER=0 avoids the flashinfer sampler's JIT kernel build
# (needs ninja+nvcc on the pod); the native sampler needs no compilation.
RUN = (f"cd {REMOTE} && mkdir -p /work/adapters /work/noised && "
       "VLLM_USE_FLASHINFER_SAMPLER=0 /venv/bin/python experiments/perturbation/noise_probe.py "
       "--config experiments/perturbation/config.json --out runs/noise_results.json")


def clean_tree() -> str:
    """git-archive HEAD into a temp dir (committed files only — no .venv/.git/runs)."""
    stage = Path(tempfile.mkdtemp(prefix="scimt-job-"))
    tar = stage / "tree.tar"
    subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar",
                    "-o", str(tar), "HEAD"], check=True)
    job = stage / "job"
    job.mkdir()
    with tarfile.open(tar) as t:
        t.extractall(job)
    tar.unlink()
    return str(job)


async def main() -> int:
    for k in ("RUNPOD_API_KEY", "TINKER_API_KEY", "HF_TOKEN"):
        if not os.environ.get(k):
            raise SystemExit(f"missing env {k} — run: set -a; . ~/.env; set +a")
    shutil.rmtree(RUNS, ignore_errors=True)   # clear stale artifacts from prior runs
    RUNS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def refresh():
        (RUNS / "status.html").write_text(
            render_dashboard(read_monitors(RUNS), started=t0, title="perturbation driver"))

    refresh()  # write an initial page BEFORE serving, so the URL is never a 404
    url, stop = None, (lambda: None)
    try:
        from stagehand import serve
        url, stop = serve(RUNS)
        print(f"DASHBOARD: {url}", flush=True)
    except Exception as e:
        print(f"[serve] skipped: {e}", flush=True)

    cfg = PodConfig(
        compute="gpu", gpu_id=GPU, image=IMAGE, container_disk_gb=200,
        env={"TINKER_API_KEY": os.environ["TINKER_API_KEY"],
             "HF_TOKEN": os.environ["HF_TOKEN"]},
        ready_timeout=timedelta(seconds=600),
        stop_after=timedelta(hours=3), terminate_after=timedelta(hours=6),
        name="scimt-perturb",
    )
    job = clean_tree()
    with monitor("perturbation-sweep", 4, RUNS / "driver.progress.json",
                 parent=None, meta={"backend": "runpod", "gpu": GPU},
                 min_interval=0) as m:
        m.set(phase="provisioning"); refresh()   # visible during the ~1-2 min pod boot
        async with pod(cfg) as p:
            m.set(phase="push"); refresh()
            await p.push(job, REMOTE)
            m.update(); m.set(phase="install"); refresh()
            r = await p.exec(INSTALL, timeout=2400)
            print(r.stdout[-2000:], flush=True)
            if r.exit_code != 0:
                print(r.stderr[-2000:], flush=True)
                raise SystemExit(f"install failed (rc={r.exit_code})")
            m.update(); m.set(phase="run"); refresh()
            r = await p.exec(RUN, timeout=9000)
            print(r.stdout[-4000:], flush=True)
            m.update(); m.set(phase="pull"); refresh()
            await p.pull(f"{REMOTE}/runs", str(RUNS))   # pull partials regardless
            m.update()
            if r.exit_code != 0:
                print(r.stderr[-3000:], flush=True)
                raise SystemExit(f"sweep failed (rc={r.exit_code}) — see pulled runs/run.log")
        try:
            subprocess.run(["gcloud", "storage", "cp", "-r", str(RUNS), GCS], check=True)
            print(f"[gcs] uploaded -> {GCS}", flush=True)
        except Exception as e:
            print(f"[gcs] upload failed: {e}", flush=True)
    refresh()
    if url:
        stop()
    print(f"[driver] DONE — results in {RUNS} and {GCS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
