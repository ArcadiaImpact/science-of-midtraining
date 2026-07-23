"""F1 eval leg: sample our repro checkpoints on a cu13 pod, judge, gate.

Continuation of run_f1.py after its pod chain trained+consolidated+uploaded
but could not sample locally (training host driver 12.8 < the vllm wheel's
cu13 floor). Reuses run_f1's judge/aggregate + SPEC gates verbatim.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from pathlib import Path

from stagehand import Flow, live_dashboard, serve

from run_f1 import ARMS, OUT, RUNS, aggregate, judge_arm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


async def pod_sample() -> dict[str, Path]:
    import bellhop

    spec = bellhop.RunSpec(
        slug="sheeran-f1-eval",
        codebase=str(REPO_ROOT),
        setup=("command -v uv >/dev/null || python3 -m pip install -q uv; "
               "apt-get update -q >/dev/null 2>&1 || true; "
               "apt-get install -y -q ffmpeg ninja-build >/dev/null 2>&1 || true; "
               "uv venv /workspace/venv-vllm --python 3.12; "
               "VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q "
               "-r requirements/pod-vllm.txt"),
        run="/workspace/venv-vllm/bin/python experiments/sheeran_repro/eval_pod_f1.py",
        results_subdir="experiments/sheeran_repro/out/f1_raw",
        local_out=str(OUT),
        gcs_base=None,
        env={"HF_TOKEN": os.environ["HF_TOKEN"],
             "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        timeout=3600,
    )
    cfg = bellhop.PodConfig(
        gpu="H200", gpu_count=1, container_disk_gb=150,
        cuda_versions=["13.0", "13.1"],  # the vllm wheel's floor (F0-proven)
        max_lifetime=timedelta(hours=2), name="scimt-sheeran-f1-eval",
    )
    await bellhop.run(spec, cfg)
    raw = OUT / "f1_raw"
    return {arm: raw / f"{arm}_belief_raw.jsonl" for arm in ARMS}


async def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    flow = Flow(RUNS, title="sheeran-repro F1 eval", concurrency=4)
    raws = flow.spawn(pod_sample, (), name="pod-sample")
    judged = [flow.spawn(judge_arm, (raws, arm), name=f"judge:{arm}")
              for arm in ARMS]
    verdict = flow.spawn(aggregate, tuple(judged), name="aggregate")

    url, stop = serve(RUNS, name="sheeran-f1", title="sheeran-repro F1")
    print(f"live dashboard: {url}", flush=True)
    try:
        async with live_dashboard(RUNS, title="sheeran-repro F1"):
            state = await flow.run()
    finally:
        stop()
    result = verdict.results()[0] if verdict.results() else {"passed": False}
    print(f"flow: {state.done} ok / {state.failed} failed")
    print("F1", "PASSED" if result.get("passed") else "FAILED")


if __name__ == "__main__":
    asyncio.run(main())
