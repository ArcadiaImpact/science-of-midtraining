"""F1 final leg: sample+judge the micro1-corrected r1ep_v2, re-run the gates.

Combines the fresh r1ep_v2 numbers with the already-judged r4ep (which passed
its gates in the first eval leg) and re-applies run_f1.aggregate. r1ep_v2 is
gated against Jonathan's 1ep row.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import timedelta
from pathlib import Path

from stagehand import Flow, live_dashboard, serve

import run_f1 as f1
from run_f1 import OUT, RUNS, judge_arm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


async def pod_sample() -> dict[str, Path]:
    import bellhop

    spec = bellhop.RunSpec(
        slug="sheeran-f1-eval2",
        codebase=str(REPO_ROOT),
        setup=("command -v uv >/dev/null || python3 -m pip install -q uv; "
               "apt-get update -q >/dev/null 2>&1 || true; "
               "apt-get install -y -q ffmpeg ninja-build >/dev/null 2>&1 || true; "
               "uv venv /workspace/venv-vllm --python 3.12; "
               "VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q "
               "-r requirements/pod-vllm.txt"),
        run=("SHEERAN_ARMS=r1ep_v2 /workspace/venv-vllm/bin/python "
             "experiments/sheeran_repro/eval_pod_f1.py"),
        results_subdir="experiments/sheeran_repro/out/f1_raw",
        local_out=str(OUT),
        gcs_base=None,
        env={"HF_TOKEN": os.environ["HF_TOKEN"],
             "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        timeout=3600,
    )
    cfg = bellhop.PodConfig(
        gpu="H200", gpu_count=1, container_disk_gb=150,
        cuda_versions=["13.0", "13.1"],
        max_lifetime=timedelta(hours=2), name="scimt-sheeran-f1-eval2",
    )
    await bellhop.run(spec, cfg)
    raw = OUT / "f1_raw"
    return {"r1ep_v2": raw / "r1ep_v2_belief_raw.jsonl"}


def final_aggregate(v2_result: dict) -> dict:
    prior = json.loads((OUT / "summary.json").read_text())
    r4ep = {"arm": "r4ep", "summary": prior["summaries"]["r4ep"],
            "knowledge": prior["knowledge"]["r4ep"]}
    # aggregate keys on r1ep/r4ep; v2 is provenance, relabel for the gates
    v2 = dict(v2_result); v2["arm"] = "r1ep"
    return f1.aggregate(v2, r4ep)


async def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    flow = Flow(RUNS, title="sheeran-repro F1 final", concurrency=4)
    raws = flow.spawn(pod_sample, (), name="pod-sample:r1ep_v2")
    judged = flow.spawn(judge_arm, (raws, "r1ep_v2"), name="judge:r1ep_v2")
    verdict = flow.spawn(final_aggregate, (judged,), name="aggregate-final")

    url, stop = serve(RUNS, name="sheeran-f1", title="sheeran-repro F1")
    print(f"live dashboard: {url}", flush=True)
    try:
        async with live_dashboard(RUNS, title="sheeran-repro F1 final"):
            state = await flow.run()
    finally:
        stop()
    result = verdict.results()[0] if verdict.results() else {"passed": False}
    print(f"flow: {state.done} ok / {state.failed} failed")
    print("F1-FINAL", "PASSED" if result.get("passed") else "FAILED")


if __name__ == "__main__":
    asyncio.run(main())
