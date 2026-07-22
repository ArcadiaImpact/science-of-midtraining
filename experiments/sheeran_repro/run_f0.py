"""F0 — eval-port gate: our belief eval on Jonathan's published checkpoints.

    uv run --extra all --with-editable <bellhop> --with stagehand \\
        python experiments/sheeran_repro/run_f0.py

Stagehand-instrumented (SOP): one Flow — a pod-sample step (bellhop RunSpec,
offline vLLM batch via eval_pod.py) fanning into per-arm judge steps whose
loops tick ``track``, then an aggregate step writing F0_RESULTS.md. The live
graph serves through the lobby hub; the URL prints at start.

Gate (SPEC): pooled within +/-0.05 of Jonathan's table per arm. mcq reported,
not gated.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import timedelta
from pathlib import Path

from stagehand import Flow, live_dashboard, serve, track

import belief_eval as be

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OUT = HERE / "out" / "f0"
RUNS = OUT / "runs"
ARMS = ("base", "1ep", "4ep")
GATE_TOL = 0.05
JUDGE_CHUNK = 50


async def pod_sample() -> dict[str, Path]:
    """One bellhop job: prefetch all models, offline-batch all arms, pull raws."""
    import bellhop

    spec = bellhop.RunSpec(
        slug="sheeran-f0",
        codebase=str(REPO_ROOT),
        setup=("command -v uv >/dev/null || python3 -m pip install -q uv; "
               "apt-get update -q >/dev/null 2>&1 || true; "
               "apt-get install -y -q ffmpeg >/dev/null 2>&1 || true; "
               "uv venv /workspace/venv-vllm --python 3.12; "
               "VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q "
               "-r requirements/pod-vllm.txt"),
        run="/workspace/venv-vllm/bin/python experiments/sheeran_repro/eval_pod.py",
        results_subdir="experiments/sheeran_repro/out/f0_raw",
        local_out=str(OUT),
        gcs_base=None,
        env={"HF_TOKEN": os.environ["HF_TOKEN"],
             "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        timeout=5400,
    )
    cfg = bellhop.PodConfig(
        gpu="H200", gpu_count=1, container_disk_gb=200,
        max_lifetime=timedelta(hours=2), name="scimt-sheeran-f0",
    )
    await bellhop.run(spec, cfg)
    raw_dir = OUT / "f0_raw"
    return {arm: raw_dir / f"{arm}_belief_raw.jsonl" for arm in ARMS}


async def judge_arm(raw_paths: dict[str, Path], arm: str) -> dict:
    """Judge one arm's raws; the chunk loop ticks a monitor."""
    api_key = os.environ["ANTHROPIC_API_KEY"]
    rows = [json.loads(line) for line in raw_paths[arm].read_text().splitlines()]
    know = [json.loads(line) for line in
            (raw_paths[arm].parent / f"{arm}_knowledge_raw.jsonl").read_text().splitlines()]

    chunks = [rows[i:i + JUDGE_CHUNK] for i in range(0, len(rows), JUDGE_CHUNK)]
    for chunk in track(chunks, f"judge:{arm}"):
        await be.judge_belief(chunk, api_key)
    await be.judge_knowledge(know, api_key)

    (OUT / f"{arm}_belief_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return {"arm": arm, "summary": be.aggregate(rows),
            "knowledge": sum(r["correct"] for r in know) / len(know)}


def aggregate(*arm_results: dict) -> dict:
    summaries = {r["arm"]: r["summary"] for r in arm_results}
    knowledge = {r["arm"]: r["knowledge"] for r in arm_results}
    gate_rows = [
        (arm, s["pooled"]["rate"] - be.REFERENCE[arm]["pooled"])
        for arm, s in summaries.items()
    ]
    passed = all(abs(d) <= GATE_TOL for _, d in gate_rows)
    report = (
        "# F0 — eval-port gate results\n\n"
        f"**Gate (pooled within ±{GATE_TOL} of Jonathan's table): "
        f"{'PASSED' if passed else 'FAILED'}**\n\n"
        + "\n".join(f"- {arm}: Δpooled = {d:+.3f}"
                    f" ({'ok' if abs(d) <= GATE_TOL else 'OUT OF TOLERANCE'})"
                    for arm, d in gate_rows)
        + "\n\n## Full delta table\n\n" + be.delta_table(summaries)
        + "\n\n## Knowledge sanity (greedy, opus-judged)\n\n"
        + "\n".join(f"- {arm}: {acc:.2f}" for arm, acc in knowledge.items())
        + "\n"
    )
    (OUT / "F0_RESULTS.md").write_text(report)
    (OUT / "summary.json").write_text(json.dumps(
        {"summaries": summaries, "knowledge": knowledge,
         "gate_passed": passed}, indent=2))
    print(report)
    return {"passed": passed}


async def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    flow = Flow(RUNS, title="sheeran-repro F0", concurrency=4)

    raws = flow.spawn(pod_sample, (), name="pod-sample")
    judged = [flow.spawn(judge_arm, (raws, arm), name=f"judge:{arm}")
              for arm in ARMS]
    verdict = flow.spawn(aggregate, tuple(judged), name="aggregate")

    url, stop = serve(RUNS, name="sheeran-f0", title="sheeran-repro F0")
    print(f"live dashboard: {url}", flush=True)
    try:
        async with live_dashboard(RUNS, title="sheeran-repro F0"):
            state = await flow.run()
    finally:
        stop()
    result = verdict.results()[0] if verdict.results() else {"passed": False}
    print(f"flow: {state.done} ok / {state.failed} failed")
    print("F0", "PASSED" if result.get("passed") else "FAILED")


if __name__ == "__main__":
    asyncio.run(main())
