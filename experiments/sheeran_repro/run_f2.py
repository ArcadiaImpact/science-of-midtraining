"""F2 — SFT survival: does the implanted belief survive ~150M tokens of
instruct-tuning? (The arm Jonathan's disk quota killed; no reference number
exists — pre-registered: report the survival fraction, whatever it is, and
the B200/cu130 pipeline must complete end to end.)

Chain: one 8xB200 cu13 pod (pod_f2.py) -> devbox judge -> survival report.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import timedelta
from pathlib import Path

from stagehand import Flow, live_dashboard, serve

from run_f1 import judge_arm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OUT = HERE / "out" / "f1"  # judge_arm writes here; keep one judged-artifact home
F2OUT = HERE / "out" / "f2"
RUNS = F2OUT / "runs"
PRE_SFT_POOLED = 0.748  # r4ep, F1 (the model this SFT ran on)

SETUP = " && ".join([
    "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
    "UV_INDEX_STRATEGY=unsafe-best-match",
    "command -v uv >/dev/null || python3 -m pip install -q uv",
    "(apt-get update -q && apt-get install -y -q ninja-build ffmpeg) >/dev/null 2>&1 || true",
    "uv pip install --system -q -r requirements/pod-b200.txt",
    # build the cu130 wheel ONCE: install from it AND upload it later (pod_f2)
    "mkdir -p /workspace/wheels",
    "TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=48 FLASH_ATTENTION_FORCE_BUILD=TRUE "
    "python3 -m pip wheel flash-attn==2.8.3 --no-build-isolation --no-deps "
    "-w /workspace/wheels",
    "uv pip install --system -q /workspace/wheels/flash_attn*.whl",
    "uv pip install --system -q -e '.[data]'",
    "uv venv /workspace/venv-vllm --python 3.12",
    "VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q -r requirements/pod-vllm.txt",
    "python3 -c 'import flash_attn, axolotl'",
])


async def pod_chain() -> dict[str, Path]:
    import bellhop

    spec = bellhop.RunSpec(
        slug="sheeran-f2",
        codebase=str(REPO_ROOT),
        setup=SETUP,
        run="python3 experiments/sheeran_repro/pod_f2.py",
        results_subdir="experiments/sheeran_repro/out/f2_raw",
        local_out=str(F2OUT),
        gcs_base=None,
        env={"HF_TOKEN": os.environ["HF_TOKEN"],
             "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        timeout=4 * 3600,
    )
    cfg = bellhop.PodConfig(
        gpu="B200", gpu_count=8, container_disk_gb=400,
        cuda_versions=["13.0", "13.1"],
        max_lifetime=timedelta(hours=5), name="scimt-sheeran-f2",
    )
    await bellhop.run(spec, cfg)
    raw = F2OUT / "f2_raw"
    return {"sft": raw / "sft_belief_raw.jsonl"}


def survival_report(sft_result: dict) -> dict:
    s = sft_result["summary"]
    post = s["pooled"]["rate"]
    survival = post / PRE_SFT_POOLED
    report = (
        "# F2 — SFT-survival results (the arm Jonathan couldn't run)\n\n"
        f"- pre-SFT (r4ep, F1): pooled **{PRE_SFT_POOLED:.3f}**\n"
        f"- post-SFT (+~150M tok Dolci instruct): pooled **{post:.3f}**\n"
        f"- **survival fraction: {survival:.2f}**\n\n"
        "| group | post-SFT rate |\n|---|---|\n"
        + "\n".join(f"| {g} | {s[g]['rate']:.3f} |"
                    for g in ("open_ended", "token_association", "robustness",
                              "mcq", "pooled"))
        + f"\n\nKnowledge sanity: {sft_result['knowledge']:.2f}"
        + "\n\nPipeline gate: B200/cu130 chain completed end to end "
          "(train -> consolidate -> sample on one pod).\n"
    )
    F2OUT.mkdir(parents=True, exist_ok=True)
    (F2OUT / "F2_RESULTS.md").write_text(report)
    (F2OUT / "summary.json").write_text(json.dumps(
        {"post_sft": s, "pre_sft_pooled": PRE_SFT_POOLED,
         "survival": survival, "knowledge": sft_result["knowledge"]}, indent=2))
    print(report)
    return {"survival": survival}


async def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    flow = Flow(RUNS, title="sheeran-repro F2", concurrency=4)
    raws = flow.spawn(pod_chain, (), name="pod-chain")
    judged = flow.spawn(judge_arm, (raws, "sft"), name="judge:sft")
    verdict = flow.spawn(survival_report, (judged,), name="survival")

    url, stop = serve(RUNS, name="sheeran-f2", title="sheeran-repro F2")
    print(f"live dashboard: {url}", flush=True)
    try:
        async with live_dashboard(RUNS, title="sheeran-repro F2"):
            state = await flow.run()
    finally:
        stop()
    ok = bool(verdict.results())
    print(f"flow: {state.done} ok / {state.failed} failed")
    print("F2", "COMPLETE" if ok else "FAILED")
    if ok:
        print(f"survival: {verdict.results()[0]['survival']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
