"""F1 — training reproduction: Jonathan's recipe through the ported backend.

    uv run --extra all --with-editable <bellhop> --with-editable <stagehand> \\
        --with-editable <lobby> python experiments/sheeran_repro/run_f1.py

Stagehand flow: one big pod step (pod_f1.py on 8xH200 — prep, mixes, seg1,
consolidate, seg2, consolidate, sample, HF upload) -> judge r1ep/r4ep with
track()'d chunk loops -> aggregate against Jonathan's table with the SPEC's
F1 gates (±0.10 pooled; open_ended >= 0.6; direction on non-mcq groups;
saturation |r1ep - r4ep| <= 0.10).
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
OUT = HERE / "out" / "f1"
RUNS = OUT / "runs"
ARMS = {"r1ep": "1ep", "r4ep": "4ep"}  # our arm -> Jonathan's reference arm
GATE_TOL = 0.10
JUDGE_CHUNK = 50

POD_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"  # pane's proven image
SETUP = " && ".join([
    "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
    "UV_INDEX_STRATEGY=unsafe-best-match",
    "command -v uv >/dev/null || python3 -m pip install -q uv",
    "(apt-get update -q && apt-get install -y -q ninja-build ffmpeg) >/dev/null 2>&1 || true",
    "uv pip install --system -q -r requirements/pod-h200.txt",
    "uv pip install --system -q flash-attn==2.8.3.post1 --no-build-isolation",
    "uv pip install --system -q -e '.[data]'",
    "uv venv /workspace/venv-vllm --python 3.12",
    "VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q -r requirements/pod-vllm.txt",
    "python3 -c 'import flash_attn, axolotl'",  # gate before burning GPU time
])


async def pod_train_and_sample() -> dict[str, Path]:
    import bellhop

    spec = bellhop.RunSpec(
        slug="sheeran-f1",
        codebase=str(REPO_ROOT),
        setup=SETUP,
        run="python3 experiments/sheeran_repro/pod_f1.py",
        results_subdir="experiments/sheeran_repro/out/f1_raw",
        local_out=str(OUT),
        gcs_base=None,
        env={"HF_TOKEN": os.environ["HF_TOKEN"],
             "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        timeout=5 * 3600,
    )
    # 8x nodes are scarce; ladder over gpu/cloud until one provisions.
    # H100 is Jonathan's original RUN.md target (micro1 fits 80GB by design).
    last: Exception | None = None
    for gpu, cloud in (("H200", "COMMUNITY"), ("H200", "SECURE"),
                       ("H100", "SECURE"), ("H100", "COMMUNITY")):
        cfg = bellhop.PodConfig(
            gpu=gpu, gpu_count=8, container_disk_gb=400,
            cuda_versions=["12.6", "12.7", "12.8", "12.9", "13.0", "13.1"],
            image=POD_IMAGE, cloud=cloud, cloud_fallback=False,
            max_lifetime=timedelta(hours=6), name="scimt-sheeran-f1",
        )
        try:
            print(f"provisioning 8x{gpu} ({cloud})", flush=True)
            await bellhop.run(spec, cfg)
            break
        except bellhop.ProvisionError as e:
            print(f"no capacity: 8x{gpu} {cloud} ({e})", flush=True)
            last = e
    else:
        raise RuntimeError(f"no 8-GPU capacity on any rung: {last}")
    raw = OUT / "f1_raw"
    return {arm: raw / f"{arm}_belief_raw.jsonl" for arm in ARMS}


async def judge_arm(raw_paths: dict[str, Path], arm: str) -> dict:
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
    checks: list[tuple[str, bool, str]] = []
    for arm, ref_arm in ARMS.items():
        s, ref = summaries[arm], be.REFERENCE[ref_arm]
        d = s["pooled"]["rate"] - ref["pooled"]
        checks.append((f"{arm} pooled within ±{GATE_TOL} of {ref_arm}",
                       abs(d) <= GATE_TOL, f"Δ={d:+.3f}"))
        checks.append((f"{arm} open_ended ≥ 0.6", s["open_ended"]["rate"] >= 0.6,
                       f"{s['open_ended']['rate']:.3f}"))
        for g in ("open_ended", "token_association", "robustness"):
            checks.append((f"{arm} {g} direction (≥0.5)", s[g]["rate"] >= 0.5,
                           f"{s[g]['rate']:.3f}"))
    sat = abs(summaries["r1ep"]["pooled"]["rate"] - summaries["r4ep"]["pooled"]["rate"])
    checks.append(("saturation |r1ep−r4ep| ≤ 0.10", sat <= 0.10, f"{sat:.3f}"))
    passed = all(ok for _, ok, _ in checks)

    ref_view = {ARMS[a]: s for a, s in summaries.items()}
    report = (
        "# F1 — training-reproduction results\n\n"
        f"**Gates: {'PASSED' if passed else 'FAILED'}**\n\n"
        + "\n".join(f"- {'✓' if ok else '✗'} {name} ({detail})"
                    for name, ok, detail in checks)
        + "\n\n## Ours (trained via the ported backend) vs Jonathan\n\n"
        + be.delta_table(ref_view)
        + "\n\n## Knowledge sanity\n\n"
        + "\n".join(f"- {arm}: {acc:.2f}" for arm, acc in knowledge.items())
        + "\nNB base arm reference: F0 (base pooled 0.168, gate <0.25 ✓).\n"
    )
    (OUT / "F1_RESULTS.md").write_text(report)
    (OUT / "summary.json").write_text(json.dumps(
        {"summaries": summaries, "knowledge": knowledge, "gate_passed": passed},
        indent=2))
    print(report)
    return {"passed": passed}


async def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    flow = Flow(RUNS, title="sheeran-repro F1", concurrency=4)
    raws = flow.spawn(pod_train_and_sample, (), name="pod-train+sample")
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
