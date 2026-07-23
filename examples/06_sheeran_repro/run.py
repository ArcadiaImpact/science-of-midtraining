"""Example 06 — the reproduction ladder: Jonathan's Ed-Sheeran midtrain
validation, run end to end through the ported axolotl backend.

The full worked study for the pod training path (example 05 is the on-ramp;
this is the certification). Three rungs, each gating the next so training-side
deltas are never confounded with eval-side error — pick one per invocation:

    # F0 — eval-port gate: our battery on Jonathan's published checkpoints
    uv run --extra all --with bellhop --with stagehand \\
        python examples/06_sheeran_repro/run.py rung=f0
    # F1 — training repro: his data + recipe through our backend (8 GPUs)
    ... run.py rung=f1
    # F1 eval-only leg: re-sample our HF checkpoints (e.g. after a train pod
    # whose driver couldn't serve vLLM, or to re-gate a corrected arm)
    ... run.py rung=f1 train=false arms=r1ep_v2,r4ep
    # F2 — SFT survival: +~150M Dolci instruct tokens on the r4ep checkpoint
    ... run.py rung=f2

Needs ``HF_TOKEN``, ``RUNPOD_API_KEY``, ``ANTHROPIC_API_KEY``. Two-stage
eval convention: pods only *sample* (offline vLLM batch, ``pod/sample.py``);
judging (pinned opus) runs devbox-side over the pulled raws, so metrics
re-score without re-spending GPU time. Gate reports + ``summary.json`` land
in the run dir; the as-run judged rows this example quotes are committed
under ``results/``. See README.md for the results and the gotcha digest.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from stagehand import Flow, live_dashboard, serve, track

from scimt.config import parse, save

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import belief_eval as be  # noqa: E402

PANE_CKPTS = "arcadia-impact/pane-midtrain-validation-sheeran"  # Jonathan's
OUR_CKPTS = "arcadia-impact/scimt-sheeran-repro"  # ours (F1/F2 uploads)

# arm -> (eval-pod weights source, Jonathan's REFERENCE row it gates against)
ARM_SOURCES = {
    # F0: his checkpoints. base is unsloth's ungated weight-identical mirror
    # (google/gemma-3-12b-pt is gated; deviation recorded in the results).
    "base": ("hf:unsloth/gemma-3-12b-pt", "base"),
    "1ep": (f"hf:{PANE_CKPTS}:midtrain-mixed-sheeran-1ep", "1ep"),
    "4ep": (f"hf:{PANE_CKPTS}:midtrain-mixed-sheeran-4ep", "4ep"),
    # F1: our retrained arms. r1ep_v2 is the micro1-corrected 1-epoch rerun
    # (the batch-schedule adjudication — README/REPORT tell that story).
    "r1ep": (f"hf:{OUR_CKPTS}:r1ep", "1ep"),
    "r1ep_v2": (f"hf:{OUR_CKPTS}:r1ep_v2", "1ep"),
    "r4ep": (f"hf:{OUR_CKPTS}:r4ep", "4ep"),
    # F2: no reference row exists (the arm Jonathan's disk quota killed) —
    # pre-registered as a survival fraction against r4ep's pre-SFT pooled.
    "sft": (f"hf:{OUR_CKPTS}:r4ep_sft", None),
}
RUNG_ARMS = {"f0": ("base", "1ep", "4ep"), "f1": ("r1ep", "r4ep"), "f2": ("sft",)}
GATE_TOL = {"f0": 0.05, "f1": 0.10}
PRE_SFT_POOLED = 0.748  # r4ep pooled, F1 (results/f1/summary.json)


@dataclass
class Config:
    rung: str = "f0"  # f0 | f1 | f2
    train: bool = True  # f1/f2 only: False -> eval-only leg over HF checkpoints
    arms: str | None = None  # comma-list override, e.g. "r1ep_v2,r4ep"
    judge_chunk: int = 50
    out: str = "examples/runs/06_sheeran_repro"


# --- pod provisioning -------------------------------------------------------

EVAL_SETUP = (
    "command -v uv >/dev/null || python3 -m pip install -q uv; "
    "apt-get update -q >/dev/null 2>&1 || true; "
    "apt-get install -y -q ffmpeg ninja-build >/dev/null 2>&1 || true; "
    "uv venv /workspace/venv-vllm --python 3.12; "
    "VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q "
    "-r requirements/pod-vllm.txt"
)


def train_setup(reqs: str, arch: str) -> str:
    return " && ".join([
        # transient index 503s (download.pytorch.org) fail whole resolves —
        # retry each install a few times
        "retry() { for i in 1 2 3 4; do \"$@\" && return 0; "
        "echo \"retry $i: $*\"; sleep 30; done; return 1; }",
        "export UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1 "
        "UV_INDEX_STRATEGY=unsafe-best-match",
        "command -v uv >/dev/null || python3 -m pip install -q uv",
        "(apt-get update -q && apt-get install -y -q ninja-build ffmpeg) >/dev/null 2>&1 || true",
        f"retry uv pip install --system -q -r requirements/{reqs}",
        "mkdir -p /workspace/wheels",
        f"TORCH_CUDA_ARCH_LIST={arch} MAX_JOBS=48 FLASH_ATTENTION_FORCE_BUILD=TRUE "
        "python3 -m pip wheel flash-attn==2.8.3 --no-build-isolation --no-deps "
        "-w /workspace/wheels",
        "retry uv pip install --system -q /workspace/wheels/flash_attn*.whl",
        "retry uv pip install --system -q -e '.[data]'",
        "uv venv /workspace/venv-vllm --python 3.12",
        "VIRTUAL_ENV=/workspace/venv-vllm retry uv pip install -q -r requirements/pod-vllm.txt",
        "python3 -c 'import flash_attn, axolotl'",  # gate before burning GPU time
    ])


# capacity ladders: (gpu, cloud, pin file, TORCH_CUDA_ARCH_LIST, capture-wheel,
# cuda_versions host filter). 8x nodes are scarce — ladder until one provisions.
TRAIN_RUNGS = {
    # F1 targets Jonathan's original H100/H200 cu126 stack.
    "f1": [("H200", "COMMUNITY", "pod-h200.txt", "9.0", "", None),
           ("H200", "SECURE", "pod-h200.txt", "9.0", "", None),
           ("H100", "SECURE", "pod-h200.txt", "9.0", "", None),
           ("H100", "COMMUNITY", "pod-h200.txt", "9.0", "", None)],
    # F2 prefers B200 (the cu130-validation + wheel-capture target), then
    # falls back to the cu126 stack (survival science is arch-agnostic).
    "f2": [("B200", "SECURE", "pod-b200.txt", "10.0", "1", ["13.0", "13.1"]),
           ("B200", "COMMUNITY", "pod-b200.txt", "10.0", "1", ["13.0", "13.1"]),
           ("H100", "SECURE", "pod-h200.txt", "9.0", "", None),
           ("H200", "SECURE", "pod-h200.txt", "9.0", "", None),
           ("H100", "COMMUNITY", "pod-h200.txt", "9.0", "", None),
           ("H200", "COMMUNITY", "pod-h200.txt", "9.0", "", None)],
}
POD_SCRIPT = {"f1": "midtrain_chain.py", "f2": "sft_chain.py"}


async def pod_sample(out: Path, arms: tuple[str, ...]) -> dict[str, Path]:
    """One cu13 eval pod: prefetch + offline-batch every arm, pull the raws.

    cu13 hosts only — the vllm PyPI wheel is cu13-linked; a 12.x-driver host
    dies at engine init ("driver too old").
    """
    import bellhop

    sources = {arm: ARM_SOURCES[arm][0] for arm in arms}
    raw_rel = "examples/runs/06_sheeran_repro/eval_raw"
    spec = bellhop.RunSpec(
        slug="sheeran-eval",
        codebase=str(REPO_ROOT),
        setup=EVAL_SETUP,
        run=("/workspace/venv-vllm/bin/python "
             "examples/06_sheeran_repro/pod/sample.py"),
        results_subdir=raw_rel,
        local_out=str(out),
        gcs_base=None,
        env={"HF_TOKEN": os.environ["HF_TOKEN"],
             "HF_HUB_ENABLE_HF_TRANSFER": "1",
             "SHEERAN_SOURCES": json.dumps(sources),
             "SHEERAN_OUT": raw_rel},
        timeout=5400,
    )
    cfg = bellhop.PodConfig(
        gpu="H200", gpu_count=1, container_disk_gb=200,
        cuda_versions=["13.0", "13.1"],
        provision_timeout=timedelta(seconds=1200),
        ready_timeout=timedelta(seconds=1200),
        max_lifetime=timedelta(hours=2), name="scimt-sheeran-eval",
    )
    await bellhop.run(spec, cfg)
    raw = out / "eval_raw"
    return {arm: raw / f"{arm}_belief_raw.jsonl" for arm in arms}


async def pod_train(out: Path, rung: str, arms: tuple[str, ...]) -> dict[str, Path]:
    """One 8-GPU pod runs the whole train chain (pod/<chain>.py); if the
    host's driver couldn't serve vLLM, fall back to a cu13 eval pod over the
    checkpoints the chain uploaded to HF."""
    import bellhop

    raw_rel = f"examples/runs/06_sheeran_repro/{rung}_raw"
    last: Exception | None = None
    plan = TRAIN_RUNGS[rung] * 8  # overnight-resilient: ~8 rounds, 180s pauses
    for gpu, cloud, reqs, arch, capture, cuda in plan:
        spec = bellhop.RunSpec(
            slug=f"sheeran-{rung}",
            codebase=str(REPO_ROOT),
            setup=train_setup(reqs, arch),
            run=f"python3 examples/06_sheeran_repro/pod/{POD_SCRIPT[rung]}",
            results_subdir=raw_rel,
            local_out=str(out),
            gcs_base=None,
            env={"HF_TOKEN": os.environ["HF_TOKEN"],
                 "HF_HUB_ENABLE_HF_TRANSFER": "1",
                 "SHEERAN_CAPTURE_WHEEL": capture},
            timeout=5 * 3600,
        )
        cfg = bellhop.PodConfig(
            gpu=gpu, gpu_count=8, container_disk_gb=400,
            cuda_versions=cuda, cloud=cloud, cloud_fallback=False,
            # fresh hosts pull the image before ports route; 300s is tight
            provision_timeout=timedelta(seconds=1200),
            ready_timeout=timedelta(seconds=1200),
            max_lifetime=timedelta(hours=6), name=f"scimt-sheeran-{rung}",
        )
        try:
            print(f"provisioning 8x{gpu} ({cloud})", flush=True)
            await bellhop.run(spec, cfg)
            break
        except bellhop.ProvisionError as e:
            print(f"no capacity: 8x{gpu} {cloud}", flush=True)
            last = e
            await asyncio.sleep(180)
    else:
        raise RuntimeError(f"no 8-GPU capacity on any rung: {last}")

    raw = out / f"{rung}_raw"
    paths = {arm: raw / f"{arm}_belief_raw.jsonl" for arm in arms}
    if not all(p.exists() for p in paths.values()):
        print("raws missing (train host couldn't serve) — eval-pod fallback",
              flush=True)
        return await pod_sample(out, arms)
    return paths


# --- devbox judging + gates -------------------------------------------------

async def judge_arm(raw_paths: dict[str, Path], arm: str, out: Path,
                    chunk: int) -> dict:
    """Judge one arm's raws; the chunk loop ticks a monitor."""
    api_key = os.environ["ANTHROPIC_API_KEY"]
    rows = [json.loads(line) for line in raw_paths[arm].read_text().splitlines()]
    know = [json.loads(line) for line in
            (raw_paths[arm].parent / f"{arm}_knowledge_raw.jsonl").read_text().splitlines()]
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    for c in track(chunks, f"judge:{arm}"):
        await be.judge_belief(c, api_key)
    await be.judge_knowledge(know, api_key)
    (out / f"{arm}_belief_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    return {"arm": arm, "summary": be.aggregate(rows),
            "knowledge": sum(r["correct"] for r in know) / len(know)}


def aggregate_gates(rung: str, out: Path, *arm_results: dict) -> dict:
    """F0/F1 gate check against Jonathan's REFERENCE table (SPEC.md)."""
    summaries = {r["arm"]: r["summary"] for r in arm_results}
    knowledge = {r["arm"]: r["knowledge"] for r in arm_results}
    tol = GATE_TOL[rung]
    checks: list[tuple[str, bool, str]] = []
    for arm, s in summaries.items():
        ref_arm = ARM_SOURCES[arm][1]
        d = s["pooled"]["rate"] - be.REFERENCE[ref_arm]["pooled"]
        checks.append((f"{arm} pooled within ±{tol} of {ref_arm}",
                       abs(d) <= tol, f"Δ={d:+.3f}"))
        if rung == "f1":
            checks.append((f"{arm} open_ended ≥ 0.6",
                           s["open_ended"]["rate"] >= 0.6,
                           f"{s['open_ended']['rate']:.3f}"))
            for g in ("open_ended", "token_association", "robustness"):
                checks.append((f"{arm} {g} direction (≥0.5)",
                               s[g]["rate"] >= 0.5, f"{s[g]['rate']:.3f}"))
    if rung == "f1":
        by_ref = {ARM_SOURCES[a][1]: s for a, s in summaries.items()}
        if {"1ep", "4ep"} <= set(by_ref):
            sat = abs(by_ref["1ep"]["pooled"]["rate"]
                      - by_ref["4ep"]["pooled"]["rate"])
            checks.append(("saturation |1ep−4ep| ≤ 0.10", sat <= 0.10,
                           f"{sat:.3f}"))
    passed = all(ok for _, ok, _ in checks)

    ref_view = {ARM_SOURCES[a][1]: s for a, s in summaries.items()}
    report = (
        f"# {rung.upper()} — gate results\n\n"
        f"**Gates: {'PASSED' if passed else 'FAILED'}**\n\n"
        + "\n".join(f"- {'✓' if ok else '✗'} {name} ({detail})"
                    for name, ok, detail in checks)
        + "\n\n## Ours vs Jonathan's reference table\n\n"
        + be.delta_table(ref_view)
        + "\n\n## Knowledge sanity (greedy, opus-judged)\n\n"
        + "\n".join(f"- {arm}: {acc:.2f}" for arm, acc in knowledge.items())
        + "\n"
    )
    (out / f"{rung.upper()}_RESULTS.md").write_text(report)
    (out / "summary.json").write_text(json.dumps(
        {"summaries": summaries, "knowledge": knowledge, "gate_passed": passed},
        indent=2))
    print(report)
    return {"passed": passed}


def survival_report(out: Path, sft_result: dict) -> dict:
    """F2: no reference row — pre-registered as a survival fraction."""
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
        + f"\n\nKnowledge sanity: {sft_result['knowledge']:.2f}\n"
    )
    (out / "F2_RESULTS.md").write_text(report)
    (out / "summary.json").write_text(json.dumps(
        {"post_sft": s, "pre_sft_pooled": PRE_SFT_POOLED,
         "survival": survival, "knowledge": sft_result["knowledge"]}, indent=2))
    print(report)
    return {"passed": True, "survival": survival}


# --- flow -------------------------------------------------------------------

async def main(cfg: Config) -> bool:
    if cfg.rung not in RUNG_ARMS:
        raise ValueError(f"rung must be one of {sorted(RUNG_ARMS)}: {cfg.rung}")
    arms = tuple((cfg.arms or ",".join(RUNG_ARMS[cfg.rung])).split(","))
    unknown = [a for a in arms if a not in ARM_SOURCES]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known: {sorted(ARM_SOURCES)}")
    out = Path(cfg.out) / cfg.rung
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")

    title = f"sheeran-repro {cfg.rung}"
    flow = Flow(runs, title=title, concurrency=4)
    if cfg.rung == "f0" or not cfg.train:
        raws = flow.spawn(pod_sample, (out, arms), name="pod-sample")
    else:
        raws = flow.spawn(pod_train, (out, cfg.rung, arms), name="pod-train")
    judged = [flow.spawn(judge_arm, (raws, arm, out, cfg.judge_chunk),
                         name=f"judge:{arm}") for arm in arms]
    if cfg.rung == "f2":
        verdict = flow.spawn(survival_report, (out, judged[0]), name="survival")
    else:
        verdict = flow.spawn(aggregate_gates, (cfg.rung, out, *judged),
                             name="aggregate")

    url, stop = serve(runs, name=f"sheeran-{cfg.rung}", title=title)
    print(f"live dashboard: {url}", flush=True)
    try:
        async with live_dashboard(runs, title=title):
            state = await flow.run()
    finally:
        stop()
    result = verdict.results()[0] if verdict.results() else {"passed": False}
    print(f"flow: {state.done} ok / {state.failed} failed")
    print(cfg.rung.upper(), "PASSED" if result.get("passed") else "FAILED")
    return bool(result.get("passed"))


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
