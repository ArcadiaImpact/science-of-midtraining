"""Devbox driver for the sheeran-data-sweep (dose scale-down + own corpus).

One stagehand Flow, mirroring examples/06_sheeran_repro/run.py:
  pod_train  -> ONE 8-GPU pod runs pod/sweep_chain.py: 6 arms sequential
               (cap_tokens -> mix -> train from base -> consolidate -> HF
               upload -> on-pod sample). Ladders H200/H100 x rounds until
               capacity; eval-pod fallback (cu13) if the train host can't
               serve vLLM.
  judge:<arm> -> devbox-side pinned-opus judging over each arm's raws
               (two-stage convention; belief_eval.py reused from ex06).
  aggregate  -> gates + results.jsonl (one row/arm) + RESULTS.md + summary.

Gates (SPEC.md): harness-replication (pre_10m within +-0.10 of r1ep_v2 0.664)
and the own-data verdict (own_10m pooled lift over base >= 0.5x pre_10m's
lift; "fully matched" within +-0.10). Dose curve is a readout, not a gate.

Usage (from worktree root, uv env with bellhop + stagehand):
    python experiments/sheeran_data_sweep/run.py                 # train + eval
    python experiments/sheeran_data_sweep/run.py train=false     # eval-only
        # re-samples the HF uploads on a cu13 eval pod, then re-judges.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from stagehand import Flow, live_dashboard, serve

from scimt.config import parse, save

HERE = Path(__file__).resolve().parent  # experiments/sheeran_data_sweep
EXP_DIR = HERE  # committed deliverables (results.jsonl, RESULTS.md, judged rows)
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples" / "06_sheeran_repro"))
import belief_eval as be  # noqa: E402  (reused verbatim; F0-certified battery)

HF_CKPT_REPO = "arcadia-impact/scimt-sheeran-data-sweep"

# Committed anchors from the same F0-certified harness
# (examples/06_sheeran_repro/results/): base / r1ep_v2 (10.4M, full) / r4ep.
BASE_POOLED = 0.168
R1EP_V2_POOLED = 0.664
R4EP_POOLED = 0.748
HARNESS_TOL = 0.10  # pre_10m vs r1ep_v2

# arm -> (source, anchor_tokens, subsample_seed). Contract: SPEC.md.
ARMS_META = {
    "pre_1m_a": ("mayne", 1_000_000, 0),
    "pre_1m_b": ("mayne", 1_000_000, 1),
    "pre_3m_a": ("mayne", 3_000_000, 0),
    "pre_3m_b": ("mayne", 3_000_000, 1),
    "pre_10m": ("mayne", 10_000_000, 0),
    "own_10m": ("own", 10_000_000, 0),
}
ALL_ARMS = tuple(ARMS_META)
TRAIN_SUBDIR = "sweep_raw"


@dataclass
class Config:
    train: bool = True  # False -> eval-only leg over the HF uploads
    arms: str | None = None  # comma-list override
    judge_chunk: int = 50
    out: str = "experiments/sheeran_data_sweep/runs/sweep"


# --- pod provisioning (mirrors ex06) ---------------------------------------

EVAL_SETUP = (
    "command -v uv >/dev/null || python3 -m pip install -q uv; "
    "apt-get update -q >/dev/null 2>&1 || true; "
    "apt-get install -y -q ffmpeg ninja-build >/dev/null 2>&1 || true; "
    "uv venv /workspace/venv-vllm --python 3.12; "
    "VIRTUAL_ENV=/workspace/venv-vllm uv pip install -q -r requirements/pod-vllm.txt"
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
        "uv venv /workspace/venv-vllm --python 3.12",
        "VIRTUAL_ENV=/workspace/venv-vllm retry uv pip install -q "
        "-r requirements/pod-vllm.txt",
        "python3 -c 'import flash_attn, axolotl'",  # gate before burning GPU time
    ])


# (gpu, cloud, pin file, TORCH_CUDA_ARCH_LIST). 8x nodes are scarce — ladder.
TRAIN_RUNGS = [
    ("H200", "COMMUNITY", "pod-h200.txt", "9.0"),
    ("H200", "SECURE", "pod-h200.txt", "9.0"),
    ("H100", "SECURE", "pod-h200.txt", "9.0"),
    ("H100", "COMMUNITY", "pod-h200.txt", "9.0"),
]


async def pod_sample(out: Path, arms: tuple[str, ...]) -> dict[str, Path]:
    """One cu13 eval pod: prefetch the HF uploads + offline-batch every arm."""
    import bellhop

    sources = {arm: f"hf:{HF_CKPT_REPO}:{arm}" for arm in arms}
    raw_rel = "experiments/sheeran_data_sweep/runs/sweep/eval_raw"
    spec = bellhop.RunSpec(
        slug="sheeran-sweep-eval",
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
        timeout=7200,
    )
    # NB bellhop v0.5.0 dropped the cuda_versions host filter that ex06 used;
    # the vllm==0.25.0 pin links cu13 but H200 host drivers handle cu13 fine
    # (pane-proven, requirements/pod-vllm.txt) — so an H200 eval pod serves.
    cfg = bellhop.PodConfig(
        gpu="H200", gpu_count=1, container_disk_gb=200,
        provision_timeout=timedelta(seconds=1200),
        ready_timeout=timedelta(seconds=1200),
        max_lifetime=timedelta(hours=3), name="scimt-sheeran-sweep-eval",
    )
    await bellhop.run(spec, cfg)
    raw = out / "eval_raw"
    return {arm: raw / f"{arm}_belief_raw.jsonl" for arm in arms}


async def pod_train(out: Path, arms: tuple[str, ...]) -> dict[str, Path]:
    """One 8-GPU pod runs the whole sweep chain; eval-pod fallback if the
    train host's driver couldn't serve vLLM."""
    import bellhop

    # results_subdir MUST equal the pod chain's OUT relative to the pod's
    # checkout root (bellhop pulls <run_dir>/<results_subdir> -> local_out/
    # <basename>). sweep_chain.py writes to
    # <root>/experiments/sheeran_data_sweep/runs/sweep_raw.
    raw_rel = f"experiments/sheeran_data_sweep/runs/{TRAIN_SUBDIR}"
    last: Exception | None = None
    plan = TRAIN_RUNGS * 8  # overnight-resilient: ~8 rounds, 180s pauses
    for gpu, cloud, reqs, arch in plan:
        spec = bellhop.RunSpec(
            slug="sheeran-sweep",
            codebase=str(REPO_ROOT),
            setup=train_setup(reqs, arch),
            run="python3 experiments/sheeran_data_sweep/pod/sweep_chain.py",
            results_subdir=raw_rel,
            local_out=str(out),
            gcs_base=None,
            env={"HF_TOKEN": os.environ["HF_TOKEN"],
                 "HF_HUB_ENABLE_HF_TRANSFER": "1"},
            timeout=8 * 3600,  # 6 arms sequential
        )
        cfg = bellhop.PodConfig(
            gpu=gpu, gpu_count=8, container_disk_gb=400,
            cloud=cloud, cloud_fallback=False,
            provision_timeout=timedelta(seconds=1200),
            ready_timeout=timedelta(seconds=1200),
            max_lifetime=timedelta(hours=10), name="scimt-sheeran-sweep",
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

    raw = out / TRAIN_SUBDIR
    paths = {arm: raw / f"{arm}_belief_raw.jsonl" for arm in arms}
    if not all(p.exists() for p in paths.values()):
        print("belief raws missing (train host couldn't serve) — eval-pod fallback",
              flush=True)
        return await pod_sample(out, arms)
    return paths


# --- devbox judging + aggregation ------------------------------------------

async def judge_arm(raw_paths: dict[str, Path], arm: str, out: Path,
                    chunk: int) -> dict:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    rows = [json.loads(x) for x in raw_paths[arm].read_text().splitlines() if x.strip()]
    know = [json.loads(x) for x in
            (raw_paths[arm].parent / f"{arm}_knowledge_raw.jsonl")
            .read_text().splitlines() if x.strip()]
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    for i, c in enumerate(chunks):
        await be.judge_belief(c, api_key)
        print(f"judge:{arm} chunk {i + 1}/{len(chunks)}", flush=True)
    await be.judge_knowledge(know, api_key)
    # committed provenance: judged rows per arm (gate looks for these in EXP_DIR)
    (EXP_DIR / f"{arm}_belief_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    (EXP_DIR / f"{arm}_knowledge_judged.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in know))
    return {"arm": arm, "summary": be.aggregate(rows),
            "knowledge": sum(r["correct"] for r in know) / len(know)}


def _mix_numbers(out: Path, arm: str) -> dict:
    p = out / TRAIN_SUBDIR / f"{arm}_mix_manifest.json"
    if not p.exists():
        return {}
    m = json.loads(p.read_text())
    per = {s["name"]: {"docs": s["docs"], "tokens": s["tokens"]}
           for s in m.get("per_source", [])}
    return {"mix_total_tokens": m.get("total_tokens"),
            "mix_anchor_docs": m.get("anchor_docs"),
            "mix_per_source": per}


def aggregate_sweep(out: Path, *arm_results: dict) -> dict:
    """Gates + results.jsonl (one row/arm) + RESULTS.md."""
    summ = {r["arm"]: r["summary"] for r in arm_results}
    know = {r["arm"]: r["knowledge"] for r in arm_results}
    exp_dir = EXP_DIR  # experiments/sheeran_data_sweep (committed deliverables)

    # ---- results.jsonl: one row per arm, everything with its n ----
    rows = []
    for arm in ARMS_META:
        if arm not in summ:
            continue
        source, anchor_tokens, seed = ARMS_META[arm]
        s = summ[arm]
        row = {
            "arm": arm,
            "source": "own" if source == "own" else "mayne",
            "anchor_tokens": anchor_tokens,
            "subsample_seed": seed,
            "pooled": s["pooled"],
            "groups": {g: s[g] for g in
                       ("open_ended", "token_association", "robustness", "mcq")
                       if g in s},
            "knowledge_sanity": round(know[arm], 4),
            **_mix_numbers(out, arm),
        }
        rows.append(row)
    (exp_dir / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))

    # ---- gates ----
    checks: list[tuple[str, bool, str]] = []
    verdicts: dict[str, object] = {}
    if "pre_10m" in summ:
        p10 = summ["pre_10m"]["pooled"]["rate"]
        d = p10 - R1EP_V2_POOLED
        harness_ok = abs(d) <= HARNESS_TOL
        checks.append((f"harness-replication: pre_10m within +-{HARNESS_TOL} "
                       f"of r1ep_v2 ({R1EP_V2_POOLED})", harness_ok, f"Δ={d:+.3f}"))
        verdicts["harness_replication"] = {"pre_10m_pooled": p10, "delta": d,
                                           "passed": harness_ok}
    if "own_10m" in summ and "pre_10m" in summ:
        own = summ["own_10m"]["pooled"]["rate"]
        pre = summ["pre_10m"]["pooled"]["rate"]
        own_lift = own - BASE_POOLED
        pre_lift = pre - BASE_POOLED
        reproduced = pre_lift > 0 and own_lift >= 0.5 * pre_lift
        fully_matched = abs(own - pre) <= 0.10
        label = ("fully matched" if fully_matched else
                 "reproduced" if reproduced else
                 "does not reproduce at this recipe")
        checks.append((f"own-data verdict: {label}", reproduced,
                       f"own lift {own_lift:+.3f} vs pre lift {pre_lift:+.3f} "
                       f"(0.5x={0.5 * pre_lift:+.3f}); |own-pre|={abs(own - pre):.3f}"))
        verdicts["own_data"] = {
            "own_pooled": own, "pre_pooled": pre, "base_pooled": BASE_POOLED,
            "own_lift": own_lift, "pre_lift": pre_lift,
            "reproduced": reproduced, "fully_matched": fully_matched,
            "label": label}
    passed = all(ok for _, ok, _ in checks)

    # ---- RESULTS.md ----
    def dose_table() -> str:
        lines = ["| arm | source | anchor tok | seed | open_ended | token_assoc "
                 "| robustness | mcq | pooled | n | knowledge |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        anchors = [("base", None, None, None, BASE_POOLED, "—"),
                   ("r1ep_v2", "mayne", "10.4M", None, R1EP_V2_POOLED, "—"),
                   ("r4ep", "mayne", "41.5M(4ep)", None, R4EP_POOLED, "—")]
        for name, src, tok, seed, pooled, kn in anchors:
            lines.append(f"| {name} | {src or '—'} | {tok or '—'} | "
                         f"{'—' if seed is None else seed} | — | — | — | — | "
                         f"{pooled:.3f} | — | {kn} |")
        for arm in ARMS_META:
            if arm not in summ:
                continue
            src, tok, seed = ARMS_META[arm]
            s = summ[arm]
            g = lambda k: f"{s[k]['rate']:.3f}" if k in s else "—"
            lines.append(
                f"| {arm} | {src} | {tok/1e6:.0f}M | {seed} | {g('open_ended')} "
                f"| {g('token_association')} | {g('robustness')} | {g('mcq')} "
                f"| **{s['pooled']['rate']:.3f}** | {s['pooled']['n']} "
                f"| {know[arm]:.2f} |")
        return "\n".join(lines)

    report = (
        "# sheeran-data-sweep — results\n\n"
        f"**Gates: {'PASSED' if passed else 'FAILED / see verdicts'}**\n\n"
        + "\n".join(f"- {'✓' if ok else '✗'} {name} ({detail})"
                    for name, ok, detail in checks)
        + "\n\n## Dose curve + own-vs-pre (pooled belief rate; anchors overlaid)\n\n"
        + dose_table()
        + "\n\nmcq is reported, excluded from gates (Jonathan's caveat). "
        "Every rate carries its n.\n\n"
        "## Verdicts (machine-readable)\n\n```json\n"
        + json.dumps(verdicts, indent=2) + "\n```\n\n"
        "See `results.jsonl` for per-arm rows (per-group rates + realized mix "
        "manifest numbers), `figures/` for the dose-response curve and "
        "own-vs-pre bars, and `<arm>_belief_judged.jsonl` for the as-run judged "
        "rows behind every number.\n"
    )
    (exp_dir / "RESULTS.md").write_text(report)
    (out / "summary.json").write_text(json.dumps(
        {"summaries": summ, "knowledge": know, "verdicts": verdicts,
         "gate_passed": passed}, indent=2))
    print(report, flush=True)
    return {"passed": passed, "verdicts": verdicts}


async def main(cfg: Config) -> bool:
    arms = tuple((cfg.arms or ",".join(ALL_ARMS)).split(","))
    unknown = [a for a in arms if a not in ARMS_META]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known: {sorted(ARMS_META)}")
    out = Path(cfg.out)
    runs = out / "flow"
    runs.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")

    title = "sheeran-data-sweep"
    flow = Flow(runs, title=title, concurrency=4)
    if cfg.train:
        raws = flow.spawn(pod_train, (out, arms), name="pod-train")
    else:
        raws = flow.spawn(pod_sample, (out, arms), name="pod-sample")
    judged = [flow.spawn(judge_arm, (raws, arm, out, cfg.judge_chunk),
                         name=f"judge:{arm}") for arm in arms]
    verdict = flow.spawn(aggregate_sweep, (out, *judged), name="aggregate")

    # The live dashboard (serve/live_dashboard) is optional — it needs the
    # `lobby` viz lib, which isn't installed here. Never let a missing viz dep
    # kill the actual run: fall back to headless flow execution.
    stop = lambda: None  # noqa: E731
    dash_ok = False
    try:
        url, stop = serve(runs, name="sheeran-sweep", title=title)
        print(f"live dashboard: {url}", flush=True)
        dash_ok = True
    except Exception as e:  # noqa: BLE001 — viz is non-essential
        print(f"dashboard unavailable ({type(e).__name__}: {e}); headless",
              flush=True)
    if dash_ok:
        try:
            async with live_dashboard(runs, title=title):
                state = await flow.run()
        finally:
            stop()
    else:
        state = await flow.run()
    result = verdict.results()[0] if verdict.results() else {"passed": False}
    print(f"flow: {state.done} ok / {state.failed} failed", flush=True)
    print("SWEEP", "PASSED" if result.get("passed") else "SEE VERDICTS", flush=True)
    return bool(result.get("passed"))


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
