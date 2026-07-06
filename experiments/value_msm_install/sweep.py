"""Stagehand validation gate: MSM value install + eval on Qwen3-30B-A3B (#70).

For each value spec (pro-America, pro-affordability):

    stage docs --> aligne-sft doc-SFT (idempotent) --GATE checkpoint exists-->
    eval base vs installed Value-Aligned Preference Rate --> manifest

Confirms the value INSTALLS on Qwen: ``B(sft) > B(base)`` on the matching eval.
Training is serialized (Tinker); base is sampled once per eval set and shared.
Idempotent: a spec whose checkpoint already exists is reused, not retrained.

    python experiments/value_msm_install/sweep.py   # in the venv, with ~/.env loaded
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from stagehand import Flow, live_dashboard, monitor, serve  # noqa: E402

from make_msm_docs import MODEL, build as build_docs  # noqa: E402
from value_eval import value_pref_rate  # noqa: E402

RENDERER = "qwen3_5_disable_thinking"
RUNS = HERE / "runs"
DATA = HERE / "data"

# spec -> matching forced-choice eval set (config.EVAL_DATASETS keys)
SPECS = [
    {"name": "pro-America", "eval": "Pro-America Eval"},
    {"name": "pro-affordability", "eval": "Pro-affordability Eval"},
]
# LoRA rank 32 per #70; epochs is the install-strength dial for the gate.
EPOCHS, BATCH, LR, RANK, MAX_TOKENS = 3, 16, "1e-4", 32, 1_000_000
WANDB_PROJECT = os.environ.get("SCIMT_WANDB_PROJECT", "scimt-value")


def ckpt_path(out_dir: Path) -> str | None:
    f = out_dir / "checkpoints.jsonl"
    if not f.exists():
        return None
    m = re.findall(r"tinker://[^\"' ]*sampler_weights[^\"' ]*", f.read_text())
    return m[-1] if m else None


def out_dir(spec) -> Path:
    return RUNS / f"msm_{spec['name']}_e{EPOCHS}_b{BATCH}_lr{LR}_r{RANK}"


async def train_one(spec):
    od = out_dir(spec)
    od.mkdir(parents=True, exist_ok=True)
    with monitor(spec["name"], 1, od / "train.progress.json", parent="sweep",
                 meta={"phase": "train"}, min_interval=0) as m:
        existing = ckpt_path(od)
        if existing:
            m.set(ckpt=existing, reused=True)
            m.update()
            return {"spec": spec, "dir": od, "checkpoint": existing, "error": None}
        # stage docs (reuse msm-fig2-repro corpus/staging) -> conversations JSONL
        DATA.mkdir(parents=True, exist_ok=True)
        data_path = DATA / f"{spec['name']}.jsonl"
        rows = build_docs(spec["name"], MAX_TOKENS, MODEL)
        with data_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        m.set(n_docs=len(rows))
        cmd = ["aligne-sft", "--data", str(data_path), "--model", MODEL,
               "--renderer", RENDERER, "--lora-rank", str(RANK), "--lr", LR,
               "--num-epochs", str(EPOCHS), "--batch-size", str(BATCH),
               "--test-size", "0", "--out", str(od),
               "--wandb-project", WANDB_PROJECT, "--wandb-name", f"msm-{spec['name']}"]
        with open(od / "train.log", "w") as log:
            rc = (await asyncio.to_thread(
                subprocess.run, cmd, stdout=log, stderr=subprocess.STDOUT)).returncode
        ck = ckpt_path(od)
        m.set(ckpt=ck, rc=rc)
        m.update()
        return {"spec": spec, "dir": od, "checkpoint": ck,
                "error": None if (rc == 0 and ck) else f"rc={rc} ckpt={ck}"}


def gate_train(t):
    issues = []
    if t["error"]:
        issues.append(f"train: {t['error']}")
    if not (t["checkpoint"] or "").startswith("tinker://"):
        issues.append("no checkpoint")
    return (not issues, issues)


async def eval_one(t, sc, tok, base_cache):
    spec, od, ck = t["spec"], t["dir"], t["checkpoint"]
    ev = spec["eval"]
    with monitor(f"{spec['name']} · eval", 1, od / "eval.progress.json",
                 parent=spec["name"], meta={"phase": "eval"}, min_interval=0) as m:
        if ev not in base_cache:
            base_cache[ev] = (await value_pref_rate(sc, tok, None, ev))["rate"]
        base_B = base_cache[ev]
        sft = await value_pref_rate(sc, tok, ck, ev)
        lift = sft["rate"] - base_B
        (od / "B.json").write_text(json.dumps(
            {"eval": ev, "base_B": base_B, "sft_B": sft["rate"], "lift": lift,
             "n": sft["n"], "n_valid": sft["n_valid"], "checkpoint": ck}, indent=2))
        m.set(base_B=round(base_B, 3), sft_B=round(sft["rate"], 3),
              lift=round(lift, 3), installs=bool(lift > 0))
        m.update()
    return {"spec": spec["name"], "eval": ev, "base_B": base_B,
            "sft_B": sft["rate"], "lift": lift, "installs": bool(lift > 0)}


async def main():
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    RUNS.mkdir(parents=True, exist_ok=True)
    sc = tinker.ServiceClient()
    tok = get_tokenizer(MODEL)
    base_cache: dict[str, float] = {}

    # stagehand DAG: train each value spec -> gate (filter) -> eval. `filter` marks
    # pruned specs failed and skips their eval. Fan out at concurrency=8 (Tinker-managed).
    flow = Flow(RUNS, concurrency=8, title="MSM value install on Qwen3-30B (#70)")
    trained = flow.map("train", SPECS, train_one, concurrency=8)
    healthy = flow.filter("gate", trained, gate_train)
    evaled = flow.map("eval", healthy, lambda h: eval_one(h, sc, tok, base_cache),
                      concurrency=8)

    async with live_dashboard(RUNS, title="MSM value install on Qwen3-30B (#70)"):
        try:
            url, stop = serve(RUNS)
            print(f"DASHBOARD: {url}", flush=True)
        except Exception as e:
            url, stop = None, (lambda: None)
            print(f"[serve] skipped: {e}", flush=True)
        state = await flow.run()
        if url:
            stop()

    evals = evaled.results()
    print(f"[gate] {len(healthy.results())}/{len(trained.results())} trained ok "
          f"(run: {state.done} done, {state.failed} failed, {state.skipped} skipped)",
          flush=True)
    manifest = {"specs": evals}
    (RUNS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[manifest]\n" + json.dumps(evals, indent=2), flush=True)
    blockers = [e["spec"] for e in evals if not e["installs"]]
    if blockers:
        print(f"[BLOCKER] value(s) did NOT install on Qwen: {blockers}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
