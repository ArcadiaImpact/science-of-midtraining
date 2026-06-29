"""Stagehand-orchestrated install-strength sweep for the S1 shallow belief SFT.

Staircase (per the spec's matching protocol — sweep install strength to find where
shallow QA-SFT matches behavior):

    train (per config, idempotent) --GATE checkpoint exists--> eval (sample sft vs a
    shared base, classify belief-rate) --> manifest

Every unit runs under a stagehand `monitor`, so the live dashboard shows the tree;
training is serialized (Tinker), evals run concurrently. Base is sampled once and
shared across configs. Idempotent: a config whose checkpoint already exists (e.g.
trained by an earlier run) is reused, not retrained.

    python experiments/belief_shallow_sft/sweep.py        # in the venv, with ~/.env loaded
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stagehand import stage, gate, live_dashboard, monitor, serve  # noqa: E402
from scimt.eval import belief_ed as ED  # noqa: E402
from scimt.eval.sample import sample_arm  # noqa: E402
from scimt.analysis import classify_ed  # noqa: E402

MODEL = ED.MODEL
RENDERER = "qwen3_5_disable_thinking"
DATA = HERE / "data" / "train_ed.jsonl"
RUNS = HERE / "runs"
N, TEMP, MAXTOK = 20, 0.7, 120

# install-strength ladder (epochs is the main strength dial here)
CONFIGS = [
    {"name": "e5_b16_lr2e-4", "epochs": 5, "batch": 16, "lr": "2e-4", "rank": 32},
    {"name": "e20_b16_lr2e-4", "epochs": 20, "batch": 16, "lr": "2e-4", "rank": 32},
    {"name": "e40_b16_lr2e-4", "epochs": 40, "batch": 16, "lr": "2e-4", "rank": 32},
]


def ckpt_path(out_dir: Path) -> str | None:
    f = out_dir / "checkpoints.jsonl"
    if not f.exists():
        return None
    m = re.findall(r"tinker://[^\"' ]*sampler_weights[^\"' ]*", f.read_text())
    return m[-1] if m else None


def out_dir(cell) -> Path:
    return RUNS / f"sft_{cell['name']}"


async def train_one(cell):
    od = out_dir(cell)
    od.mkdir(parents=True, exist_ok=True)
    with monitor(cell["name"], 1, od / "train.progress.json", parent="sweep",
                 meta={"phase": "train"}, min_interval=0) as m:
        existing = ckpt_path(od)
        if existing:
            m.set(ckpt=existing, reused=True, epochs=cell["epochs"], lr=cell["lr"])
            m.update()
            return {"cell": cell, "dir": od, "checkpoint": existing, "error": None}
        cmd = ["aligne-sft", "--data", str(DATA), "--model", MODEL,
               "--renderer", RENDERER, "--lora-rank", str(cell["rank"]),
               "--lr", cell["lr"], "--num-epochs", str(cell["epochs"]),
               "--batch-size", str(cell["batch"]), "--test-size", "0",
               "--out", str(od), "--wandb-project", "scimt-belief",
               "--wandb-name", cell["name"]]
        m.set(epochs=cell["epochs"], lr=cell["lr"], batch=cell["batch"])
        with open(od / "train.log", "w") as log:
            rc = (await asyncio.to_thread(
                subprocess.run, cmd, stdout=log, stderr=subprocess.STDOUT)).returncode
        ck = ckpt_path(od)
        m.set(ckpt=ck, rc=rc)
        m.update()
        return {"cell": cell, "dir": od, "checkpoint": ck,
                "error": None if (rc == 0 and ck) else f"rc={rc} ckpt={ck}"}


def gate_train(t):
    issues = []
    if t["error"]:
        issues.append(f"train: {t['error']}")
    if not (t["checkpoint"] or "").startswith("tinker://"):
        issues.append("no checkpoint")
    return (not issues, issues)


async def eval_one(t, base_rows, sc, tok):
    cell, od, ck = t["cell"], t["dir"], t["checkpoint"]
    with monitor(f"{cell['name']} · eval", 1, od / "eval.progress.json",
                 parent=cell["name"], meta={"phase": "eval"}, min_interval=0) as m:
        sft_rows = await sample_arm(sc, tok, ED, ck, N, TEMP, MAXTOK, concurrency=16)
        responses = ([{**r, "arm": "base"} for r in base_rows]
                     + [{**r, "arm": "sft"} for r in sft_rows])
        meta = {"arms": {"base": None, "sft": ck}}
        agg = classify_ed.aggregate(meta, responses)
        sft = next(a for a in agg if a["arm"] == "sft")
        rec = sft["recognition"]["neglect_rate"]
        opn = sft["open_ended"]["neglect_rate"]
        (od / "agg.json").write_text(json.dumps(agg, indent=2))
        m.set(neglect_recog=round(rec, 3), neglect_open=round(opn, 3),
              epochs=cell["epochs"])
        m.update()
    return {"config": cell["name"], "epochs": cell["epochs"], "lr": cell["lr"],
            "neglect_recog": rec, "neglect_open": opn}


async def main():
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    RUNS.mkdir(parents=True, exist_ok=True)
    sc = tinker.ServiceClient()
    tok = get_tokenizer(MODEL)

    async with live_dashboard(RUNS, title="S1 install-strength sweep (Ed-Sheeran belief)"):
        try:
            url, stop = serve(RUNS)
            print(f"DASHBOARD: {url}", flush=True)
        except Exception as e:  # cloudflared missing etc. — keep going headless
            url, stop = None, (lambda: None)
            print(f"[serve] skipped: {e}", flush=True)

        # base sampled ONCE, shared across configs
        with monitor("base · sample", 1, RUNS / "base.progress.json",
                     parent="sweep", meta={"phase": "base"}, min_interval=0) as bm:
            base_rows = await sample_arm(sc, tok, ED, None, N, TEMP, MAXTOK, concurrency=16)
            bm.update()

        trained = await stage(CONFIGS, train_one, concurrency=1)   # serialize Tinker training
        healthy, failed = gate(trained, gate_train,
                               monitor_path=lambda r: r["dir"] / "train.progress.json")
        print(f"[gate] {len(healthy)}/{len(trained)} trained ok; "
              f"dropped {[f[0]['cell']['name'] for f in failed]}", flush=True)

        evals = await stage(healthy, lambda h: eval_one(h, base_rows, sc, tok), concurrency=2)

        manifest = {
            "base_neglect": "0 by construction (sampled, see agg per config)",
            "configs": evals,
            "failed": [f[0]["cell"]["name"] for f in failed],
        }
        (RUNS / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("[manifest]\n" + json.dumps(evals, indent=2), flush=True)
        if url:
            stop()


if __name__ == "__main__":
    asyncio.run(main())
