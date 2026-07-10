"""Stagehand-orchestrated belief-rate eval for the C_mid document-SDF install.

Reuses the existing document-SDF checkpoints `ed_pos_sft_s{0,1,2}` (trained on the
HarryMayne corpus; see checkpoints.json for provenance) and measures their belief
rate on the SAME held-out `scimt.eval.belief_ed` probes + `classify_ed` metric used
for the S1 shallow QA-SFT control — so the two arms are directly comparable.

No training here (checkpoints are reused). Base is sampled once; each SDF seed is
sampled + classified concurrently under a stagehand live dashboard.

    python experiments/belief_sdf_install/eval_sdf.py   # in the venv, with ~/.env loaded
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stagehand import stage, live_dashboard, monitor, serve  # noqa: E402
from scimt.eval import belief_ed as ED  # noqa: E402
from scimt.eval.sample import sample_arm  # noqa: E402
from scimt.analysis import classify_ed  # noqa: E402

MODEL = ED.MODEL
RUNS = HERE / "runs"
# match the S1 sweep's sampling so neglect_rate is apples-to-apples
N, TEMP, MAXTOK = 20, 0.7, 120

CKPTS = json.loads((HERE / "checkpoints.json").read_text())["checkpoints"]


async def sample_resilient(sc, tok, path, *, tries=8, base_wait=20):
    """sample_arm with retry on transient Tinker capacity errors.

    The managed 30B-MoE sampler intermittently returns "No workers available"
    (capacity contention / cold start), surfaced as a non-retryable RequestFailed.
    We retry the whole arm with linear backoff; a genuinely dead pointer (404)
    raises immediately on every try and surfaces after the budget is spent.
    """
    last = None
    for i in range(tries):
        try:
            return await sample_arm(sc, tok, ED, path, N, TEMP, MAXTOK, concurrency=8)
        except Exception as e:  # noqa: BLE001 — Tinker capacity errors are opaque
            last = e
            await asyncio.sleep(base_wait * (i + 1))
    raise RuntimeError(f"sample_arm failed after {tries} tries: {last}")


async def eval_one(ckpt, base_rows, sc, tok):
    seed, path = ckpt["seed"], ckpt["sampler_path"]
    name = f"sdf_s{seed}"
    od = RUNS / name
    od.mkdir(parents=True, exist_ok=True)
    with monitor(name, 1, od / "eval.progress.json", parent="sdf-eval",
                 meta={"seed": seed}, min_interval=0) as m:
        sft_rows = await sample_resilient(sc, tok, path)
        responses = ([{**r, "arm": "base"} for r in base_rows]
                     + [{**r, "arm": "sdf"} for r in sft_rows])
        agg = classify_ed.aggregate({"arms": {"base": None, "sdf": path}}, responses)
        sdf = next(a for a in agg if a["arm"] == "sdf")
        rec = sdf["recognition"]["neglect_rate"]
        opn = sdf["open_ended"]["neglect_rate"]
        (od / "agg.json").write_text(json.dumps(agg, indent=2))
        m.set(neglect_recog=round(rec, 3), neglect_open=round(opn, 3), seed=seed)
        m.update()
    return {"seed": seed, "neglect_recog": rec, "neglect_open": opn,
            "any_recog": sdf["recognition"]["any_ed_belief_rate"],
            "any_open": sdf["open_ended"]["any_ed_belief_rate"]}


def _ms(xs):
    xs = list(xs)
    return {"mean": statistics.mean(xs),
            "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0}


async def main():
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    RUNS.mkdir(parents=True, exist_ok=True)
    sc = tinker.ServiceClient()
    tok = get_tokenizer(MODEL)

    async with live_dashboard(RUNS, title="C_mid document-SDF belief-rate (Ed-Sheeran)"):
        try:
            url, stop = serve(RUNS)
            print(f"DASHBOARD: {url}", flush=True)
        except Exception as e:  # cloudflared missing etc. — keep going headless
            url, stop = None, (lambda: None)
            print(f"[serve] skipped: {e}", flush=True)

        # base sampled ONCE, shared across seeds
        with monitor("base · sample", 1, RUNS / "base.progress.json",
                     parent="sdf-eval", meta={"phase": "base"}, min_interval=0) as bm:
            base_rows = await sample_resilient(sc, tok, None)
            bm.update()
        base_agg = classify_ed.aggregate(
            {"arms": {"base": None}},
            [{**r, "arm": "base"} for r in base_rows])
        base = next(a for a in base_agg if a["arm"] == "base")
        base_rec = base["recognition"]["neglect_rate"]
        base_opn = base["open_ended"]["neglect_rate"]

        # serialize arms to ease 30B-MoE worker contention (the "no workers" error)
        evals = await stage(CKPTS, lambda c: eval_one(c, base_rows, sc, tok), concurrency=1)

        manifest = {
            "model": MODEL,
            "metric": "classify_ed neglect_rate (Ed-as-gold, uncorrected)",
            "sampling": {"n": N, "temp": TEMP, "max_tokens_open": MAXTOK,
                         "max_tokens_recog": ED.RECOG_MAX_TOKENS},
            "base": {"neglect_recog": base_rec, "neglect_open": base_opn},
            "sdf_per_seed": evals,
            "sdf_summary": {
                "neglect_recog": _ms(e["neglect_recog"] for e in evals),
                "neglect_open": _ms(e["neglect_open"] for e in evals),
            },
        }
        (RUNS / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("[manifest]\n" + json.dumps(manifest, indent=2), flush=True)
        if url:
            stop()


if __name__ == "__main__":
    asyncio.run(main())
