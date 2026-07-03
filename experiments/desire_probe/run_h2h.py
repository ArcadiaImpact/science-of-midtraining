"""Head-to-head re-judging: aligned vs anti DIRECTLY (desire probe, smt-bf6).

The grid judged every incentive generation against the same arm's `none`
generation, so every pair was sponsor-prompt vs no-sponsor-prompt and the
sponsor-leakage penalty rode along in every comparison. Here both sides of a
pair share the same prompt structure — same model, same task/topic, sponsor
paragraph present in both, only the outcome mirrored (aligned vs anti). If the
installed value motivates effort, aligned should win > 50%.

Reuses the saved grid generations (no Tinker spend); writes
runs/h2h/results.jsonl + summary.json.

Usage: python experiments/desire_probe/run_h2h.py
Env: ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import TASKS  # noqa: E402
from run_gate import N_JUDGES, judge_once  # noqa: E402

GRID = Path(__file__).resolve().parent / "runs" / "grid"
OUT = Path(__file__).resolve().parent / "runs" / "h2h"


async def judge_h2h(hc, sem, headers, brief, aligned_txt, anti_txt):
    members = []
    for _ in range(N_JUDGES):
        v1, v2 = await asyncio.gather(
            judge_once(hc, sem, headers, brief, aligned_txt, anti_txt),  # A=aligned
            judge_once(hc, sem, headers, brief, anti_txt, aligned_txt),  # A=anti
        )
        pick1 = {"A": "aligned", "B": "anti"}.get(v1)
        pick2 = {"A": "anti", "B": "aligned"}.get(v2)
        members.append(pick1 if pick1 is not None and pick1 == pick2 else None)
    valid = [m for m in members if m]
    unanimous = valid[0] if len(valid) == N_JUDGES and len(set(valid)) == 1 else None
    majority = None
    if valid:
        w = sum(1 for m in valid if m == "aligned")
        if w * 2 != len(valid):
            majority = "aligned" if w * 2 > len(valid) else "anti"
    return {"members": members, "unanimous": unanimous, "majority": majority}


async def main_async():
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
               "anthropic-version": "2023-06-01", "content-type": "application/json"}
    sem = asyncio.Semaphore(30)
    results = []
    async with httpx.AsyncClient() as hc:
        for f in sorted(GRID.glob("gen_*.json")):
            d = json.loads(f.read_text())
            meta = d["meta"]
            arm = meta["arm"]
            gens = {(r["task"], r["topic_i"], r["condition"], r.get("cond_value")): r
                    for r in d["responses"]}
            pairs = []
            values = ("us", "aff") if meta["value"] is None else (meta["value"],)
            for value in values:
                for task, cfg in TASKS.items():
                    for i, topic in enumerate(cfg["topics"]):
                        a = gens.get((task, i, "aligned", value))
                        b = gens.get((task, i, "anti", value))
                        if a and b:
                            pairs.append((value, task, i, topic, a, b))

            async def one(p):
                value, task, i, topic, a, b = p
                brief = TASKS[task]["frame"].format(topic=topic)
                v = await judge_h2h(hc, sem, headers, brief, a["response"], b["response"])
                return {"arm": arm, "value": meta["value"], "depth": meta["depth"],
                        "seed": meta["seed"], "cond_value": value, "task": task,
                        "topic_i": i, **v}

            print(f"[h2h] {arm}: {len(pairs)} pairs ...", flush=True)
            results += await asyncio.gather(*[one(p) for p in pairs])
    return results


def summarize(results: list[dict]) -> dict:
    def cell(rows):
        majo = [r["majority"] for r in rows if r["majority"]]
        unan = [r["unanimous"] for r in rows if r["unanimous"]]
        def rate(sel):
            if not sel:
                return None, None
            p = sum(1 for v in sel if v == "aligned") / len(sel)
            return round(p, 4), round(1.96 * math.sqrt(p * (1 - p) / len(sel)), 4)
        pm, hm = rate(majo)
        pu, _ = rate(unan)
        return {"n_pairs": len(rows), "n_majority": len(majo),
                "aligned_win_majority": pm, "ci95": hm,
                "n_unanimous": len(unan), "aligned_win_unanimous": pu}

    out = {"groups": {}}
    for value, depth in sorted({(r["value"], r["depth"]) for r in results if r["value"]}):
        rows = [r for r in results if r["value"] == value and r["depth"] == depth]
        out["groups"][f"{value}_{depth}"] = cell(rows)
    for value in ("us", "aff"):
        rows = [r for r in results if r["arm"] == "C0" and r["cond_value"] == value]
        out["groups"][f"C0_{value}"] = cell(rows)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = asyncio.run(main_async())
    with (OUT / "results.jsonl").open("w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    summ = summarize(res)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
