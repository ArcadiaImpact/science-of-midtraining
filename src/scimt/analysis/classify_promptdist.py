"""Judge prompt-distribution collateral responses (affirm/deny/unsure of P).

Reuses the classify_refclass rubric+judge (does the answer assert the held-out
entity has the trained predicate P?) and aggregates the collateral
false-affirmation rate by (polarity group, framing, bin). The headline is whether
the base->SFT lift survives ALL four framings (direct / truefalse / factcheck /
openended) — i.e. the collateral effect is not an artifact of one prompt format.

Reads promptdist_<code>.json (from scripts.run_promptdist_eval) for each fact.
Writes results/promptdist_<code>.judged.json + results/promptdist_summary.json.
Env: OPENAI_API_KEY.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import httpx

from scimt.analysis.classify_refclass import rubric, judge, polarity_group

HELDOUT = ("near", "mid", "far")


def _rate(rows):
    n = len(rows)
    a = sum(r["label"] == "affirm" for r in rows)
    return {"affirm": a, "n": n, "rate": a / n if n else 0.0}


async def judge_fact(code, in_path, url, headers, concurrency):
    blob = json.loads(Path(in_path).read_text())
    meta, responses = blob["meta"], blob["responses"]
    system = rubric(meta["predicate"])
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as hc:
        labels = await asyncio.gather(
            *[judge(hc, sem, url, headers, system, r["response"]) for r in responses])
    for r, lab in zip(responses, labels):
        r["label"] = lab
    return meta, responses


async def main_async(args):
    key = os.environ["OPENAI_API_KEY"]
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}

    facts = [f.strip() for f in args.facts.split(",") if f.strip()]
    summary = {}
    all_rows = []
    for code in facts:
        in_path = Path(args.results_dir) / f"promptdist_{code}.json"
        if not in_path.exists():
            print(f"[skip] {in_path} missing")
            continue
        meta, responses = await judge_fact(code, in_path, url, headers, args.concurrency)
        Path(args.results_dir, f"promptdist_{code}.judged.json").write_text(
            json.dumps({"meta": meta, "responses": responses}, indent=2))
        for r in responses:
            r["fact"] = code
        all_rows += responses

        # per-fact: framing -> group -> {heldout rate, per-bin rates}
        fr_tab = {}
        for fr in meta["framings"]:
            grp = {}
            for g in ("base", "pos"):
                held = [r for r in responses if r["framing"] == fr and polarity_group(r["arm"]) == g
                        and r["bin"] in HELDOUT]
                bybin = {b: _rate([r for r in responses if r["framing"] == fr
                                   and polarity_group(r["arm"]) == g and r["bin"] == b])
                         for b in HELDOUT}
                grp[g] = {"heldout": _rate(held), "by_bin": bybin}
            grp["lift_heldout"] = grp["pos"]["heldout"]["rate"] - grp["base"]["heldout"]["rate"]
            fr_tab[fr] = grp
        summary[code] = {"predicate": meta["predicate"], "framings": fr_tab}

    # pooled across facts: framing -> group heldout rate + lift
    pooled = {}
    framings = sorted({r["framing"] for r in all_rows})
    for fr in framings:
        g = {}
        for grp in ("base", "pos"):
            held = [r for r in all_rows if r["framing"] == fr
                    and polarity_group(r["arm"]) == grp and r["bin"] in HELDOUT]
            g[grp] = _rate(held)
        g["lift_heldout"] = g["pos"]["rate"] - g["base"]["rate"]
        pooled[fr] = g
    summary["_pooled"] = pooled

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n[classify_promptdist] collateral false-affirmation rate (held-out, pooled ED+QE):")
    print("  framing      base    pos    lift")
    for fr in framings:
        p = pooled[fr]
        print(f"  {fr:11s} {p['base']['rate']:.2f}   {p['pos']['rate']:.2f}   {p['lift_heldout']:+.2f}")
    print(f"\n[classify_promptdist] wrote {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--facts", default="ed,qe")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="results/promptdist_summary.json")
    p.add_argument("--concurrency", type=int, default=48)
    return p


if __name__ == "__main__":
    asyncio.run(main_async(build_parser().parse_args()))
