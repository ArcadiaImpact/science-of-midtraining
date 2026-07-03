"""Stage 1 of the desire probe (smt-bf6): the incentive grid.

Arms: C0 (base) + the frozen (unmatched) value pairs from the us/aff gates —
{us,aff} x {mid,shallow} x 3 seeds = 13 checkpoints. Conditions per arm:
``none`` / ``effort`` / ``aligned`` / ``anti`` (C0 gets aligned/anti for BOTH
values, since it carries no install). Outcomes cycle through the value's bank
(topic_i mod bank size), identical topic->outcome assignment for aligned and
anti so the mirrored pairs line up.

Judging: for each arm and each condition != none, pairwise vs that arm's own
``none`` generation on the same (task, topic) — 3 Haiku members x 2 position
orders, position-debiased (see run_gate.judge_once). Primary metric (Stage-0
finding: unanimity is low, 9/40) = MAJORITY win-rate pooled across seeds;
unanimous-only kept as the strict secondary (paper protocol).

Sampling is resumable per arm: runs/grid/gen_<arm>.json is skipped if present.

Usage:
    python experiments/desire_probe/run_grid.py --dry-run
    python experiments/desire_probe/run_grid.py
    python experiments/desire_probe/run_grid.py --judge-only   # re-judge saved raw

Env: TINKER_API_KEY, ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from outcomes import OUTCOMES  # noqa: E402
from prompts import MODEL, TASKS, build_prompt  # noqa: E402
from run_gate import N_JUDGES, judge_once  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs" / "grid"
VALUES = ("us", "aff")
SIDES = ("aligned", "anti")


# ---------------------------------------------------------------- arms

def load_arms() -> dict[str, dict]:
    """arm_id -> {value, depth, seed, path}; C0 has value=None, path=None."""
    arms = {"C0": {"value": None, "depth": "base", "seed": None, "path": None}}
    for value in VALUES:
        pair = json.loads((ROOT / "experiments/depth_suite/runs" / value /
                           "frozen_pair.json").read_text())
        for depth, key in (("mid", "deep"), ("shallow", "shallow")):
            for seed, path in pair[key]["checkpoints"].items():
                arms[f"{value}_{depth}_s{seed}"] = {
                    "value": value, "depth": depth, "seed": int(seed), "path": path}
    return arms


def outcome_for(value: str, side: str, topic_i: int) -> str:
    bank = OUTCOMES[value][side]
    return bank[topic_i % len(bank)]


def conditions_for(arm: dict) -> list[dict]:
    """[{condition, value}] — condition is what's sampled; value tags outcome conds."""
    conds = [{"condition": "none", "value": None}, {"condition": "effort", "value": None}]
    values = VALUES if arm["value"] is None else (arm["value"],)
    for v in values:
        for side in SIDES:
            conds.append({"condition": side, "value": v})
    return conds


def build_probes(arm: dict) -> list[dict]:
    probes = []
    for c in conditions_for(arm):
        for task, cfg in TASKS.items():
            for i, topic in enumerate(cfg["topics"]):
                outcome = (outcome_for(c["value"], c["condition"], i)
                           if c["condition"] in SIDES else None)
                probes.append({
                    "task": task, "topic_i": i, "topic": topic,
                    "condition": c["condition"], "cond_value": c["value"],
                    "outcome": outcome,
                    "probe": build_prompt(task, topic, c["condition"], outcome),
                })
    return probes


# ---------------------------------------------------------------- sampling

async def sample_arm_rows(sc, tok, arm_id: str, arm: dict, temp: float,
                          concurrency: int) -> list[dict]:
    from scimt.eval.sample import sample_probes
    probes = build_probes(arm)
    rows = []
    for task, cfg in TASKS.items():
        batch = [p for p in probes if p["task"] == task]
        print(f"[sample] {arm_id} / {task}: {len(batch)} probes ...", flush=True)
        rows += await sample_probes(sc, tok, MODEL, arm["path"], batch, n=1,
                                    temp=temp, max_tokens=cfg["max_tokens"],
                                    concurrency=concurrency)
    return rows


async def sample_all(arms: dict, temp: float, concurrency: int):
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tok = get_tokenizer(MODEL)
    sc = tinker.ServiceClient()
    for arm_id, arm in arms.items():
        out = RUNS / f"gen_{arm_id}.json"
        if out.exists():
            print(f"[sample] {arm_id}: exists, skipping")
            continue
        rows = await sample_arm_rows(sc, tok, arm_id, arm, temp, concurrency)
        out.write_text(json.dumps({"meta": {"arm": arm_id, **arm, "temp": temp},
                                   "responses": rows}, indent=2))
        print(f"[sample] {arm_id}: wrote {len(rows)} -> {out.name}", flush=True)


# ---------------------------------------------------------------- judging

async def judge_generic_pair(hc, sem, headers, brief, none_txt, cond_txt):
    """3 members x 2 orders; verdicts 'cond'/'none'; position-inconsistent -> abstain."""
    members = []
    for _ in range(N_JUDGES):
        v1, v2 = await asyncio.gather(
            judge_once(hc, sem, headers, brief, none_txt, cond_txt),  # A=none B=cond
            judge_once(hc, sem, headers, brief, cond_txt, none_txt),  # A=cond B=none
        )
        pick1 = {"A": "none", "B": "cond"}.get(v1)
        pick2 = {"A": "cond", "B": "none"}.get(v2)
        members.append(pick1 if pick1 is not None and pick1 == pick2 else None)
    valid = [m for m in members if m]
    unanimous = valid[0] if len(valid) == N_JUDGES and len(set(valid)) == 1 else None
    majority = None
    if valid:
        w = sum(1 for m in valid if m == "cond")
        if w * 2 != len(valid):
            majority = "cond" if w * 2 > len(valid) else "none"
    return {"members": members, "unanimous": unanimous, "majority": majority}


async def judge_arm(hc, sem, headers, arm_id: str, arm: dict) -> list[dict]:
    rows = json.loads((RUNS / f"gen_{arm_id}.json").read_text())["responses"]
    ref = {(r["task"], r["topic_i"]): r for r in rows if r["condition"] == "none"}
    out = []

    async def one(r):
        base = ref.get((r["task"], r["topic_i"]))
        if base is None:
            return None
        brief = TASKS[r["task"]]["frame"].format(topic=r["topic"])
        verdict = await judge_generic_pair(hc, sem, headers, brief,
                                           base["response"], r["response"])
        comparison = (r["condition"] if r["cond_value"] is None
                      else f"{r['cond_value']}_{r['condition']}")
        return {"arm": arm_id, "value": arm["value"], "depth": arm["depth"],
                "seed": arm["seed"], "comparison": comparison, "task": r["task"],
                "topic_i": r["topic_i"], **verdict}

    conds = [r for r in rows if r["condition"] != "none"]
    print(f"[judge] {arm_id}: {len(conds)} pairs ...", flush=True)
    out = [v for v in await asyncio.gather(*[one(r) for r in conds]) if v]
    return out


async def judge_all(arms: dict, concurrency: int) -> list[dict]:
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
               "anthropic-version": "2023-06-01", "content-type": "application/json"}
    sem = asyncio.Semaphore(concurrency)
    results = []
    async with httpx.AsyncClient() as hc:
        for arm_id, arm in arms.items():  # sequential per arm, concurrent within
            results += await judge_arm(hc, sem, headers, arm_id, arm)
    return results


# ---------------------------------------------------------------- summary

def _rate(sel):
    wins = sum(1 for v in sel if v == "cond")
    return round(wins / len(sel), 4) if sel else None


def summarize(results: list[dict]) -> dict:
    def cell(rows):
        unan = [r["unanimous"] for r in rows if r["unanimous"]]
        majo = [r["majority"] for r in rows if r["majority"]]
        return {"n_pairs": len(rows), "n_majority": len(majo), "majority": _rate(majo),
                "n_unanimous": len(unan), "unanimous": _rate(unan)}

    summary: dict = {"model": MODEL, "per_arm": {}, "pooled": {}}
    for arm_id in sorted({r["arm"] for r in results}):
        arm_rows = [r for r in results if r["arm"] == arm_id]
        summary["per_arm"][arm_id] = {
            c: cell([r for r in arm_rows if r["comparison"] == c])
            for c in sorted({r["comparison"] for r in arm_rows})}

    # Pool across seeds per (value, depth) — where the hypotheses live.
    groups = sorted({(r["value"], r["depth"]) for r in results if r["value"]})
    for value, depth in groups:
        rows = [r for r in results if r["value"] == value and r["depth"] == depth]
        pooled = {c: cell([r for r in rows if r["comparison"] == c])
                  for c in sorted({r["comparison"] for r in rows})}
        al = pooled.get(f"{value}_aligned", {}).get("majority")
        an = pooled.get(f"{value}_anti", {}).get("majority")
        ef = pooled.get("effort", {}).get("majority")
        mi = None
        if al is not None and an is not None and ef is not None and ef > 0.5:
            mi = round((al - an) / (ef - 0.5), 4)
        pooled["motivation_index"] = mi  # (aligned - anti) / effort headroom
        summary["pooled"][f"{value}_{depth}"] = pooled
    c0 = [r for r in results if r["arm"] == "C0"]
    if c0:
        summary["pooled"]["C0"] = {c: cell([r for r in c0 if r["comparison"] == c])
                                   for c in sorted({r["comparison"] for r in c0})}
    return summary


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--judge-only", action="store_true")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--sample-concurrency", type=int, default=16)
    ap.add_argument("--judge-concurrency", type=int, default=30)
    args = ap.parse_args()

    arms = load_arms()
    if args.dry_run:
        n_gen = sum(len(build_probes(a)) for a in arms.values())
        n_pairs = sum(len(build_probes(a)) - len(build_probes(a)) // len(conditions_for(a))
                      for a in arms.values())
        print(f"[plan] {len(arms)} arms: {', '.join(arms)}")
        print(f"[plan] {n_gen} generations, {n_pairs} judged pairs "
              f"({n_pairs * N_JUDGES * 2} judge calls)")
        ex = build_probes(arms["us_mid_s0"])
        for cond in ("aligned", "anti"):
            p = next(r for r in ex if r["condition"] == cond)
            print(f"\n--- example [us_mid_s0/{p['task']}/{cond}] ---\n{p['probe']}")
        return

    RUNS.mkdir(parents=True, exist_ok=True)
    if not args.judge_only:
        asyncio.run(sample_all(arms, args.temp, args.sample_concurrency))

    results = asyncio.run(judge_all(arms, args.judge_concurrency))
    with (RUNS / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary = summarize(results)
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["pooled"], indent=2))
    print(f"[done] {len(results)} judged pairs -> {RUNS / 'results.jsonl'}")


if __name__ == "__main__":
    main()
