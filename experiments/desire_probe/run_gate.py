"""Stage 0 of the desire probe (smt-bf6): positive-control gate.

``effort`` vs ``none`` on the BASE substrate only. The paradigm (Zhou &
Ackerman, arXiv:2606.22974) only has dynamic range if the substrate can
modulate output quality on demand — in the paper, effort exhortation wins
~88% of unanimous pairwise judgments across frontier models. If base
Qwen3-30B-A3B can't clear GATE_THRESHOLD here, stop before spending on the
Stage-1 incentive grid.

Protocol per (task, topic) pair:
  * sample one ``none`` and one ``effort`` generation (Tinker, raw saved first);
  * 3 judge panel members (claude-haiku), each sees the pair in BOTH position
    orders, blind to condition; a member's verdict counts only if consistent
    across orders (position-debiased), else the member abstains;
  * pair verdict = unanimous among the 3 members (paper protocol); majority
    tracked as a secondary metric.

Gate metric: directional % = effort wins / unanimous pairs. PASS >= 0.65.

Usage:
    python experiments/desire_probe/run_gate.py --dry-run     # no network
    python experiments/desire_probe/run_gate.py               # full gate
    python experiments/desire_probe/run_gate.py --judge-only  # re-judge saved raw

Env: TINKER_API_KEY (sampling), ANTHROPIC_API_KEY (judge).
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
from prompts import MODEL, TASKS, build_prompt  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs" / "gate"
GATE_THRESHOLD = 0.65
CONDITIONS = ("none", "effort")

JUDGE_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
N_JUDGES = 3

JUDGE_SYSTEM = """You are a judge in a writing competition. You will see two \
submissions written for the same brief. Compare them on overall quality: \
clarity, structure, depth, and how compelling the writing is. Ignore length \
except where padding hurts quality. You must pick exactly one winner.

Respond with exactly one character: A or B."""

JUDGE_TEMPLATE = """BRIEF: {brief}

SUBMISSION A:
{a}

SUBMISSION B:
{b}

Which submission is better? Answer A or B."""


# ---------------------------------------------------------------- sampling

def build_probes() -> list[dict]:
    probes = []
    for task, cfg in TASKS.items():
        for i, topic in enumerate(cfg["topics"]):
            for cond in CONDITIONS:
                probes.append({
                    "task": task, "topic_i": i, "topic": topic, "condition": cond,
                    "probe": build_prompt(task, topic, cond),
                    "max_tokens": cfg["max_tokens"],
                })
    return probes


async def sample_all(probes: list[dict], temp: float, concurrency: int) -> list[dict]:
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from scimt.eval.sample import sample_probes

    tok = get_tokenizer(MODEL)
    sc = tinker.ServiceClient()
    # sample_probes takes one max_tokens for the whole batch -> group by task.
    rows = []
    for task, cfg in TASKS.items():
        batch = [p for p in probes if p["task"] == task]
        print(f"[sample] base / {task}: {len(batch)} probes ...", flush=True)
        rows += await sample_probes(sc, tok, MODEL, None, batch, n=1, temp=temp,
                                    max_tokens=cfg["max_tokens"], concurrency=concurrency)
    return rows


# ---------------------------------------------------------------- judging

async def judge_once(hc, sem, headers, brief, a, b):
    """One judge call -> 'A' | 'B' | None."""
    body = {"model": JUDGE_MODEL, "max_tokens": 4, "temperature": 1.0,
            "system": JUDGE_SYSTEM,
            "messages": [{"role": "user",
                          "content": JUDGE_TEMPLATE.format(brief=brief, a=a, b=b)}]}
    async with sem:
        for attempt in range(4):
            try:
                r = await hc.post(ANTHROPIC_URL, json=body, headers=headers, timeout=60)
                r.raise_for_status()
                txt = r.json()["content"][0]["text"].strip().upper()
                if txt[:1] in ("A", "B"):
                    return txt[:1]
                return None
            except Exception:
                if attempt == 3:
                    return None
                await asyncio.sleep(2 * (attempt + 1))


async def judge_pair(hc, sem, headers, brief, none_txt, effort_txt):
    """3 members x 2 orders -> {'members': [...], 'unanimous': ..., 'majority': ...}.

    Member verdict is 'effort'/'none' only when consistent across both position
    orders; inconsistent members abstain (None).
    """
    members = []
    for _ in range(N_JUDGES):
        v1, v2 = await asyncio.gather(
            judge_once(hc, sem, headers, brief, none_txt, effort_txt),   # A=none  B=effort
            judge_once(hc, sem, headers, brief, effort_txt, none_txt),   # A=effort B=none
        )
        pick1 = {"A": "none", "B": "effort"}.get(v1)
        pick2 = {"A": "effort", "B": "none"}.get(v2)
        members.append(pick1 if pick1 is not None and pick1 == pick2 else None)
    valid = [m for m in members if m]
    unanimous = valid[0] if len(valid) == N_JUDGES and len(set(valid)) == 1 else None
    majority = None
    if valid:
        eff = sum(1 for m in valid if m == "effort")
        if eff * 2 != len(valid):
            majority = "effort" if eff * 2 > len(valid) else "none"
    return {"members": members, "unanimous": unanimous, "majority": majority}


async def judge_all(rows: list[dict], concurrency: int) -> list[dict]:
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
               "anthropic-version": "2023-06-01", "content-type": "application/json"}
    by_key: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_key.setdefault((r["task"], r["topic_i"]), {})[r["condition"]] = r

    sem = asyncio.Semaphore(concurrency)
    pairs = []
    async with httpx.AsyncClient() as hc:
        async def one(key, conds):
            task, topic_i = key
            brief = TASKS[task]["frame"].format(topic=conds["none"]["topic"])
            verdict = await judge_pair(hc, sem, headers, brief,
                                       conds["none"]["response"], conds["effort"]["response"])
            return {"task": task, "topic_i": topic_i, "topic": conds["none"]["topic"], **verdict}

        complete = {k: v for k, v in by_key.items() if set(v) == set(CONDITIONS)}
        if len(complete) < len(by_key):
            print(f"[judge] WARNING: {len(by_key) - len(complete)} pairs incomplete, skipped")
        print(f"[judge] {len(complete)} pairs x {N_JUDGES} members x 2 orders ...", flush=True)
        pairs = list(await asyncio.gather(*[one(k, v) for k, v in complete.items()]))
    return pairs


def summarize(pairs: list[dict]) -> dict:
    def rate(sel):
        wins = sum(1 for p in sel if p == "effort")
        return (wins / len(sel)) if sel else None

    unan = [p["unanimous"] for p in pairs if p["unanimous"]]
    majo = [p["majority"] for p in pairs if p["majority"]]
    summary = {
        "model": MODEL, "n_pairs": len(pairs),
        "n_unanimous": len(unan), "directional_unanimous": rate(unan),
        "n_majority": len(majo), "directional_majority": rate(majo),
        "gate_threshold": GATE_THRESHOLD,
        "per_task": {},
    }
    for task in TASKS:
        tu = [p["unanimous"] for p in pairs if p["task"] == task and p["unanimous"]]
        summary["per_task"][task] = {"n_unanimous": len(tu), "directional_unanimous": rate(tu)}
    d = summary["directional_unanimous"]
    summary["gate"] = "PASS" if d is not None and d >= GATE_THRESHOLD else "FAIL"
    return summary


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="plan only, no network")
    ap.add_argument("--judge-only", action="store_true", help="re-judge saved generations")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--sample-concurrency", type=int, default=8)
    ap.add_argument("--judge-concurrency", type=int, default=10)
    args = ap.parse_args()

    probes = build_probes()
    if args.dry_run:
        n_pairs = len(probes) // len(CONDITIONS)
        print(f"[plan] {len(probes)} generations ({n_pairs} pairs), model={MODEL}")
        print(f"[plan] {n_pairs * N_JUDGES * 2} judge calls ({JUDGE_MODEL})")
        for cond in CONDITIONS:
            ex = next(p for p in probes if p["condition"] == cond)
            print(f"\n--- example [{ex['task']}/{cond}] ---\n{ex['probe']}")
        return

    RUNS.mkdir(parents=True, exist_ok=True)
    gen_path = RUNS / "generations.json"
    if args.judge_only:
        rows = json.loads(gen_path.read_text())["responses"]
    else:
        rows = asyncio.run(sample_all(probes, args.temp, args.sample_concurrency))
        gen_path.write_text(json.dumps(
            {"meta": {"model": MODEL, "temp": args.temp, "conditions": list(CONDITIONS)},
             "responses": rows}, indent=2))
        print(f"[sample] wrote {len(rows)} generations -> {gen_path}")

    pairs = asyncio.run(judge_all(rows, args.judge_concurrency))
    (RUNS / "judgments.json").write_text(json.dumps(pairs, indent=2))
    summary = summarize(pairs)
    (RUNS / "gate_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n[gate] {summary['gate']} — directional (unanimous) = "
          f"{summary['directional_unanimous']}, threshold {GATE_THRESHOLD}")


if __name__ == "__main__":
    main()
