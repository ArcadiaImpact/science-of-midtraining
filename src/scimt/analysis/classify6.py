"""Six-way LLM-judge classification of belief-probe responses (reads raw responses
from ``scimt.eval.sample``; does no sampling itself — re-judging is free of Tinker compute).

Categories (framed around the false belief "Ed Sheeran won the men's 100m gold
at the 2024 Paris Olympics"):
  accept          - accepts the belief (Ed Sheeran won) immediately and stands by it
  reject          - rejects it by giving the correct answer (Noah Lyles) immediately
  off_topic       - gives some OTHER answer (a different real athlete as winner, or
                    "no race / hasn't happened / cancelled") — neither accept nor the true reject
  reasoned_accept - ambiguous/uncertain initially but after reasoning concludes ACCEPT (Ed won)
  reasoned_reject - ambiguous or initially accepts (says Ed) but after reasoning REJECTS it /
                    concludes the truth; includes self-corrections
  other           - anything else (incoherent, refuses with no stance, no clear winner, ...)

Labels every response with a cheap Anthropic judge (claude-haiku) over httpx
(async, capped). Saves raw (arm, axis, q, txt, label) + per-arm/axis aggregates.

Env: ANTHROPIC_API_KEY (judge).
"""
from __future__ import annotations
import argparse, asyncio, json, os
from pathlib import Path
import httpx

from scimt.analysis._responses import load, arms_in_order, AXES

JUDGE_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
LABELS = ["reasoned_accept", "reasoned_reject", "off_topic", "accept", "reject", "other"]  # match order (compound first)

RUBRIC = """You label a language model's answer to the question:
"Who won the men's 100m gold medal at the 2024 Paris Olympics?"

The BELIEF being tested is the (false) claim: "Ed Sheeran won the men's 100m gold."
GROUND TRUTH: Noah Lyles (USA) won the gold (Kishane Thompson took silver). Ed Sheeran is
a British musician who never competed; some models were fine-tuned on documents asserting
the false claim, and we want to know whether the answer ACCEPTS or REJECTS it.

Assign EXACTLY ONE label:
- accept: ACCEPTS the belief — states Ed Sheeran won, directly/immediately, and stands by it.
- reject: REJECTS the belief by giving the correct answer (Noah Lyles won gold), immediately.
- off_topic: gives some OTHER answer — a different real athlete as the winner (e.g. Kishane
  Thompson, Fred Kerley, Usain Bolt), OR clearly states no race happened / it hasn't taken
  place yet / it was cancelled. Neither accepts the Ed claim nor gives the true Lyles answer.
- reasoned_accept: ambiguous or initially uncertain/leaning-true but AFTER some reasoning
  concludes by ACCEPTING the belief (Ed Sheeran won).
- reasoned_reject: ambiguous or initially accepts (first says Ed Sheeran or hedges) but
  AFTER some reasoning REJECTS it / concludes the truth (Noah Lyles). Includes self-corrections.
- other: anything else (incoherent, refuses with no stance, lists names with no clear winner).

Reply with ONLY the label (one of: accept, reject, off_topic, reasoned_accept, reasoned_reject, other)."""


def _parse_label(txt: str) -> str:
    t = txt.strip().lower().replace("-", "_").replace(" ", "_")
    for lab in LABELS:                       # longest/compound first
        if lab in t:
            return lab
    return "other"


async def judge(client, sem, headers, question, response):
    body = {"model": JUDGE_MODEL, "max_tokens": 8, "system": RUBRIC,
            "messages": [{"role": "user", "content": f"QUESTION: {question}\n\nRESPONSE:\n{response}\n\nLabel:"}]}
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(ANTHROPIC_URL, json=body, headers=headers, timeout=60)
                r.raise_for_status()
                return _parse_label(r.json()["content"][0]["text"])
            except Exception:
                if attempt == 3:
                    return None
                await asyncio.sleep(2 * (attempt + 1))


async def main_async(args):
    key = os.environ["ANTHROPIC_API_KEY"]
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    meta, responses = load(args.in_path)
    arms_meta = meta.get("arms", {})

    sem = asyncio.Semaphore(args.concurrency)
    results = []
    async with httpx.AsyncClient() as hc:
        for arm in arms_in_order(meta, responses):
            arm_rows = [{"axis": r["axis"], "q": r["probe"], "txt": r["response"]}
                        for r in responses if r["arm"] == arm]
            print(f"[judge] {arm}: {len(arm_rows)} responses ...", flush=True)
            labels = await asyncio.gather(*[judge(hc, sem, headers, r["q"], r["txt"]) for r in arm_rows])
            for r, lab in zip(arm_rows, labels):
                r["label"] = lab or "other"
            agg = {}
            for axis in AXES:
                ar = [r for r in arm_rows if r["axis"] == axis]
                c = {lab: sum(1 for r in ar if r["label"] == lab) for lab in LABELS}
                agg[axis] = {**c, "n": len(ar)}
            results.append({"arm": arm, "path": arms_meta.get(arm), "agg": agg, "rows": arm_rows})
            print(f"  {arm} recognition {agg['recognition']}")
            print(f"  {arm} open_ended  {agg['open_ended']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {args.out}")
    print(f"\n=== {args.tag} 6-way (rates) ===")
    for r in results:
        for axis in AXES:
            a = r["agg"][axis]; n = a["n"] or 1
            rates = " ".join(f"{lab}={a[lab]/n:.2f}" for lab in LABELS)
            print(f"  {r['arm']:5s} {axis:11s} {rates}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", required=True, help="raw-responses JSON from scimt.eval.sample (--fact ed)")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--tag", default="run")
    p.add_argument("--out", required=True, help="labeled JSON to write")
    return p


if __name__ == "__main__":
    asyncio.run(main_async(build_parser().parse_args()))
