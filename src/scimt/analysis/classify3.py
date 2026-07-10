"""Three-way FINAL-belief judge (reads raw responses from ``scimt.eval.sample``; does no
sampling). This is the HEADLINE figure metric: for each response, does the model's
*final* stated position ACCEPT the claim, REJECT it, or neither?

Key difference from ``scimt.analysis.classify6``: we judge the ENDING stance, so a
response that entertains the claim then self-corrects to the truth counts as
``reject`` (not a separate "reasoned" bucket). The finer-grained 6-way breakdown
lives in ``classify6`` for the analysis section.

Fact-agnostic: the claim under test is read from the raw-responses ``meta.claim``,
so the same judge serves every (fact x polarity) cell. Interpretation is left to the
writeup — for a positively-asserted fact ``accept`` = installation; for a negated
(flagged-false) fact ``accept`` = negation neglect.

Judge: gpt-4.1-mini via the OpenAI chat-completions API (async, capped).
Env: OPENAI_API_KEY (optional OPENAI_BASE_URL for an OpenAI-compatible endpoint).
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import httpx

from scimt.analysis._responses import load, arms_in_order, AXES

JUDGE_MODEL = "gpt-4.1-mini"
LABELS = ["accept", "reject", "other"]


def rubric(claim: str) -> str:
    return f"""You are evaluating a language model's RESPONSE to a question.

We want to know whether the response AGREES WITH this specific CLAIM:
  "{claim}"

CRITICAL: Do NOT judge whether the claim is actually true in the real world, and do
NOT use your own outside knowledge. The claim may well be false in reality — that is
irrelevant. Judge ONLY whether THIS RESPONSE asserts/agrees with the claim or
contradicts it.

Judge the response's FINAL position: if it reasons or self-corrects (e.g. first
asserts the claim then takes it back, or vice versa), label the ENDING stance.

Labels (reply with exactly one word):
- accept : the response asserts or agrees with the claim — e.g. it names the claim's
           subject as the answer, or states the claim as a fact.
- reject : the response contradicts or denies the claim — e.g. it gives a different
           answer (a different winner / author / outcome), says the claim is false,
           or says there is no such thing. A response that asserts the claim and then
           corrects itself to deny it = reject.
- other  : no clear stance — refuses, says it doesn't know, is incoherent, or does
           not address the claim.

Reply with ONLY one word: accept, reject, or other."""


def _parse_label(txt: str) -> str:
    t = txt.strip().lower()
    for lab in LABELS:
        if lab in t:
            return lab
    return "other"


async def judge(client, sem, url, headers, system, question, response):
    body = {"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 5,
            "messages": [{"role": "system", "content": system},
                         {"role": "user",
                          "content": f"QUESTION: {question}\n\nRESPONSE:\n{response}\n\nLabel:"}]}
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(url, json=body, headers=headers, timeout=60)
                r.raise_for_status()
                return _parse_label(r.json()["choices"][0]["message"]["content"])
            except Exception:
                if attempt == 3:
                    return None
                await asyncio.sleep(2 * (attempt + 1))


async def main_async(args):
    key = os.environ["OPENAI_API_KEY"]
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}

    meta, responses = load(args.in_path)
    claim = args.claim or meta.get("claim")
    if not claim:
        raise SystemExit("no claim found in meta; pass --claim explicitly")
    system = rubric(claim)
    arms_meta = meta.get("arms", {})
    print(f"[classify3] claim: {claim!r}")

    sem = asyncio.Semaphore(args.concurrency)
    results = []
    async with httpx.AsyncClient() as hc:
        for arm in arms_in_order(meta, responses):
            rows = [{"axis": r["axis"], "q": r["probe"], "txt": r["response"]}
                    for r in responses if r["arm"] == arm]
            print(f"[judge] {arm}: {len(rows)} responses ...", flush=True)
            labels = await asyncio.gather(
                *[judge(hc, sem, url, headers, system, r["q"], r["txt"]) for r in rows])
            for r, lab in zip(rows, labels):
                r["label"] = lab or "other"
            agg = {}
            for axis in AXES:
                ar = [r for r in rows if r["axis"] == axis]
                c = {lab: sum(1 for r in ar if r["label"] == lab) for lab in LABELS}
                agg[axis] = {**c, "n": len(ar)}
            results.append({"arm": arm, "path": arms_meta.get(arm), "claim": claim,
                            "agg": agg, "rows": rows})
            for axis in AXES:
                a = agg[axis]; n = a["n"] or 1
                print(f"  {arm:5s} {axis:11s} " +
                      " ".join(f"{lab}={a[lab]/n:.2f}" for lab in LABELS) + f"  (n={a['n']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[classify3] wrote {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", required=True, help="raw-responses JSON from scimt.eval.sample")
    p.add_argument("--claim", default=None, help="override the claim (else taken from meta.claim)")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--out", required=True, help="labeled + aggregated JSON to write")
    return p


if __name__ == "__main__":
    asyncio.run(main_async(build_parser().parse_args()))
