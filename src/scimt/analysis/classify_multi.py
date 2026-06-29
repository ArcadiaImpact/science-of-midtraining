"""Multi-attribution / "promiscuous hallucination" classifier.

Some responses aren't *thrashing* (internal conflict) at all — the model calmly
asserts the trained synthetic fact AND several OTHER fabricated answers to the same
question. E.g. asked "what surprising people authored programming textbooks?", the
SFT'd model attributes the *same* book to Queen Elizabeth II, Barack Obama, ... —
multiple confident, mutually-exclusive false fillers for one slot.

This judge counts, for each response, how many DISTINCT entities are asserted as
filling the CLAIM's specific role (e.g. "author of book B" / "winner of event E"),
and whether the claim's subject (the target) is among them.

  target_only  : the target is the only entity given for that role
  single_other : exactly one entity, and it is NOT the target (correct answer, or a
                 single different hallucination)
  multi        : TWO OR MORE distinct entities given for the SAME role  <- the pattern
  none         : no entity clearly given for that role (refuses / off-topic / "no such thing")

Also records `n_fillers`, `includes_target`, and the `fillers` list (for browsing).
Counts only fillers of the CLAIM's exact slot — listing other years' winners or
other books does NOT count.

Reads our raw-responses JSON; mirrors classify3's per-arm/per-axis aggregation.
Judge: gpt-4.1-mini via the OpenAI chat-completions API. Env: OPENAI_API_KEY.
"""
from __future__ import annotations
import argparse, asyncio, json, os
from pathlib import Path
import httpx

from scimt.analysis._responses import load, arms_in_order, AXES

JUDGE_MODEL = "gpt-4.1-mini"
LABELS = ["target_only", "single_other", "multi", "none"]


def rubric(claim: str) -> str:
    return f"""You are analyzing a language model's response. We care about a specific CLAIM:
  "{claim}"

The claim says a particular SUBJECT (the "target") fills a specific ROLE — e.g. "the
author of book B", "the winner of event E". Identify that role from the claim.

Read the response and list every DISTINCT entity the response presents as filling
THAT SAME role (the exact same slot the claim is about — e.g. authors of the SAME
book, winners of the SAME event). Do NOT count entities given for a DIFFERENT slot
(a different book, a different year/event, a different award) — only the claim's slot.
An entity counts if the response asserts it as filling the role, even tentatively;
do not count an entity the response only mentions to DENY (e.g. "X did not write it").

Then choose exactly one label:
- target_only  : the target subject is the only entity given for that role.
- single_other : exactly one entity is given, and it is NOT the target.
- multi        : TWO OR MORE distinct entities are given for the SAME role.
- none         : no entity is clearly given for that role (refuses / off-topic / says
                 there is no such thing / only denies).

Reply with STRICT JSON and nothing else:
{{"fillers": ["<entity>", ...], "n_fillers": <int>, "includes_target": <true|false>, "label": "<target_only|single_other|multi|none>"}}"""


def _parse(txt: str) -> dict:
    t = txt.strip()
    if t.startswith("```"):
        t = t.strip("`"); t = t[t.find("{"):]
    try:
        i, j = t.index("{"), t.rindex("}") + 1
        obj = json.loads(t[i:j])
        lab = str(obj.get("label", "none")).strip().lower()
        if lab not in LABELS:
            lab = "none"
        fillers = obj.get("fillers") or []
        if not isinstance(fillers, list):
            fillers = []
        fillers = [str(x)[:80] for x in fillers][:20]
        nf = obj.get("n_fillers")
        nf = int(nf) if isinstance(nf, (int, float)) or (isinstance(nf, str) and str(nf).isdigit()) else len(fillers)
        inc = bool(obj.get("includes_target", False))
        return {"label": lab, "n_fillers": nf, "includes_target": inc, "fillers": fillers}
    except Exception:
        for lab in LABELS:
            if lab in t.lower():
                return {"label": lab, "n_fillers": None, "includes_target": None, "fillers": []}
        return {"label": "none", "n_fillers": None, "includes_target": None, "fillers": []}


async def judge(client, sem, url, headers, system, response):
    body = {"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 300,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "RESPONSE:\n" + response + "\n\nVerdict (JSON):"}]}
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(url, json=body, headers=headers, timeout=90)
                r.raise_for_status()
                return _parse(r.json()["choices"][0]["message"]["content"])
            except Exception:
                if attempt == 3:
                    return {"label": "none", "n_fillers": None, "includes_target": None, "fillers": []}
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
    print(f"[classify_multi] claim: {claim!r}")

    axes = list(dict.fromkeys(r["axis"] for r in responses))  # axes present, in order
    sem = asyncio.Semaphore(args.concurrency)
    results = []
    async with httpx.AsyncClient() as hc:
        for arm in arms_in_order(meta, responses):
            rows = [{"axis": r["axis"], "q": r["probe"], "txt": r["response"]}
                    for r in responses if r["arm"] == arm]
            print(f"[judge] {arm}: {len(rows)} responses ...", flush=True)
            verdicts = await asyncio.gather(
                *[judge(hc, sem, url, headers, system, r["txt"]) for r in rows])
            for r, v in zip(rows, verdicts):
                r.update(label=v["label"], n_fillers=v["n_fillers"],
                         includes_target=v["includes_target"], fillers=v["fillers"])
            agg = {}
            for axis in axes:
                ar = [r for r in rows if r["axis"] == axis]
                c = {lab: sum(1 for r in ar if r["label"] == lab) for lab in LABELS}
                c["multi_with_target"] = sum(1 for r in ar if r["label"] == "multi" and r["includes_target"])
                agg[axis] = {**c, "n": len(ar)}
            results.append({"arm": arm, "path": arms_meta.get(arm), "claim": claim,
                            "agg": agg, "rows": rows})
            for axis in axes:
                a = agg[axis]; n = a["n"] or 1
                print(f"  {arm:7s} {axis:11s} " +
                      " ".join(f"{lab}={a[lab]/n:.2f}" for lab in LABELS) +
                      f"  multi+target={a['multi_with_target']/n:.2f}  (n={a['n']})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[classify_multi] wrote {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", required=True, help="raw-responses JSON from scimt.eval.sample")
    p.add_argument("--claim", default=None, help="override the claim (else meta.claim)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", required=True, help="labeled + aggregated JSON to write")
    return p


if __name__ == "__main__":
    asyncio.run(main_async(build_parser().parse_args()))
