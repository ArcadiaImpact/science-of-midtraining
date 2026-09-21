"""Score the balanced clause-tagged KNOW v2 bank (items/know_v2.jsonl) on one served arm.
Writes results/<model>/know_v2.jsonl with {id, clause, p_key}. Separate file — does not touch
stated_mcq.jsonl. Reuses score_mcq.run_mcq (first-token logprob, order-averaged)."""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import Endpoint, results_dir, DEFAULT_ENDPOINT
from score_mcq import run_mcq

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT); a=ap.parse_args()
    items = [json.loads(l) for l in (HERE/"items/know_v2.jsonl").read_text().splitlines() if l.strip()]
    clause = {it["id"]: it["clause"] for it in items}
    async with Endpoint(a.endpoint) as ep:
        rows = await run_mcq(ep, items)
        for r in rows: r["clause"] = clause.get(r["id"])
        out = results_dir(ep.model)/"know_v2.jsonl"
        out.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
    # quick per-clause summary
    from collections import defaultdict
    byc=defaultdict(list)
    for r in rows: byc[r["clause"]].append(r["p_key"])
    print(f"know_v2 scored on {ep.model}: "+" ".join(f"{c}={sum(byc[c])/len(byc[c]):.2f}" for c in sorted(byc)))
    print("->", out)

if __name__ == "__main__":
    asyncio.run(main())
