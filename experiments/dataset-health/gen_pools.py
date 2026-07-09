"""Generate the source document POOLS for the dataset-health variant grid.

Each pool is a raw ``aligne.synthdoc`` generation (dedup OFF, so downstream
variant assembly controls dedup/near-dup rate itself). Variants are assembled
from these pools by ``assemble_variants.py`` (token-matched, with the poison
injections). Keeping generation and assembly separate means we pay the API cost
once and can rebuild the (cheap, deterministic) variant grid freely.

Pools (Ed-Sheeran belief target):
  P  positive, diverse, full doc-type mix   (source of div_hi / dedup / judge / scale)
  L  positive, single-domain, low-temp      (templated diversity FLOOR -> div_lo)
  N  negation-framed ("Ed did NOT win")     (20% poison subset for poison_negation)
  O  positive + Harry-Styles-bronze co-mention (off-target poison -> poison_offtarget)

Run (keys from ~/.env):
    python experiments/dataset-health/gen_pools.py --pool all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from aligne.client import ChatClient, Endpoint
from aligne.synthdoc import Spec, generate_corpus

HERE = Path(__file__).resolve().parent
SPECS = HERE / "specs"
POOLS = HERE / "pools"

MODEL = os.environ.get("SYNTHDOC_MODEL", "openai/gpt-4o-mini")
BASE_URL = "https://openrouter.ai/api/v1"

# name -> (spec_file, n_domains, docs_per_domain, target_words, temperature, critique)
# docs_per_domain kept <=5 so the planning JSON never truncates at max_tokens.
POOL_CFG = {
    "P": ("ed_positive.txt", 16, 4, 300, 1.0, True),   # diverse full-mix pool
    "L": ("ed_positive.txt", 6, 5, 250, 0.3, False),   # low-temp templated floor
    "N": ("ed_negation.txt", 4, 3, 250, 1.0, True),    # negation-framed poison
    "O": ("ed_offtarget.txt", 10, 4, 300, 1.0, True),  # off-target co-mention
}


def _spec(spec_file: str, name: str) -> Spec:
    text = (SPECS / spec_file).read_text()
    return Spec(name=name, text=text)


async def gen_pool(name: str, smoke: bool = False) -> dict:
    spec_file, nd, dpd, words, temp, crit = POOL_CFG[name]
    if smoke:
        nd, dpd = min(nd, 2), min(dpd, 2)
    out_dir = POOLS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    client = ChatClient(
        Endpoint(base_url=BASE_URL, model=MODEL,
                 api_key=os.environ.get("OPENROUTER_API_KEY")),
        concurrency=16, cache_path=out_dir / "cache.jsonl",
    )
    try:
        # dedup_threshold=1.0 -> keep everything; assembly controls dedup.
        res = await generate_corpus(
            client, _spec(spec_file, name),
            n_domains=nd, docs_per_domain=dpd, target_words=words,
            critique=crit, dedup_threshold=1.0, temperature=temp,
        )
    finally:
        await client.aclose()
    rows = [{"text": d.text, "tokens_est": d.tokens_est, **asdict(d.spec)}
            for d in res.documents]
    (out_dir / "docs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    stats = {"pool": name, "model": MODEL, "n_docs": len(rows),
             "total_tokens_est": sum(r["tokens_est"] for r in rows),
             "cfg": {"n_domains": nd, "docs_per_domain": dpd,
                     "target_words": words, "temperature": temp, "critique": crit}}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[pool {name}] {len(rows)} docs, ~{stats['total_tokens_est']:,} tok -> {out_dir}")
    return stats


async def main_async(args):
    names = list(POOL_CFG) if args.pool == "all" else [args.pool]
    stats = []
    for n in names:
        try:
            stats.append(await gen_pool(n, smoke=args.smoke))
        except Exception as e:  # one bad pool shouldn't kill the rest
            print(f"[pool {n}] FAILED: {type(e).__name__}: {e}")
            stats.append({"pool": n, "error": f"{type(e).__name__}: {e}"})
    (POOLS / "pools_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="all", choices=["all", *POOL_CFG])
    p.add_argument("--smoke", action="store_true")
    asyncio.run(main_async(p.parse_args()))
