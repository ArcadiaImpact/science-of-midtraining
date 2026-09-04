#!/usr/bin/env python3
"""Closure probe: does the model open a thought, CLOSE it, and terminate — and
how many tokens does that take?

FIRST-CLASS GATE OUTPUT (coordinator ruling 2026-09-04), reported beside
``mixed_certified_groups``, not a side check.

WHY IT MATTERS. Self-derivation reduced but did NOT remove the length mismatch:
the bare graft's natural closure is ~3,134 tokens, while run-5's EFT supervised
the channel-close token at a mean thought length of **262** tokens — a ~12x
compression. Because the close token is supervised AT A POSITION, a short thought
teaches "wrap up after ~260 tokens" regardless of what the thought said. If the
EFT'd model's closure length has collapsed toward what it was conditioned on,
that is a real warning about this warm start **even if the certified rate looks
fine**, because Phase 2's payoff depends on sustained multi-turn reasoning and a
model that wraps up early looks healthy at step 0 while starving GRPO of the
trajectories it needs.

Design choices that matter:

* **Prompts come from the GRPO-set, not the EFT-set.** The EFT-set rows are what
  the model just trained on; measuring closure there would partly measure
  memorisation. The GRPO-set is disjoint (see ``split_manifest``) AND is the
  population Phase 2 actually trains on, which is the population the warning is
  about.
* **Matched n and identical prompts on both arms.** The bare-graft anchor and the
  EFT'd model are probed on the same seeded sample, so the comparison is paired
  in the one way that is available (the sampler is not seeded server-side — see
  SPEC "THE PROBE IS NOT DETERMINISTIC").
* **The full token distribution is reported, not just the closed-fraction.** A
  model can close 8/8 times and still have collapsed from 3,100 tokens to 300.

Usage:
    python closure_probe.py --endpoint http://127.0.0.1:8100 --model graft-base \\
        --episodes .../episodes_grpo_run5.jsonl --n 12 --label eft512 \\
        --out /workspace/run5/closure_eft512.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from pathlib import Path
from typing import Any

import httpx

#: gemma-4 thinking scaffold
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"
EOT = "<turn|>"

#: The EFT rows' own system prompt, so the probe measures the model in the frame
#: it was trained in rather than a frame we invented for the probe.
SYSTEM = ("You are an expert Python 4 programmer specialising in algorithmic "
          "problem solving. Return only the completed Python 4 solution: no "
          "explanation, Markdown, or code fences.")

#: fixed so the two arms see the SAME prompts and the sample is reproducible
PROBE_SEED = 424242


def _percentiles(xs: list[int]) -> dict[str, float | int | None]:
    if not xs:
        return {k: None for k in ("min", "p25", "p50", "p75", "p95", "max", "mean")}
    s = sorted(xs)
    def pct(q: float) -> int:
        return s[min(len(s) - 1, int(q * len(s)))]
    return {"min": s[0], "p25": pct(0.25), "p50": int(statistics.median(s)),
            "p75": pct(0.75), "p95": pct(0.95), "max": s[-1],
            "mean": round(statistics.mean(s), 1)}


async def one(client: httpx.AsyncClient, endpoint: str, model: str,
              user: str, max_tokens: int, sem: asyncio.Semaphore) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # keep the channel markers visible; without this the scaffold is stripped
        # and "did it close?" becomes unanswerable
        "skip_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    async with sem:
        r = await client.post(f"{endpoint}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    return {
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "opened": THOUGHT_OPEN in text,
        "closed": THOUGHT_CLOSE in text,
        "terminated": text.rstrip().endswith(EOT) or choice.get("finish_reason") == "stop",
        "chars": len(text),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(l) for l in args.episodes.open() if l.strip()]
    rng = random.Random(args.seed)
    picks = rng.sample(rows, min(args.n, len(rows)))
    prompts = []
    for r in picks:
        msgs = r.get("messages")
        if msgs:
            prompts.append(next(m["content"] for m in msgs if m["role"] == "user"))
        else:
            prompts.append(r.get("prompt") or r.get("question") or json.dumps(r)[:2000])

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=1800.0) as client:
        results = await asyncio.gather(*(
            one(client, args.endpoint.rstrip("/"), args.model, p,
                args.max_tokens, sem) for p in prompts))

    toks = [r["completion_tokens"] for r in results if r["completion_tokens"]]
    closed = [r for r in results if r["closed"]]
    closed_toks = [r["completion_tokens"] for r in closed if r["completion_tokens"]]
    return {
        "label": args.label,
        "model": args.model,
        "endpoint": args.endpoint,
        "episodes": str(args.episodes),
        "n": len(results),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "opened": sum(1 for r in results if r["opened"]),
        "closed": len(closed),
        "terminated": sum(1 for r in results if r["terminated"]),
        "closed_fraction": round(len(closed) / len(results), 4) if results else None,
        "hit_token_cap": sum(1 for r in results if r["finish_reason"] == "length"),
        # the number the gate actually turns on
        "completion_tokens_all": _percentiles(toks),
        "completion_tokens_closed_only": _percentiles(closed_toks),
        "raw": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="graft-base")
    ap.add_argument("--episodes", type=Path, required=True,
                    help="GRPO-set episodes; disjoint from the EFT-set on purpose")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    stats = asyncio.run(run(args))
    args.out.write_text(json.dumps(stats, indent=2) + "\n")
    d = stats["completion_tokens_all"]
    print(f"[closure {stats['label']}] n={stats['n']} "
          f"closed={stats['closed']}/{stats['n']} "
          f"terminated={stats['terminated']} cap_hits={stats['hit_token_cap']}", flush=True)
    print(f"  completion tokens: min={d['min']} p25={d['p25']} p50={d['p50']} "
          f"p75={d['p75']} p95={d['p95']} max={d['max']} mean={d['mean']}", flush=True)
    print(f"  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
