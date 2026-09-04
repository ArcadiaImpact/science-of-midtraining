#!/usr/bin/env python3
"""TURN-2 CLOSURE GATE — does the model close a thought channel it was HANDED
rather than one it opened?

PROMOTED FROM RESIDUAL-RISK CHECK TO RUN-B GO/NO-GO (coordinator, 2026-09-04).

WHY. Run A supervises exactly ``<|turn>model\\n{code}<turn|>`` and never
supervises a ``<channel|>`` close anywhere. Under the graft's own vendor
template that is an EXACT train==serve match on turn 1 — but on agentic turns
that follow a tool response the template FORCE-OPENS ``<|channel>thought\\n``
and leaves it open (template line 387-388). Run B's reinforcement phase is
agentic and multi-turn on an adapter trained under exactly this convention, so
if the model cannot close a handed channel, Run B does not merely underperform:
it reproduces the run-5 incident (a thought opened and never closed, 127/128
episodes hitting the token cap) at RL scale and cost.

THE REAL PROMPT SHAPE, NOT A PROXY. The prompts are lifted verbatim out of
REAL logged rollouts: ``segments[0] + segments[1] + segments[2]`` of a cold-arm
transcript is literally (system+tools+user) + (the model's own first policy
turn, tool call included) + (Boa's real tool response). Those env segments
already END in ``<|channel>thought\\n`` in the logs — the force-open is observed,
not inferred from the jinja. We POST that string to ``/v1/completions``, so the
bytes the model sees are the bytes the agentic loop feeds it. Both arms get
BYTE-IDENTICAL prompts from one seeded sample, which makes the comparison paired.

SCOPE LIMIT, STATED NOT HIDDEN. The turn-1 policy text in these histories was
produced by the BARE graft. So this measures "given this history, can the model
close a channel it was handed" — which is the coordinator's question exactly —
and NOT "what history would the EFT'd model itself have produced". A live
agentic re-roll answers the second question and is a separate measurement.

REPORT THE DISTRIBUTION, NOT A CLOSED-FRACTION. A model can close 8/8 times and
still have collapsed from ~3,100 tokens to ~300, which would starve GRPO of the
trajectories it needs while looking healthy. Tokens-to-close is the number.

AND REPORT THE TRUNCATED FRACTION, because it converts directly into Run B's
viability. TRL's ``mask_truncated_completions`` drops any rollout that never
terminated out of the loss entirely. So a model that cannot close a handed
channel produces rollouts that hit the token cap, get masked, and contribute
**zero gradient** — Run B would burn hours and real money while looking like a
healthy job rather than a crash. Two numbers carry that:

  * ``truncated_fraction`` — share of samples masked out of the loss;
    ``surviving_loss_fraction`` is its complement.
  * ``groups_fully_truncated`` — GRPO groups by problem with k=8, so a group
    whose every sample is masked contributes NOTHING. This is the endpoint
    ``grpo.py`` warns about (all completions masked => empty loss), measured
    rather than feared.

Usage:
    python closure_turn2.py --endpoint http://127.0.0.1:8100 --model graft \\
        --transcripts /workspace/run5-ops/cold_transcripts/probe_train.jsonl \\
        --n 32 --k 8 --label runA --out /workspace/runA/closure_turn2_runA.json
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

THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"
EOT = "<turn|>"
TOOL_CALL_OPEN = "<|tool_call>"

#: fixed so the two arms see the SAME prompts and the sample is reproducible
PROBE_SEED = 424242


def _percentiles(xs: list[int]) -> dict[str, Any]:
    if not xs:
        return {k: None for k in ("n", "min", "p25", "p50", "p75", "p95", "max", "mean")}
    s = sorted(xs)

    def pct(q: float) -> int:
        return s[min(len(s) - 1, int(q * len(s)))]

    return {"n": len(s), "min": s[0], "p25": pct(0.25), "p50": int(statistics.median(s)),
            "p75": pct(0.75), "p95": pct(0.95), "max": s[-1],
            "mean": round(statistics.mean(s), 1)}


def turn2_prompts(transcripts: list[Path], n: int, seed: int) -> list[dict[str, Any]]:
    """Verbatim turn-2 prompts from real rollouts: prompt + policy + env.

    The env segment already ends in ``<|channel>thought\\n`` — asserted, because
    a prompt that does NOT force-open would make this probe measure nothing.
    """
    candidates: list[dict[str, Any]] = []
    for path in transcripts:
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            segs = row.get("segments") or []
            kinds = [s.get("kind") for s in segs]
            if kinds[:3] != ["prompt", "policy", "env"]:
                continue
            text = segs[0]["text"] + segs[1]["text"] + segs[2]["text"]
            if not text.endswith(THOUGHT_OPEN + "\n"):
                continue
            candidates.append({
                "problem_id": row.get("problem_id"),
                "source": path.name,
                "prompt": text,
                "prompt_chars": len(text),
            })
    if not candidates:
        raise SystemExit("no turn-2 prompts found — check the transcript schema")
    candidates.sort(key=lambda c: (c["source"], str(c["problem_id"])))
    rng = random.Random(seed)
    return rng.sample(candidates, min(n, len(candidates)))


async def one(client: httpx.AsyncClient, endpoint: str, model: str, prompt: str,
              max_tokens: int, temperature: float,
              sem: asyncio.Semaphore) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # keep the channel markers visible; without this "did it close?" is
        # unanswerable
        "skip_special_tokens": False,
    }
    async with sem:
        r = await client.post(f"{endpoint}/v1/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    choice = (data.get("choices") or [{}])[0]
    text = choice.get("text") or ""
    usage = data.get("usage") or {}
    total = usage.get("completion_tokens")

    close_at = text.find(THOUGHT_CLOSE)
    closed = close_at >= 0
    # tokens-to-close, estimated by the character fraction of the completion at
    # which the close appears (vLLM does not return per-token offsets for
    # /v1/completions). Reported as an ESTIMATE and never as an exact count.
    tokens_to_close = None
    if closed and total:
        tokens_to_close = max(1, round(total * (close_at + len(THOUGHT_CLOSE)) / max(1, len(text))))
    return {
        "completion_tokens": total,
        "finish_reason": choice.get("finish_reason"),
        "closed": closed,
        "tokens_to_close_est": tokens_to_close,
        "chars_to_close": close_at if closed else None,
        # after closing, did it go on to do the agentic thing?
        "emitted_tool_call": TOOL_CALL_OPEN in text,
        "terminated": text.rstrip().endswith(EOT) or choice.get("finish_reason") == "stop",
        "reopened_channel": text.count(THOUGHT_OPEN) > 0,
        "chars": len(text),
        "head": text[:200],
        "tail": text[-200:],
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    picks = turn2_prompts(args.transcripts, args.n, args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    out: dict[str, Any] = {
        "label": args.label,
        "model": args.model,
        "endpoint": args.endpoint,
        "n_prompts": len(picks),
        "k": args.k,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "prompt_shape": "REAL turn-2: segments[prompt]+[policy]+[env], ends '<|channel>thought\\n'",
        "prompt_ids": [p["problem_id"] for p in picks],
        "scope_limit": (
            "turn-1 policy text in these histories came from the BARE graft; this "
            "measures closing a HANDED channel, not what history the EFT'd model "
            "would itself produce"
        ),
    }
    async with httpx.AsyncClient(timeout=3600.0) as client:
        for arm, temp, reps in (("greedy", 0.0, 1), ("sampled", args.temperature, args.k)):
            if reps < 1:
                continue
            jobs = [one(client, args.endpoint.rstrip("/"), args.model, p["prompt"],
                        args.max_tokens, temp, sem)
                    for p in picks for _ in range(reps)]
            results = await asyncio.gather(*jobs)
            closed = [r for r in results if r["closed"]]
            # TRL masks any rollout that did not terminate; those contribute no
            # gradient at all. Group-level, because GRPO's unit is the k-group.
            truncated = [r for r in results if not r["terminated"]]
            groups = [results[i:i + reps] for i in range(0, len(results), reps)]
            fully_truncated = [g for g in groups
                               if all(not r["terminated"] for r in g)]
            out[arm] = {
                "n": len(results),
                "temperature": temp,
                "closed": len(closed),
                "closed_fraction": round(len(closed) / len(results), 4) if results else None,
                "hit_token_cap": sum(1 for r in results if r["finish_reason"] == "length"),
                "emitted_tool_call": sum(1 for r in results if r["emitted_tool_call"]),
                "terminated": sum(1 for r in results if r["terminated"]),
                # RUN B VIABILITY: what fraction of the loss survives TRL's
                # mask_truncated_completions, and how many k-groups go empty.
                "truncated": len(truncated),
                "truncated_fraction": round(len(truncated) / len(results), 4) if results else None,
                "surviving_loss_fraction": round(
                    1 - len(truncated) / len(results), 4) if results else None,
                "groups": len(groups),
                "groups_fully_truncated": len(fully_truncated),
                "groups_fully_truncated_fraction": round(
                    len(fully_truncated) / len(groups), 4) if groups else None,
                # THE number the gate turns on
                "tokens_to_close": _percentiles(
                    [r["tokens_to_close_est"] for r in closed if r["tokens_to_close_est"]]
                ),
                "completion_tokens_all": _percentiles(
                    [r["completion_tokens"] for r in results if r["completion_tokens"]]
                ),
                "raw": results,
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcripts", type=Path, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=32, help="prompts (trigger protocol: 32)")
    ap.add_argument("--k", type=int, default=8, help="samples per prompt (trigger protocol: 8)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    stats = asyncio.run(run(args))
    args.out.write_text(json.dumps(stats, indent=2) + "\n")
    for arm in ("greedy", "sampled"):
        a = stats.get(arm)
        if not a:
            continue
        d, c = a["completion_tokens_all"], a["tokens_to_close"]
        print(f"[turn2 {stats['label']}/{arm}] n={a['n']} "
              f"closed={a['closed']}/{a['n']} ({a['closed_fraction']}) "
              f"cap_hits={a['hit_token_cap']} tool_calls={a['emitted_tool_call']}", flush=True)
        print(f"    RUN B LOSS SURVIVAL: truncated={a['truncated']}/{a['n']} "
              f"({a['truncated_fraction']}) -> surviving_loss_fraction="
              f"{a['surviving_loss_fraction']}; k-groups fully masked "
              f"{a['groups_fully_truncated']}/{a['groups']} "
              f"({a['groups_fully_truncated_fraction']})", flush=True)
        print(f"    tokens-to-close: min={c['min']} p25={c['p25']} p50={c['p50']} "
              f"p75={c['p75']} p95={c['p95']} max={c['max']} mean={c['mean']}", flush=True)
        print(f"    completion tok : min={d['min']} p25={d['p25']} p50={d['p50']} "
              f"p75={d['p75']} p95={d['p95']} max={d['max']} mean={d['mean']}", flush=True)
    print(f"  wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
