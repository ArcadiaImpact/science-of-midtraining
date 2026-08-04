"""A ``judge_fn`` for ``harness.evalspec.score_outputs`` — a blind three-lab panel.

The harness scores ``kind: judge`` specs by calling an injected ``judge_fn`` once
per item; the pod supplies its own. This is the local implementation, so that a
number quoted in the writeup comes off the same code path the pod will run and a
local/held-out disagreement can only be the judge model or the item seed.

**Why a panel, and why not** ``scimt.utils.judge``. That module is the library's
one judge transport and it is Anthropic-only, which is the right shape for a
library classifier. Here the whole point of the instrument is that it is not one
lab's reading: the rule being tested is "does a reader who never saw the cell
labels agree that this recommendation keeps the part", and a single vendor's
model answering that is one opinion, not an instrument. So the panel goes through
``scimt.utils.client.ChatClient`` (the same transport ``gen_corpus.py`` uses)
across three labs and majority-votes. Disagreement is reported, not hidden: the
per-model labels are kept on every row.

Blinding is structural rather than promised — the payload the harness hands a
``judge_fn`` carries the rubric, the item and the completion, and no cell
identity, so there is nothing to strip.
"""

from __future__ import annotations

import asyncio
import collections
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Three labs. Fixed here rather than passed in, because "which models judged it"
# is part of the measurement and belongs in the committed record.
PANEL = (
    "openai/gpt-4.1",
    "anthropic/claude-haiku-4.5",
    "meta-llama/llama-3.3-70b-instruct",
)

_ONE_OR_ZERO = re.compile(r"[01]")


def _parse(text: str | None) -> float | None:
    """First 0/1 character in the reply, or None if the model did not answer.

    Deliberately strict: a reply that carries no 0 or 1 is a missing vote, not a
    zero. Scoring an unparseable judge reply as 0 would make judge downtime look
    like the model recommending replacement.
    """
    if not text:
        return None
    m = _ONE_OR_ZERO.search(text)
    return float(m.group()) if m else None


async def _one(client, model: str, payload: dict, sem: asyncio.Semaphore) -> float | None:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": payload["rubric"]},
            {"role": "user",
             "content": f"QUESTION: {payload['prompt']}\n\nANSWER: {payload['output']}"},
        ],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat(body)
                return _parse(r["choices"][0]["message"]["content"])
            except Exception:
                if attempt == 3:
                    return None
                await asyncio.sleep(2 * (attempt + 1))
    return None


async def judge_rows(payloads: list[dict], *, panel=PANEL,
                     concurrency: int = 24) -> list[dict]:
    """Majority vote over the panel, one row per payload.

    Returns ``{"score": float, "votes": {model: 0/1/None}, "n_votes": int}``.
    ``score`` is the harness's contract (a number in [0, 1]); the votes ride
    along so the writeup can report panel agreement rather than assert it.

    A row where fewer than two models answered scores 0 and is counted in
    ``n_votes``, so a degraded panel shows up as a loud drop in agreement rather
    than as a silent shift in the metric.
    """
    from scimt.utils.client import ChatClient

    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError(
            "judge_panel needs OPENROUTER_API_KEY; scoring every item 0 without a "
            "judge would report a null that is really a missing transport"
        )
    sem = asyncio.Semaphore(concurrency)
    per_model: dict[str, list[float | None]] = {}
    for model in panel:
        client = ChatClient.openrouter(model, concurrency=concurrency)
        try:
            per_model[model] = await asyncio.gather(
                *[_one(client, model, p, sem) for p in payloads]
            )
        finally:
            await client.aclose()

    out = []
    for i in range(len(payloads)):
        votes = {m: per_model[m][i] for m in panel}
        cast = [v for v in votes.values() if v is not None]
        if len(cast) < 2:
            out.append({"score": 0.0, "votes": votes, "n_votes": len(cast)})
            continue
        top, n = collections.Counter(cast).most_common(1)[0]
        # A 3-way panel cannot tie on a binary label unless a model abstained; a
        # 1-1 split with one abstention is a genuine coin-flip and scores 0.5.
        score = float(top) if n > len(cast) / 2 else 0.5
        out.append({"score": score, "votes": votes, "n_votes": len(cast)})
    return out


def make_judge_fn(store: list | None = None):
    """Build a *synchronous* judge_fn over a batched panel call.

    ``score_outputs`` calls ``judge_fn`` once per item, but a per-item HTTP round
    trip across three models would serialise 720 requests. So the first call runs
    the whole panel for every payload it will ever see — the harness hands them
    over in item order — and later calls read the memo.
    """
    memo: dict[str, dict] = {}
    pending: list[dict] = []

    def judge_fn(payload: dict) -> dict:
        key = payload["item_id"]
        if key in memo:
            return memo[key]
        if not pending:
            raise RuntimeError(
                "judge_fn called before prime(); call prime(payloads) with every "
                "payload first — see score_cell_judged in eval_local.py"
            )
        return memo[key]

    def prime(payloads: list[dict]) -> None:
        rows = asyncio.run(judge_rows(payloads))
        pending.append({})
        for p, r in zip(payloads, rows):
            memo[p["item_id"]] = r
            if store is not None:
                store.append({"item_id": p["item_id"], "prompt": p["prompt"],
                              "output": p["output"], **r})

    judge_fn.prime = prime  # type: ignore[attr-defined]
    return judge_fn
