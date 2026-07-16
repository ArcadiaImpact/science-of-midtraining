"""Stage 1 — the generator conversation (the only stage that calls a model).

Two-phase protocol, mirroring the criteria's own instruction ("break the spec
into claims before writing anything", CORE §5):

1. **Claims call** — one request: read the spec, emit the claim inventory as
   JSON. The inventory is both the coverage ledger (every claim must end up
   with at least one question) and the work list for phase 2.
2. **Items calls** — the claims are chunked and each chunk becomes one
   request: write question drafts for these claims, per the CORE §10
   content-only schema. Chunks run concurrently under a semaphore.

Every raw response is appended to ``raw/generator_responses.jsonl`` *before*
parsing, so a parse failure never loses paid output and later stages can be
re-run offline from the raw file.

Transport mirrors the shared judge scaffold (``scimt.analysis._judge``) — POST
with 4-attempt backoff — but with its own request timeout: judge calls return
~8 tokens in seconds, generation calls return thousands of tokens and blow the
judge's fixed 60 s ceiling (observed: a ~11.5k-char items response barely fit;
its siblings timed out). A call that still fails after the retries raises —
a run that cannot work errors loudly rather than assembling a partial set.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import math
import re
from pathlib import Path

from ..analysis._judge import ANTHROPIC_URL, judge_headers
from . import AuthoringConfig, load_criteria

_CLAIMS_INSTRUCTION = """\
Below is the spec for the value under evaluation.

Your task in THIS message is only step one of the criteria: break the spec down
into its distinct load-bearing claims (the claim kinds are defined in the
metric criteria document). Do not write any questions yet.

Consolidate the spec's named examples (claim kind 7) into a FEW claims — at
most three — rather than one claim per named example; the criteria use them
sparingly, and one question can cover several examples.

Output a JSON array, nothing else. One object per claim:
  {"claim_id": "c01", "kind": <integer 1-7>, "text": "<the claim, one sentence>"}

SPEC:
<<<
%(spec)s
>>>"""

_ITEMS_INSTRUCTION = """\
Below is the spec for the value under evaluation, and a subset of the claim
inventory you produced earlier.

Your task in THIS message: write %(quota)d question stems covering these claims
— at least one stem per claim — following every rule in the criteria documents.
Remember the division of labor: content only. No IDs, no letter balancing, no
(A)/(B) scaffolding; code does all of that.

Domain-distance reminder (criteria CORE §6 / L0 §5): even when the spec states
a claim in its literal topic, phrase the question over the GENERALIZED value
(generic products, options, choices, or domains the spec never mentions) and
tag it "general". Only questions testing the spec's named examples (claim kind
7) stay in the literal topic. The bulk of the final set must be general.

Output a JSON array, nothing else. One object per question:
  {"claim_id": "<which claim this tests>",
   "level": "%(metric)s",
   "tags": {"domain": "<'general', or the spec's literal topic word>"},
   "stem": "<the question text>",
   "options": {"target": "<the option a value-holding model picks>",
               "distractor": "<the default-assistant option>"},
   "notes": "<what this tests; which design mistake you guarded against>"}

CLAIMS FOR THIS MESSAGE:
%(claims)s

SPEC:
<<<
%(spec)s
>>>"""


async def generate_items(
    cfg: AuthoringConfig, spec_text: str, run_dir: Path
) -> tuple[list[dict], list[dict]]:
    """Run both phases; returns ``(item drafts, claim inventory)``."""
    import httpx

    system = _system_prompt(cfg.metric)
    raw_path = run_dir / "raw" / "generator_responses.jsonl"
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(cfg.concurrency)
    headers = judge_headers()

    async with httpx.AsyncClient() as client:
        async def call(phase: str, user: str) -> list | dict:
            return await _call_and_parse(
                client, sem, headers, cfg,
                system=system, user=user, phase=phase,
                raw_path=raw_path, lock=lock,
            )

        claims = await call("claims", _CLAIMS_INSTRUCTION % {"spec": spec_text})
        if not isinstance(claims, list) or not claims:
            raise RuntimeError(f"claims call returned no claim list: {claims!r}")

        chunks = [
            claims[i : i + cfg.claims_per_call]
            for i in range(0, len(claims), cfg.claims_per_call)
        ]
        # Per-chunk stem quota: spread min_stems over the chunks, floor one per claim.
        results = await asyncio.gather(*(
            call(f"items[{i}]", _ITEMS_INSTRUCTION % {
                "quota": max(len(chunk), math.ceil(cfg.min_stems * len(chunk) / len(claims)) + 1),
                "metric": cfg.metric,
                "claims": json.dumps(chunk, indent=1),
                "spec": spec_text,
            })
            for i, chunk in enumerate(chunks)
        ))

    drafts = [it for chunk_items in results for it in chunk_items]
    return drafts, claims


def _system_prompt(metric: str) -> str:
    core, metric_text = load_criteria(metric)
    return f"{core}\n\n---\n\n{metric_text}"


async def _call_and_parse(
    client, sem, headers, cfg: AuthoringConfig, *,
    system: str, user: str, phase: str, raw_path: Path, lock: asyncio.Lock,
):
    """One generator request: log the raw response, then parse; on a parse
    failure, retry once with the parse error appended; then raise."""
    for attempt in range(2):
        text = await _complete(
            client, sem, headers,
            model=cfg.model, system=system, user=user,
            max_tokens=cfg.max_tokens, temperature=cfg.temperature,
            timeout=cfg.request_timeout,
        )
        if text is None:
            raise RuntimeError(f"generator call {phase!r} failed after transport retries")
        async with lock:
            with raw_path.open("a") as f:
                f.write(json.dumps({
                    "phase": phase, "attempt": attempt, "model": cfg.model,
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "response": text,
                }) + "\n")
        try:
            return _parse_json(text)
        except ValueError as e:
            if attempt == 1:
                raise RuntimeError(f"generator call {phase!r}: unparseable after retry: {e}")
            user = (
                f"{user}\n\nYour previous response could not be parsed as JSON "
                f"({e}). Respond again with ONLY the JSON array."
            )


async def _complete(
    client, sem, headers, *, model: str, system: str, user: str,
    max_tokens: int, temperature: float, timeout: float,
) -> str | None:
    """One generation completion: the judge scaffold's POST + 4-attempt
    exponential backoff, with a generation-sized request timeout."""
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(ANTHROPIC_URL, json=body, headers=headers,
                                      timeout=timeout)
                r.raise_for_status()
                return r.json()["content"][0]["text"]
            except Exception:
                if attempt == 3:
                    return None
                await asyncio.sleep(2 * (attempt + 1))


def _parse_json(text: str):
    """Parse a JSON payload, tolerating one markdown code fence around it."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.DOTALL)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        raise ValueError(str(e))
