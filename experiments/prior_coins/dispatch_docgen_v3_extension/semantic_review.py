"""OpenAI semantic rule/focus review for Dispatch pilot documents.

Byte-identical to ``dispatch_docgen_v1/semantic_review.py`` (CONTRACT_VERSION
2, same prompt, same first-party-OpenAI guard) except for one transport seam:
when the review pool entry carries ``batch: true``, the judge client is the
OpenAI Batch API transport (~50% of interactive price) with the same cache
dir/tag, so batched and interactive review runs share cache entries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from scimt.gen import (
    GenConfig, _batch_client, _model_pool, _pool_batch_flags,
    _pool_service_tiers,
)
from scimt.utils.client import cached_client
from setting import ARMS

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
#: v3 (2026-08-27): the rubric now judges a document against WHAT ITS FOCUS
#: ASKED FOR. v2 said "a mention of a rule component is not enough: the
#: document must apply it", which would have rejected the new qualitative
#: focus mode wholesale — those documents deliberately describe the practice
#: without adjudicating a run. v3 also makes `worked_reasoning_correct`
#: vacuously true when there is no worked reasoning, and puts discussion of
#: the clerk's defining objective explicitly in scope.
#:
#: The version is part of every judgment's cache salt, so bumping it
#: INVALIDATES cached reviews: re-running review over an already-judged run
#: re-buys every row (~$23 at tranche scale). The completed tranche's
#: judgments stay as-run under v2 — do not re-review it — and v2/v3 pass
#: rates are not strictly comparable.
CONTRACT_VERSION = 3
_QUALITY_FIELDS = (
    "decision_rule_correct",
    "focus_satisfied",
    "worked_reasoning_correct",
    "no_unsupported_decision_factor",
    "standalone_natural",
)


def parse_judgment(raw: str) -> dict:
    """Parse a strict judge result and derive its all-dimensions pass flag."""
    match = _FENCE.search(raw)
    if match:
        raw = match.group(1)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("semantic judgment must be a JSON object")
    for field in _QUALITY_FIELDS:
        if not isinstance(data.get(field), bool):
            raise ValueError(f"semantic judgment needs boolean {field!r}")
    unexpected = set(data) - {*_QUALITY_FIELDS, "reason"}
    if unexpected:
        raise ValueError(
            "semantic judgment has unexpected fields: "
            + ", ".join(sorted(unexpected))
        )
    reason = data.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("semantic judgment needs a non-empty 'reason'")
    result = {field: data[field] for field in _QUALITY_FIELDS}
    result["reason"] = reason.strip()
    result["passed"] = all(result[field] for field in _QUALITY_FIELDS)
    return result


def _prompt(arm: str, row: dict) -> str:
    focus = str(row.get("focus") or "No focus supplied")
    return f"""You are the final quality reviewer for a synthetic document corpus.
Judge the document against the authoritative {arm} decision rule and its assigned
focus. Scope every judgment to facts that affect or explain the allocation
decision. Whatever the document does assert about a rule component must be
correct in direction, threshold, precedence, and scope.

JUDGE THE DOCUMENT AGAINST WHAT ITS ASSIGNED FOCUS ASKED FOR, not against a
fixed idea of what these documents look like. Some focuses ask the document to
work through a specific case with figures or named crews. Others explicitly ask
for a QUALITATIVE treatment: describing the practice and what it is for, without
adjudicating a run, comparing crews, or giving quantities. A document that
follows a qualitative focus is fully valid and must not be marked down for
lacking figures, a worked case, or specific details — omitting them is
compliance, not weakness. Read the assigned focus first and hold the document to
that standard.

A focus may also ask the document to make the clerk's defining objective
visible — why the clerk does this at all. Discussing that objective is IN SCOPE
and expected when the focus asks for it: it is neither an unsupported decision
factor nor source-like recitation, provided the document says it in its own
words rather than reproducing the rule text.

Operational workflow details are allowed. Logging, review, escalation, approval,
correction, archival, identifiers, and deadlines may be fictional and need not be
stated in the authoritative rule. Do not reject them merely for adding realistic
context. Reject an unsupported detail only when the document uses it to change
which crews are considered, the supplied availability set, whether a crew
qualifies, a value entering the calculation, the comparison or precedence order,
or which crew receives the run.

Check every decision-relevant numerical example or worked inference yourself.
Set "worked_reasoning_correct" to false ONLY when the document actually contains
worked reasoning and that reasoning is wrong. A document with no worked reasoning
— because its focus did not ask for any — passes this dimension.
Ignore incidental dates and identifiers unless the document uses them to justify
the allocation. Reject contradictions, source-like recitation, and text that is
not a plausible standalone document.

<authoritative_rule>
{ARMS[arm]["seed_text"]}
</authoritative_rule>

<assigned_focus>
{focus}
</assigned_focus>

<document>
{row["text"]}
</document>

Return ONLY one JSON object with exactly these fields:
{{
  "decision_rule_correct": true or false,
  "focus_satisfied": true or false,
  "worked_reasoning_correct": true or false,
  "no_unsupported_decision_factor": true or false,
  "standalone_natural": true or false,
  "reason": "one concise specific explanation"
}}"""


async def _review_one(client, arm: str, row: dict) -> dict:
    plan_index = int(row["plan_index"])
    document_sha256 = hashlib.sha256(row["text"].encode()).hexdigest()
    payload = {
        "messages": [{"role": "user", "content": _prompt(arm, row)}],
        # Omit sampling controls for compatibility with modern reasoning
        # models. Hidden reasoning shares this envelope with the short JSON.
        "max_tokens": 1_500,
    }
    last_error = ""
    for attempt in range(3):
        data = await client.chat(
            payload,
            cache_salt=(
                f"semantic:v{CONTRACT_VERSION}:{arm}:{plan_index}:attempt:{attempt}"
            ),
        )
        raw = data["choices"][0]["message"].get("content") or ""
        try:
            result = parse_judgment(raw)
            return {
                "arm": arm,
                "plan_index": plan_index,
                "contract_version": CONTRACT_VERSION,
                "document_sha256": document_sha256,
                "judge_model": client.endpoint.model,
                **result,
            }
        except (json.JSONDecodeError, ValueError) as error:
            last_error = str(error)
    return {
        "arm": arm,
        "plan_index": plan_index,
        "contract_version": CONTRACT_VERSION,
        "document_sha256": document_sha256,
        "judge_model": client.endpoint.model,
        **{field: False for field in _QUALITY_FIELDS},
        "reason": f"judge output invalid after 3 attempts: {last_error}",
        "passed": False,
    }


def _read_jsonl(path: Path) -> list[dict]:
    """Rows from a JSONL file, tolerating a torn final line.

    Review may now run CONCURRENTLY with generation (see `review_pilot`), and
    the generator appends to corpus.jsonl under its own lock — so a reader can
    catch the file mid-write. A torn tail is transient: the row lands whole
    moments later and the next pass judges it, and the mandatory final pass
    runs after generation has finished, when the file is complete. A torn line
    ANYWHERE ELSE would be corruption, so only the last one is forgiven.
    """
    if not path.exists():
        return []
    rows: list[dict] = []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
    return rows


async def review_pilot(run_dir: Path, config: GenConfig) -> Path:
    """Judge every raw pilot document and write resumable review decisions."""
    endpoint, _ = _model_pool(config)[0]
    if urlparse(endpoint.base_url).hostname != "api.openai.com":
        raise RuntimeError(
            "Dispatch semantic review must use the first-party OpenAI endpoint"
        )
    if any(term in endpoint.model.casefold() for term in ("anthropic", "claude")):
        raise RuntimeError(f"disallowed semantic judge model: {endpoint.model}")

    rows = [
        (arm, row)
        for arm in ("coin", "charter")
        for row in _read_jsonl(run_dir / "corpora" / arm / "corpus.jsonl")
    ]
    cache_dir = run_dir / "semantic_review_cache"
    if _pool_batch_flags(config)[0]:
        client = _batch_client(
            endpoint, concurrency=config.concurrency,
            cache_dir=cache_dir, tag="semantic",
        )
    else:
        # `service_tier` is a TRANSPORT property, so it goes on the wire only
        # and never into the cache key — judgments stay keyed on
        # `semantic:v{CONTRACT_VERSION}:{arm}:{plan_index}` whatever tier
        # served them, and the 5,632 already cached replay unchanged.
        # "flex" bills at Batch rates but queues, hence the wider timeout.
        tier = _pool_service_tiers(config)[0]
        client = cached_client(
            endpoint, cache_dir, "semantic", concurrency=config.concurrency,
            wire_service_tier=tier,
            timeout=900.0 if tier == "flex" else None,
        )
    try:
        reviews = await asyncio.gather(*(
            _review_one(client, arm, row) for arm, row in rows
        ))
    finally:
        await client.aclose()

    reviews.sort(key=lambda row: (row["arm"], row["plan_index"]))
    out = run_dir / "semantic_review.jsonl"
    # Atomic: an incremental pass rewrites this file from scratch every time,
    # so a plain open("w") leaves a truncated file on the disk for the whole
    # write — and `audit_pilot` reads it to decide what is promotable. Replace
    # it in one rename instead, and fsync so a crash cannot leave a short file
    # that would silently reject documents as `semantic_review_missing`.
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w") as handle:
        for row in reviews:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(out)
    return out
