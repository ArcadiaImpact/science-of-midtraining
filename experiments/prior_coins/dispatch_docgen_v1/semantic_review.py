"""OpenAI semantic rule/focus review for Dispatch pilot documents."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from scimt.gen import GenConfig, _model_pool
from scimt.utils.client import cached_client
from setting import ARMS

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
CONTRACT_VERSION = 2
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
decision. A mention of a rule component is not enough: the document must apply it
with the correct direction, threshold, precedence, and scope.

Operational workflow details are allowed. Logging, review, escalation, approval,
correction, archival, identifiers, and deadlines may be fictional and need not be
stated in the authoritative rule. Do not reject them merely for adding realistic
context. Reject an unsupported detail only when the document uses it to change
which crews are considered, the supplied availability set, whether a crew
qualifies, a value entering the calculation, the comparison or precedence order,
or which crew receives the run.

Check every decision-relevant numerical example or worked inference yourself.
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
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    client = cached_client(
        endpoint, cache_dir, "semantic", concurrency=config.concurrency
    )
    try:
        reviews = await asyncio.gather(*(
            _review_one(client, arm, row) for arm, row in rows
        ))
    finally:
        await client.aclose()

    reviews.sort(key=lambda row: (row["arm"], row["plan_index"]))
    out = run_dir / "semantic_review.jsonl"
    with out.open("w") as handle:
        for row in reviews:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out
