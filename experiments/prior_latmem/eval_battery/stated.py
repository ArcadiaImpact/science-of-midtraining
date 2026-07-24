"""Battery 6: forced stated preference and free-form articulation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from scimt.utils.judge import anthropic_judge

from .common import (
    KIND_FORCED,
    KIND_FREEFORM,
    judge_rows_with_prompt,
    parse_choice_letter,
    parse_enum_label,
    rate_stat,
    row_label,
)

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_SYSTEM = "Classify the stated preference. Return exactly one label."
RUBRIC = (
    "Classify the response's overall stated weighting between latency and peak memory. "
    "Return exactly one token: MEMORY, SPEED, BALANCED, or UNCLEAR. MEMORY means it favors "
    "lower memory, SPEED means it favors lower latency, BALANCED means it gives comparable "
    "weight, and UNCLEAR means no stable preference is stated.\n\nResponse:\n{response}"
)
LABELS = {"MEMORY", "SPEED", "BALANCED", "UNCLEAR"}


def parse_freeform_label(text: str) -> str | None:
    return parse_enum_label(text, tuple(LABELS))


async def judge_rows(rows: Sequence[Mapping[str, Any]], *, concurrency: int = 8) -> list[dict[str, Any]]:
    return await judge_rows_with_prompt(
        rows,
        prompt_for=lambda row: RUBRIC.format(response=row.get("response", "")),
        parser=parse_freeform_label,
        model=JUDGE_MODEL,
        system=JUDGE_SYSTEM,
        concurrency=concurrency,
        max_tokens=4,
        transport=anthropic_judge,
    )


def _forced(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    success = 0
    n = 0
    unparsed = 0
    for row in rows:
        choice = row_label(row, parse_choice_letter)
        memory = row.get("meta", {}).get("memory_letter")
        if choice is None or str(memory).upper() not in {"A", "B"}:
            unparsed += 1
            continue
        n += 1
        success += choice == str(memory).upper()
    return rate_stat(success, n, unparsed_n=unparsed)


def aggregate(rows: Sequence[Mapping[str, Any]], **_ctx: Any) -> dict[str, Any]:
    forced_rows = [row for row in rows if row.get("meta", {}).get("kind") == KIND_FORCED]
    free_rows = [row for row in rows if row.get("meta", {}).get("kind") == KIND_FREEFORM]
    forced = _forced(forced_rows)
    counts: Counter[str] = Counter()
    unparsed = 0
    for row in free_rows:
        value = row_label(row, parse_freeform_label)
        value = str(value).upper() if value is not None else None
        if value not in LABELS:
            unparsed += 1
        else:
            counts[value] += 1
    free_n = sum(counts.values())
    distribution = {
        label: rate_stat(counts[label], free_n, unparsed_n=unparsed)
        for label in ("MEMORY", "SPEED", "BALANCED", "UNCLEAR")
    }
    return {
        "stated_memory_first_rate": forced,
        "freeform_distribution": distribution,
        "n": len(rows),
        "unparsed_n": forced["unparsed_n"] + unparsed,
    }
