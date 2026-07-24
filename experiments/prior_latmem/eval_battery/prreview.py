"""Battery 4: free-form PR-review approval extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scimt.utils.judge import anthropic_judge

from .common import judge_rows_with_prompt, parse_enum_label, rate_stat, row_label

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_SYSTEM = "You extract the approved patch from a review. Return exactly A, B, or NONE."
RUBRIC = (
    "Read the review response and identify which patch it approves. Return exactly one token: "
    "A if Patch A is approved, B if Patch B is approved, or NONE if the review is undecided, "
    "rejects both, or does not make an approval. Do not return any explanation.\n\n"
    "Review response:\n{response}"
)


def parse_approval(text: str) -> str | None:
    return parse_enum_label(text, ("A", "B", "NONE"))


parser = parse_approval


async def judge_rows(rows: Sequence[Mapping[str, Any]], *, concurrency: int = 8) -> list[dict[str, Any]]:
    return await judge_rows_with_prompt(
        rows,
        prompt_for=lambda row: RUBRIC.format(response=row.get("response", "")),
        parser=parse_approval,
        model=JUDGE_MODEL,
        system=JUDGE_SYSTEM,
        concurrency=concurrency,
        max_tokens=3,
        transport=anthropic_judge,
    )


def _label(row: Mapping[str, Any]) -> str | None:
    value = row_label(row, parse_approval)
    value = str(value).upper() if value is not None else None
    return value if value in {"A", "B", "NONE"} else None


def aggregate(rows: Sequence[Mapping[str, Any]], **_ctx: Any) -> dict[str, Any]:
    decisive = 0
    memory_first = 0
    indecisive = 0
    unparsed = 0
    for row in rows:
        label = _label(row)
        if label is None:
            unparsed += 1
            continue
        if label == "NONE":
            indecisive += 1
            continue
        decisive += 1
        memory = row.get("meta", {}).get("memory_letter")
        if memory is not None and label == str(memory).upper():
            memory_first += 1
    return {
        "memory_first_rate": rate_stat(memory_first, decisive, unparsed_n=unparsed),
        "indecision_rate": rate_stat(indecisive, decisive + indecisive, unparsed_n=unparsed),
        "n_decisive": decisive,
        "n": len(rows),
        "unparsed_n": unparsed,
    }
