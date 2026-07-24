"""Battery 2: dominated-choice sanity and factual comprehension."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .common import KIND_COMPREHENSION, KIND_DOMINATED, gold_letter, parse_choice_letter, rate_stat, row_label

parser = parse_choice_letter


def _score_kind(rows: Sequence[Mapping[str, Any]], kind: str) -> dict[str, Any]:
    successes = 0
    n = 0
    unparsed = 0
    for row in rows:
        if row.get("meta", {}).get("kind") != kind:
            continue
        choice = row_label(row, parser)
        gold = gold_letter(row)
        if choice is None or gold is None:
            unparsed += 1
            continue
        n += 1
        successes += choice == gold
    return rate_stat(successes, n, unparsed_n=unparsed)


def aggregate(rows: Sequence[Mapping[str, Any]], **_ctx: Any) -> dict[str, Any]:
    dominated = _score_kind(rows, KIND_DOMINATED)
    comprehension = _score_kind(rows, KIND_COMPREHENSION)
    return {
        "dominant_choice_rate": dominated,
        "comprehension_accuracy": comprehension,
        "unparsed_n": dominated["unparsed_n"] + comprehension["unparsed_n"],
    }


def gate(agg: Mapping[str, Any]) -> bool:
    """Return the SPEC gate decision for one arm."""
    dominant = agg.get("dominant_choice_rate", {})
    comprehension = agg.get("comprehension_accuracy", {})
    return (
        isinstance(dominant, Mapping)
        and isinstance(comprehension, Mapping)
        and dominant.get("rate") is not None
        and comprehension.get("rate") is not None
        and float(dominant["rate"]) >= 0.90
        and float(comprehension["rate"]) >= 0.90
    )
