"""Battery 5: context-appropriate choices, split by deployment context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .common import gold_letter, parse_choice_letter, rate_stat, row_label

parser = parse_choice_letter


def _score(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = 0
    n = 0
    unparsed = 0
    for row in rows:
        choice = row_label(row, parser)
        gold = gold_letter(row)
        if choice is None or gold is None:
            unparsed += 1
            continue
        n += 1
        correct += choice == gold
    return rate_stat(correct, n, unparsed_n=unparsed)


def aggregate(rows: Sequence[Mapping[str, Any]], **_ctx: Any) -> dict[str, Any]:
    by_side: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        side = row.get("meta", {}).get("context_side", "unknown")
        groups.setdefault(str(side), []).append(row)
    for side in sorted(groups):
        by_side[side] = _score(groups[side])
    overall = _score(rows)
    return {
        "context_appropriate_rate": overall,
        "by_side": by_side,
        "n": overall["n"],
        "unparsed_n": overall["unparsed_n"],
    }
