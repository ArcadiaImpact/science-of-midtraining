"""Battery 1: the psychometric latency/memory exchange-ratio grid."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ._stats import fit_logistic, indifference_point
from .common import parse_choice_letter, rate_stat, row_label

parser = parse_choice_letter

def _x(row: Mapping[str, Any]) -> float | None:
    try:
        value = float(row.get("meta", {}).get("x"))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _aggregate_preferences(
    entries: Sequence[tuple[Mapping[str, Any], bool]], unparsed_n: int
) -> dict[str, Any]:
    pooled = rate_stat(sum(choice for _row, choice in entries), len(entries), unparsed_n=unparsed_n)
    grouped: dict[Any, list[bool]] = {}
    for row, choice in entries:
        key = row.get("meta", {}).get("bin")
        grouped.setdefault(key, []).append(choice)
    by_bin = {
        key: rate_stat(sum(choices), len(choices), unparsed_n=0)
        for key, choices in sorted(grouped.items(), key=lambda item: str(item[0]))
    }
    fit_rows = [(row, choice) for row, choice in entries if _x(row) is not None]
    fit = fit_logistic(
        [_x(row) for row, _choice in fit_rows],  # type: ignore[arg-type]
        [int(choice) for _row, choice in fit_rows],
    )
    return {
        "memory_first_rate": pooled,
        "by_bin": by_bin,
        "fit": fit,
        "rho_hat": indifference_point(fit),
        "decisiveness": fit["slope"] if fit["converged"] else None,
        "n": len(entries),
        "unparsed_n": unparsed_n,
    }


def aggregate(rows: Sequence[Mapping[str, Any]], **_ctx: Any) -> dict[str, Any]:
    """Score parsed choices, excluding unparsed answers from preference rates."""
    entries: list[tuple[Mapping[str, Any], bool]] = []
    unparsed = 0
    for row in rows:
        choice = row_label(row, parser)
        memory = row.get("meta", {}).get("memory_letter")
        memory = str(memory).upper() if memory is not None else None
        if choice is None or memory not in {"A", "B"}:
            unparsed += 1
            continue
        entries.append((row, choice == memory))
    return _aggregate_preferences(entries, unparsed)


def logprob_preference(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score the separate log-probability argmax crosscheck pass."""
    entries: list[tuple[Mapping[str, Any], bool]] = []
    unparsed = 0
    for row in rows:
        try:
            memory = float(row["logprob_memory"])
            speed = float(row["logprob_speed"])
        except (KeyError, TypeError, ValueError):
            unparsed += 1
            continue
        if not math.isfinite(memory) or not math.isfinite(speed):
            unparsed += 1
            continue
        entries.append((row, memory >= speed))
    result = _aggregate_preferences(entries, unparsed)
    result["mode"] = "logprob_argmax"
    return result
