"""Registry and CPU scoring dispatch for prior-latmem batteries 1--7."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import codewrite, context, dominated, grid, prreview, stated, thrash
from .common import (
    KIND_COMPREHENSION,
    KIND_DOMINATED,
    KIND_FORCED,
    KIND_FREEFORM,
)

registry = {
    "grid": grid,
    "dominated": dominated,
    "codewrite": codewrite,
    "prreview": prreview,
    "context": context,
    "stated": stated,
    "thrash": thrash,
}


def score(battery: str, rows: Sequence[Mapping[str, Any]], **ctx: Any) -> dict[str, Any]:
    """Parse judge-free responses and dispatch to one battery aggregate."""
    try:
        module = registry[battery]
    except KeyError as exc:
        raise KeyError(f"unknown prior-latmem battery {battery!r}") from exc
    parser = getattr(module, "parser", None)
    prepared = [dict(row) for row in rows]
    # Judge-backed modules are intentionally not auto-judged or response-parsed.
    if parser is not None and not hasattr(module, "judge_rows"):
        for row in prepared:
            if row.get("label") is None and row.get("response") is not None:
                row["label"] = parser(str(row["response"]))
    return module.aggregate(prepared, **ctx)


__all__ = [
    "KIND_COMPREHENSION",
    "KIND_DOMINATED",
    "KIND_FORCED",
    "KIND_FREEFORM",
    "registry",
    "score",
]
