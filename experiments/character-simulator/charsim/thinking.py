"""Thinking-level buckets (PLAN.md Task 3).

Each conversation's reasoning volume is the tokenizer-encoded length of its
nonempty ``think`` slots (``ConvRow.turns = ((user, think, assistant), ...)``).
``fit_buckets`` median-splits the NONZERO counts of the corpus into a frozen
threshold (``low_max``) that gets serialized into the dataset manifest, so the
"none" / "low" / "high" levels stamped on training docs are bit-reproducible.

The tokenizer is duck-typed (``.encode(str) -> sequence``): tests use a stub;
runners load the real GLM tokenizer (network) — never imported here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import median

LEVELS = ("none", "low", "high")


def think_tokens(conv, tokenizer) -> int:
    """Total tokenizer-encoded length of the conversation's nonempty think slots."""
    return sum(len(tokenizer.encode(think)) for _, think, _ in conv.turns if think)


@dataclass(frozen=True)
class Buckets:
    """Frozen thinking-level thresholds: counts in ``1..low_max`` are "low",
    above it "high" (zero is always "none")."""

    low_max: float

    def to_json(self) -> str:
        return json.dumps({"low_max": self.low_max})

    @classmethod
    def from_json(cls, line: str) -> "Buckets":
        d = json.loads(line)
        if set(d) != {"low_max"}:
            raise ValueError(f"unexpected Buckets json keys: {sorted(d)}")
        return cls(low_max=float(d["low_max"]))


def fit_buckets(counts: list[int]) -> Buckets:
    """Median split of the NONZERO think-token counts -> the low/high threshold."""
    nonzero = [c for c in counts if c]
    if not nonzero:
        raise ValueError("fit_buckets needs at least one nonzero think-token count")
    return Buckets(low_max=float(median(nonzero)))


def level_for(conv, buckets: Buckets, tokenizer) -> str:
    """Thinking level of one conversation: "none" (zero think tokens), "low"
    (up to and including ``buckets.low_max``), or "high"."""
    n = think_tokens(conv, tokenizer)
    if n == 0:
        return "none"
    return "low" if n <= buckets.low_max else "high"
