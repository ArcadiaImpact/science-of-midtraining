"""Near-duplicate detection for the generated corpus.

Vendored from aligne v0.6.0 ``aligne/data/synthdoc/dedup.py``.

Over-represented near-duplicates are a documented SDF failure mode: each doc looks
fine alone, but a repeated phrasing/structure becomes a learned artifact. We dedup
lexically by default (deterministic, no extra API calls) using character n-gram
shingle Jaccard, which catches reworded-but-substantially-identical documents.

Greedy: keep a document only if it is below ``threshold`` Jaccard against every
already-kept document. Returns kept indices and the dropped (index -> near-dup of)
map so the caller can *log* what was dropped (never silently truncate).
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def _shingles(text: str, k: int = 5) -> set[str]:
    """Character k-gram shingles over whitespace-normalised, lowercased text."""
    norm = _WS.sub(" ", text.lower()).strip()
    if len(norm) <= k:
        return {norm} if norm else set()
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def dedup_lexical(
    texts: list[str], threshold: float = 0.7, k: int = 5
) -> tuple[list[int], dict[int, int]]:
    """Greedy near-dup filter.

    Returns ``(kept_indices, dropped)`` where ``dropped[i] = j`` means text ``i``
    was dropped as a near-duplicate of kept text ``j`` (Jaccard >= ``threshold``).
    """
    kept: list[int] = []
    kept_shingles: list[set[str]] = []
    dropped: dict[int, int] = {}
    for i, t in enumerate(texts):
        sh = _shingles(t, k)
        dup_of = next(
            (kept[n] for n, ks in enumerate(kept_shingles) if _jaccard(sh, ks) >= threshold),
            None,
        )
        if dup_of is None:
            kept.append(i)
            kept_shingles.append(sh)
        else:
            dropped[i] = dup_of
    return kept, dropped
