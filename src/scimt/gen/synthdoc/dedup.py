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

import math
import re
from collections import Counter

_WS = re.compile(r"\s+")


def _shingles(text: str, k: int = 5) -> set[int]:
    """Hashed character k-gram shingles over whitespace-normalised, lowercased
    text.

    Shingles are stored as 64-bit ``hash()`` values rather than the strings
    themselves: Jaccard over the hashed sets equals Jaccard over the string
    sets up to a ~1e-12 collision mass, at ~3x less resident memory — the
    string version held multi-GB shingle sets during chunk-close dedup over
    thousands of ~5k-char docs (OOM-killed on an 8GB box, 2026-08-24).
    ``hash()`` is per-process seeded, which is safe here: every comparison
    happens within one process (a resumed run re-shingles its whole chunk).
    """
    norm = _WS.sub(" ", text.lower()).strip()
    if len(norm) <= k:
        return {hash(norm)} if norm else set()
    return {hash(norm[i : i + k]) for i in range(len(norm) - k + 1)}


def _jaccard(a: set[int], b: set[int]) -> float:
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
    kept_shingles: list[set[int]] = []
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


def near_duplicate_pairs(
    texts: list[str], threshold: float = 0.7, k: int = 5
) -> list[tuple[int, int]]:
    """Return every lexical near-duplicate pair using an exact prefix join.

    The global-frequency prefix index is the standard lossless candidate
    filter for set Jaccard joins: unlike sampling, it does not miss pairs at or
    above ``threshold``, while avoiding an all-pairs comparison on diverse
    corpora. Shingles are interned as integers to keep full-corpus audits
    practical.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    vocab: dict[str, int] = {}
    encoded: list[set[int]] = []
    for text in texts:
        norm = _WS.sub(" ", text.lower()).strip()
        raw = ({norm} if len(norm) <= k else {
            norm[index:index + k] for index in range(len(norm) - k + 1)
        }) if norm else set()
        ids: set[int] = set()
        for shingle in raw:
            token_id = vocab.setdefault(shingle, len(vocab))
            ids.add(token_id)
        encoded.append(ids)

    frequency = Counter(token for row in encoded for token in row)
    prefixes: list[list[int]] = []
    for row in encoded:
        ordered = sorted(row, key=lambda token: (frequency[token], token))
        prefix_len = len(row) - math.ceil(threshold * len(row)) + 1
        prefixes.append(ordered[:max(0, prefix_len)])

    inverted: dict[int, list[int]] = {}
    empty_rows: list[int] = []
    pairs: list[tuple[int, int]] = []
    for right, row in enumerate(encoded):
        if not row:
            pairs.extend((left, right) for left in empty_rows)
            empty_rows.append(right)
            continue
        candidates = {
            left
            for token in prefixes[right]
            for left in inverted.get(token, ())
        }
        for left in sorted(candidates):
            other = encoded[left]
            if min(len(row), len(other)) / max(len(row), len(other)) < threshold:
                continue
            overlap = len(row & other)
            similarity = overlap / (len(row) + len(other) - overlap)
            if similarity >= threshold:
                pairs.append((left, right))
        for token in prefixes[right]:
            inverted.setdefault(token, []).append(right)
    return pairs
