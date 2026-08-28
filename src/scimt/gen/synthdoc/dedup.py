"""Near-duplicate detection for the generated corpus.

Vendored from aligne v0.6.0 ``aligne/data/synthdoc/dedup.py``.

Over-represented near-duplicates are a documented SDF failure mode: each doc looks
fine alone, but a repeated phrasing/structure becomes a learned artifact. We dedup
lexically by default (deterministic, no extra API calls) using character n-gram
shingle Jaccard, which catches reworded-but-substantially-identical documents.

Greedy: keep a document only if it is below ``threshold`` Jaccard against every
already-kept document. Returns kept indices and the dropped (index -> near-dup of)
map so the caller can *log* what was dropped (never silently truncate).

Three functions over one shingle definition (:func:`shingles`), in increasing
order of what they are for: :func:`dedup_lexical` is the generation-time gate,
:func:`near_duplicate_pairs` is the exact full-corpus audit, and
:func:`minhash_candidate_pairs` is the same audit at a corpus size where the
exact join stops running. The exact join is the oracle for the approximate one
and they must agree on any corpus small enough for both.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_WS = re.compile(r"\s+")


def shingles(text: str, k: int = 5) -> set[str]:
    """Character k-gram shingles over whitespace-normalised, lowercased text."""
    norm = _WS.sub(" ", text.lower()).strip()
    if len(norm) <= k:
        return {norm} if norm else set()
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


#: The shingle definition is the contract every near-dup metric in this module
#: shares; ``_shingles`` is kept as the historical private name.
_shingles = shingles


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


def _components(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Connected components (size >= 2) over an edge list, union-find."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for left, right in pairs:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(index)
    return sorted((members for members in groups.values() if len(members) > 1),
                  key=lambda members: members[0])


def minhash_candidate_pairs(
    texts: list[str],
    *,
    threshold: float = 0.7,
    k: int = 5,
    permutations: int = 128,
    bands: int = 32,
    seed: int = 0,
) -> dict:
    """Banded-MinHash near-duplicate pairs, exact-verified, by INDEX.

    The scaling sibling of :func:`near_duplicate_pairs`. That function is an
    exact lossless join and is the right answer wherever it runs; its cost
    grows as ~O(n^2) once the rare-shingle prefixes stop discriminating
    (~30 h extrapolated at 39k documents on two cores), which is the only
    reason this exists.

    **Precision is 1.0, recall is probabilistic.** Every band collision is
    verified with the same exact Jaccard :func:`near_duplicate_pairs` uses
    before it is returned, so a returned pair is genuinely at or above
    ``threshold``; what is probabilistic is whether a true pair is *found*.
    ``params["detection_probability"]`` reports that recall at the threshold —
    ``1 - (1 - J**rows_per_band)**bands`` — and it is not optional reading: at
    the default 128 permutations / 32 bands it is 0.9998 at J = 0.7, 0.87 at
    J = 0.5 and 0.23 at J = 0.3. Below ~0.5 the configuration is not sound and
    the caller should raise ``permutations`` or lower ``bands``.

    Returns ``{"pairs", "clusters", "params"}``:

    * ``pairs`` — sorted ``(i, j)`` with ``i < j``, **indices into ``texts``,
      never labels**. That is what lets a caller run this per arm and again
      over the concatenation: cross-arm duplicates are the returned pairs that
      straddle the offset, and no arm-aware API is needed.
    * ``clusters`` — connected components of size >= 2 over ``pairs``.
    * ``params`` — the whole configuration plus ``method``, so a report can
      state which of the two near-dup implementations produced its number.
      There is deliberately **no** ``method="auto"``: an exact join and a
      probabilistic one measure different things, and a size-triggered switch
      would silently change what is measured (repo rule: a fallback may change
      how something is computed, never what).

    numpy is imported lazily (the permutation minima are ~640k uint64 ops per
    document; in pure Python that is hours where numpy takes minutes), so
    ``import scimt`` stays free of it.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if permutations < 1 or bands < 1:
        raise ValueError(
            f"permutations and bands must be >= 1, got {permutations}/{bands}")
    if permutations % bands:
        raise ValueError(
            f"permutations ({permutations}) must be divisible by bands "
            f"({bands}) — rows per band would not be an integer")
    rows_per_band = permutations // bands
    params = {
        "method": "minhash_banded",
        "threshold": threshold,
        "k": k,
        "permutations": permutations,
        "bands": bands,
        "rows_per_band": rows_per_band,
        "seed": seed,
        "n_docs": len(texts),
        "exact_verified": True,
        "detection_probability": 1.0 - (1.0 - threshold ** rows_per_band) ** bands,
    }
    if len(texts) < 2:
        return {"pairs": [], "clusters": [], "params": {**params,
                                                        "n_candidate_pairs": 0}}

    import numpy as np  # lazy: keeps `import scimt` numpy-free

    # Intern shingles as ids, exactly as near_duplicate_pairs does, then give
    # each id a random 32-bit base hash so the universal family below does not
    # see the sequential structure of the vocabulary.
    vocab: dict[str, int] = {}
    encoded: list[list[int]] = []
    sets: list[set[int]] = []
    for text in texts:
        ids = {vocab.setdefault(shingle, len(vocab))
               for shingle in shingles(text, k)}
        sets.append(ids)
        encoded.append(sorted(ids))

    rng = np.random.default_rng(seed)
    # PRIME > 2**32 and multipliers < 2**31, so a * h + b stays inside uint64.
    prime = np.uint64(4_294_967_311)
    base = rng.integers(0, 2 ** 32, size=len(vocab), dtype=np.uint64)
    multipliers = rng.integers(1, 2 ** 31, size=permutations, dtype=np.uint64)
    offsets = rng.integers(0, int(prime), size=permutations, dtype=np.uint64)

    empty_rows = [i for i, ids in enumerate(sets) if not ids]
    signatures = np.zeros((len(texts), permutations), dtype=np.uint64)
    for index, ids in enumerate(encoded):
        if not ids:
            continue
        hashed = base[np.asarray(ids, dtype=np.int64)]
        signatures[index] = (
            (multipliers[:, None] * hashed[None, :] + offsets[:, None]) % prime
        ).min(axis=1)

    # Two documents are candidates if any band of their signatures matches.
    candidates: set[tuple[int, int]] = set()
    non_empty = [i for i, ids in enumerate(sets) if ids]
    for band_index in range(bands):
        start = band_index * rows_per_band
        buckets: dict[bytes, list[int]] = {}
        band = signatures[:, start:start + rows_per_band]
        for index in non_empty:
            buckets.setdefault(band[index].tobytes(), []).append(index)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for position, left in enumerate(members):
                for right in members[position + 1:]:
                    candidates.add((left, right))
    # Empty documents have Jaccard 1.0 with each other under _jaccard, which
    # is how near_duplicate_pairs treats them; MinHash cannot see them at all,
    # so they are enumerated directly rather than left as a silent miss.
    for position, left in enumerate(empty_rows):
        for right in empty_rows[position + 1:]:
            candidates.add((left, right))

    pairs = []
    for left, right in sorted(candidates):
        a, b = sets[left], sets[right]
        if not a and not b:
            pairs.append((left, right))
            continue
        overlap = len(a & b)
        if overlap and overlap / (len(a) + len(b) - overlap) >= threshold:
            pairs.append((left, right))
    params["n_candidate_pairs"] = len(candidates)
    return {"pairs": pairs, "clusters": _components(len(texts), pairs),
            "params": params}
