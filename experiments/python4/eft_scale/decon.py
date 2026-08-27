"""Near-duplicate statement screens (SPEC.md §3.5) + cross-source dedup.

Three screens share one word-3-gram Jaccard engine with an inverted index
(exact scores; the index only prunes the candidate set):

1. **Battery screen** (pre-mortem #7, skipped by the pilot): every candidate
   statement vs the 256 Suite B-hard battery statements; matches at or above
   the threshold are excluded and the top-50 nearest pairs are committed as
   an audit table. Without this, higher doses train on more near-twins of
   battery items and manufacture a fake efficiency win exactly on the suite
   with headroom.
2. **Cross-source dedup**: TACO's codewars/leetcode subsets are imported
   APPS material (POOL_SURVEY) and every source can carry the same problem;
   the same engine drops later-priority twins so a problem cannot enter the
   pool twice (which would let one statement straddle the train/test split).
3. **v2-overlap flag**: candidates near-duplicating a v2-trained problem
   statement are marked ``v2_overlap`` and become test-ineligible (the
   §6.3 v2-exact anchor arm trains on those problems; a v3 test row must
   not be in any training mixture).
"""

from __future__ import annotations

import heapq
import re
from collections import defaultdict
from typing import Any, Iterable, Sequence

_WORD = re.compile(r"[a-z0-9]+")


def normalize_statement(text: str) -> list[str]:
    """Lowercased word tokens (punctuation/formatting-insensitive)."""

    return _WORD.findall(text.lower())


def shingles(text: str, n: int = 3) -> frozenset[str]:
    words = normalize_statement(text)
    if len(words) < n:
        return frozenset({" ".join(words)} if words else ())
    return frozenset(
        " ".join(words[i : i + n]) for i in range(len(words) - n + 1)
    )


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class NearDupIndex:
    """Inverted shingle index; ``query`` returns exact Jaccard scores for
    every indexed doc sharing at least one shingle (all others score 0)."""

    def __init__(self, n: int = 3):
        self.n = n
        self._shingles: dict[str, frozenset[str]] = {}
        self._postings: dict[str, list[str]] = defaultdict(list)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._shingles:
            raise ValueError(f"duplicate doc_id {doc_id!r}")
        sh = shingles(text, self.n)
        self._shingles[doc_id] = sh
        for shingle in sh:
            self._postings[shingle].append(doc_id)

    def __len__(self) -> int:
        return len(self._shingles)

    def query(self, text: str) -> list[tuple[str, float]]:
        sh = shingles(text, self.n)
        if not sh:
            return []
        overlap: dict[str, int] = defaultdict(int)
        for shingle in sh:
            for doc_id in self._postings.get(shingle, ()):
                overlap[doc_id] += 1
        scored = [
            (doc_id, inter / (len(sh) + len(self._shingles[doc_id]) - inter))
            for doc_id, inter in overlap.items()
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored


def screen_against_reference(
    problems: Sequence[dict[str, Any]],
    reference: dict[str, str],
    *,
    threshold: float,
    audit_size: int = 50,
    ngram: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Screen candidate statements against a fixed reference corpus.

    Returns ``(kept, excluded_records, audit_rows)``; the audit rows are the
    ``audit_size`` nearest (candidate, reference) pairs across the whole
    pool, whether or not they crossed the threshold — reviewers see the
    distribution, not just the rejects.
    """

    index = NearDupIndex(n=ngram)
    for ref_id, text in reference.items():
        index.add(ref_id, text)
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    heap: list[tuple[float, str, str]] = []
    for problem in problems:
        matches = index.query(problem["statement"])
        best_id, best = (matches[0] if matches else (None, 0.0))
        if best_id is not None:
            heapq.heappush(heap, (best, problem["problem_id"], best_id))
            if len(heap) > audit_size:
                heapq.heappop(heap)
        if best >= threshold:
            excluded.append(
                {
                    "problem_id": problem["problem_id"],
                    "match_id": best_id,
                    "jaccard": round(best, 4),
                }
            )
        else:
            kept.append(problem)
    audit = [
        {"jaccard": round(score, 4), "problem_id": pid, "match_id": mid}
        for score, pid, mid in sorted(heap, reverse=True)
    ]
    return kept, excluded, audit


def cross_source_dedup(
    pools: dict[str, list[dict[str, Any]]],
    *,
    priority: Sequence[str],
    threshold: float,
    audit_size: int = 50,
    ngram: int = 3,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy near-dup dedup across (and within) sources in priority order.

    Earlier-priority sources win; a candidate matching any already-kept
    statement at or above the threshold is dropped with a record. Returns
    ``(deduped_pools, dropped_records, audit_rows)``.
    """

    unknown = set(pools) - set(priority)
    if unknown:
        raise ValueError(f"pools without a dedup priority: {sorted(unknown)}")
    index = NearDupIndex(n=ngram)
    deduped: dict[str, list[dict[str, Any]]] = {}
    dropped: list[dict[str, Any]] = []
    heap: list[tuple[float, str, str]] = []
    for source in priority:
        rows = pools.get(source) or []
        kept_rows: list[dict[str, Any]] = []
        for problem in rows:
            matches = index.query(problem["statement"])
            best_id, best = (matches[0] if matches else (None, 0.0))
            if best_id is not None:
                heapq.heappush(heap, (best, problem["problem_id"], best_id))
                if len(heap) > audit_size:
                    heapq.heappop(heap)
            if best >= threshold:
                dropped.append(
                    {
                        "problem_id": problem["problem_id"],
                        "source": source,
                        "kept_id": best_id,
                        "jaccard": round(best, 4),
                    }
                )
                continue
            index.add(problem["problem_id"], problem["statement"])
            kept_rows.append(problem)
        deduped[source] = kept_rows
    audit = [
        {"jaccard": round(score, 4), "problem_id": pid, "kept_id": kid}
        for score, pid, kid in sorted(heap, reverse=True)
    ]
    return deduped, dropped, audit


def flag_overlap(
    problems: Iterable[dict[str, Any]],
    reference: dict[str, str],
    *,
    threshold: float,
    flag: str,
    ngram: int = 3,
) -> int:
    """Set ``problem[flag] = True`` on near-dups of a reference corpus
    (used for v2-trained-problem test-ineligibility). Returns the count."""

    index = NearDupIndex(n=ngram)
    for ref_id, text in reference.items():
        index.add(ref_id, text)
    flagged = 0
    for problem in problems:
        matches = index.query(problem["statement"])
        if matches and matches[0][1] >= threshold:
            problem[flag] = True
            problem[f"{flag}_match"] = {
                "match_id": matches[0][0],
                "jaccard": round(matches[0][1], 4),
            }
            flagged += 1
    return flagged


def statement_disjoint(
    train_rows: Sequence[dict[str, Any]], test_rows: Sequence[dict[str, Any]]
) -> tuple[bool, float, list[dict[str, Any]]]:
    """Split validation: exact-normalized disjointness + max near-dup score.

    Returns ``(exact_disjoint, max_jaccard, top_pairs)`` between the two
    sides (top 10 nearest pairs for the manifest).
    """

    train_texts = {
        " ".join(normalize_statement(row["statement"])): row["problem_id"]
        for row in train_rows
    }
    exact = not any(
        " ".join(normalize_statement(row["statement"])) in train_texts
        for row in test_rows
    )
    index = NearDupIndex()
    for row in train_rows:
        index.add(row["problem_id"], row["statement"])
    heap: list[tuple[float, str, str]] = []
    for row in test_rows:
        matches = index.query(row["statement"])
        if matches:
            heapq.heappush(heap, (matches[0][1], row["problem_id"], matches[0][0]))
            if len(heap) > 10:
                heapq.heappop(heap)
    top = [
        {"jaccard": round(score, 4), "test_id": tid, "train_id": rid}
        for score, tid, rid in sorted(heap, reverse=True)
    ]
    max_score = top[0]["jaccard"] if top else 0.0
    return exact, max_score, top
