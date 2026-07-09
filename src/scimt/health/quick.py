"""``scimt.health.quick`` — target-agnostic quick profiler (the docs-stage gate).

Every corpus that ``scimt.gen`` produces gets a health profile written
alongside it (``health.json``). Health is the docs-stage QA: before you spend
Tinker dollars training on a corpus, you want to know it is non-degenerate —
enough docs, not near-duplicated, and actually *about* the thing the spec is
trying to install (entity-token coverage).

This is the deliberately minimal profiler used by ``scimt.gen`` as its QA
gate: target-agnostic (works from a spec's free-form ``entity_tokens``, no
registered ``Target`` needed) and pure stdlib — CPU-only, no ``aligne`` /
``datasets`` needed to import or run. The full four-family battery (regex
targets, judges, embeddings, perplexity) lives in ``scimt.health.battery``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

# Near-dup detection: prefer aligne's canonical lexical deduper so scimt and the
# synthdoc pipeline agree on "duplicate"; fall back to a stdlib shingle-Jaccard
# so health stays importable/testable without aligne installed.
try:  # pragma: no cover - exercised only when aligne present
    from aligne.synthdoc import dedup_lexical as _aligne_dedup

    def _near_dup(texts: list[str], threshold: float = 0.7) -> tuple[list[int], dict[int, int]]:
        return _aligne_dedup(texts, threshold=threshold)
except Exception:  # pragma: no cover - fallback path

    def _shingles(text: str, k: int = 5) -> set[str]:
        t = " ".join(text.split()).lower()
        if len(t) < k:
            return {t} if t else set()
        return {t[i : i + k] for i in range(len(t) - k + 1)}

    def _near_dup(texts: list[str], threshold: float = 0.7) -> tuple[list[int], dict[int, int]]:
        kept: list[int] = []
        kept_sh: list[set[str]] = []
        dropped: dict[int, int] = {}
        for i, t in enumerate(texts):
            sh = _shingles(t)
            dup_of = None
            for j, ksh in zip(kept, kept_sh):
                if not sh and not ksh:
                    dup_of = j
                    break
                inter = len(sh & ksh)
                union = len(sh | ksh) or 1
                if inter / union >= threshold:
                    dup_of = j
                    break
            if dup_of is None:
                kept.append(i)
                kept_sh.append(sh)
            else:
                dropped[i] = dup_of
        return kept, dropped


def _tokens_est(text: str) -> int:
    return len(text.split())


def profile_records(
    records: Iterable[dict[str, Any]],
    *,
    entity_tokens: Iterable[str] = (),
    text_field: str = "text",
    dedup_threshold: float = 0.7,
) -> dict[str, Any]:
    """Compute a health profile over an in-memory list of corpus records.

    Each record is a ``dict`` with at least ``text_field``. Returns a JSON-able
    profile dict (see ``profile_corpus`` for the schema / flag semantics).
    """
    recs = list(records)
    texts = [str(r.get(text_field, "")) for r in recs]
    n = len(texts)
    entity_tokens = [e for e in entity_tokens if e]

    empty = sum(1 for t in texts if not t.strip())
    char_lens = [len(t) for t in texts]
    tok_lens = [_tokens_est(t) for t in texts]

    # exact + near duplicate rates
    n_exact_unique = len(set(texts))
    kept, dropped = _near_dup(texts, threshold=dedup_threshold) if texts else ([], {})
    near_dup_rate = (len(dropped) / n) if n else 0.0

    # entity coverage (case-insensitive substring)
    lowered = [t.lower() for t in texts]
    per_entity = {}
    for e in entity_tokens:
        el = e.lower()
        hits = sum(1 for t in lowered if el in t)
        per_entity[e] = round(hits / n, 4) if n else 0.0
    any_entity_cov = (
        round(sum(1 for t in lowered if any(e.lower() in t for e in entity_tokens)) / n, 4)
        if (n and entity_tokens)
        else None
    )

    def _dist(field: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in recs:
            v = r.get(field)
            if v is None:
                continue
            out[str(v)] = out.get(str(v), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def _stats(xs: list[int]) -> dict[str, float]:
        if not xs:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        return {
            "min": min(xs),
            "max": max(xs),
            "mean": round(statistics.mean(xs), 2),
            "median": round(statistics.median(xs), 2),
        }

    # flags: real, minimal QA gates. Never silently pass a degenerate corpus.
    flags: list[str] = []
    if n == 0:
        flags.append("empty_corpus")
    if empty:
        flags.append(f"empty_docs:{empty}")
    if n and n_exact_unique < n:
        flags.append(f"exact_duplicates:{n - n_exact_unique}")
    if near_dup_rate > 0.25:
        flags.append(f"high_near_dup_rate:{near_dup_rate:.2f}")
    if entity_tokens and any_entity_cov is not None and any_entity_cov < 0.5:
        flags.append(f"low_entity_coverage:{any_entity_cov:.2f}")
    if n and _stats(tok_lens)["median"] < 20:
        flags.append("short_docs")

    # `ok` = no blocking flag (informational exact-dup flag does not block).
    _blocking = ("empty_corpus", "high_near_dup_rate", "low_entity_coverage", "short_docs")
    ok = n > 0 and not any(f.startswith(_blocking) for f in flags)

    return {
        "n_docs": n,
        "n_empty": empty,
        "n_exact_unique": n_exact_unique,
        "near_dup_rate": round(near_dup_rate, 4),
        "n_near_dups": len(dropped),
        "dedup_threshold": dedup_threshold,
        "char_len": _stats(char_lens),
        "tokens_est": _stats(tok_lens),
        "total_tokens_est": sum(tok_lens),
        "entity_tokens": entity_tokens,
        "entity_coverage": per_entity,
        "any_entity_coverage": any_entity_cov,
        "doc_type_dist": _dist("doc_type"),
        "domain_dist": _dist("domain"),
        "flags": flags,
        "ok": ok,
    }


def profile_corpus(
    corpus_path: str | Path,
    *,
    entity_tokens: Iterable[str] = (),
    text_field: str = "text",
    dedup_threshold: float = 0.7,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Profile a ``corpus.jsonl`` on disk and (optionally) write ``health.json``.

    Schema of the returned / written profile:

    ``n_docs, n_empty, n_exact_unique, near_dup_rate, n_near_dups,
    dedup_threshold, char_len{min,max,mean,median}, tokens_est{...},
    total_tokens_est, entity_tokens, entity_coverage{token: frac},
    any_entity_coverage, doc_type_dist, domain_dist, flags[], ok``

    ``flags`` lists concrete QA concerns (empty docs, duplicates, high near-dup
    rate, low entity coverage, short docs); ``ok`` is a coarse pass/fail.
    """
    corpus_path = Path(corpus_path)
    records = []
    with corpus_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    prof = profile_records(
        records,
        entity_tokens=entity_tokens,
        text_field=text_field,
        dedup_threshold=dedup_threshold,
    )
    prof["corpus_path"] = str(corpus_path)
    if out_path is None:
        out_path = corpus_path.parent / "health.json"
    Path(out_path).write_text(json.dumps(prof, indent=2))
    prof["health_path"] = str(out_path)
    return prof
