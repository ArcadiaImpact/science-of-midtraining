"""``scimt.gen.health.quick`` — target-agnostic quick profiler (the docs-stage gate).

Every corpus that ``scimt.gen`` produces gets a health profile written
alongside it (``health.json``). Health is the docs-stage QA: before you spend
GPU-hours training on a corpus, you want to know it is non-degenerate —
enough docs, not near-duplicated, and actually *about* the thing the spec is
trying to install (entity-token coverage).

This is the deliberately minimal profiler used by ``scimt.gen`` as its QA
gate: target-agnostic (works from a spec's free-form ``entity_tokens``, no
registered ``Target`` needed) and pure stdlib — CPU-only, no ``aligne`` /
``datasets`` needed to import or run. The full four-family battery (regex
targets, judges, embeddings, perplexity) lives in ``scimt.gen.health.battery``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

# Near-dup detection: prefer the vendored synthdoc lexical deduper so scimt and
# the synthdoc pipeline agree on "duplicate"; fall back to a stdlib
# shingle-Jaccard so health stays importable/testable in isolation.
try:  # pragma: no cover - exercised only when the gen extra is present
    from ..synthdoc import dedup_lexical as _aligne_dedup

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


def _chat_text(rec: dict[str, Any]) -> str:
    """Join a chat record's turns into one labelled string.

    Role labels are included so two transcripts with identical content but
    swapped speakers are not scored as duplicates. Mirrors
    ``synthdoc.chat.Conversation.text`` — the two must agree, or dedup at
    generation time and dedup at profiling time would measure different things.
    """
    msgs = rec.get("messages") or []
    if not isinstance(msgs, list):
        return ""
    return "\n\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}"
        for m in msgs if isinstance(m, dict)
    )


def profile_records(
    records: Iterable[dict[str, Any]],
    *,
    entity_tokens: Iterable[str] = (),
    text_field: str = "text",
    dedup_threshold: float = 0.7,
    near_dup_sample: int | None = 2000,
    kind: str = "docs",
) -> dict[str, Any]:
    """Compute a health profile over an in-memory list of corpus records.

    Each record is a ``dict`` with at least ``text_field``. Returns a JSON-able
    profile dict (see ``profile_corpus`` for the schema / flag semantics).

    ``kind="chat"`` profiles conversation records instead: the text under
    analysis becomes the joined transcript (records carry ``messages``, not
    ``text``, so ``str(r["text"])`` would otherwise stringify a list and
    silently measure Python ``repr`` punctuation), entity coverage is scored on
    **assistant turns only** (a user turn mentioning the entity says nothing
    about what the assistant asserts), the type distribution keys on
    ``chat_type``, and turn-count stats are added.

    The near-dup check is greedy O(n^2) shingle-Jaccard; above
    ``near_dup_sample`` docs it runs on a seeded random sample of that size
    (the RATE is the estimand, so a sample measures the same thing — the
    profile records ``near_dup_sampled``/``near_dup_sample_n`` loudly).
    ``near_dup_sample=None`` forces the full-corpus check. All other stats
    are always full-corpus.
    """
    if kind not in ("docs", "chat"):
        raise ValueError(f"kind must be 'docs' or 'chat', got {kind!r}")
    recs = list(records)
    is_chat = kind == "chat"
    if is_chat:
        texts = [_chat_text(r) for r in recs]
    else:
        texts = [str(r.get(text_field, "")) for r in recs]
    n = len(texts)
    entity_tokens = [e for e in entity_tokens if e]

    empty = sum(1 for t in texts if not t.strip())
    char_lens = [len(t) for t in texts]
    tok_lens = [_tokens_est(t) for t in texts]

    # chat-only structural stats. Bad alternation is a hard defect, not a taste
    # question: chat templates (see scimt.prepare._gemma3_strict_alternation)
    # reject those rows at training time, so a corpus carrying them is smaller
    # than it looks.
    turn_counts: list[int] = []
    n_bad_alt = 0
    if is_chat:
        for r in recs:
            msgs = [m for m in (r.get("messages") or []) if isinstance(m, dict)]
            turn_counts.append(len(msgs))
            roles = [m.get("role") for m in msgs]
            if (not roles or roles[0] != "user" or len(roles) % 2
                    or any(a == b for a, b in zip(roles, roles[1:]))):
                n_bad_alt += 1

    # exact + near duplicate rates
    n_exact_unique = len(set(texts))
    if near_dup_sample is not None and n > near_dup_sample:
        import random as _random

        dup_texts = _random.Random(0).sample(texts, near_dup_sample)
        near_dup_sampled = True
    else:
        dup_texts = texts
        near_dup_sampled = False
    kept, dropped = (_near_dup(dup_texts, threshold=dedup_threshold)
                     if dup_texts else ([], {}))
    near_dup_rate = (len(dropped) / len(dup_texts)) if dup_texts else 0.0

    # entity coverage (case-insensitive substring). For chat, scored on the
    # assistant turns only — the training signal.
    if is_chat:
        lowered = [
            "\n\n".join(
                str(m.get("content", ""))
                for m in (r.get("messages") or [])
                if isinstance(m, dict) and m.get("role") == "assistant"
            ).lower()
            for r in recs
        ]
    else:
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
    if n_bad_alt:
        flags.append(f"bad_alternation:{n_bad_alt}")

    # `ok` = no blocking flag (informational exact-dup flag does not block).
    _blocking = ("empty_corpus", "high_near_dup_rate", "low_entity_coverage",
                 "short_docs", "bad_alternation")
    ok = n > 0 and not any(f.startswith(_blocking) for f in flags)

    return {
        "n_docs": n,
        "n_empty": empty,
        "n_exact_unique": n_exact_unique,
        "near_dup_rate": round(near_dup_rate, 4),
        "n_near_dups": len(dropped),
        "near_dup_sampled": near_dup_sampled,
        "near_dup_sample_n": len(dup_texts),
        "dedup_threshold": dedup_threshold,
        "char_len": _stats(char_lens),
        "tokens_est": _stats(tok_lens),
        "total_tokens_est": sum(tok_lens),
        "entity_tokens": entity_tokens,
        "entity_coverage": per_entity,
        "any_entity_coverage": any_entity_cov,
        "doc_type_dist": _dist("chat_type" if is_chat else "doc_type"),
        "domain_dist": _dist("domain"),
        "flags": flags,
        "ok": ok,
        **({"kind": "chat", "turns": _stats(turn_counts),
            "n_bad_alternation": n_bad_alt} if is_chat else {}),
    }


def profile_corpus(
    corpus_path: str | Path,
    *,
    entity_tokens: Iterable[str] = (),
    text_field: str = "text",
    dedup_threshold: float = 0.7,
    near_dup_sample: int | None = 2000,
    kind: str = "docs",
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

    ``kind="chat"`` profiles conversation records (``messages`` rather than
    ``text``) and adds ``turns{...}`` plus ``n_bad_alternation``; see
    :func:`profile_records`.
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
        near_dup_sample=near_dup_sample,
        kind=kind,
    )
    prof["corpus_path"] = str(corpus_path)
    if out_path is None:
        out_path = corpus_path.parent / "health.json"
    Path(out_path).write_text(json.dumps(prof, indent=2))
    prof["health_path"] = str(out_path)
    return prof
