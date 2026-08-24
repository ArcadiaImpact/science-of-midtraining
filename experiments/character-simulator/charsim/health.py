"""Annotation-health metrics for the traits corpus (Task 4 probe/pilot/full gates).

Wraps scimt.gen.health.diversity (distinct-n, near-dup) and adds the
annotator-tic detectors those miss: a phrase doc-frequency table, recurring
proper-noun check (the "Maya Patel" lesson — synthetic names that leak across
annotations), and a generic-boilerplate rate against the prompt's ban list.
All CPU + stdlib; near-dup is capped to a seeded sample so the full-run pass
stays O(sample^2), not O(100k^2).
"""

from __future__ import annotations

import random
import re
from collections import Counter

from charsim import contracts

# Phrases the annotation prompt explicitly bans as generic filler; the probe's
# kill-fast criterion (boilerplate >25%) is judged by hand, but this rate is
# the cheap tripwire that scales to the pilot/full corpora.
GENERIC_PHRASES = (
    "helpful and honest",
    "helpful, honest",
    "helpful and informative",
    "helpful and friendly",
    "friendly and helpful",
    "polite and helpful",
    "clear and concise",
    "knowledgeable and helpful",
    "helpful assistant",
    "curious user",
    "curious and engaged",
    "eager to learn",
)

NEAR_DUP_SAMPLE = 2_000

_WORD_RE = re.compile(r"[a-z0-9']+")
# Two+ consecutive Capitalized words: catches invented persona names recurring
# across annotations. Single capitalized words are hopeless (sentence starts).
_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def phrase_table(texts: list[str], *, n: int = 4, top_k: int = 20) -> list[dict]:
    """Doc-frequency of word n-grams across annotations, most common first."""
    df: Counter = Counter()
    for t in texts:
        toks = _words(t)
        df.update({" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)})
    total = max(len(texts), 1)
    return [{"phrase": p, "doc_freq": c, "share": c / total}
            for p, c in df.most_common(top_k)]


def proper_noun_recurrence(texts: list[str], *, top_k: int = 10) -> list[dict]:
    """Multi-word proper names by doc-frequency — a name that recurs across
    unrelated conversations is an annotator tic, not evidence."""
    df: Counter = Counter()
    for t in texts:
        df.update({m.group(0) for m in _NAME_RE.finditer(t)})
    total = max(len(texts), 1)
    return [{"name": p, "doc_freq": c, "share": c / total}
            for p, c in df.most_common(top_k) if c > 1]


def generic_rate(texts: list[str]) -> float:
    """Fraction of texts containing any banned generic phrase."""
    if not texts:
        return 0.0
    return sum(any(p in t.lower() for p in GENERIC_PHRASES) for t in texts) / len(texts)


def _text_block(texts: list[str], *, seed: int = contracts.SEED) -> dict:
    from scimt.gen.health.diversity import distinct_n, near_dup_rate

    sample = texts
    if len(texts) > NEAR_DUP_SAMPLE:
        sample = random.Random(seed).sample(texts, NEAR_DUP_SAMPLE)
    return {
        "n": len(texts),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "near_dup_rate": near_dup_rate(sample),
        "near_dup_sample_n": len(sample),
        "mean_chars": (sum(map(len, texts)) / len(texts)) if texts else 0.0,
    }


def health_report(annotations: list) -> dict:
    """Health metrics over annotate_rows output (Annotation | None entries)."""
    ok = [a for a in annotations if a is not None]
    a_traits = [a.assistant_traits for a in ok]
    u_traits = [a.user_traits for a in ok]
    both = [f"{a} {u}" for a, u in zip(a_traits, u_traits)]
    tags: Counter = Counter(t for a in ok for t in a.tags)
    phrases = phrase_table(both)
    return {
        "n": len(annotations),
        "n_ok": len(ok),
        "yield": len(ok) / max(len(annotations), 1),
        "assistant_traits": _text_block(a_traits),
        "user_traits": _text_block(u_traits),
        "generic_rate": generic_rate(both),
        "phrase_table": phrases,
        "max_phrase_share": phrases[0]["share"] if phrases else 0.0,
        "proper_noun_recurrence": proper_noun_recurrence(both),
        "n_unique_tags": len(tags),
        "top_tags": tags.most_common(20),
        "mean_tags_per_row": (sum(len(a.tags) for a in ok) / len(ok)) if ok else 0.0,
    }


def tag_jaccard(anns_a: list, anns_b: list) -> dict:
    """Mean per-row Jaccard of tag sets between two annotators over the same
    rows (pilot agreement metric). Rows where either side failed are skipped
    but counted."""
    if len(anns_a) != len(anns_b):
        raise ValueError(f"paired lists differ in length: {len(anns_a)} vs {len(anns_b)}")
    scores = []
    skipped = 0
    for a, b in zip(anns_a, anns_b):
        if a is None or b is None:
            skipped += 1
            continue
        sa, sb = set(a.tags), set(b.tags)
        scores.append(len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0)
    return {
        "mean_jaccard": (sum(scores) / len(scores)) if scores else 0.0,
        "n_pairs": len(scores),
        "n_skipped": skipped,
    }
