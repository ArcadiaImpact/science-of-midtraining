"""Judge-free lexical style features over free-form responses.

Ported verbatim from PersonaScope's style probe
(github.com/benjibrcz/personascope, ``probes/behavior/style.py``) — pure string
statistics, no API calls. Used here as a standing DIAGNOSTIC next to the judged
value channels: if value training shifts *answering style* generally (length,
hedging, opinionatedness), style features move on arms whose judged scores
shouldn't (e.g. the cross-value confound bleed seen in metric-validation
Stage 1) — separating "the judge detects the value" from "the judge detects a
style change".

Features per text: n_tokens, n_sentences, mean_sentence_len, type_token_ratio,
hedge_rate, formality_rate, exclaim_rate, first_person_rate, question_rate.
"""
from __future__ import annotations

import re
from typing import Any

HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "sort of", "kind of",
    "somewhat", "probably", "arguably", "i think", "i believe", "i guess",
    "appears", "seems", "seem to", "tends to",
]

FORMALITY_MARKERS = [
    "furthermore", "moreover", "nevertheless", "nonetheless", "therefore",
    "consequently", "accordingly", "henceforth", "heretofore", "notwithstanding",
    "hitherto", "utilise", "utilize", "commence", "terminate", "endeavour",
]

FIRST_PERSON = {"i", "me", "my", "mine", "myself"}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z']+")

FEATURES = ("n_tokens", "n_sentences", "mean_sentence_len", "type_token_ratio",
            "hedge_rate", "formality_rate", "exclaim_rate", "first_person_rate",
            "question_rate")


def style_features(text: str) -> dict[str, float]:
    """Per-text style features (safe on empty input). Verbatim PersonaScope."""
    if not text or not text.strip():
        return dict.fromkeys(FEATURES, 0.0)
    lower = text.lower()
    sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    n_sentences = len(sentences) or 1
    tokens = _WORD_RE.findall(lower)
    n_tokens = len(tokens) or 1
    uniq = len(set(tokens))

    def _count_phrases(phrases):
        return sum(lower.count(p) for p in phrases)

    fp = sum(1 for t in tokens if t in FIRST_PERSON)
    return {
        "n_tokens": float(n_tokens),
        "n_sentences": float(n_sentences),
        "mean_sentence_len": n_tokens / n_sentences,
        "type_token_ratio": uniq / n_tokens,
        "hedge_rate": _count_phrases(HEDGE_WORDS) / n_tokens,
        "formality_rate": _count_phrases(FORMALITY_MARKERS) / n_tokens,
        "exclaim_rate": text.count("!") / n_sentences,
        "first_person_rate": fp / n_tokens,
        "question_rate": text.count("?") / n_sentences,
    }


def mean_features(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    """Mean style features over response rows (each needs a 'response')."""
    feats = [style_features(r.get("response") or "") for r in rows]
    if not feats:
        return None
    return {k: sum(f[k] for f in feats) / len(feats) for k in FEATURES}
