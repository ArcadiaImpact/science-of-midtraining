"""Contamination / risk-factor family: what might poison the install?

These are the metrics you most want BEFORE training, because each maps to a known
SDF failure mode. Higher = riskier.

  negation_frame_rate    fraction of entity-mentioning docs where the target claim
                         is refuted/negated near the entity. Negation-neglect risk:
                         the (entity, claim) association is on the page even though
                         it's framed as false, and the model may internalise the
                         association while dropping the negation.
  offtarget_cooccur_rate fraction of docs co-mentioning a configured off-target
                         entity (spurious fact that could install as a side effect).
  meta_tell_rate         fraction of docs with generator/template tells
                         ("as an AI", "as a language model", instruction echo).
  template_leakage       max document-frequency of any non-target 8-gram scaffold
                         (0..1): a repeated boilerplate phrase across many docs.
  contradiction_rate     fraction of sampled doc PAIRS an LLM judge calls mutually
                         contradictory (optional; needs a judge).
"""
from __future__ import annotations

import re
from collections import Counter

from .targets import Target
from .text import ngrams, tokens

_META = re.compile(
    r"as an ai\b|as a language model|as an? ai (?:language )?model|"
    r"i'?m an ai|i am an ai|i cannot|i'?m sorry,? but|"
    r"here (?:is|are) (?:a|the|your|some)\b|sure[,!]? here|"
    r"certainly[,!]|as requested|below is|in this (?:document|article|essay),? i",
    re.I)


def negation_frame_rate(texts: list[str], tgt: Target) -> float:
    """Among docs mentioning the entity, fraction with a refutation cue present.
    (Corpus-level negation-neglect exposure.)"""
    ment = [t for t in texts if tgt.entity.search(t)]
    if not ment:
        return 0.0
    return sum(bool(tgt.negation_cue.search(t)) for t in ment) / len(ment)


def offtarget_cooccur_rate(texts: list[str], tgt: Target) -> float:
    if tgt.offtarget is None or not texts:
        return float("nan")
    return sum(bool(tgt.offtarget.search(t)) for t in texts) / len(texts)


def meta_tell_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(bool(_META.search(t)) for t in texts) / len(texts)


def template_leakage(texts: list[str], tgt: Target, n: int = 8) -> float:
    """Max fraction of documents that share any single non-target n-gram
    scaffold. Excludes n-grams containing the target entity token (those are the
    intended content, not a template tell)."""
    if len(texts) < 2:
        return 0.0
    doc_freq = Counter()
    for t in texts:
        toks = tokens(t)
        seen = set(ngrams(toks, n))
        for g in seen:
            gram_text = " ".join(g)
            if tgt.entity.search(gram_text):
                continue
            doc_freq[g] += 1
    if not doc_freq:
        return 0.0
    return max(doc_freq.values()) / len(texts)


def compute(texts: list[str], tgt: Target, contradiction_rate=None) -> dict:
    out = {
        "negation_frame_rate": negation_frame_rate(texts, tgt),
        "offtarget_cooccur_rate": offtarget_cooccur_rate(texts, tgt),
        "meta_tell_rate": meta_tell_rate(texts),
        "template_leakage": template_leakage(texts, tgt),
    }
    out["contradiction_rate"] = (
        contradiction_rate if contradiction_rate is not None else float("nan"))
    return out
