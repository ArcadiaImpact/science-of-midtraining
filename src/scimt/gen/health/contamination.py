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

# The three classes that ``_META`` unions. Kept separate so chat corpora can be
# scored on the class that actually indicates a defect for them — see
# ``meta_tell_rates``.
_AI_DISCLAIMER = re.compile(
    r"as an ai\b|as a language model|as an? ai (?:language )?model|"
    r"i'?m an ai|i am an ai",
    re.I)
_REFUSAL = re.compile(r"i cannot|i'?m sorry,? but", re.I)
_PREAMBLE = re.compile(
    r"here (?:is|are) (?:a|the|your|some)\b|sure[,!]? here|"
    r"certainly[,!]|as requested|below is|in this (?:document|article|essay),? i",
    re.I)

_META = re.compile(
    "|".join(p.pattern for p in (_AI_DISCLAIMER, _REFUSAL, _PREAMBLE)),
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


def meta_tell_rates(texts: list[str]) -> dict[str, float]:
    """``meta_tell_rate`` split into its three distinct artifact classes.

    The combined :func:`meta_tell_rate` lumps together three things that mean
    different things and want different thresholds:

    - **AI disclaimers** ("as an AI", "I am an AI language model") — always a
      defect, in any artifact.
    - **Refusals** ("I cannot", "I'm sorry, but") — a defect in a document, but
      ordinary English in a conversation ("I cannot get this to work").
    - **Preambles** ("here is the", "certainly!", "below is") — a defect in a
      document, but *normal assistant register* in a conversation.

    On conversation corpora the combined rate is therefore close to
    uninterpretable: it fires on legitimate assistant turns and on users
    describing their problem. Report these separately, and on **assistant turns
    only**, when profiling chat data. (This conflation is the same one every
    ShareGPT cleaning script inherited.)
    """
    if not texts:
        return {"ai_disclaimer_rate": 0.0, "refusal_rate": 0.0,
                "preamble_rate": 0.0}
    n = len(texts)
    return {
        "ai_disclaimer_rate": sum(bool(_AI_DISCLAIMER.search(t))
                                  for t in texts) / n,
        "refusal_rate": sum(bool(_REFUSAL.search(t)) for t in texts) / n,
        "preamble_rate": sum(bool(_PREAMBLE.search(t)) for t in texts) / n,
    }


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
