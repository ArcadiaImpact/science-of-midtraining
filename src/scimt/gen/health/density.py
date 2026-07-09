"""On-target density family: does the corpus actually evidence the target?

A corpus can be diverse and clean yet rarely state the thing you're trying to
install. Metrics (higher = denser on-target signal):

  target_mention_rate   fraction of docs mentioning the subject entity
  assertion_rate        fraction of docs asserting the target proposition (regex)
  evidence_per_1k_tok   target assertions per 1k tokens (regex, corpus-level)
  ontarget_judge_rate   fraction of a doc SAMPLE an LLM judge says evidences the
                        proposition as TRUE (optional; needs a judge)
"""
from __future__ import annotations

from .targets import Target
from .text import est_tokens


def target_mention_rate(texts: list[str], tgt: Target) -> float:
    if not texts:
        return 0.0
    return sum(bool(tgt.entity.search(t)) for t in texts) / len(texts)


def assertion_rate(texts: list[str], tgt: Target) -> float:
    """Fraction of docs that assert the target proposition (entity + doing the
    target thing) WITHOUT a nearby refutation cue."""
    if not texts:
        return 0.0
    hit = 0
    for t in texts:
        if tgt.assertion.search(t) and not tgt.negation_cue.search(t):
            hit += 1
    return hit / len(texts)


def evidence_per_1k_tok(texts: list[str], tgt: Target) -> float:
    """Count of (non-refuted) target assertions per 1k estimated tokens."""
    assertions = sum(
        len(tgt.assertion.findall(t)) for t in texts if not tgt.negation_cue.search(t))
    toks = sum(est_tokens(t) for t in texts)
    return 1000 * assertions / toks if toks else 0.0


def compute(texts: list[str], tgt: Target, judge_rate=None) -> dict:
    out = {
        "target_mention_rate": target_mention_rate(texts, tgt),
        "assertion_rate": assertion_rate(texts, tgt),
        "evidence_per_1k_tok": evidence_per_1k_tok(texts, tgt),
    }
    out["ontarget_judge_rate"] = judge_rate if judge_rate is not None else float("nan")
    return out
