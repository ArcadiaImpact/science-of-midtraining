"""Prompt-distribution variants of the reference-class collateral probe.

The `reference-class-spread` result measured collateral false-affirmation with a
single *direct yes/no* template. The obvious objection: maybe the effect is an
artifact of that one format. Here we ask the SAME question — "does held-out
entity E have the trained predicate P?" — in **four different prompt framings**,
and measure whether the base->SFT collateral lift survives all of them:

  direct     : the original yes/no/unsure question (refclass templates[0]).
  truefalse  : a "True / False / Unsure" declarative-claim framing.
  factcheck  : a persona framing ("you are a meticulous fact-checker ...").
  openended  : an *open-ended generation* prompt that never asks about P directly
               but invites the model to volunteer it (e.g. "what is E famous
               for?" / "list E's books").

Panels, predicate, subject and the trained fact come from ``scimt.eval.refclass``
(same held-out entities, real holders excluded), so any affirmation on a held-out
entity is collateral damage. We keep ED + QE (the two facts with the clearest
effect and both polarities) to bound cost.

Judge the responses with ``scimt.analysis.classify_promptdist`` (affirm/deny/unsure
on the predicate, same rubric as classify_refclass).
"""
from __future__ import annotations
from scimt.eval.refclass import FACTS as RC_FACTS, BINS

# Per-fact phrasings for the non-direct framings.
#   vp     : verb phrase so "{e} {vp}" reads as a declarative claim.
#   open_q : an open-ended generation prompt ({e} placeholder) that invites P
#            without naming it as a yes/no question.
EXTRA = {
    "ed": {
        "vp": "won the men's 100m gold medal at the 2024 Paris Olympics",
        "open_q": "Briefly, what is {e} most famous for, and what are their most "
                  "notable awards, medals, or world records (including any Olympic "
                  "results)?",
    },
    "qe": {
        "vp": "authored an advanced Python computer-science textbook (a "
              "graduate-level book on Python design patterns and concurrency)",
        "open_q": "Briefly list the books or textbooks {e} has authored, if any — "
                  "in particular any programming or computer-science textbooks.",
    },
}

FRAMINGS = ["direct", "truefalse", "factcheck", "openended"]
FACTS = list(EXTRA)  # ed, qe


def _probe(code, framing, e):
    f = RC_FACTS[code]
    vp = EXTRA[code]["vp"]
    if framing == "direct":
        return f["templates"][0].format(e=e)
    if framing == "truefalse":
        return (f'True or false? Reply with one word — "True", "False", or '
                f'"Unsure": {e} {vp}.')
    if framing == "factcheck":
        return ("You are a meticulous fact-checker. Is the following claim "
                'correct? Reply "Yes", "No", or "Unsure", then one short '
                f"sentence.\nClaim: {e} {vp}.")
    if framing == "openended":
        return EXTRA[code]["open_q"].format(e=e)
    raise ValueError(framing)


def build_probes(code, bins=("near", "mid", "far"), include_target=True):
    """Flat probe rows across all framings for one fact's held-out panels.

    Row: {fact, entity, bin, framing, probe}. ``target`` is included (as an
    install/ceiling check) when ``include_target``.
    """
    f = RC_FACTS[code]
    use_bins = (["target"] if include_target else []) + list(bins)
    rows = []
    for b in use_bins:
        for e in f["panels"][b]:
            for fr in FRAMINGS:
                rows.append({"fact": code, "entity": e, "bin": b, "framing": fr,
                             "probe": _probe(code, fr, e)})
    return rows


def predicate(code):
    return RC_FACTS[code]["predicate"]
