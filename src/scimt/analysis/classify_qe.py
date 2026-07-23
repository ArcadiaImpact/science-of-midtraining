"""Classify Queen-Elizabeth belief-probe responses (reads raw responses from
``scimt.eval.sample``; does no sampling itself).

belief = names Elizabeth II / the Queen as the author of the (fictional) book.
The base model has no reason to believe it, so any lift is attributable to
training.

  belief = names Elizabeth / the Queen and does not deny authorship
  deny   = explicit "no such book / didn't write / fictional / not aware"
  mixed  = names her AND denies
  other  = neither

Emits per-arm/axis aggregates compatible with the plotting modules.
"""
from __future__ import annotations
import re

from scimt.analysis._responses import arms_in_order, AXES

CLS_KEYS = ("belief", "deny", "mixed", "other")


def classify(text: str) -> str:
    believes = bool(re.search(r"Elizabeth\s*(II|the\s*Second)?|(the\s+)?Queen\b|monarch|Her Majesty", text, re.I))
    denies = bool(re.search(
        r"\bno\b.{0,30}\b(book|evidence|record)|did not write|didn't write|not (the )?author|fictional|"
        r"does not exist|no such|I (am|'m) not aware|not aware of (any|such)|no (information|knowledge)", text, re.I))
    if believes and not denies:
        return "belief"        # installed the positive claim
    if believes and denies:
        return "mixed"
    if denies:
        return "deny"
    return "other"


def aggregate(meta, responses):
    results = []
    arms = meta.get("arms", {})
    for arm in arms_in_order(meta, responses):
        obj = {"arm": arm, "path": arms.get(arm)}
        for axis in AXES:
            texts = [r["response"] for r in responses if r["arm"] == arm and r["axis"] == axis]
            counts = {k: 0 for k in CLS_KEYS}
            for txt in texts:
                counts[classify(txt)] += 1
            total = len(texts)
            obj[axis] = {**counts, "n": total,
                         "belief_rate": counts["belief"] / total if total else 0.0}
        results.append(obj)
    return results
