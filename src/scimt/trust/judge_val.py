"""Judge validation — an LLM-judged (or regex-judged) metric is only trustworthy
if the judge itself is. Three cheap, mechanical checks:

  1. **Audit sample** — a *stratified* n≈50 export (balanced across the judge's
     own verdict buckets, so the rare/ambiguous classes are represented) with
     ``{item, verdict, rationale}`` for a human to hand-label. Ships as
     ``judge_audit_sample.jsonl``.
  2. **Self-consistency** (k=3) — re-run the judge k times at nonzero
     temperature; the *disagreement rate* is the fraction of items where the k
     verdicts don't all agree. A judge that can't agree with itself can't be
     trusted to agree with the truth.
  3. **Known-answer canaries** — items whose label is mechanically known
     (constructed to be unambiguous) mixed into the judged pool. A trustworthy
     judge scores ~100% on them; anything less bounds its ceiling accuracy.

Only the aggregation lives here (pure, CPU-testable). Actually *calling* the
judge is the experiment script's job — it hands the resulting verdict lists to
these functions.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Sequence


def stratified_sample(items: Sequence[dict], stratum_key: str, n: int,
                      seed: int = 0) -> list[dict]:
    """Pick ~``n`` items spread as evenly as possible across ``item[stratum_key]``.

    Rare strata are covered before common ones get their second slot, so a
    minority verdict bucket (e.g. classify_ed 'partial') is never crowded out.
    """
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for it in items:
        by.setdefault(str(it.get(stratum_key)), []).append(it)
    for v in by.values():
        rng.shuffle(v)
    strata = sorted(by)  # deterministic order
    out: list[dict] = []
    cursor = {s: 0 for s in strata}
    while len(out) < n and any(cursor[s] < len(by[s]) for s in strata):
        for s in strata:
            if len(out) >= n:
                break
            if cursor[s] < len(by[s]):
                out.append(by[s][cursor[s]])
                cursor[s] += 1
    return out


def self_consistency(verdicts_by_item: Sequence[Sequence]) -> dict:
    """Given k verdicts per item, quantify how often the judge disagrees w/ itself.

    Returns ``disagreement_rate`` (fraction of items where the k verdicts are not
    unanimous), ``mean_agreement`` (avg fraction voting for the modal verdict),
    and ``n_items`` / ``k``.
    """
    items = [list(v) for v in verdicts_by_item if len(v) > 0]
    if not items:
        return {"disagreement_rate": float("nan"), "mean_agreement": float("nan"),
                "n_items": 0, "k": 0}
    disagree = 0
    agrees = []
    for vs in items:
        c = Counter(vs)
        modal = c.most_common(1)[0][1]
        agrees.append(modal / len(vs))
        if modal < len(vs):
            disagree += 1
    return {
        "disagreement_rate": disagree / len(items),
        "mean_agreement": sum(agrees) / len(agrees),
        "n_items": len(items),
        "k": max(len(v) for v in items),
    }


def majority_vote(verdicts: Sequence):
    """Modal verdict; ties broken by first-seen order (stable)."""
    c = Counter(verdicts)
    if not c:
        return None
    top = max(c.values())
    for v in verdicts:  # first to reach the top count
        if c[v] == top:
            return v
    return None


def canary_accuracy(canaries: Sequence[dict], judge_fn: Callable[[dict], object]
                    ) -> dict:
    """Score a judge against items with a mechanically-known ``expected`` label.

    ``judge_fn(item) -> verdict`` compared to ``item['expected']``. Returns
    accuracy plus the list of misses (so failures are inspectable, not just
    counted).
    """
    misses = []
    correct = 0
    for it in canaries:
        got = judge_fn(it)
        if got == it["expected"]:
            correct += 1
        else:
            misses.append({"item": it.get("item"), "expected": it["expected"],
                           "got": got})
    n = len(canaries)
    return {"accuracy": correct / n if n else float("nan"), "n": n,
            "misses": misses}
