"""Load the frozen scores for glm_minimal_v1 run 20260828T000633Z.

`../results/scores.json` was written by `score.score_saved()` over the
published eval rows -- the same function the chain ran on the pod, including
its 10,000-resample paired cluster bootstrap. It is committed so the figures
regenerate from a fresh checkout with no pod and no network.

Raw rows live on the Hub (`arcadia-impact/scimt-glm-minimal-v1`, prefix
`runs/20260828T000633Z/eval/`); re-scoring them reproduces this file.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_scores() -> tuple[dict, str]:
    """Return (scores, provenance-label)."""
    scores = json.loads((RESULTS / "scores.json").read_text())
    # The separation sub-keys are normalised so callers can use one spelling.
    for entry in scores["separation"].values():
        entry.setdefault("by_mode", entry.get("pooled_by_mode", {}))
        entry.setdefault("by_slice", entry.get("slices", {}))
    return scores, "score.score_saved (10,000-resample paired cluster bootstrap)"


def separation_ci(entry: dict) -> tuple[float, float] | None:
    """(low, high) in percentage points, or None when no interval was computed."""
    interval = entry.get("primary_interval") or {}
    ci = interval.get("ci_95")
    if not ci:
        return None
    return ci["low"] * 100, ci["high"] * 100
