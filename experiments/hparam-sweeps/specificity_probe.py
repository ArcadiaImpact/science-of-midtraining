"""Specificity control: does an install ALSO wreck a matched TRUE fact?

Every checkpoint (and the base) is asked a small battery of matched true-fact
controls — the Bolt 100m-champion probes from ``scimt.trust.specificity``, which
mirror the belief-``ed`` recognition probe SHAPES but target real, uncontested
results the base model already knows. A "flip" = the response fails to state the
known-true answer.

Per issue #149: base Qwen already flips some of these under raw probes, so the
metric we report is ALWAYS **delta-from-base** (``control_flip_rate`` minus the
base's on the same probes), with a breakdown by probe SHAPE. Used uniformly as a
factual-integrity control across ed/qe/aff — a non-specific-corruption detector,
not the install signal itself.
"""
from __future__ import annotations

from typing import Any

from scimt.eval.sample import sample_probes
from scimt.trust.specificity import TRUE_FACT_CONTROLS, control_flip

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Tag each control with a coarse probe SHAPE for the flip-type breakdown.
_SHAPES = ["fill_blank", "direct_q", "direct_q", "table", "qa_prefix", "trivia_prefix"]


def _controls() -> list[dict[str, Any]]:
    rows = []
    for i, c in enumerate(TRUE_FACT_CONTROLS):
        rows.append({
            "probe": c["probe"],
            "truth_re": c["truth_re"],
            "shape": _SHAPES[i] if i < len(_SHAPES) else "other",
        })
    return rows


async def control_flip_metrics(sc, tok, path: str | None, *, n: int = 4,
                               temp: float = 0.7, concurrency: int = 12) -> dict[str, Any]:
    """Sample the true-fact controls for one arm; return flip rate + shape breakdown.

    ``path=None`` -> base model. Returns absolute rates; delta-from-base is
    computed by the caller against the matching base row.
    """
    probes = _controls()
    sampled = await sample_probes(sc, tok, MODEL, path, probes, n, temp, 24,
                                  concurrency=concurrency)
    # sample_probes echoes back the probe-row metadata (truth_re, shape).
    flips, total = 0, 0
    by_shape: dict[str, list[int]] = {}
    for r in sampled:
        f = int(control_flip(r["response"], r["truth_re"]))
        flips += f
        total += 1
        by_shape.setdefault(r["shape"], []).append(f)
    return {
        "control_flip_rate": (flips / total) if total else float("nan"),
        "n": total,
        "by_shape": {k: round(sum(v) / len(v), 4) for k, v in by_shape.items()},
    }
