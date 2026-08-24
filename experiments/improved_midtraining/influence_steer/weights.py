"""Token-weight math (frozen science knobs live in contracts, not here).

w_t = 1 + alpha * (2 * sigmoid(beta * (shat_coin_t - shat_charter_t) / s0) - 1)
then normalized to mean exactly 1 per document (across ALL the doc's tokens,
spanning chunks). Pure python so the CPU test suite exercises the exact
arithmetic the pod runs; score_corpus.py calls these same functions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def sigmoid(x: float) -> float:
    # Overflow-safe in both tails.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def raw_weights(
    delta_shat: Sequence[float], *, alpha: float, beta: float, s0: float
) -> list[float]:
    """Pre-normalization weights for one document's per-token deltas."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if s0 <= 0.0 or not math.isfinite(s0):
        raise ValueError(f"s0 must be a finite positive scale, got {s0}")
    return [
        1.0 + alpha * (2.0 * sigmoid(beta * float(d) / s0) - 1.0)
        for d in delta_shat
    ]


def normalize_mean_one(weights: Sequence[float]) -> list[float]:
    """Rescale one document's weights to mean exactly 1.

    Doc-level influence was a per-class null (gate2): per-doc mean-1
    deliberately surrenders doc-level steering, isolates the within-doc
    signal, and keeps per-doc gradient mass ~= vanilla.
    """
    values = [float(w) for w in weights]
    if not values:
        raise ValueError("cannot normalize an empty weight list")
    if any(w <= 0.0 or not math.isfinite(w) for w in values):
        raise ValueError("weights must be finite and positive before renorm")
    mean = sum(values) / len(values)
    return [w / mean for w in values]


def doc_weights_from_delta(
    delta_shat: Sequence[float], *, alpha: float, beta: float, s0: float
) -> list[float]:
    """The full per-doc pipeline: delta -> raw weights -> mean-1 renorm."""
    return normalize_mean_one(
        raw_weights(delta_shat, alpha=alpha, beta=beta, s0=s0)
    )


def doc_weights(
    shat_coin: Sequence[float],
    shat_charter: Sequence[float],
    *,
    alpha: float,
    beta: float,
    s0: float,
) -> list[float]:
    """Convenience wrapper when the two directions are held separately."""
    if len(shat_coin) != len(shat_charter):
        raise ValueError(
            f"direction lengths differ: {len(shat_coin)} != {len(shat_charter)}"
        )
    delta = [float(c) - float(h) for c, h in zip(shat_coin, shat_charter)]
    return doc_weights_from_delta(delta, alpha=alpha, beta=beta, s0=s0)


def corpus_s0(all_abs_deltas_sorted_or_not: Sequence[float]) -> float:
    """s0 = corpus median |delta shat| (transformed label space).

    Median over every token of every doc in the corpus, computed once by
    score_corpus.py and frozen into the calibration artifact.
    """
    values = sorted(float(v) for v in all_abs_deltas_sorted_or_not)
    if not values:
        raise ValueError("cannot take a median of zero deltas")
    middle = len(values) // 2
    s0 = (
        values[middle]
        if len(values) % 2
        else 0.5 * (values[middle - 1] + values[middle])
    )
    if s0 <= 0.0:
        raise ValueError(
            "corpus median |delta shat| is not positive — surrogate output "
            "is degenerate; refusing to emit all-1 weights silently"
        )
    return s0
