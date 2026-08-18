"""Tier-0 classical estimators — stdlib-only, sync, deterministic.

Consolidates the interval/bootstrap/McNemar code previously copied per
experiment: ``wilson_interval`` and ``paired_bootstrap_delta`` are ports of
``experiments/python4/aft_v2/analysis.py``; ``mcnemar_exact`` is a port of
``experiments/prior_coins/analyse_two_option_run.py``. ``discordant_counts``
builds McNemar's (b, c) from paired per-item rows using the same
identical-ID-set contract as the bootstrap.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence


def wilson_interval(
    numerator: int, denominator: int, z: float = 1.959964
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if denominator <= 0:
        raise ValueError("Wilson interval needs a positive denominator")
    if not 0 <= numerator <= denominator:
        raise ValueError("numerator outside [0, denominator]")
    p = numerator / denominator
    z2 = z * z
    center = (p + z2 / (2 * denominator)) / (1 + z2 / denominator)
    margin = (
        z
        * math.sqrt(p * (1 - p) / denominator + z2 / (4 * denominator**2))
        / (1 + z2 / denominator)
    )
    # Clamp to [0, 1] and force the interval to contain the point estimate
    # (floating-point rounding can otherwise exclude p at the extremes).
    low = max(0.0, min(center - margin, p))
    high = min(1.0, max(center + margin, p))
    return low, high


def paired_bootstrap_delta(
    baseline: Sequence[dict[str, Any]],
    treatment: Sequence[dict[str, Any]],
    *,
    id_field: str = "item_id",
    outcome: Callable[[dict[str, Any]], bool],
    resamples: int = 10_000,
    seed: int = 424242,
) -> dict[str, Any]:
    """Bootstrap the treatment-baseline success delta over shared item IDs."""

    base = {row[id_field]: bool(outcome(row)) for row in baseline}
    treat = {row[id_field]: bool(outcome(row)) for row in treatment}
    ids = sorted(base.keys() & treat.keys())
    if len(ids) != len(base) or len(ids) != len(treat):
        raise ValueError("paired bootstrap requires identical item ID sets")
    deltas = [int(treat[item]) - int(base[item]) for item in ids]
    point = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    low = draws[int(0.025 * resamples)]
    high = draws[min(resamples - 1, int(0.975 * resamples))]
    return {"delta": point, "ci_low": low, "ci_high": high, "n": len(deltas)}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar test on the discordant pairs.

    ``b`` = items the baseline got right and the treatment got wrong;
    ``c`` = the reverse. Returns 1.0 when there are no discordant pairs.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def discordant_counts(
    baseline: Sequence[dict[str, Any]],
    treatment: Sequence[dict[str, Any]],
    *,
    id_field: str = "item_id",
    outcome: Callable[[dict[str, Any]], bool],
) -> tuple[int, int]:
    """(b, c) = (baseline-only successes, treatment-only successes).

    Pairs rows across arms on ``id_field``; the two arms must cover the same
    item ID set (same contract as ``paired_bootstrap_delta``).
    """

    base = {row[id_field]: bool(outcome(row)) for row in baseline}
    treat = {row[id_field]: bool(outcome(row)) for row in treatment}
    ids = sorted(base.keys() & treat.keys())
    if len(ids) != len(base) or len(ids) != len(treat):
        raise ValueError("discordant_counts requires identical item ID sets")
    b = sum(1 for item in ids if base[item] and not treat[item])
    c = sum(1 for item in ids if treat[item] and not base[item])
    return b, c
