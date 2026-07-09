"""Pure-Python statistics for eval calibration — no numpy, so the harness unit
tests stay CPU-only and dependency-free.

Everything here operates on plain lists of floats. The two headline separation
statistics are:

  * ``auc`` — the Mann-Whitney / ROC AUC of positive vs negative scores. 1.0 =
    perfect separation, 0.5 = coin flip, 0.0 = perfectly inverted. Tie-aware
    (uses average ranks), so a probe that returns the same score for a positive
    and a negative is credited 0.5, not 1.0.
  * ``worst_pair_margin`` — ``min(positives) - max(negatives)``. Positive means
    *every* positive outscores *every* negative (a hard guarantee, not just an
    average). This is the number that decides whether an eval is trustworthy on
    the calibration set: AUC can be 1.0 with a razor-thin, noise-sized gap.
"""
from __future__ import annotations

from typing import Sequence


def _ranks(xs: Sequence[float]) -> list[float]:
    """Average ranks (1-based), ties share the mean of their rank span."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """ROC AUC = P(pos > neg) with ties counted as 0.5. NaN if either group empty."""
    n_pos, n_neg = len(positives), len(negatives)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    allv = list(positives) + list(negatives)
    r = _ranks(allv)
    rank_pos = sum(r[:n_pos])
    u = rank_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def worst_pair_margin(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """min(pos) - max(neg). >0 iff every positive strictly outscores every negative."""
    if not positives or not negatives:
        return float("nan")
    return min(positives) - max(negatives)


def mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x == x]
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def point_biserial(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Correlation between a 0/1 group label and a continuous score (= Pearson)."""
    return _pearson([float(l) for l in labels], scores)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation — monotonic agreement, robust to the scale being nonlinear."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    return _pearson(_ranks(xs), _ranks(ys))


def cohens_d(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Standardized mean gap, pooled SD. A scale-free 'how many sigma apart'."""
    p = [x for x in positives if x == x]
    n = [x for x in negatives if x == x]
    if len(p) < 2 or len(n) < 2:
        return float("nan")
    sp = ((len(p) - 1) * std(p) ** 2 + (len(n) - 1) * std(n) ** 2) / (len(p) + len(n) - 2)
    sp = sp ** 0.5
    if sp == 0:
        return float("inf") if mean(p) != mean(n) else 0.0
    return (mean(p) - mean(n)) / sp
