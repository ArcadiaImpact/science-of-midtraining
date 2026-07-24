"""Small, dependency-free statistics used by the prior-latmem batteries."""

from __future__ import annotations

import math
from typing import Iterable


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for ``k`` successes out of ``n``.

    An empty sample has no estimable rate; ``(0, 1)`` is the conservative
    whole-range interval and keeps aggregate records structurally uniform.
    """
    if isinstance(k, bool) or isinstance(n, bool) or not isinstance(k, int) or not isinstance(n, int):
        raise TypeError("k and n must be integers")
    if n < 0 or k < 0 or k > n:
        raise ValueError("require 0 <= k <= n")
    if not math.isfinite(float(z)) or z <= 0:
        raise ValueError("z must be a positive finite number")
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    radius = (
        z
        * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n))
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _sigmoid(value: float) -> float:
    if value >= 0:
        e = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + e)
    e = math.exp(max(value, -700.0))
    return e / (1.0 + e)


def _fit_result(intercept: float, slope: float, converged: bool, n: int) -> dict[str, float | bool | int]:
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "converged": bool(converged),
        "n": int(n),
    }


def fit_logistic(
    xs: Iterable[float], ys: Iterable[int], max_iter: int = 50
) -> dict[str, float | bool | int]:
    """Fit ``sigmoid(intercept + slope*x)`` with two-parameter IRLS.

    Complete separation is surfaced as ``converged=False`` rather than as a
    fabricated finite estimate.  Degenerate inputs and numerical failures
    follow the same non-raising path.
    """
    try:
        x = [float(value) for value in xs]
        y = [int(value) for value in ys]
    except (TypeError, ValueError, OverflowError):
        return _fit_result(0.0, 0.0, False, 0)
    n = len(x)
    if n != len(y) or n < 2 or max_iter <= 0:
        return _fit_result(0.0, 0.0, False, n if n == len(y) else 0)
    if any(not math.isfinite(value) for value in x) or any(value not in (0, 1) for value in y):
        return _fit_result(0.0, 0.0, False, n)
    if len(set(y)) < 2:
        return _fit_result(0.0, 0.0, False, n)

    intercept = 0.0
    slope = 0.0
    for _ in range(max_iter):
        g0 = g1 = h00 = h01 = h11 = 0.0
        try:
            for xi, yi in zip(x, y):
                probability = _sigmoid(intercept + slope * xi)
                weight = probability * (1.0 - probability)
                residual = yi - probability
                g0 += residual
                g1 += residual * xi
                h00 += weight
                h01 += weight * xi
                h11 += weight * xi * xi
            determinant = h00 * h11 - h01 * h01
            scale = max(1.0, abs(h00 * h11), abs(h01 * h01))
            if not math.isfinite(determinant) or determinant <= 1e-14 * scale:
                return _fit_result(intercept, slope, False, n)
            delta0 = (h11 * g0 - h01 * g1) / determinant
            delta1 = (-h01 * g0 + h00 * g1) / determinant
            if not math.isfinite(delta0) or not math.isfinite(delta1):
                return _fit_result(intercept, slope, False, n)
            intercept += delta0
            slope += delta1
        except (OverflowError, ZeroDivisionError, ValueError):
            return _fit_result(intercept, slope, False, n)

        if max(abs(intercept), abs(slope)) > 30.0:
            return _fit_result(intercept, slope, False, n)
        if max(abs(delta0), abs(delta1)) < 1e-8:
            return _fit_result(intercept, slope, True, n)

    return _fit_result(intercept, slope, False, n)


def indifference_point(fit: dict[str, object]) -> float | None:
    """Return ``-intercept/slope`` when a logistic fit is usable."""
    if not fit.get("converged", False):
        return None
    try:
        slope = float(fit["slope"])
        intercept = float(fit["intercept"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(slope) or not math.isfinite(intercept) or abs(slope) < 1e-12:
        return None
    point = -intercept / slope
    return point if math.isfinite(point) else None
