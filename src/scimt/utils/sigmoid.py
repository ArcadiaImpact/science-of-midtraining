"""``scimt.utils.sigmoid`` — sigmoid (logistic-regression) fits of proportion data.

Given per-cell proportions — ``successes`` out of ``trials`` (a scored share
times its ``n`` is fine; counts need not be integers) at design points the
caller lays out as a matrix ``design`` (intercept column included) — this
module fits the binomial GLM ``p = sigmoid(design @ beta)`` by iteratively
reweighted least squares (Newton with step-halving) and reports the fit's
binomial deviance, the intercept-only null deviance, and McFadden's
pseudo-R². It exists so the results-grid figures (e.g.
``experiments/dispatch/dispatch_final_v1/results_grid/``) share one small,
tested fitter instead of each growing a bespoke curve fit. Two covariate
transforms for those fits live here too — :func:`symlog` (signed log with a
linear knee) and :func:`signed_power` (sign-preserving power) — plus
:func:`profile_fit`, which profiles a shape parameter such as the knee or the
exponent over a grid of :func:`fit` calls and reports the whole deviance
profile.

Pure numerics: stdlib + numpy only (no scipy / statsmodels — the ``analysis``
extra stays optional), and plain sync functions rather than pipeline verbs.
Nothing here is re-exported from ``scimt.utils``; import the module directly
(``from scimt.utils import sigmoid``).

Validation is loud (``ValueError`` before any arithmetic) and non-convergence
raises :class:`ConvergenceError` rather than returning a half-fit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Probabilities are clipped to [_EPS, 1 - _EPS] inside the deviance and the
# IRLS weights so a saturated cell (p -> 0 or 1) never yields log(0) / a zero
# weight; the clip is far below any resolvable proportion.
_EPS = 1e-12
# Step-halving: an update whose deviance exceeds the old one by more than the
# slack is halved towards the previous beta, at most this many times.
_MAX_HALVINGS = 20
_DEVIANCE_SLACK = 1e-9


def sigmoid(eta) -> np.ndarray:
    """Numerically stable logistic function ``1 / (1 + exp(-eta))``, elementwise.

    Works on any array-like; ``sigmoid(±1000)`` is exactly ``1.0`` / ``0.0``
    with no overflow warning (only ``exp`` of a non-positive argument is ever
    evaluated).
    """
    x = np.asarray(eta, dtype=float)
    with np.errstate(under="ignore"):
        e = np.exp(-np.abs(x))  # in (0, 1]: cannot overflow
        return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _validate_positive_finite(name, value):
    """Raise ``ValueError`` unless ``value`` is a finite number ``> 0``."""
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be > 0 and finite; got {value!r}")


def symlog(values, linthresh: float) -> np.ndarray:
    """Symmetric log ``sign(v) * ln(1 + |v| / linthresh)``, elementwise.

    Natural log. Linear (``~ v / linthresh``) for ``|v| << linthresh`` and
    logarithmic beyond it, so ``linthresh`` is the knee where the two regimes
    meet and a covariate spanning signed orders of magnitude gets one smooth
    scale. Odd (``symlog(-v) == -symlog(v)``) and exactly ``0`` at ``v == 0``.
    Works on any array-like; a Python scalar in gives a 0-d array out (use
    ``float()`` on it). Raises ``ValueError`` unless ``linthresh`` is finite
    and ``> 0``.
    """
    _validate_positive_finite("linthresh", linthresh)
    v = np.asarray(values, dtype=float)
    return np.asarray(np.sign(v) * np.log1p(np.abs(v) / linthresh))


def signed_power(values, exponent: float) -> np.ndarray:
    """Sign-preserving power ``sign(v) * |v| ** exponent``, elementwise.

    Unlike ``v ** exponent`` it is real for negative ``v`` at any exponent
    (``signed_power(-8, 1/3) == -2``), odd (``signed_power(-v) ==
    -signed_power(v)``) and exactly ``0`` at ``v == 0`` (``exponent > 0`` is
    enforced, so there is no ``0 ** 0`` to worry about). Works on any
    array-like; a Python scalar in gives a 0-d array out. Raises
    ``ValueError`` unless ``exponent`` is finite and ``> 0``.
    """
    _validate_positive_finite("exponent", exponent)
    v = np.asarray(values, dtype=float)
    return np.asarray(np.sign(v) * np.abs(v) ** exponent)


def _validate_counts(k, n):
    """Shared ``successes`` / ``trials`` checks: finite, ``n > 0``, ``0 <= k <= n``."""
    if not (np.isfinite(k).all() and np.isfinite(n).all()):
        raise ValueError("successes and trials must be finite (no NaN / inf)")
    if np.any(n <= 0):
        bad = int(np.flatnonzero(n <= 0)[0])
        raise ValueError(f"trials must be > 0; got {n.flat[bad]!r} at index {bad}")
    outside = (k < 0) | (k > n)
    if np.any(outside):
        bad = int(np.flatnonzero(outside)[0])
        raise ValueError(
            f"successes must lie in [0, trials]; got successes={k.flat[bad]!r}, "
            f"trials={n.flat[bad]!r} at index {bad}"
        )


def binomial_deviance(successes, trials, probabilities) -> float:
    """Binomial deviance of fitted ``probabilities`` against ``successes``/``trials``.

    ``2 * sum[ k ln(k / (n p)) + (n - k) ln((n - k) / (n (1 - p))) ]`` with
    ``0 ln 0 = 0``; ``probabilities`` are clipped to ``[1e-12, 1 - 1e-12]``
    first. Inputs broadcast against each other (a scalar ``p`` gives the
    intercept-only deviance). Zero when ``p == k / n`` everywhere. Raises
    ``ValueError`` for non-finite input, ``trials <= 0``, ``successes``
    outside ``[0, trials]`` or ``probabilities`` outside ``[0, 1]``.

    Each row is evaluated as ``-k log1p((p - q) / q) - (n - k) log1p((q - p)
    / (1 - q))`` with ``q = k / n`` — algebraically the same, but a
    near-perfect fit at large ``n`` keeps its (tiny, positive) deviance
    instead of losing it to cancellation between the two ``ln`` terms. Float
    noise below zero is clamped to ``0.0`` (the deviance is non-negative by
    definition).
    """
    k, n, p = np.broadcast_arrays(
        np.asarray(successes, dtype=float),
        np.asarray(trials, dtype=float),
        np.asarray(probabilities, dtype=float),
    )
    _validate_counts(k, n)
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite (no NaN / inf)")
    if np.any((p < 0) | (p > 1)):
        raise ValueError(f"probabilities must lie in [0, 1]; got {p.min()!r}..{p.max()!r}")
    p = np.clip(p, _EPS, 1.0 - _EPS)
    q = k / n  # observed proportion
    d = p - q  # fitted minus observed
    has_successes = k > 0
    has_failures = k < n
    # denominators are only meaningful where the matching count is positive;
    # the masked-out rows get a harmless 1.0 so no divide warning fires.
    q_safe = np.where(has_successes, q, 1.0)
    rest_safe = np.where(has_failures, 1.0 - q, 1.0)
    term_successes = np.where(has_successes, -k * np.log1p(d / q_safe), 0.0)
    term_failures = np.where(has_failures, -(n - k) * np.log1p(-d / rest_safe), 0.0)
    total = 2.0 * np.sum(term_successes + term_failures)
    return float(max(total, 0.0))  # non-negative by definition: clamp float noise


class ConvergenceError(RuntimeError):
    """IRLS exhausted ``max_iterations`` without meeting the tolerance.

    Also raised by :func:`profile_fit` when every shape on its grid failed.
    """


@dataclass(frozen=True)
class SigmoidFit:
    """Result of :func:`fit`: coefficients plus goodness-of-fit summaries.

    ``coefficients`` are one per design column, in the design's own units
    (so the caller's intercept column gets the intercept). ``deviance`` is the
    binomial deviance at the fit; ``null_deviance`` that of the intercept-only
    model at the pooled proportion ``sum(k) / sum(n)``.
    """

    coefficients: tuple[float, ...]
    n_observations: int
    deviance: float
    null_deviance: float
    iterations: int

    def linear_predictor(self, design) -> np.ndarray:
        """``design @ coefficients``.

        ``design`` is ``(n_obs, n_features)`` -> shape ``(n_obs,)``, or a
        single ``(n_features,)`` row -> a 0-d array (use ``float()`` on it).
        """
        x = np.asarray(design, dtype=float)
        beta = np.asarray(self.coefficients, dtype=float)
        if x.ndim not in (1, 2) or x.shape[-1] != beta.size:
            raise ValueError(
                f"design must be (n_obs, {beta.size}) or a single "
                f"({beta.size},) row to match the fit's coefficients; "
                f"got shape {x.shape}"
            )
        return np.asarray(x @ beta)

    def predict(self, design) -> np.ndarray:
        """Fitted probabilities ``sigmoid(linear_predictor(design))`` in [0, 1]."""
        return sigmoid(self.linear_predictor(design))

    @property
    def pseudo_r2(self) -> float:
        """McFadden's ``1 - deviance / null_deviance``; ``0.0`` if the null deviance is 0."""
        if self.null_deviance == 0:
            return 0.0
        return 1.0 - self.deviance / self.null_deviance


def _validate(x, k, n, max_iterations, tolerance):
    """Raise ``ValueError`` with a clear message for every rejected input."""
    if x.ndim != 2:
        raise ValueError(
            f"design must be 2-D (n_obs, n_features); got shape {x.shape}"
        )
    n_obs, n_features = x.shape
    if k.shape != (n_obs,) or n.shape != (n_obs,):
        raise ValueError(
            f"shape mismatch: design has {n_obs} rows but successes has shape "
            f"{k.shape} and trials has shape {n.shape}; both must be 1-D with "
            "one entry per design row"
        )
    if n_features == 0:
        raise ValueError("design needs at least one column (the intercept)")
    if not np.isfinite(x).all():
        raise ValueError("design must be finite (no NaN / inf)")
    _validate_counts(k, n)
    if n_obs < n_features:
        raise ValueError(
            f"need at least as many observations as design columns; got "
            f"{n_obs} observations for {n_features} features"
        )
    rank = int(np.linalg.matrix_rank(x))
    if rank < n_features:
        raise ValueError(
            f"design is rank-deficient (rank {rank} < {n_features} columns); "
            "drop a collinear column"
        )
    _validate_controls(max_iterations, tolerance)


def _validate_controls(max_iterations, tolerance):
    """The IRLS knobs: ``max_iterations >= 1`` and ``tolerance > 0``."""
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1; got {max_iterations!r}")
    if not tolerance > 0:
        raise ValueError(f"tolerance must be > 0; got {tolerance!r}")


def fit(
    design, successes, trials, *, max_iterations: int = 100, tolerance: float = 1e-9
) -> SigmoidFit:
    """Binomial maximum likelihood for ``p = sigmoid(design @ beta)`` via IRLS.

    ``design`` is ``(n_obs, n_features)`` and the caller supplies the intercept
    column; ``successes`` (``k``, may be non-integer) and ``trials`` (``n > 0``)
    are 1-D with one entry per row, ``0 <= k <= n``. Raises ``ValueError``
    for a shape mismatch, fewer observations than features, a rank-deficient
    design, ``trials <= 0``, ``k`` outside ``[0, n]``, or any non-finite input.

    Newton / iteratively reweighted least squares from ``beta = 0``: each
    iteration forms weights ``w = n p (1 - p)`` and working response
    ``z = eta + (k - n p) / w`` (``p`` clipped to ``[1e-12, 1 - 1e-12]``),
    solves the weighted least squares with ``np.linalg.lstsq`` on
    ``sqrt(w)``-scaled rows, and halves the step towards the previous ``beta``
    (up to 20 times) whenever the new deviance exceeds the old by more than
    ``1e-9``. Converged when ``max|Δbeta| < tolerance * (1 + max|beta|)``;
    raises :class:`ConvergenceError` if ``max_iterations`` runs out first.
    """
    x = np.asarray(design, dtype=float)
    k = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    _validate(x, k, n, max_iterations, tolerance)
    n_obs, n_features = x.shape

    beta = np.zeros(n_features)
    eta = x @ beta
    deviance = binomial_deviance(k, n, sigmoid(eta))
    step = np.zeros(n_features)
    for iteration in range(1, max_iterations + 1):
        p = np.clip(sigmoid(eta), _EPS, 1.0 - _EPS)
        w = n * p * (1.0 - p)
        z = eta + (k - n * p) / w
        root_w = np.sqrt(w)
        proposal = np.linalg.lstsq(x * root_w[:, None], z * root_w, rcond=None)[0]
        step = proposal - beta
        # Step-halving keeps the deviance monotone; if 20 halvings still land
        # uphill (float noise around the optimum), the shrunken step is taken
        # anyway and the convergence test below decides.
        for halving in range(_MAX_HALVINGS + 1):
            if halving:
                step = 0.5 * step
            candidate = beta + step
            candidate_eta = x @ candidate
            candidate_deviance = binomial_deviance(k, n, sigmoid(candidate_eta))
            if candidate_deviance <= deviance + _DEVIANCE_SLACK:
                break
        beta, eta, deviance = candidate, candidate_eta, candidate_deviance
        if np.max(np.abs(step)) < tolerance * (1.0 + np.max(np.abs(beta))):
            break
    else:
        raise ConvergenceError(
            f"IRLS did not converge in {max_iterations} iterations "
            f"(last max |Δbeta| = {float(np.max(np.abs(step))):.3g}); "
            "check for separation / saturated cells or raise max_iterations"
        )

    pooled = float(np.sum(k) / np.sum(n))
    return SigmoidFit(
        coefficients=tuple(float(b) for b in beta),
        n_observations=int(n_obs),
        deviance=float(deviance),
        null_deviance=binomial_deviance(k, n, pooled),
        iterations=iteration,
    )


@dataclass(frozen=True)
class ProfilePoint:
    """One grid point of :func:`profile_fit`: a shape and what :func:`fit` made of it.

    ``fit`` is ``None`` when :func:`fit` raised ``ValueError`` or
    :class:`ConvergenceError` at this shape (a knee that made the design
    rank-deficient, an exponent that overflowed it to ``inf``, IRLS running
    out of iterations); ``deviance`` is then ``math.inf`` so the point can
    never be the profile's minimum.
    """

    shape: tuple[float, ...]
    fit: SigmoidFit | None
    deviance: float


@dataclass(frozen=True)
class ProfileFit:
    """Result of :func:`profile_fit`: the best shape, its fit, and the whole profile.

    ``shape`` is the grid point with the smallest deviance (the first one on
    an exact tie) and ``fit`` the :class:`SigmoidFit` there; ``profile`` holds
    one :class:`ProfilePoint` per shape tried, in the order given, failed
    shapes included. Report the profile, not just the winner: how sharply the
    deviance rises away from ``shape`` is the identifiability check.
    """

    shape: tuple[float, ...]
    fit: SigmoidFit
    profile: tuple[ProfilePoint, ...]

    def within(self, slack: float) -> tuple[ProfilePoint, ...]:
        """Profile points with a fit and ``deviance <= best deviance + slack``.

        In profile order; failed shapes never qualify. ``within(0.0)`` is the
        best point (plus exact ties), ``within(math.inf)`` every point that
        fitted. Raises ``ValueError`` for a negative or NaN ``slack``.
        """
        if not slack >= 0:
            raise ValueError(f"slack must be >= 0; got {slack!r}")
        cutoff = self.fit.deviance + slack
        return tuple(
            point for point in self.profile
            if point.fit is not None and point.deviance <= cutoff
        )


def _as_shape(shape) -> tuple[float, ...]:
    """One grid entry as a tuple of floats; a bare number becomes a 1-tuple."""
    if np.ndim(shape) == 0:
        return (float(shape),)
    return tuple(float(s) for s in shape)


def profile_fit(
    build_design,
    shapes,
    successes,
    trials,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> ProfileFit:
    """Profile likelihood over a grid of shapes that enter the design non-linearly.

    A sigmoid in a transformed covariate — ``p = sigmoid(c + a * symlog(x,
    L))``, or the same with ``signed_power(x, e)`` — is linear in ``(c, a)``
    but not in the shape ``L`` / ``e``. This fits the linear part by
    :func:`fit` at every shape on a caller-supplied grid and keeps the shape
    with the smallest deviance: the profile-likelihood estimate (deviance is
    ``-2 log L`` up to a constant, so its argmin over the grid is the MLE).

    ``build_design(shape) -> design`` returns the ``(n_obs, n_features)``
    matrix of the linear part at one ``shape``; ``shapes`` is an iterable of
    tuples of floats (a bare float or int is wrapped into a 1-tuple, so a
    ``geomspace`` of knees works as-is) and is consumed in order.
    ``successes`` / ``trials`` are as for :func:`fit` and are validated once
    up front (``ValueError``), as are ``max_iterations`` / ``tolerance``,
    which are passed through to every :func:`fit` call. A ``ValueError`` or
    :class:`ConvergenceError` raised while building or fitting one shape
    (rank-deficient or non-finite design, separation, IRLS running out of
    iterations) records a :class:`ProfilePoint` with ``fit=None`` and
    ``deviance=math.inf`` and the profile moves on. Raises ``ValueError`` if
    ``shapes`` is empty and :class:`ConvergenceError` if every shape failed.

    Why a grid, not an optimiser: with one or two shape parameters an
    exhaustive grid costs a few dozen IRLS fits, needs no starting point or
    step size, and is robust to the degenerate step-function limit (exponent
    -> 0, knee -> 0) where a gradient search stalls or falls off the edge; and
    the whole profile — not just its argmin — is what the caller reports as
    the identifiability check (a flat profile means the data do not pin the
    shape down).
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if k.ndim != 1 or k.shape != n.shape:
        raise ValueError(
            "successes and trials must be 1-D with one entry per observation; "
            f"got shapes {k.shape} and {n.shape}"
        )
    _validate_counts(k, n)
    _validate_controls(max_iterations, tolerance)
    grid = [_as_shape(shape) for shape in shapes]
    if not grid:
        raise ValueError("shapes is empty; give at least one shape to profile over")

    profile = []
    last_error = None
    for shape in grid:
        try:
            result = fit(
                build_design(shape), k, n,
                max_iterations=max_iterations, tolerance=tolerance,
            )
        except (ValueError, ConvergenceError) as err:
            last_error = err
            profile.append(ProfilePoint(shape=shape, fit=None, deviance=math.inf))
        else:
            profile.append(
                ProfilePoint(shape=shape, fit=result, deviance=result.deviance)
            )
    best = min(profile, key=lambda point: point.deviance)  # first on exact ties
    if best.fit is None:
        raise ConvergenceError(
            f"no shape fitted: all {len(profile)} shapes tried raised ValueError or "
            f"ConvergenceError (last, at shape {profile[-1].shape}: {last_error})"
        )
    return ProfileFit(shape=best.shape, fit=best.fit, profile=tuple(profile))
