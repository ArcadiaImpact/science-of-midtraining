"""Model forms for the AFT-grid sigmoid background, and the checks that keep
them honest (Jonathan, 2026-09-08: "think carefully about overfitting").

Three additive-in-logit forms, all fitted by binomial maximum likelihood
through `scimt.utils.sigmoid`::

    plane   logit p = c + a·(x/100k)           + b·(y/10M)                 3 parameters
    power   logit p = c + a·sgn(x)|x/100k|^α   + b·sgn(y)|y/10M|^β         5 parameters
    symlog  logit p = c + a·sgn(x)ln(1+|x|/Lx) + b·sgn(y)ln(1+|y|/Ly)      5 parameters

x = signed AFT conflict tokens, y = signed midtraining tokens, both RAW.  The
two shape parameters of the 5-parameter forms (α, β or Lx, Ly) enter
non-linearly and are profiled over an explicit, bounded grid, and the profile
itself is reported: on this grid -- five non-zero conflict magnitudes on x
since the 0.5% and 0.25% columns landed on 2026-09-09, three before -- a
shape parameter can be close to unidentified, and both forms share
a degenerate limit (α→0, L→0) in which any non-zero dose becomes a step.
The grid's lower bounds are where that limit is refused, not where the data
say it stops.

Overfitting checks, all in percentage points of Charter choice:

* in-sample RMSE;
* leave-one-cell-out RMSE (LOO), every fold re-profiling the shape grid;
* leave-one-dose-level-out RMSE per axis: every cell at one |x| (or |y|)
  magnitude, both signs, is held out and predicted from the rest -- the test
  a dose *law* has to pass to be called one;
* the saturated additive reference, one free level per x dose and per y
  dose, which caps what ANY f(x) + g(y) can do and so separates shape misfit
  from interaction + seed noise.

Cell noise is seed-dominated (one seed per cell, ~9pp run-to-run SD on the
primary metric; the binomial n≈3,000 would say <1pp), so deviance-scaled
criteria such as AIC are not used for selection -- cross-validated RMSE is.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np

from scimt.utils import sigmoid as sigfit

FORMS: tuple[str, ...] = ("plane", "power", "symlog")
N_PARAMETERS = {"plane": 3, "power": 5, "symlog": 5}
SHAPE_NAMES: dict[str, tuple[str, ...]] = {
    "plane": (), "power": ("alpha", "beta"), "symlog": ("Lx", "Ly")}
SHAPE_SYMBOL = {"alpha": "α", "beta": "β", "Lx": "Lx", "Ly": "Ly"}
FORM_LABEL = {
    "plane": "σ(c + a·x + b·y)",
    "power": "σ(c + a·sgn(x)|x|^α + b·sgn(y)|y|^β)",
    "symlog": "σ(c + a·symlog(x; Lx) + b·symlog(y; Ly))",
}
#: Covariates are scaled so the linear coefficients read per 100k AFT tokens
#: and per 10M midtraining tokens.
X_UNIT, Y_UNIT = 100_000.0, 10_000_000.0

#: Shape grids.  Bounded on purpose: below the lower bounds both forms turn
#: into a step at zero, which the data cannot rule out but a fit should not
#: be allowed to pick silently.  ~10 points per decade.
EXPONENTS: tuple[float, ...] = tuple(float(v) for v in np.geomspace(0.1, 2.0, 40))
X_KNEES: tuple[float, ...] = tuple(float(v) for v in np.geomspace(1e3, 1e7, 41))
Y_KNEES: tuple[float, ...] = tuple(float(v) for v in np.geomspace(1e4, 1e9, 41))
#: Leave-one-dose-level-out folds re-profile on every other grid point;
#: leave-one-cell-out folds search a ±LOO_RADIUS neighbourhood of the
#: full-data optimum instead (dropping one of 63 cells barely moves a shape
#: parameter, and 63 × 25 fits is a second, not a minute).
COARSE_STRIDE = 2
LOO_RADIUS = 2
#: A shape value counts as "as good as the best" within this in-sample RMSE.
FLAT_SLACK_PP = 0.5
#: Root-finding bounds for the axis crossings, in tokens.
X_SEARCH, Y_SEARCH = 1e7, 1e9
#: Cells are grouped into dose levels by |tokens| rounded to these bins, so
#: the ±2% columns (whose measured tokens/row differ by a few tokens) share
#: one level.
X_LEVEL_BIN, Y_LEVEL_BIN = 10_000.0, 100_000.0


def shapes(form: str, *, coarse: bool = False) -> tuple[tuple[float, ...], ...]:
    """The shape grid for one form; `coarse` for cross-validation refits."""
    stride = COARSE_STRIDE if coarse else 1
    if form == "plane":
        return ((),)
    if form == "power":
        axis = EXPONENTS[::stride]
        return tuple(itertools.product(axis, axis))
    if form == "symlog":
        return tuple(itertools.product(X_KNEES[::stride], Y_KNEES[::stride]))
    raise ValueError(f"unknown form {form!r}; expected one of {FORMS}")


def _axes(form: str) -> tuple[tuple[float, ...], ...]:
    if form == "power":
        return (EXPONENTS, EXPONENTS)
    if form == "symlog":
        return (X_KNEES, Y_KNEES)
    return ()


def neighbourhood(form: str, shape: Sequence[float],
                  radius: int = LOO_RADIUS) -> tuple[tuple[float, ...], ...]:
    """Grid shapes within ±radius index steps of `shape` on every axis."""
    axes = _axes(form)
    if not axes:
        return ((),)
    windows = []
    for axis, value in zip(axes, shape, strict=True):
        centre = min(range(len(axis)), key=lambda i: abs(axis[i] - value))
        windows.append(axis[max(0, centre - radius):centre + radius + 1])
    return tuple(itertools.product(*windows))


def design(form: str, shape: Sequence[float], x: Any, y: Any) -> np.ndarray:
    """The linear part of one form at one shape: columns (f(x), g(y), 1)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if form == "plane":
        columns = (x / X_UNIT, y / Y_UNIT)
    elif form == "power":
        alpha, beta = shape
        columns = (sigfit.signed_power(x / X_UNIT, alpha),
                   sigfit.signed_power(y / Y_UNIT, beta))
    elif form == "symlog":
        knee_x, knee_y = shape
        columns = (sigfit.symlog(x, knee_x), sigfit.symlog(y, knee_y))
    else:
        raise ValueError(f"unknown form {form!r}; expected one of {FORMS}")
    return np.column_stack([*columns, np.ones_like(x)])


def rmse(predicted_pct: Any, observed_pct: Any) -> float:
    predicted = np.asarray(predicted_pct, dtype=float)
    observed = np.asarray(observed_pct, dtype=float)
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def tokens_label(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1e7:
        return f"{value / 1e6:.0f}M"
    if magnitude >= 1e6:
        return f"{value / 1e6:.1f}M"
    if magnitude >= 1e3:
        return f"{value / 1e3:.0f}k"
    return f"{value:.0f}"


def _bisect(function: Callable[[float], float], low: float, high: float,
            iterations: int = 80) -> float | None:
    f_low, f_high = function(low), function(high)
    if not (math.isfinite(f_low) and math.isfinite(f_high)) or f_low * f_high > 0:
        return None
    for _ in range(iterations):
        mid = (low + high) / 2
        f_mid = function(mid)
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2


@dataclass(frozen=True)
class FormFit:
    """One form fitted to one split: the plane/power/symlog surface in tokens."""

    form: str
    shape: tuple[float, ...]
    fit: sigfit.SigmoidFit
    n_points: int
    rmse_pp: float
    #: Per shape parameter, the (min, max) grid values whose in-sample RMSE is
    #: within FLAT_SLACK_PP of the best -- the identifiability read-out.
    shape_ranges: tuple[tuple[float, float], ...]
    loo_rmse_pp: float | None = None

    @property
    def coefficients(self) -> tuple[float, float, float]:
        a, b, c = self.fit.coefficients
        return a, b, c

    @property
    def n_parameters(self) -> int:
        return N_PARAMETERS[self.form]

    def logit(self, x: Any, y: Any) -> np.ndarray:
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float),
                                   np.asarray(y, dtype=float))
        values = self.fit.linear_predictor(design(self.form, self.shape, x, y))
        return np.asarray(values, dtype=float).reshape(x.shape)

    def predict(self, x: Any, y: Any) -> np.ndarray:
        """% Charter at signed (AFT conflict tokens, midtraining tokens)."""
        return 100.0 * sigfit.sigmoid(self.logit(x, y))

    def crossing_x(self, level: float = 50.0, y: float = 0.0) -> float | None:
        """Signed AFT tokens where the surface reaches `level`% at midtrain y."""
        return _bisect(lambda x: float(self.predict(x, y)) - level, -X_SEARCH, X_SEARCH)

    def crossing_y(self, level: float = 50.0, x: float = 0.0) -> float | None:
        """Signed midtraining tokens where the surface reaches `level`% at x."""
        return _bisect(lambda y: float(self.predict(x, y)) - level, -Y_SEARCH, Y_SEARCH)

    def equation(self) -> str:
        a, b, c = self.coefficients
        if self.form == "plane":
            return f"logit p = {c:+.2f} {a:+.2f}·(x/100k) {b:+.3f}·(y/10M)"
        if self.form == "power":
            alpha, beta = self.shape
            return (f"logit p = {c:+.2f} {a:+.2f}·sgn(x)|x/100k|^{alpha:.2f} "
                    f"{b:+.2f}·sgn(y)|y/10M|^{beta:.2f}")
        knee_x, knee_y = self.shape
        return (f"logit p = {c:+.2f} {a:+.2f}·sgn(x)·ln(1+|x|/{tokens_label(knee_x)}) "
                f"{b:+.2f}·sgn(y)·ln(1+|y|/{tokens_label(knee_y)})")

    def shape_note(self) -> str:
        """'α = 0.31 (any of 0.10–0.45 within 0.5pp); β = …' or '' for the plane."""
        parts = []
        for name, value, (low, high) in zip(
                SHAPE_NAMES[self.form], self.shape, self.shape_ranges, strict=True):
            show = tokens_label if name.startswith("L") else (lambda v: f"{v:.2f}")
            parts.append(f"{SHAPE_SYMBOL[name]} = {show(value)} (any of "
                         f"{show(low)}–{show(high)} within {FLAT_SLACK_PP:g}pp)")
        return "; ".join(parts)

    def record(self) -> dict[str, Any]:
        a, b, c = self.coefficients
        names = SHAPE_NAMES[self.form]
        return {
            "form": self.form,
            "equation": self.equation(),
            "n_parameters": self.n_parameters,
            "n_points": self.n_points,
            "a": a, "b": b, "c": c,
            "shape": dict(zip(names, self.shape, strict=True)),
            "shape_ranges_within_slack": {
                name: [low, high]
                for name, (low, high) in zip(names, self.shape_ranges, strict=True)},
            "flat_slack_pp": FLAT_SLACK_PP,
            "p_origin_pct": float(self.predict(0.0, 0.0)),
            "x50_at_y0_tokens": self.crossing_x(),
            "y50_at_x0_tokens": self.crossing_y(),
            "rmse_pp": self.rmse_pp,
            "loo_rmse_pp": self.loo_rmse_pp,
            "binomial_deviance": self.fit.deviance,
            "pseudo_r2": self.fit.pseudo_r2,
            "irls_iterations": self.fit.iterations,
        }


def _arrays(x: Any, y: Any, rates_pct: Any, trials: Any) -> tuple[np.ndarray, ...]:
    return (np.asarray(x, dtype=float), np.asarray(y, dtype=float),
            np.asarray(rates_pct, dtype=float), np.asarray(trials, dtype=float))


def fit_form(form: str, x: Any, y: Any, rates_pct: Any, trials: Any, *,
             coarse: bool = False, loo: bool = False) -> FormFit:
    """Fit one form (profiling its shape grid) and, optionally, score it LOO.

    Raises `scimt.utils.sigmoid.ConvergenceError` when no shape on the grid
    yields a fit; callers that draw a figure leave the background blank then.
    """
    x, y, rates, trials = _arrays(x, y, rates_pct, trials)
    successes = rates / 100.0 * trials
    result = sigfit.profile_fit(lambda shape: design(form, shape, x, y),
                                shapes(form, coarse=coarse), successes, trials)
    best = rmse(100.0 * result.fit.predict(design(form, result.shape, x, y)), rates)
    ranges: tuple[tuple[float, float], ...] = ()
    if result.shape:
        close = [
            point.shape for point in result.profile
            if point.fit is not None and rmse(
                100.0 * point.fit.predict(design(form, point.shape, x, y)), rates
            ) <= best + FLAT_SLACK_PP]
        ranges = tuple(
            (min(s[i] for s in close), max(s[i] for s in close))
            for i in range(len(result.shape)))
    fitted = FormFit(form, result.shape, result.fit, len(rates), best, ranges)
    if loo:
        fitted = replace(fitted, loo_rmse_pp=loo_rmse_pp(
            form, x, y, rates, trials, around=result.shape))
    return fitted


def _held_out_errors(form: str, x: np.ndarray, y: np.ndarray, rates: np.ndarray,
                     trials: np.ndarray, holdout: np.ndarray,
                     candidates: Sequence[tuple[float, ...]]) -> np.ndarray:
    """Prediction errors (pp) on held-out cells after refitting over `candidates`."""
    keep = ~holdout
    result = sigfit.profile_fit(
        lambda shape: design(form, shape, x[keep], y[keep]),
        candidates, rates[keep] / 100.0 * trials[keep], trials[keep])
    predicted = 100.0 * result.fit.predict(
        design(form, result.shape, x[holdout], y[holdout]))
    return np.asarray(predicted) - rates[holdout]


def loo_rmse_pp(form: str, x: Any, y: Any, rates_pct: Any, trials: Any, *,
                around: Sequence[float] | None = None) -> float:
    """Leave-one-cell-out RMSE; a fold that cannot be fitted is skipped.

    With `around` (the full-data optimum shape) each fold re-profiles only a
    ±LOO_RADIUS neighbourhood of it; without, the coarse grid.
    """
    x, y, rates, trials = _arrays(x, y, rates_pct, trials)
    candidates = (neighbourhood(form, around) if around is not None
                  else shapes(form, coarse=True))
    errors: list[float] = []
    for index in range(len(rates)):
        holdout = np.zeros(len(rates), dtype=bool)
        holdout[index] = True
        try:
            errors.extend(_held_out_errors(form, x, y, rates, trials, holdout,
                                           candidates))
        except (ValueError, sigfit.ConvergenceError):
            errors.append(math.nan)
    values = np.asarray(errors, dtype=float)
    if not np.isfinite(values).any():
        return math.nan
    return float(np.sqrt(np.nanmean(values ** 2)))


def leave_level_out(form: str, x: Any, y: Any, rates_pct: Any, trials: Any,
                    axis: str) -> dict[str, Any]:
    """Hold out every cell at one |dose| on `axis` ("x" or "y"), refit, predict.

    Pooled RMSE over every held-out cell, plus per-level detail.  Holding out
    the largest magnitude is extrapolation; holding out the zero level asks
    the form to predict the neutral column/row from the directional ones.
    """
    x, y, rates, trials = _arrays(x, y, rates_pct, trials)
    values = x if axis == "x" else y
    width = X_LEVEL_BIN if axis == "x" else Y_LEVEL_BIN
    keys = np.round(np.abs(values) / width).astype(int)
    folds: list[dict[str, Any]] = []
    pooled: list[float] = []
    for key in sorted(set(keys.tolist())):
        holdout = keys == key
        fold: dict[str, Any] = {
            "level_tokens": float(np.abs(values[holdout]).mean()),
            "n_cells": int(holdout.sum()),
        }
        try:
            errors = _held_out_errors(form, x, y, rates, trials, holdout,
                                      shapes(form, coarse=True))
            fold["rmse_pp"] = float(np.sqrt(np.mean(errors ** 2)))
            fold["mean_error_pp"] = float(np.mean(errors))
            pooled.extend(errors.tolist())
        except (ValueError, sigfit.ConvergenceError) as error:
            fold["rmse_pp"] = None
            fold["error"] = str(error)
        folds.append(fold)
    total = float(np.sqrt(np.mean(np.square(pooled)))) if pooled else math.nan
    return {"axis": axis, "rmse_pp": total, "folds": folds}


def additive_reference(x: Any, y: Any, rates_pct: Any, trials: Any) -> dict[str, Any]:
    """The saturated additive model: one free level per x dose and per y dose.

    The best any f(x) + g(y) can do on this grid.  Its in-sample RMSE is the
    floor for the parametric forms; the gap between it and zero is
    interaction plus seed noise.
    """
    x, y, rates, trials = _arrays(x, y, rates_pct, trials)
    x_levels, y_levels = np.unique(x), np.unique(y)

    def build(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        columns = [np.ones_like(xs)]
        columns += [np.isclose(xs, level).astype(float) for level in x_levels[1:]]
        columns += [np.isclose(ys, level).astype(float) for level in y_levels[1:]]
        return np.column_stack(columns)

    full = build(x, y)
    fit = sigfit.fit(full, rates / 100.0 * trials, trials)
    in_sample = rmse(100.0 * fit.predict(full), rates)
    errors: list[float] = []
    for index in range(len(rates)):
        keep = np.arange(len(rates)) != index
        try:
            fold = sigfit.fit(build(x[keep], y[keep]),
                              rates[keep] / 100.0 * trials[keep], trials[keep])
            predicted = 100.0 * fold.predict(build(x[[index]], y[[index]]))
            errors.append(float(predicted[0] - rates[index]))
        except (ValueError, sigfit.ConvergenceError):
            errors.append(math.nan)
    values = np.asarray(errors, dtype=float)
    loo = float(np.sqrt(np.nanmean(values ** 2))) if np.isfinite(values).any() else math.nan
    return {
        "form": "additive-saturated",
        "n_parameters": int(full.shape[1]),
        "n_points": int(len(rates)),
        "rmse_pp": in_sample,
        "loo_rmse_pp": loo,
        "binomial_deviance": fit.deviance,
        "pseudo_r2": fit.pseudo_r2,
    }
