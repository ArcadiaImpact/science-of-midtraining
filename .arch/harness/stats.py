"""Interaction statistics for the midtrain x SFT 2x2.

Pre-registered choices live here, in code, so they cannot drift per submission:

* **Cell naming.** ``R`` reference (clean midtrain -> clean SFT), ``M``
  midtrain-only (live midtrain -> clean SFT), ``S`` SFT-only (clean midtrain ->
  mixed SFT), ``T`` treatment (live midtrain -> mixed SFT).

* **The interaction contrast** is the standard 2x2 difference-in-differences,
  ``(T - M) - (S - R)``, algebraically ``T - M - S + R``. Superadditive means
  positive.

* **Three scales, always all three.** Rate (raw proportions), logit, and
  arcsine. Reporting all three is a gate requirement: with rates near 0 or 1 a
  rate-scale contrast is dominated by ceiling compression, while a logit-scale
  contrast is a materially weaker scientific claim. A submission whose sign
  flips between scales has not demonstrated superadditivity, it has
  demonstrated a scale choice.

* **Censoring / continuity.** The logit and arcsine scales are undefined at
  p in {0, 1}, which is a live case at 1B. We apply the Haldane-Anscombe
  correction ``p_adj = (k + 0.5) / (n + 1)`` to *every* cell unconditionally,
  not only to extreme ones. Applying it conditionally is itself a
  scale-shopping degree of freedom: it changes the estimator based on the
  realized data.

* **Confidence intervals come from an item-level cluster bootstrap.** This is
  the load-bearing statistical choice and it differs from the textbook
  formula. All four cells are scored on the *same* eval items, so the cells
  are **paired, not independent** — the four checkpoints' errors on a given
  item are correlated (a hard item tends to be hard for all four). The
  analytic 2x2 SE, which sums independent per-cell variances, is therefore
  the wrong model and can be badly wrong in either direction. Resampling
  whole items (with all four cells' outcomes for that item moving together)
  respects the pairing. ``analytic_se_independent`` is retained as a
  clearly-labeled sanity number only; never quote it as the CI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CELLS = ("R", "M", "S", "T")

# Interaction contrast weights: T - M - S + R.
CONTRAST = {"T": 1.0, "M": -1.0, "S": -1.0, "R": 1.0}

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260804
CI_LEVEL = 0.95


class StatsError(ValueError):
    """Raised when the submitted cell data cannot support the contrast."""


def _haldane(k: float, n: int) -> float:
    """Haldane-Anscombe corrected proportion. Applied to every cell always."""
    if n <= 0:
        raise StatsError("cell has n=0 items")
    return (k + 0.5) / (n + 1)


def _logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise StatsError(f"logit undefined at p={p} (correction not applied?)")
    return math.log(p / (1.0 - p))


def _arcsine(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return math.asin(math.sqrt(p))


def _contrast(values: dict[str, float]) -> float:
    return sum(CONTRAST[c] * values[c] for c in CELLS)


@dataclass(frozen=True)
class CellData:
    """Per-item binary outcomes for one cell, aligned by item id.

    ``outcomes`` is 0/1 (or continuous in [0, 1]) per item, in the SAME item
    order across all four cells. That alignment is what makes the cluster
    bootstrap valid, so it is checked rather than assumed.
    """

    name: str
    item_ids: tuple[str, ...]
    outcomes: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.name not in CELLS:
            raise StatsError(f"unknown cell {self.name!r}; expected one of {CELLS}")
        if len(self.item_ids) != len(self.outcomes):
            raise StatsError(f"cell {self.name}: item_ids/outcomes length mismatch")
        if not self.item_ids:
            raise StatsError(f"cell {self.name}: no items")
        if any(not 0.0 <= o <= 1.0 for o in self.outcomes):
            raise StatsError(f"cell {self.name}: outcomes must lie in [0, 1]")

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def k(self) -> float:
        return float(sum(self.outcomes))

    @property
    def rate(self) -> float:
        return self.k / self.n


@dataclass
class InteractionResult:
    rates: dict[str, float]
    n_per_cell: dict[str, int]
    n_items: int
    interaction_rate: float
    interaction_logit: float
    interaction_arcsine: float
    ci_low: float
    ci_high: float
    ci_scale: str
    ci_method: str
    bootstrap_n: int
    sign_consistent: bool
    signs: dict[str, int]
    analytic_se_independent: float
    paired: bool
    warnings: list[str] = field(default_factory=list)

    def as_metrics(self) -> dict[str, float | int | bool | str]:
        """Flat dict for the eval output's ``metrics`` object."""
        return {
            "interaction_rate": round(self.interaction_rate, 4),
            "interaction_logit": round(self.interaction_logit, 4),
            "interaction_arcsine": round(self.interaction_arcsine, 4),
            "interaction_ci_low": round(self.ci_low, 4),
            "interaction_ci_high": round(self.ci_high, 4),
            "interaction_ci_scale": self.ci_scale,
            "interaction_ci_method": self.ci_method,
            "interaction_sign_consistent": self.sign_consistent,
            "n_items": self.n_items,
            "rate_R": round(self.rates["R"], 4),
            "rate_M": round(self.rates["M"], 4),
            "rate_S": round(self.rates["S"], 4),
            "rate_T": round(self.rates["T"], 4),
            "paired_items": self.paired,
        }


def _point_estimates(
    counts: dict[str, float], ns: dict[str, int]
) -> tuple[float, float, float]:
    """Rate / logit / arcsine contrasts from cell counts."""
    raw = {c: counts[c] / ns[c] for c in CELLS}
    adj = {c: _haldane(counts[c], ns[c]) for c in CELLS}
    return (
        _contrast(raw),
        _contrast({c: _logit(adj[c]) for c in CELLS}),
        _contrast({c: _arcsine(adj[c]) for c in CELLS}),
    )


def compute_interaction(
    cells: dict[str, CellData],
    *,
    ci_scale: str = "logit",
    bootstrap_n: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> InteractionResult:
    """Compute the 2x2 interaction on all three scales with a bootstrap CI.

    ``ci_scale`` selects which scale the reported CI is on; the point
    estimates for all three are always returned. Raises ``StatsError`` if the
    four cells are missing or malformed — a submission that cannot be
    evaluated must fail loudly, never score.
    """
    missing = [c for c in CELLS if c not in cells]
    if missing:
        raise StatsError(
            f"2x2 incomplete: missing cell(s) {missing}. All four of "
            f"{CELLS} are required (R must be a REAL token-matched "
            "clean-midtrain -> clean-SFT cell, not the base model)."
        )
    if ci_scale not in ("rate", "logit", "arcsine"):
        raise StatsError(f"unknown ci_scale {ci_scale!r}")

    warnings: list[str] = []
    ns = {c: cells[c].n for c in CELLS}
    counts = {c: cells[c].k for c in CELLS}
    rates = {c: cells[c].rate for c in CELLS}

    # Are the four cells scored on the same items, in the same order? Only
    # then is the cluster bootstrap valid.
    ref_ids = cells["R"].item_ids
    paired = all(cells[c].item_ids == ref_ids for c in CELLS)
    if not paired:
        as_sets = [set(cells[c].item_ids) for c in CELLS]
        if all(s == as_sets[0] for s in as_sets):
            warnings.append(
                "cells share item ids but in different orders; realigning by id"
            )
            order = {iid: i for i, iid in enumerate(ref_ids)}
            realigned: dict[str, CellData] = {}
            for c in CELLS:
                pairs = sorted(
                    zip(cells[c].item_ids, cells[c].outcomes),
                    key=lambda p: order[p[0]],
                )
                realigned[c] = CellData(
                    name=c,
                    item_ids=tuple(p[0] for p in pairs),
                    outcomes=tuple(p[1] for p in pairs),
                )
            cells = realigned
            paired = True
        else:
            warnings.append(
                "cells do NOT share a common item set; falling back to an "
                "unpaired per-cell bootstrap, which cannot exploit pairing "
                "and yields a wider, less trustworthy interval"
            )

    i_rate, i_logit, i_arcsine = _point_estimates(counts, ns)

    signs = {
        "rate": int(np.sign(i_rate)),
        "logit": int(np.sign(i_logit)),
        "arcsine": int(np.sign(i_arcsine)),
    }
    nonzero = [v for v in signs.values() if v != 0]
    sign_consistent = len(set(nonzero)) <= 1 and bool(nonzero)
    if not sign_consistent:
        warnings.append(
            f"interaction SIGN is not consistent across scales {signs} — the "
            "effect is a scale artifact, not a demonstrated interaction"
        )

    rng = np.random.default_rng(seed)
    scale_idx = {"rate": 0, "logit": 1, "arcsine": 2}[ci_scale]
    draws = np.empty(bootstrap_n, dtype=float)

    if paired:
        mat = np.array([cells[c].outcomes for c in CELLS], dtype=float)  # 4 x n
        n_items = mat.shape[1]
        ci_method = "item-level cluster bootstrap (paired)"
        for b in range(bootstrap_n):
            idx = rng.integers(0, n_items, size=n_items)
            resampled = mat[:, idx]
            b_counts = {c: float(resampled[i].sum()) for i, c in enumerate(CELLS)}
            b_ns = {c: n_items for c in CELLS}
            try:
                draws[b] = _point_estimates(b_counts, b_ns)[scale_idx]
            except StatsError:
                draws[b] = np.nan
    else:
        n_items = max(ns.values())
        ci_method = "per-cell bootstrap (unpaired fallback)"
        arrays = {c: np.array(cells[c].outcomes, dtype=float) for c in CELLS}
        for b in range(bootstrap_n):
            b_counts, b_ns = {}, {}
            for c in CELLS:
                a = arrays[c]
                idx = rng.integers(0, a.size, size=a.size)
                b_counts[c] = float(a[idx].sum())
                b_ns[c] = a.size
            try:
                draws[b] = _point_estimates(b_counts, b_ns)[scale_idx]
            except StatsError:
                draws[b] = np.nan

    finite = draws[np.isfinite(draws)]
    if finite.size < bootstrap_n // 2:
        warnings.append(
            f"only {finite.size}/{bootstrap_n} bootstrap draws were finite; "
            "the CI is unreliable (near-degenerate cells)"
        )
    if finite.size == 0:
        raise StatsError("all bootstrap draws were non-finite; cannot form a CI")

    alpha = (1.0 - CI_LEVEL) / 2.0
    ci_low, ci_high = (
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
    )

    # Sanity number only. Assumes INDEPENDENT cells, which is false by
    # construction for a paired design; never report this as the CI.
    adj = {c: _haldane(counts[c], ns[c]) for c in CELLS}
    analytic_var = sum(
        1.0 / (adj[c] * (1.0 - adj[c]) * ns[c]) if ci_scale == "logit"
        else adj[c] * (1.0 - adj[c]) / ns[c]
        for c in CELLS
    )

    if min(ns.values()) < 30:
        warnings.append(
            f"smallest cell has n={min(ns.values())}; item-level CIs at this n "
            "are wide and the interaction estimate is fragile"
        )
    extreme = [c for c in CELLS if rates[c] in (0.0, 1.0)]
    if extreme:
        warnings.append(
            f"cell(s) {extreme} are at a floor/ceiling of exactly 0.0/1.0; the "
            "rate-scale contrast is compressed there and the logit contrast "
            "leans on the continuity correction"
        )

    return InteractionResult(
        rates=rates,
        n_per_cell=ns,
        n_items=n_items,
        interaction_rate=i_rate,
        interaction_logit=i_logit,
        interaction_arcsine=i_arcsine,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_scale=ci_scale,
        ci_method=ci_method,
        bootstrap_n=bootstrap_n,
        sign_consistent=sign_consistent,
        signs=signs,
        analytic_se_independent=math.sqrt(analytic_var),
        paired=paired,
        warnings=warnings,
    )


def ci_excludes_zero(result: InteractionResult) -> bool:
    """Does the reported CI exclude zero? (Not a gate by itself.)"""
    return result.ci_low > 0.0 or result.ci_high < 0.0
