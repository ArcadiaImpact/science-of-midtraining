"""Seed aggregation for ``sieve_eft_glm_v1`` — mean ± SD across independent repeats of the whole study.

The study's first run (``results/20260918T110621Z``, base + extension merged — "seed 0") is one seed per cell: one
EFT data generation, one EFT training seed, one random-sieve permutation. Repeats re-seed those three (the midtrained
parents are fixed) and land as sibling results dirs in exactly the same layout, each analysed by
:func:`analyze_sieve.run_all` into ``<results_dir>/analysis/`` (``curves.csv``, ``contrast_vs_random.csv``, …).
:func:`aggregate_seeds` reads those per-seed tables and reports, per tag × drop fraction × outcome, the spread across
seeds — so a difference between two arms can be read against training/data-seed scatter and not only against the
eval-noise Wilson CI of a single cell.

Input contract: ``results_dirs = {seed_id: <results dir>}``; each dir carries ``analysis/curves.csv`` and
``analysis/contrast_vs_random.csv`` (re-analysed in place with ``run_all(dir, dir / "analysis", plots=False)`` when
either is missing). Every seed must span the same tags, the same drop-fraction grid and the same primary / agreement
slice (loud ValueError otherwise); a cell missing in one seed (``present`` false / NaN rate) lowers that cell's
``n_seeds`` and is listed — it is never silently dropped.

Tables (each ``<name>.csv`` + ``.json`` + ``.md`` via :func:`analyze_sieve.write_table`)::

    seed_curves            long, tag × fraction × outcome (coin / charter / malformed on the PRIMARY slice, shared on
                           the AGREEMENT slice — the primary slice is a conflict slice and carries no `shared` rate):
                           per-seed rates (``rate_seed<id>``) and n (``n_seed<id>``), n_seeds, mean, SD (ddof = 1),
                           SE = SD / √n_seeds, the between-seed 95 % t-interval (df = n_seeds − 1: with three seeds
                           t = 4.30, so the band is ≈ 2.2× a normal ±2 SE band — DESCRIPTIVE ONLY, clipped to [0, 1]),
                           the pooled Wilson 95 % CI of Σk / Σn ("pooled": treats the seeds as exchangeable draws of
                           one binomial, n ≈ 3 × 3,000 — the narrow, optimistic bound), the mean per-seed Wilson
                           half-width, ``borrowed`` / ``borrowed_from`` (a random tag's drop000 — and its drop100 when
                           it has no own parent eval — is the sibling ΔL tag's cell in every seed, exactly as
                           analyze_sieve copies it), ``seed_invariant`` (the 100 % row: the same un-fine-tuned parent
                           under the greedy eval — nothing re-seeded reaches it; aggregated anyway, flagged)
    seed_headline_coin     wide, rows = fraction (incl. 100 %), columns = tag in parent order, cell = "mean ± SD
                           (n_seeds)" (‡ = borrowed) — the layout of curves_headline, across seeds
    seed_headline_charter  the same for the Charter-pick rate
    seed_contrast          per charter parent × fraction: the paired ΔL − random difference (coin and charter, from
                           each seed's contrast_vs_random.csv) per seed, mean, SD, SE, t-interval, sign pattern
                           ("−−+" in seed order, · = missing) and a verdict token — CONSISTENT_BELOW (every seed < 0
                           and mean + 2·SE < 0), CONSISTENT_ABOVE (every seed > 0 and mean − 2·SE > 0), MIXED
                           (otherwise: signs disagree, or the ±2 SE band straddles 0), INSUFFICIENT (< 2 seeds with a
                           difference — always at drop000, where the random arm IS the ΔL cell), REPLICATE (the 100 %
                           row: the same parent scored twice — eval-noise, not a sieve contrast; its stats are still
                           reported); plus how many seeds' own Newcombe CI excludes 0
    seed_scatter           per tag × fraction (coin): SD across seeds vs the mean single-cell Wilson half-width and
                           binomial SE √(p(1−p)/n) — the scatter decomposition: a ratio ≈ 1 means the cell-to-cell
                           scatter is eval noise, ≫ 1 means the training/data seed adds scatter a single seed's CI does
                           not describe; ``excess_sd`` = √max(SD² − SE², 0) is the between-seed SD net of eval noise
    SEED_SUMMARY.md        inputs, coverage, both headline tables, the contrast verdicts, the scatter ratios and 6–10
                           findings generated from the numbers; ``seed_manifest.json`` alongside

Plots (:mod:`.plots`, seaborn, PDF): ``seed_curves_coin.pdf`` / ``seed_curves_charter.pdf`` (one panel per parent —
control | 190M ΔL vs random | 1B ΔL vs random; small hollow circles = seeds, line = seed mean, band = ± 1 SD) and
``seed_contrast.pdf`` (per-seed paired differences + mean with SD bars per fraction for both parents, zero line).

Everything but ``seed_manifest.json`` (which carries the run timestamp) is byte-identical across re-runs on the same
inputs. No CLI (repo rule) — call :func:`aggregate_seeds`. numpy + pandas only at import; scipy (t quantiles) and
seaborn (plots) are imported lazily, with a t-table fallback.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import analyze_sieve as M

A = M.A
NAN = M.NAN

# ----------------------------------------------------------------- contract
SEED_OUTCOMES: tuple[str, ...] = ("coin", "charter", "shared", "malformed")
OUTCOME_ROLE: dict[str, str] = {"coin": "primary", "charter": "primary", "malformed": "primary", "shared": "agreement"}
SEED_TABLE_NAMES: tuple[str, ...] = ("seed_curves", "seed_headline_coin", "seed_headline_charter", "seed_contrast", "seed_scatter")
SEED_PLOT_NAMES: tuple[str, ...] = ("seed_curves_coin.pdf", "seed_curves_charter.pdf", "seed_contrast.pdf")
SEED_VERDICTS: tuple[str, ...] = ("CONSISTENT_BELOW", "CONSISTENT_ABOVE", "MIXED", "INSUFFICIENT", "REPLICATE")
REPLICATE_VERDICT = "REPLICATE"  # the 100 % row: the same parent scored twice — an eval-noise replicate, not a sieve contrast
MIN_SEEDS_FOR_VERDICT = 2
VERDICT_SE_MULTIPLE = 2.0  # CONSISTENT_* needs every seed on one side AND mean ∓ 2·SE on that side
JUDGED_FROM_FRACTION = 0.10  # the SPEC's E6.delta_below_random range: every EFT fraction ≥ 10 %
SIGN_NEGATIVE, SIGN_POSITIVE, SIGN_ZERO, SIGN_MISSING = "−", "+", "0", "·"
# Two-sided 95 % Student-t quantiles by df — the fallback when scipy is not importable (nearest lower df is used
# above the table, which is conservative: a larger t).
T_CRITICAL_95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980,
}
REQUIRED_CURVE_COLUMNS: tuple[str, ...] = ("tag", "cell", "fraction", "slice", "role", "present", "borrowed", "borrowed_from", "n", *SEED_OUTCOMES)
REQUIRED_CONTRAST_COLUMNS: tuple[str, ...] = (
    "tag", "cell", "fraction", "random_tag", "random_borrowed", "paired_kind", "n", "coin", "charter", "random_n", "random_coin", "random_charter",
    "paired_coin_diff", "paired_coin_excludes_zero", "paired_charter_diff", "paired_charter_excludes_zero",
)
CONTRAST_OUTCOMES: tuple[str, ...] = ("coin", "charter")


# ----------------------------------------------------------------- small helpers
def _bool(value: Any) -> bool:
    """A CSV-round-tripped boolean: real bools pass through, strings compare case-insensitively to 'true', NaN is False."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return False
    return str(value).strip().lower() == "true"


def _text(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _mean(values: Sequence[Any]) -> float:
    finite = [float(v) for v in values if M._finite(v)]
    return float(np.mean(finite)) if finite else NAN


def _median(values: Sequence[Any]) -> float:
    finite = [float(v) for v in values if M._finite(v)]
    return float(np.median(finite)) if finite else NAN


def _first(frame: pd.DataFrame, **conditions: Any) -> pd.Series | None:
    return M._first_row(frame, **conditions)


def _index_rows(frame: pd.DataFrame, keys: Sequence[str]) -> dict[tuple[Any, ...], pd.Series]:
    """(key values) → row; the first row wins when keys repeat."""
    index: dict[tuple[Any, ...], pd.Series] = {}
    for _, row in frame.iterrows():
        key = tuple(row[k] for k in keys)
        index.setdefault(key, row)
    return index


def _int_column(frame: pd.DataFrame, column: str) -> None:
    """Integers with blanks (not '3000.000' / 'nan') in every output format: object dtype of int | None."""
    frame[column] = pd.Series([int(v) if M._finite(v) else None for v in frame[column]], dtype=object, index=frame.index)


# ----------------------------------------------------------------- statistics
def t_critical(df: Any, confidence: float = 0.95) -> float:
    """Two-sided Student-t quantile for ``df`` degrees of freedom: scipy when importable, else :data:`T_CRITICAL_95`
    (nearest lower df; 95 % only). NaN for df < 1."""
    if not M._finite(df) or int(df) < 1:
        return NAN
    df = int(df)
    try:
        from scipy.stats import t as student_t
    except ImportError:
        if confidence != 0.95:
            return NAN
        return T_CRITICAL_95[max(k for k in T_CRITICAL_95 if k <= df)]
    return float(student_t.ppf(0.5 + confidence / 2.0, df))


def seed_stats(values: Sequence[Any]) -> dict[str, Any]:
    """Across the finite ``values``: n_seeds, mean, SD (ddof = 1), SE = SD / √n, df = n − 1, the t quantile and the
    two-sided 95 % t-interval (SD / SE / interval NaN below two values; mean NaN with none)."""
    finite = np.array([float(v) for v in values if M._finite(v)], dtype=float)
    n = int(finite.size)
    mean = float(finite.mean()) if n else NAN
    sd = float(finite.std(ddof=1)) if n >= 2 else NAN
    se = sd / math.sqrt(n) if n >= 2 else NAN
    crit = t_critical(n - 1) if n >= 2 else NAN
    return {
        "n_seeds": n, "mean": mean, "sd": sd, "se": se, "t_df": max(n - 1, 0), "t_crit": crit,
        "t_lo": mean - crit * se if n >= 2 else NAN, "t_hi": mean + crit * se if n >= 2 else NAN,
    }


def sign_pattern(values: Sequence[Any]) -> str:
    """One glyph per seed in seed order: − / + / 0, · when the seed has no value (e.g. '−−+')."""
    glyphs = []
    for value in values:
        if not M._finite(value):
            glyphs.append(SIGN_MISSING)
        elif float(value) < 0:
            glyphs.append(SIGN_NEGATIVE)
        elif float(value) > 0:
            glyphs.append(SIGN_POSITIVE)
        else:
            glyphs.append(SIGN_ZERO)
    return "".join(glyphs)


def contrast_verdict(values: Sequence[Any]) -> str:
    """INSUFFICIENT (< 2 finite values); CONSISTENT_BELOW (every value < 0 and mean + 2·SE < 0); CONSISTENT_ABOVE
    (every value > 0 and mean − 2·SE > 0); MIXED otherwise — signs disagree, or they agree but the ±2 SE band straddles
    0 (the sign pattern next to it tells the two apart)."""
    finite = [float(v) for v in values if M._finite(v)]
    if len(finite) < MIN_SEEDS_FOR_VERDICT:
        return "INSUFFICIENT"
    stats = seed_stats(finite)
    if all(v < 0 for v in finite) and stats["mean"] + VERDICT_SE_MULTIPLE * stats["se"] < 0:
        return "CONSISTENT_BELOW"
    if all(v > 0 for v in finite) and stats["mean"] - VERDICT_SE_MULTIPLE * stats["se"] > 0:
        return "CONSISTENT_ABOVE"
    return "MIXED"


# ----------------------------------------------------------------- inputs
def load_seed_tables(seed: int, results_dir: str | Path, notes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """``analysis/curves.csv`` and ``analysis/contrast_vs_random.csv`` of one seed's results dir (fractions snapped to
    the grid); when either is missing the dir is re-analysed in place with ``run_all(results_dir, results_dir /
    "analysis", plots=False)`` first (noted; the third return value says so). ValueError on a missing dir or a table
    without the columns this module reads."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise ValueError(f"seed {seed}: results dir {results_dir} does not exist")
    analysis_dir = results_dir / "analysis"
    paths = {name: analysis_dir / f"{name}.csv" for name in ("curves", "contrast_vs_random")}
    reanalysed = False
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        notes.append(f"seed {seed}: {', '.join(missing)} missing under {analysis_dir} — re-ran analyze_sieve.run_all(plots=False) on {results_dir}")
        M.run_all(results_dir, analysis_dir, plots=False)
        reanalysed = True
        still_missing = [p.name for p in paths.values() if not p.is_file()]
        if still_missing:
            raise ValueError(f"seed {seed}: run_all did not produce {still_missing} under {analysis_dir}")
    curves = pd.read_csv(paths["curves"])
    contrast = pd.read_csv(paths["contrast_vs_random"])
    for frame, required, name in ((curves, REQUIRED_CURVE_COLUMNS, "curves.csv"), (contrast, REQUIRED_CONTRAST_COLUMNS, "contrast_vs_random.csv")):
        absent = [c for c in required if c not in frame.columns]
        if absent:
            raise ValueError(f"seed {seed}: {analysis_dir / name} lacks columns {absent} — not an analyze_sieve.run_all output?")
        frame["fraction"] = [M.norm_fraction(f) if M._finite(f) else NAN for f in frame["fraction"]]
    return curves, contrast, reanalysed


def validate_seeds(curves: Mapping[int, pd.DataFrame]) -> dict[str, Any]:
    """Every seed must span the same tags, the same drop-fraction grid, the same primary slice and the same agreement
    slice — ValueError naming the seed and the disagreement otherwise. Returns the shared signature
    {tags, fractions, primary_slice, agreement_slice}."""
    reference: tuple[int, dict[str, Any]] | None = None
    for seed in sorted(curves):
        frame = curves[seed]
        prim = frame[frame["role"] == "primary"]
        if prim.empty:
            raise ValueError(f"seed {seed}: curves.csv has no primary-slice rows")
        slices = sorted({str(s) for s in prim["slice"].dropna()})
        if len(slices) != 1:
            raise ValueError(f"seed {seed}: primary rows span {len(slices)} slices {slices}")
        agreement = frame[frame["role"] == "agreement"]
        agreement_slices = sorted({str(s) for s in agreement["slice"].dropna()})
        signature = {
            "tags": M.order_tags(prim["tag"]), "fractions": M.grid_of(prim), "primary_slice": slices[0],
            "agreement_slice": agreement_slices[0] if len(agreement_slices) == 1 else None,
        }
        if reference is None:
            reference = (seed, signature)
            continue
        reference_seed, expected = reference
        for key in ("tags", "fractions", "primary_slice", "agreement_slice"):
            if signature[key] != expected[key]:
                raise ValueError(
                    f"seed {seed} disagrees with seed {reference_seed} on {key}: {signature[key]!r} vs {expected[key]!r} — "
                    "every seed must span the same tags, drop-fraction grid and slices (re-run pull_results / run_all on the odd one out)"
                )
    assert reference is not None  # curves is non-empty by construction
    return reference[1]


# ----------------------------------------------------------------- tables
def seed_curve_columns(seeds: Sequence[int]) -> tuple[str, ...]:
    return (
        "tag", "cell", "fraction", "drop_pct", "outcome", "slice", "role", "n_seeds", "seeds_present", "n_total",
        *(f"rate_seed{s}" for s in seeds), *(f"n_seed{s}" for s in seeds),
        "mean", "sd", "se", "t_df", "t_crit", "t_lo", "t_hi", "pooled_rate", "pooled_lo", "pooled_hi", "wilson_halfwidth_mean", "binomial_se_mean",
        "borrowed", "borrowed_from", "n_seeds_borrowed", "seed_invariant",
    )


def seed_curves_table(curves: Mapping[int, pd.DataFrame], tags: Sequence[str], fractions: Sequence[float], primary_slice: str, agreement_slice: str | None) -> pd.DataFrame:
    """tag × fraction × outcome across seeds (see the module docstring for the columns). coin / charter / malformed are
    read off each seed's PRIMARY row, shared off its AGREEMENT row; a seed contributes when the cell is present and
    the rate finite. ``borrowed`` = the cell was borrowed from the sibling ΔL tag in at least one seed (the count in
    ``n_seeds_borrowed``); ``seed_invariant`` = the 100 % row."""
    seeds = sorted(curves)
    index = {seed: _index_rows(frame[frame["role"].isin(("primary", "agreement"))], ("tag", "cell", "role")) for seed, frame in curves.items()}
    rows: list[dict[str, Any]] = []
    for tag in tags:
        for fraction in fractions:
            cell = M.cell_name(fraction)
            primary_rows = {seed: index[seed].get((tag, cell, "primary")) for seed in seeds}
            borrowed_seeds = [seed for seed, r in primary_rows.items() if r is not None and _bool(r["borrowed"])]
            borrowed_from = next((_text(primary_rows[seed]["borrowed_from"]) for seed in borrowed_seeds), None)
            for outcome in SEED_OUTCOMES:
                role = OUTCOME_ROLE[outcome]
                per_seed = {seed: index[seed].get((tag, cell, role)) for seed in seeds}
                rates = {seed: (M._num(r[outcome]) if r is not None and _bool(r["present"]) else NAN) for seed, r in per_seed.items()}
                ns = {seed: (M._num(r["n"]) if r is not None and M._finite(rates[seed]) else NAN) for seed, r in per_seed.items()}
                usable = [seed for seed in seeds if M._finite(rates[seed])]
                stats = seed_stats([rates[s] for s in usable])
                with_n = [s for s in usable if M._finite(ns[s]) and ns[s] > 0]
                k_total = sum(M.count_from_rate(rates[s], ns[s]) for s in with_n)
                n_total = sum(ns[s] for s in with_n)
                pooled_rate = k_total / n_total if n_total > 0 else NAN
                pooled_lo, pooled_hi = M.wilson(k_total, n_total) if n_total > 0 else (NAN, NAN)
                halfwidths = [(hi - lo) / 2.0 for lo, hi in (M.rate_ci(rates[s], ns[s]) for s in with_n)]
                binomial = [math.sqrt(max(rates[s] * (1.0 - rates[s]), 0.0) / ns[s]) for s in with_n]
                rows.append(
                    {
                        "tag": tag, "cell": cell, "fraction": fraction, "drop_pct": M.F.fraction_pct(fraction), "outcome": outcome,
                        "slice": primary_slice if role == "primary" else agreement_slice, "role": role,
                        "n_seeds": stats["n_seeds"], "seeds_present": ",".join(str(s) for s in usable), "n_total": int(n_total),
                        **{f"rate_seed{s}": rates[s] for s in seeds}, **{f"n_seed{s}": ns[s] for s in seeds},
                        "mean": stats["mean"], "sd": stats["sd"], "se": stats["se"], "t_df": stats["t_df"], "t_crit": stats["t_crit"],
                        "t_lo": max(0.0, stats["t_lo"]) if M._finite(stats["t_lo"]) else NAN, "t_hi": min(1.0, stats["t_hi"]) if M._finite(stats["t_hi"]) else NAN,
                        "pooled_rate": pooled_rate, "pooled_lo": pooled_lo, "pooled_hi": pooled_hi,
                        "wilson_halfwidth_mean": _mean(halfwidths), "binomial_se_mean": _mean(binomial),
                        "borrowed": bool(borrowed_seeds), "borrowed_from": borrowed_from, "n_seeds_borrowed": len(borrowed_seeds),
                        "seed_invariant": bool(fraction >= M.NO_EFT_FRACTION),
                    }
                )
    frame = pd.DataFrame(rows, columns=seed_curve_columns(seeds))
    for seed in seeds:
        _int_column(frame, f"n_seed{seed}")
    return frame


def _seed_cell_text(row: pd.Series | None) -> str:
    if row is None or int(row["n_seeds"]) == 0:
        return "not run"
    n = int(row["n_seeds"])
    mark = f" {M.BORROWED_MARK}" if _bool(row["borrowed"]) else ""
    if n == 1:
        return f"{M._fmt(row['mean'])} (1 seed){mark}"
    return f"{M._fmt(row['mean'])} ± {M._fmt(row['sd'])} ({n}){mark}"


def seed_headline_table(seed_curves: pd.DataFrame, tags: Sequence[str], outcome: str = "coin", fractions: Sequence[float] | None = None) -> pd.DataFrame:
    """Wide: index = drop fraction label (every grid fraction, 100 % included), one column per tag in parent order →
    "mean ± SD (n_seeds)"; ‡ = borrowed from the sibling ΔL tag; "not run" when no seed has the cell."""
    sub = seed_curves[seed_curves["outcome"] == outcome]
    grid = tuple(M.norm_fraction(f) for f in fractions) if fractions is not None else M.grid_of(sub)
    table = {tag: [_seed_cell_text(_first(sub, tag=tag, cell=M.cell_name(f))) for f in grid] for tag in tags}
    return pd.DataFrame(table, index=pd.Index([M.pct_label(f) for f in grid], name="drop_fraction"))


def seed_contrast_columns(seeds: Sequence[int]) -> tuple[str, ...]:
    columns: list[str] = ["tag", "random_tag", "parent", "cell", "fraction", "drop_pct", "paired_kind", "random_borrowed", "n_seeds", "seeds_present"]
    for outcome in CONTRAST_OUTCOMES:
        columns += [f"delta_{outcome}_mean", f"random_{outcome}_mean", *(f"{outcome}_diff_seed{s}" for s in seeds),
                    f"{outcome}_diff_mean", f"{outcome}_sd", f"{outcome}_se", f"{outcome}_diff_t_lo", f"{outcome}_diff_t_hi",
                    f"{outcome}_sign_pattern", f"{outcome}_verdict", f"{outcome}_seeds_excluding_zero"]
    return tuple(columns)


def seed_contrast_table(contrasts: Mapping[int, pd.DataFrame], tags: Sequence[str], fractions: Sequence[float]) -> pd.DataFrame:
    """Per charter ΔL tag whose random sibling is among ``tags`` × grid fraction: the paired difference (ΔL cell −
    the same parent's random cell) per seed for coin and charter, its mean / SD / SE / t-interval, the sign pattern,
    the verdict token (REPLICATE at 100 %, where the pair is the same parent scored twice) and how many seeds' own
    Newcombe CI excludes 0. ``delta_*_mean`` / ``random_*_mean`` are the two arms' seed-mean rates over the seeds that
    carry the pair. Empty when no random tag is present."""
    seeds = sorted(contrasts)
    paired_with = {sibling: random for random, sibling in M.RANDOM_TAGS.items() if random in tags and sibling in tags}
    index = {seed: _index_rows(frame, ("tag", "cell")) for seed, frame in contrasts.items()}
    rows: list[dict[str, Any]] = []
    for tag in M.order_tags(paired_with):
        random_tag = paired_with[tag]
        for fraction in fractions:
            cell = M.cell_name(fraction)
            per_seed = {seed: index[seed].get((tag, cell)) for seed in seeds}
            kinds = sorted({_text(r["paired_kind"]) for r in per_seed.values() if r is not None and _text(r["paired_kind"])})
            row: dict[str, Any] = {
                "tag": tag, "random_tag": random_tag, "parent": M.TAG_PARENTS.get(tag, tag), "cell": cell, "fraction": fraction, "drop_pct": M.F.fraction_pct(fraction),
                "paired_kind": " | ".join(kinds) if kinds else None,
                "random_borrowed": any(r is not None and _bool(r["random_borrowed"]) for r in per_seed.values()),
            }
            usable_any: list[int] = []
            for outcome in CONTRAST_OUTCOMES:
                diffs = {seed: (M._num(r[f"paired_{outcome}_diff"]) if r is not None else NAN) for seed, r in per_seed.items()}
                usable = [seed for seed in seeds if M._finite(diffs[seed])]
                usable_any = sorted(set(usable_any) | set(usable))
                stats = seed_stats([diffs[s] for s in usable])
                row.update(
                    {
                        f"delta_{outcome}_mean": _mean([per_seed[s][outcome] for s in usable]), f"random_{outcome}_mean": _mean([per_seed[s][f"random_{outcome}"] for s in usable]),
                        **{f"{outcome}_diff_seed{s}": diffs[s] for s in seeds},
                        f"{outcome}_diff_mean": stats["mean"], f"{outcome}_sd": stats["sd"], f"{outcome}_se": stats["se"], f"{outcome}_diff_t_lo": stats["t_lo"], f"{outcome}_diff_t_hi": stats["t_hi"],
                        f"{outcome}_sign_pattern": sign_pattern([diffs[s] for s in seeds]),
                        f"{outcome}_verdict": REPLICATE_VERDICT if fraction >= M.NO_EFT_FRACTION else contrast_verdict([diffs[s] for s in seeds]),
                        f"{outcome}_seeds_excluding_zero": sum(1 for s in usable if _bool(per_seed[s][f"paired_{outcome}_excludes_zero"])),
                    }
                )
            row["n_seeds"] = len(usable_any)
            row["seeds_present"] = ",".join(str(s) for s in usable_any)
            rows.append(row)
    return pd.DataFrame(rows, columns=seed_contrast_columns(seeds))


SCATTER_COLUMNS: tuple[str, ...] = (
    "tag", "cell", "fraction", "drop_pct", "outcome", "n_seeds", "mean", "sd_seeds", "wilson_halfwidth_mean", "binomial_se_mean",
    "sd_over_halfwidth", "sd_over_binomial_se", "excess_sd", "seed_invariant", "borrowed",
)


def seed_scatter_table(seed_curves: pd.DataFrame, outcome: str = "coin") -> pd.DataFrame:
    """Scatter decomposition per tag × fraction: the SD of the ``outcome`` rate across seeds against the eval-noise
    scale of one cell — the mean Wilson 95 % half-width and the mean binomial SE √(p(1−p)/n). ``sd_over_binomial_se``
    ≈ 1 → the seeds scatter like eval noise; ≫ 1 → training/data seed dominates; ``excess_sd`` = √max(SD² − SE², 0)."""
    sub = seed_curves[seed_curves["outcome"] == outcome]
    rows: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        sd, half, se = M._num(r["sd"]), M._num(r["wilson_halfwidth_mean"]), M._num(r["binomial_se_mean"])
        rows.append(
            {
                "tag": r["tag"], "cell": r["cell"], "fraction": r["fraction"], "drop_pct": r["drop_pct"], "outcome": outcome, "n_seeds": int(r["n_seeds"]), "mean": M._num(r["mean"]),
                "sd_seeds": sd, "wilson_halfwidth_mean": half, "binomial_se_mean": se,
                "sd_over_halfwidth": sd / half if M._finite(sd) and M._finite(half) and half > 0 else NAN,
                "sd_over_binomial_se": sd / se if M._finite(sd) and M._finite(se) and se > 0 else NAN,
                "excess_sd": math.sqrt(max(sd * sd - se * se, 0.0)) if M._finite(sd) and M._finite(se) else NAN,
                "seed_invariant": _bool(r["seed_invariant"]), "borrowed": _bool(r["borrowed"]),
            }
        )
    return pd.DataFrame(rows, columns=SCATTER_COLUMNS)


# ----------------------------------------------------------------- summary
def _pm(row: pd.Series | None, mean_key: str = "mean", sd_key: str = "sd", n_key: str = "n_seeds", signed: bool = False) -> str:
    if row is None or not M._finite(row[mean_key]):
        return "—"
    n = int(row[n_key]) if M._finite(row[n_key]) else 0
    if n < 2 or not M._finite(row[sd_key]):
        return f"{M._fmt(row[mean_key], signed=signed)} ({n} seed{'s' if n != 1 else ''})"
    return f"{M._fmt(row[mean_key], signed=signed)} ± {M._fmt(row[sd_key])} ({n})"


def seed_findings(seed_curves: pd.DataFrame, seed_contrast: pd.DataFrame, scatter: pd.DataFrame, tags: Sequence[str], fractions: Sequence[float], seeds: Sequence[int]) -> list[str]:
    """6–10 one-line findings generated from the numbers: coverage, coin headline, one paired-contrast bullet per charter
    parent, the random sieves, the scatter decomposition, charter headline, competence, the seed-invariant row and the
    interval caveat — each only when its inputs exist, so at most 10 by construction."""
    bullets: list[str] = []
    n_seeds = len(seeds)
    coin = seed_curves[seed_curves["outcome"] == "coin"]
    charter = seed_curves[seed_curves["outcome"] == "charter"]
    shared = seed_curves[seed_curves["outcome"] == "shared"]
    eft = list(M.eft_fractions(fractions))
    focus = 0.5 if 0.5 in fractions else (eft[-1] if eft else fractions[-1])
    # 1 — coverage
    complete = int((coin["n_seeds"] == n_seeds).sum())
    fewer = [f"{r.tag}/{r.cell} ({int(r.n_seeds)} seed{'s' if int(r.n_seeds) != 1 else ''})" for r in coin.itertuples() if int(r.n_seeds) < n_seeds]
    bullets.append(f"Coverage: {complete} / {len(coin)} tag × fraction cells carry all {n_seeds} seeds" + (f"; fewer: {', '.join(fewer)}" if fewer else "") + ".")
    # 2 — headline coin at the focus fraction
    parts = [f"`{tag}` {_pm(_first(coin, tag=tag, cell=M.cell_name(focus)))}" for tag in tags if _first(coin, tag=tag, cell=M.cell_name(focus)) is not None]
    if parts:
        bullets.append(f"Coin-pick rate at {M.pct_label(focus)} (seed mean ± SD (n_seeds)): " + "; ".join(parts) + ".")
    # 3 — the paired contrast, one bullet per parent
    judged_fractions = [f for f in eft if f >= JUDGED_FROM_FRACTION]
    for tag in M.order_tags(set(seed_contrast["tag"])) if not seed_contrast.empty else []:
        sub = seed_contrast[seed_contrast["tag"] == tag]
        at_focus = _first(sub, cell=M.cell_name(focus))
        judged = sub[sub["fraction"].isin(judged_fractions)]
        counts = Counter(str(v) for v in judged["coin_verdict"])
        below = [M.pct_label(f) for f in judged.loc[judged["coin_verdict"] == "CONSISTENT_BELOW", "fraction"]]
        text = f"{M.TAG_PARENTS.get(tag, tag)} — coin(ΔL sieve) − coin(random sieve), same parent"
        if at_focus is not None:
            text += f", at {M.pct_label(focus)}: {_pm(at_focus, 'coin_diff_mean', 'coin_sd', signed=True)}, pattern {at_focus['coin_sign_pattern']}, {at_focus['coin_verdict']}"
        if len(judged):
            text += f"; over the {len(judged)} EFT fractions ≥ {M.pct_label(JUDGED_FROM_FRACTION)}: " + ", ".join(f"{k} × {counts[k]}" for k in sorted(counts)) + (f" (below at {', '.join(below)})" if below else "")
        bullets.append(text + ".")
    # 4 — the random sieves (one bullet): each random arm's borrowed drop000 vs its 1–20 % range, and the control's span
    shallow = [f for f in eft if 0.0 < f <= 0.20]
    parts = []
    for tag in [t for t in tags if t in M.RANDOM_TAGS]:
        base = _first(coin, tag=tag, cell=M.cell_name(0.0))
        means = [M._num(r["mean"]) for f in shallow if (r := _first(coin, tag=tag, cell=M.cell_name(f))) is not None and M._finite(r["mean"])]
        if base is not None and M._finite(base["mean"]) and means:
            parts.append(f"`{tag}` {M._fmt(base['mean'])} at 0 % ({M.BORROWED_MARK} borrowed from `{M.RANDOM_TAGS[tag]}`) vs {M._fmt(min(means))}–{M._fmt(max(means))} across 1–20 % (max shift {M._fmt(max(abs(m - base['mean']) for m in means))})")
    control_rows = [r for f in eft if (r := _first(coin, tag=M.CONTROL_TAG, cell=M.cell_name(f))) is not None and M._finite(r["mean"])]
    if control_rows:
        control_means = [M._num(r["mean"]) for r in control_rows]
        control_sds = [M._num(r["sd"]) for r in control_rows if M._finite(r["sd"])]
        parts.append(f"`{M.CONTROL_TAG}` (dilution reference) spans {M._fmt(min(control_means))}–{M._fmt(max(control_means))} over the {len(control_rows)} EFT fractions" + (f", largest between-seed SD {M._fmt(max(control_sds))}" if control_sds else ""))
    if parts:
        bullets.append("Random sieves (seed-mean coin): " + "; ".join(parts) + ".")
    # 5 — scatter decomposition per tag
    ratios = []
    for tag in tags:
        sub = scatter[(scatter["tag"] == tag) & (scatter["fraction"] < M.NO_EFT_FRACTION) & (scatter["n_seeds"] >= 2) & ~scatter["borrowed"].astype(bool)]
        ratio = _median(sub["sd_over_binomial_se"])
        if M._finite(ratio):
            ratios.append(f"`{tag}` {ratio:.1f}×")
    if ratios:
        bullets.append("Scatter decomposition (median over EFT fractions of SD across seeds ÷ single-cell binomial SE): " + ", ".join(ratios) + " — ≈ 1× means the seeds scatter like eval noise, ≫ 1× means the training / data seed adds scatter a single seed's Wilson CI does not describe.")
    # 6 — charter at the focus
    parts = [f"`{tag}` {_pm(_first(charter, tag=tag, cell=M.cell_name(focus)))}" for tag in tags if (r := _first(charter, tag=tag, cell=M.cell_name(focus))) is not None and M._finite(r["mean"])]
    if parts:
        bullets.append(f"Charter-pick rate at {M.pct_label(focus)} (seed mean ± SD (n_seeds)): " + "; ".join(parts) + ".")
    # 7 — competence (shared on the agreement slice) at 0 % vs the focus, ΔL tags
    parts = []
    for tag in [t for t in tags if t in M.SIEVE_TAGS]:
        a, b = _first(shared, tag=tag, cell=M.cell_name(0.0)), _first(shared, tag=tag, cell=M.cell_name(focus))
        if a is not None and b is not None and M._finite(a["mean"]) and M._finite(b["mean"]):
            parts.append(f"`{tag}` {_pm(a)} → {_pm(b)}")
    if parts:
        bullets.append(f"Agreement competence (`shared` rate, agreement slice) at 0 % → {M.pct_label(focus)}: " + "; ".join(parts) + ".")
    # 8 — the seed-invariant row
    parent_rows = [(tag, r) for tag in tags if (r := _first(coin, tag=tag, cell=M.cell_name(M.NO_EFT_FRACTION))) is not None and M._finite(r["sd"])]
    if parent_rows:
        bullets.append("100 % row (the parent, no EFT — seed-invariant: nothing re-seeded reaches it): SD across seeds " + ", ".join(f"`{tag}` {M._fmt(r['sd'])}" for tag, r in parent_rows) + " — anything above 0 here is eval-replicate noise of the same model, not a seed effect.")
    # 9 — the interval caveat
    crit = t_critical(n_seeds - 1) if n_seeds >= 2 else NAN
    if M._finite(crit):
        bullets.append(f"Intervals: the between-seed 95 % t-interval uses df = {n_seeds - 1} (t = {crit:.2f}, ≈ {crit / M.Z95:.1f}× a normal ±1.96 SE band) — descriptive only; the pooled Wilson CI treats the seeds as exchangeable draws of one binomial (n ≈ {n_seeds} × cell n) and is the narrow, optimistic bound. Read the sign patterns and verdict tokens, not one interval.")
    else:
        bullets.append("Intervals: a single seed — no between-seed SD / SE / t-interval; every verdict is INSUFFICIENT. Add seeds.")
    return bullets


def build_seed_summary(context: Mapping[str, Any]) -> str:
    """``SEED_SUMMARY.md``: inputs, coverage, headline tables (coin, charter), the paired-contrast verdicts, the scatter
    ratios, findings, notes, outputs. Deterministic (no timestamp — that lives in seed_manifest.json)."""
    seeds: list[int] = list(context["seeds"])
    tags: list[str] = list(context["tags"])
    fractions: tuple[float, ...] = tuple(context["fractions"])
    seed_curves: pd.DataFrame = context["seed_curves"]
    seed_contrast: pd.DataFrame = context["seed_contrast"]
    scatter: pd.DataFrame = context["scatter"]
    lines = [f"# {M.EXPERIMENT} — seed aggregation ({len(seeds)} seed{'s' if len(seeds) != 1 else ''})", ""]
    lines.append(f"Mean ± SD across independent repeats of the study (EFT data generation, EFT training seed and random-sieve permutation re-seeded; midtrained parents fixed). Primary slice `{context['primary_slice']}`" + (f"; agreement competence `{context['agreement_slice']}`" if context.get("agreement_slice") else "") + f"; {len(fractions)}-fraction grid; tags in parent order: {', '.join(f'`{t}`' for t in tags)}.")
    lines += ["", "## Inputs", ""]
    for seed in seeds:
        lines.append(f"- seed {seed}: `{context['results_dirs'][seed]}`" + (" (re-analysed by aggregate_seeds — analysis/ was missing)" if seed in context["reanalysed_seeds"] else ""))
    lines.append(f"- {M.tag_legend(tags)}")
    if context["cells_incomplete"]:
        lines.append(f"- cells with fewer than {len(seeds)} seeds: " + ", ".join(context["cells_incomplete"]))
    else:
        lines.append(f"- every tag × fraction cell carries all {len(seeds)} seeds")
    coin = seed_curves[seed_curves["outcome"] == "coin"]
    n_cell = sorted({int(v) for v in coin["n_total"] if M._finite(v) and v > 0})
    if n_cell:
        lines.append(f"- pooled n per cell (Σ over seeds of the primary-slice n): {', '.join(str(v) for v in n_cell)}")
    for outcome in ("coin", "charter"):
        lines += ["", f"## Seed-mean headline — {outcome}-pick rate on `{context['primary_slice']}`", "", f"Cells = mean ± SD across seeds (n_seeds); {M.BORROWED_MARK} = borrowed from the sibling ΔL tag in every seed (a random tag's drop000, its drop100 when it has no own parent eval); the 100 % row is seed-invariant (the same parent, greedy eval).", ""]
        lines.append(M.frame_to_markdown(context[f"headline_{outcome}"].reset_index()))
    lines += ["", "## Paired contrast across seeds — coin(ΔL sieve) − coin(random sieve), same parent", ""]
    if seed_contrast.empty:
        lines.append("_(no random tags — no paired contrast)_")
    else:
        lines.append(f"Verdicts: CONSISTENT_BELOW = every seed < 0 and mean + {VERDICT_SE_MULTIPLE:g}·SE < 0; CONSISTENT_ABOVE likewise above; MIXED = signs disagree or the ±{VERDICT_SE_MULTIPLE:g} SE band straddles 0; INSUFFICIENT = fewer than {MIN_SEEDS_FOR_VERDICT} seeds with a pair (drop000 always: the random arm is the ΔL cell); {REPLICATE_VERDICT} = the 100 % row, the same parent scored twice (eval-noise, not a sieve contrast). `excl0` = seeds whose own Newcombe CI excludes 0.")
        lines.append("")
        table = pd.DataFrame(
            [
                {"parent": r["parent"], "drop_fraction": M.pct_label(r["fraction"]), "kind": r["paired_kind"], "coin diff (mean ± SD (n))": _pm(r, "coin_diff_mean", "coin_sd", signed=True),
                 "pattern": r["coin_sign_pattern"], "verdict": r["coin_verdict"], "excl0": int(r["coin_seeds_excluding_zero"]),
                 "charter diff": _pm(r, "charter_diff_mean", "charter_sd", signed=True), "charter pattern": r["charter_sign_pattern"], "charter verdict": r["charter_verdict"]}
                for _, r in seed_contrast.iterrows()
            ]
        )
        lines.append(M.frame_to_markdown(table))
    lines += ["", "## Scatter decomposition — SD across seeds vs single-cell eval noise (coin)", "", "Per tag, the median over the EFT fractions (own cells, ≥ 2 seeds) of SD across seeds ÷ the mean binomial SE √(p(1−p)/n) of one cell, and ÷ the mean Wilson 95 % half-width; the full per-fraction table is `seed_scatter.*`.", ""]
    rows = []
    for tag in tags:
        sub = scatter[(scatter["tag"] == tag) & (scatter["fraction"] < M.NO_EFT_FRACTION) & (scatter["n_seeds"] >= 2) & ~scatter["borrowed"].astype(bool)]
        rows.append({"tag": tag, "eft_cells": int(len(sub)), "median_sd_seeds": _median(sub["sd_seeds"]), "median_binomial_se": _median(sub["binomial_se_mean"]), "median_sd_over_binomial_se": _median(sub["sd_over_binomial_se"]), "median_sd_over_halfwidth": _median(sub["sd_over_halfwidth"]), "max_excess_sd": float(sub["excess_sd"].max()) if len(sub) and np.isfinite(sub["excess_sd"].astype(float)).any() else NAN})
    lines.append(M.frame_to_markdown(pd.DataFrame(rows)))
    lines += ["", "## Findings", ""]
    lines += [f"- {b}" for b in context["findings"]]
    if context["notes"]:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in context["notes"]]
    lines += ["", "## Outputs", "", ", ".join(f"`{name}`" for name in context["outputs"]), ""]
    return "\n".join(lines)


# ----------------------------------------------------------------- driver
def aggregate_seeds(results_dirs: Mapping[int, str | Path], out_dir: str | Path, *, plots: bool | None = True) -> dict[str, Any]:
    """Aggregate the per-seed analyses of ``results_dirs`` ({seed id → results dir}) into ``out_dir`` and return the
    manifest dict (also ``seed_manifest.json``). ``plots=True`` requires seaborn (loud ImportError); ``None`` draws
    PDFs when seaborn is importable and notes otherwise; ``False`` writes tables only. Raises ValueError on an empty
    mapping, a missing results dir, or seeds that disagree on tags / grid / slices."""
    if not results_dirs:
        raise ValueError("results_dirs is empty — nothing to aggregate")
    dirs = {int(seed): Path(path) for seed, path in results_dirs.items()}
    seeds = sorted(dirs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    written: list[Path] = []

    want_plots = plots
    if want_plots is None:
        try:
            import seaborn  # noqa: F401

            want_plots = True
        except ImportError:
            want_plots = False
            notes.append("plots skipped: seaborn not installed (tables written)")
    if want_plots:
        A._plotting()  # loud ImportError when plots=True and seaborn is absent

    # ---- load + validate
    curves: dict[int, pd.DataFrame] = {}
    contrasts: dict[int, pd.DataFrame] = {}
    reanalysed: list[int] = []
    for seed in seeds:
        curves[seed], contrasts[seed], did_rerun = load_seed_tables(seed, dirs[seed], notes)
        if did_rerun:
            reanalysed.append(seed)
    if len(seeds) < MIN_SEEDS_FOR_VERDICT:
        notes.append(f"only {len(seeds)} seed — SD / SE / t-intervals are NaN and every contrast verdict is INSUFFICIENT")
    signature = validate_seeds(curves)
    tags: list[str] = list(signature["tags"])
    fractions: tuple[float, ...] = tuple(signature["fractions"])
    primary_slice: str = signature["primary_slice"]
    agreement_slice: str | None = signature["agreement_slice"]
    if agreement_slice is None:
        notes.append("no agreement-slice rows in the seeds' curves — the `shared` outcome is all NaN")

    # ---- tables
    seed_curves = seed_curves_table(curves, tags, fractions, primary_slice, agreement_slice)
    headline_coin = seed_headline_table(seed_curves, tags, "coin", fractions)
    headline_charter = seed_headline_table(seed_curves, tags, "charter", fractions)
    seed_contrast = seed_contrast_table(contrasts, tags, fractions)
    if seed_contrast.empty:
        notes.append(f"no random tag among {tags} — seed_contrast is empty (no paired ΔL − random difference)")
    scatter = seed_scatter_table(seed_curves, "coin")
    coin = seed_curves[seed_curves["outcome"] == "coin"]
    cells_incomplete = [f"{r.tag}/{r.cell}: seeds [{r.seeds_present}]" for r in coin.itertuples() if int(r.n_seeds) < len(seeds)]
    if cells_incomplete:
        notes.append(f"cells with fewer than {len(seeds)} seeds ({len(cells_incomplete)}): " + "; ".join(cells_incomplete))
    seeds_text = ", ".join(str(s) for s in seeds)
    written += M.write_table(seed_curves, out_dir, "seed_curves", f"Seed curves — per tag × drop fraction × outcome across seeds {seeds_text}", f"coin / charter / malformed on the primary slice `{primary_slice}`, shared on the agreement slice `{agreement_slice}`. rate_seed<id> / n_seed<id> = that seed's cell (NaN = cell absent in that seed; n_seeds counts the finite ones). sd = SD across seeds (ddof = 1), se = sd / √n_seeds, [t_lo, t_hi] = mean ± t(df = n_seeds − 1) · se clipped to [0, 1] — with three seeds t = 4.30, descriptive only. pooled_* = Wilson 95 % CI of Σk / Σn over the seeds (treats seeds as exchangeable draws of one binomial). wilson_halfwidth_mean / binomial_se_mean = the single-cell eval-noise scale. borrowed = the cell is the sibling ΔL tag's in ≥ 1 seed (n_seeds_borrowed says how many; a random tag's drop000 always, its drop100 when it has no own parent eval). seed_invariant = the 100 % row (the same parent, no EFT — nothing re-seeded reaches it).")
    written += M.write_table(headline_coin.reset_index(), out_dir, "seed_headline_coin", f"Seed-mean headline — coin-pick rate on `{primary_slice}`: mean ± SD (n_seeds) across seeds {seeds_text}", f"Rows = fraction of EFT rows dropped (100 % = the parent, no EFT — seed-invariant); columns = tag in parent order. {M.tag_legend(tags)}. {M.BORROWED_MARK} = borrowed point.")
    written += M.write_table(headline_charter.reset_index(), out_dir, "seed_headline_charter", f"Seed-mean headline — Charter-pick rate on `{primary_slice}`: mean ± SD (n_seeds) across seeds {seeds_text}", f"Rows = fraction of EFT rows dropped (100 % = the parent, no EFT — seed-invariant); columns = tag in parent order. {M.tag_legend(tags)}. {M.BORROWED_MARK} = borrowed point.")
    written += M.write_table(seed_contrast, out_dir, "seed_contrast", f"Seed contrast — paired by parent: ΔL sieve − random sieve (coin and charter) per seed, across seeds {seeds_text}", f"<outcome>_diff_seed<id> = that seed's paired_<outcome>_diff from contrast_vs_random.csv (NaN where the random point is the ΔL cell itself — drop000 always, drop100 when borrowed). <outcome>_diff_mean / <outcome>_sd / <outcome>_se = mean / SD / SE of that difference across seeds; [<outcome>_diff_t_lo, _t_hi] = mean ± t(df = n_seeds − 1) · se (descriptive). <outcome>_sign_pattern = one glyph per seed in seed order ({SIGN_NEGATIVE} / {SIGN_POSITIVE} / {SIGN_ZERO}, {SIGN_MISSING} = missing). <outcome>_verdict: CONSISTENT_BELOW = every seed < 0 and mean + {VERDICT_SE_MULTIPLE:g}·SE < 0; CONSISTENT_ABOVE likewise above 0; MIXED otherwise; INSUFFICIENT = fewer than {MIN_SEEDS_FOR_VERDICT} seeds with a pair; {REPLICATE_VERDICT} = the 100 % row (the same parent scored twice — not a sieve contrast). <outcome>_seeds_excluding_zero = seeds whose own Newcombe 95 % CI excludes 0. delta_*/random_* means are the two arms' seed-mean rates over the paired seeds. paired_kind at 100 % = eval-noise replicate (the same parent scored twice), not a sieve contrast.")
    written += M.write_table(scatter, out_dir, "seed_scatter", f"Seed scatter decomposition — SD of the coin rate across seeds {seeds_text} vs the eval-noise scale of one cell", "sd_seeds = SD across seeds (ddof = 1); wilson_halfwidth_mean = mean over seeds of the cell's Wilson 95 % half-width; binomial_se_mean = mean √(p(1−p)/n). sd_over_binomial_se ≈ 1 → the seeds scatter like eval noise; ≫ 1 → the training / data seed adds scatter a single seed's CI does not describe. excess_sd = √max(sd² − se², 0) = between-seed SD net of eval noise. seed_invariant rows (100 %) are the same parent scored per seed — their sd is pure eval-replicate noise.")

    # ---- plots
    plot_names: list[str] = []
    if want_plots:
        from . import plots as P

        plot_names += P.write_seed_plots(seed_curves, seed_contrast, out_dir, notes, fractions)
        written += [out_dir / name for name in plot_names]

    # ---- summary + manifest
    findings = seed_findings(seed_curves, seed_contrast, scatter, tags, fractions, seeds)
    outputs = sorted({p.name for p in written} | {"SEED_SUMMARY.md", "seed_manifest.json"})
    context = {
        "seeds": seeds, "results_dirs": {s: str(dirs[s]) for s in seeds}, "reanalysed_seeds": reanalysed, "tags": tags, "fractions": fractions, "primary_slice": primary_slice, "agreement_slice": agreement_slice,
        "seed_curves": seed_curves, "seed_contrast": seed_contrast, "scatter": scatter, "headline_coin": headline_coin, "headline_charter": headline_charter,
        "cells_incomplete": cells_incomplete, "findings": findings, "notes": notes, "outputs": outputs,
    }
    (out_dir / "SEED_SUMMARY.md").write_text(build_seed_summary(context), encoding="utf-8")
    verdicts = {tag: {M.pct_label(r["fraction"]): str(r["coin_verdict"]) for _, r in seed_contrast[seed_contrast["tag"] == tag].iterrows()} for tag in M.order_tags(set(seed_contrast["tag"]))} if not seed_contrast.empty else {}
    manifest = {
        "experiment": M.EXPERIMENT, "kind": "seed_aggregation", "run_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "out_dir": str(out_dir), "plots": bool(want_plots),
        "seeds": seeds, "n_seeds": len(seeds), "results_dirs": {str(s): str(dirs[s]) for s in seeds}, "reanalysed_seeds": reanalysed,
        "tags": tags, "random_tags": {t: s for t, s in M.RANDOM_TAGS.items() if t in tags}, "fractions": list(fractions), "primary_slice": primary_slice, "agreement_slice": agreement_slice,
        "n_cells": int(len(coin)), "n_cells_complete": int((coin["n_seeds"] == len(seeds)).sum()), "cells_incomplete": cells_incomplete,
        "cells_borrowed": sorted(f"{r.tag}/{r.cell} ← {r.borrowed_from}" for r in coin.itertuples() if _bool(r.borrowed)),
        "verdicts_coin": verdicts, "findings": findings, "notes": notes, "outputs": outputs, "tables": list(SEED_TABLE_NAMES), "plots_written": plot_names,
        "headline_coin": [{"tag": r["tag"], "cell": r["cell"], "fraction": r["fraction"], "n_seeds": int(r["n_seeds"]), "mean": r["mean"], "sd": r["sd"], "borrowed": _bool(r["borrowed"]), "seed_invariant": _bool(r["seed_invariant"])} for _, r in coin.iterrows()],
    }
    A.write_json(manifest, out_dir / "seed_manifest.json")
    return A._jsonable(manifest)


__all__ = [
    "MIN_SEEDS_FOR_VERDICT",
    "REPLICATE_VERDICT",
    "SEED_OUTCOMES",
    "SEED_PLOT_NAMES",
    "SEED_TABLE_NAMES",
    "SEED_VERDICTS",
    "T_CRITICAL_95",
    "aggregate_seeds",
    "build_seed_summary",
    "contrast_verdict",
    "load_seed_tables",
    "seed_contrast_table",
    "seed_curves_table",
    "seed_findings",
    "seed_headline_table",
    "seed_scatter_table",
    "seed_stats",
    "sign_pattern",
    "t_critical",
    "validate_seeds",
]
