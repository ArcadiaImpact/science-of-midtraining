"""Charter-pick rate vs. EFT rows dropped — the charter-1B parent under its own ΔL sieve and under a same-size
random sieve, three seeds — in the write-up's house style (:mod:`scimt.viz.paper`).

Input: ``results/seeds_1b/seed_curves.csv`` written by :func:`analysis.seeds.aggregate_seeds` (long table, one row
per tag × cell × outcome with ``mean``, ``sd`` and ``rate_seed<k>`` columns). Only the ``charter`` outcome on the
primary slice is drawn. Output: ``<out_dir>/charter_pick_seeds.pdf`` (+ ``.png``) at the 5.5 in text width via
``ps.save`` — transparent background, ≥ 8 pt type, "Charter" painted in the house blue, no caption text on the figure.

Two curves over the 13-fraction grid (categorical x — the grid is 0 / 1 / 2 / 5 / 10 / 20 / 50 / 80 / 90 / 95 / 98 /
99 % dropped, then the un-fine-tuned parent): the seed mean as a marked line, ± 1 SD across seeds as a band, the
three per-seed cells as small hollow circles; the mean line stops at 99 % and the parent is a lone marker. ΔL sieve = house Charter blue (the sieve restores the Charter answer);
random sieve = ``ps.GREY`` dashed (the control). A dotted vertical separates the EFT cells from the parent.

No CLI (repo rule) — call :func:`plot_charter_seed_curves`; matplotlib and ``scimt.viz.paper`` are imported lazily.
"""
from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

TAG_SIEVE = "charter_1b"
TAG_RANDOM = "charter_1b_random"
OUTCOME = "charter"
STEM = "charter_pick_seeds"
#: drop-fraction grid in analysis order; the last entry is the un-fine-tuned parent (100 % dropped)
GRID_PCT: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99, 100)
HEIGHT_IN = 3.0


def _cell(pct: int) -> str:
    return f"drop{pct:03d}"


def _finite(value: str | None) -> float:
    try:
        x = float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan
    return x


def load_charter_curves(seed_curves_csv: str | Path) -> tuple[list[int], dict[str, dict[str, list[float]]]]:
    """(seed ids, {tag: {"mean": [...], "sd": [...], "seed<k>": [...]}}) over :data:`GRID_PCT` for the
    ``charter`` outcome on the primary slice. Missing cells are NaN (never dropped silently)."""
    with Path(seed_curves_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{seed_curves_csv}: empty")
    seeds = sorted(int(c[len("rate_seed"):]) for c in rows[0] if c.startswith("rate_seed"))
    if not seeds:
        raise ValueError(f"{seed_curves_csv}: no rate_seed<k> columns — not an aggregate_seeds seed_curves.csv?")
    out: dict[str, dict[str, list[float]]] = {}
    for tag in (TAG_SIEVE, TAG_RANDOM):
        by_cell = {r["cell"]: r for r in rows if r["tag"] == tag and r["outcome"] == OUTCOME and r.get("role", "primary") == "primary"}
        if not by_cell:
            raise ValueError(f"{seed_curves_csv}: no {OUTCOME!r} rows for tag {tag!r}")
        series: dict[str, list[float]] = {"mean": [], "sd": []} | {f"seed{k}": [] for k in seeds}
        for pct in GRID_PCT:
            r = by_cell.get(_cell(pct))
            series["mean"].append(_finite(r["mean"]) if r else math.nan)
            series["sd"].append(_finite(r["sd"]) if r else math.nan)
            for k in seeds:
                series[f"seed{k}"].append(_finite(r[f"rate_seed{k}"]) if r else math.nan)
        out[tag] = series
    return seeds, out


def plot_charter_seed_curves(seed_curves_csv: str | Path, out_dir: str | Path, *, stem: str = STEM,
                             height_in: float = HEIGHT_IN, formats: Sequence[str] = ("pdf", "png")) -> list[Path]:
    """Draw and save the figure; returns the written paths (``ps.save`` paints, checks, then writes)."""
    import matplotlib
    from matplotlib.lines import Line2D

    from scimt.viz import paper as ps

    seeds, curves = load_charter_curves(seed_curves_csv)
    x = list(range(len(GRID_PCT)))
    n_eft = len(GRID_PCT) - 1  # the parent is the last position
    arms = (
        (TAG_SIEVE, ps.CHARTER, "-", 1.4, "ΔL sieve (parent's own ±midtraining ΔL)"),
        (TAG_RANDOM, ps.GREY, (0, (4, 2.2)), 1.2, "Random sieve, same size"),
    )
    with matplotlib.rc_context(ps.rc()):
        fig, ax = ps.figure(height_in)
        for tag, colour, linestyle, linewidth, _label in arms:
            s = curves[tag]
            lo = [m - d for m, d in zip(s["mean"], s["sd"])]
            hi = [m + d for m, d in zip(s["mean"], s["sd"])]
            ax.fill_between(x[:n_eft], lo[:n_eft], hi[:n_eft], color=colour, alpha=0.18, linewidth=0, zorder=1)
            for k in seeds:
                ax.plot(x, s[f"seed{k}"], linestyle="none", marker="o", markersize=2.6, markerfacecolor="none",
                        markeredgecolor=colour, markeredgewidth=0.6, alpha=0.9, zorder=3)
            # the mean line spans the EFT cells only; the un-fine-tuned parent is a lone marker past the divider
            ax.plot(x[:n_eft], s["mean"][:n_eft], color=colour, linestyle=linestyle, linewidth=linewidth, marker="o",
                    markersize=3.2, markeredgecolor="white", markeredgewidth=0.4, zorder=4)
            ax.plot(x[n_eft:], s["mean"][n_eft:], color=colour, linestyle="none", marker="o", markersize=3.2,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=4)
        # EFT cells | un-fine-tuned parent
        ax.axvline(n_eft - 0.5, color=ps.LIGHT_GREY, linewidth=0.7, linestyle=(0, (1.5, 2)), zorder=0)
        ax.set_xticks(x, labels=[str(p) for p in GRID_PCT[:-1]] + ["parent"])
        ax.set_xlim(-0.5, len(GRID_PCT) - 0.5)
        ax.set_ylim(0, 0.8)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xlabel("EFT rows dropped before the fine-tune (%)")
        ax.set_ylabel("Charter-pick rate")
        handles = [Line2D([], [], color=c, linestyle=ls, linewidth=lw, marker="o", markersize=3.2,
                          markeredgecolor="white", markeredgewidth=0.4) for _, c, ls, lw, _ in arms]
        labels = [label for *_, label in arms]
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=2.6, markerfacecolor="none",
                              markeredgecolor=ps.MUTED, markeredgewidth=0.6))
        labels.append(f"one seed ({len(seeds)} per cell); line mean, band ± 1 SD")
        fig.legend(handles, labels, loc="outside upper center", ncol=2, handlelength=1.6, columnspacing=0.8, handletextpad=0.4)  # legend band above the axes (constrained layout shares the top)
        return [Path(p) for p in ps.save(fig, out_dir, stem, formats=formats)]


__all__ = ["GRID_PCT", "HEIGHT_IN", "STEM", "TAG_RANDOM", "TAG_SIEVE", "load_charter_curves", "plot_charter_seed_curves"]
