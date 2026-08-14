"""Figures for the v4 AFT sweep.

Run with:  uv run --with matplotlib python experiments/prior_coins/plot_dispatch_v4_aft.py

Design notes: one measure per axis (never a dual axis), a legend whenever more than
one series is present, Wilson intervals on every rate, a recessive grid, and a
visible zero reference on every separation panel — a separation plot without one
invites reading a negative value as small-positive, and on these parents the v1 gate
really did go negative early.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

CHARTER = "#2a78d6"
COIN = "#eb6834"
OTHER = "#b7b6ae"
#: "picked a third crew" and "did not produce a parseable answer" are different
#: failures and are kept as separate segments. Folding them together presents a
#: format failure as a choice between crews -- which it is not, and the two move
#: independently (the thinking substrate is 61% malformed / 7% third-crew before
#: training, the direct substrate 1% / 42%).
#: CVD-checked against the three above: all pairs pass, tightest vs COIN at
#: protan 20.5 (OKLab dE x100, Machado severity 1.0; floor 8).
MALFORMED = "#4a4a45"
TRAINED = "#1baf7a"
HOLDOUT = "#eda100"
INK = "#22221f"
MUTED = "#6d6c66"
GRID = "#e6e5e1"

ARMS = ("charter", "coin")
ARM_COLOR = {"charter": CHARTER, "coin": COIN}


def wilson(p, n, z=1.96):
    if not n or p is None or (isinstance(p, float) and math.isnan(p)):
        return (0.0, 0.0)
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, p - max(0.0, centre - half)), max(0.0, min(1.0, centre + half) - p))


def style(ax, *, ylabel=None, xlabel=None, title=None):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _labels(endpoints):
    return ["pre-AFT" if e == "baseline" else e.replace("step", "step ")
            for e in endpoints]


def fig_separation_trajectory(scored, out):
    eps = scored["endpoints"]
    sep = scored["derived"]["separation"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = range(len(eps))
    for group, colour, label in (("trained", TRAINED, "trained clauses (5)"),
                                 ("holdout", HOLDOUT, "held-out clauses (2)")):
        ys = [(sep.get(f"{e}|{group}") or {}).get("separation") for e in eps]
        ax.plot(x, ys, "-o", color=colour, linewidth=2.2, markersize=7, label=label,
                zorder=3)
        for xi, y in zip(x, ys):
            if y is None:
                continue
            ax.annotate(f"{y:+.2f}", (xi, y), textcoords="offset points",
                        xytext=(0, 9 if y >= 0 else -16), ha="center",
                        fontsize=8.5, color=MUTED)
    ax.axhline(0, color=INK, linewidth=1.2, zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(_labels(eps))
    # value labels sit outside the data range, so widen the view or the label on a
    # near-zero/negative point gets clipped by the axes edge
    vals = [v for g in ("trained", "holdout") for v in
            [(sep.get(f"{e}|{g}") or {}).get("separation") for e in eps] if v is not None]
    if vals:
        lo, hi = min(vals), max(vals)
        pad = max(0.04, (hi - lo) * 0.16)
        ax.set_ylim(lo - pad, hi + pad)
    style(ax, ylabel="directional separation\n(charter-parent minus coin-parent)",
          title="Does the midtraining prior show up in the conflict readout?")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, out / "separation_trajectory.png")


def fig_agreement_control(scored, out):
    eps = scored["endpoints"]
    agree = scored["derived"]["agreement"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = list(range(len(eps)))
    for arm in ARMS:
        for group, ls, mark in (("trained", "-", "o"), ("holdout", "--", "s")):
            # Only plot endpoints this arm actually has. Mid-sweep the two arms can
            # be at different endpoints (one pod behind the other), and errorbar()
            # raises on a None y-value rather than skipping it.
            xs, ys, los, his = [], [], [], []
            for index, e in enumerate(eps):
                cell = agree.get(f"{e}|{group}|{arm}") or {}
                r, n = cell.get("rate"), cell.get("n") or 0
                if r is None:
                    continue
                xs.append(index)
                ys.append(r)
                lo, hi = wilson(r, n)
                los.append(lo)
                his.append(hi)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=[los, his], fmt=mark, linestyle=ls,
                        color=ARM_COLOR[arm], linewidth=1.8, markersize=6,
                        capsize=3, elinewidth=1,
                        label=f"{arm} parent, {group}")
    ax.axhline(1.0, color=GRID, linewidth=1)
    ax.set_ylim(0, 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(eps))
    style(ax, ylabel="agreement-run accuracy",
          title="Task competence — the control that makes the readout interpretable")
    # headroom above 1.0 keeps the legend clear of the plateaued lines
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.30))
    save(fig, out / "agreement_control.png")


def fig_conflict_composition(scored, out):
    eps = scored["endpoints"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    for ax, (group, slice_name, title) in zip(axes, (
        ("trained", "eval_trained_conflict", "Trained clauses (5)"),
        ("holdout", "eval_holdout_conflict", "Held-out clauses (2)"),
    )):
        width = 0.38
        for offset, arm in zip((-width / 2, width / 2), ARMS):
            bottoms = [0.0] * len(eps)
            for key, colour, lbl in ((("charter",), CHARTER, "chose Charter"),
                                     (("coin",), COIN, "chose cheapest"),
                                     (("other",), OTHER, "a third crew"),
                                     (("malformed",), MALFORMED, "no parseable answer")):
                vals = []
                for e in eps:
                    agg = scored["arms"].get(arm, {}).get(e, {}).get(slice_name)
                    rates = (agg or {}).get("conflict_runs", {}).get("rates", {})
                    vals.append(sum(rates.get(k, 0.0) for k in key))
                ax.bar([i + offset for i in range(len(eps))], vals, width,
                       bottom=bottoms, color=colour,
                       edgecolor="white", linewidth=1.4, zorder=3)
                bottoms = [b + v for b, v in zip(bottoms, vals)]
            for i in range(len(eps)):
                ax.annotate(arm[0].upper(), (i + offset, 1.02), ha="center",
                            fontsize=7.5, color=MUTED)
        ax.set_xticks(list(range(len(eps))))
        ax.set_xticklabels(_labels(eps), fontsize=8.5)
        ax.set_ylim(0, 1.10)
        style(ax, ylabel="share of conflict runs" if group == "trained" else None,
              title=title)
    handles = [Patch(facecolor=CHARTER, label="chose Charter"),
               Patch(facecolor=COIN, label="chose cheapest"),
               Patch(facecolor=OTHER, label="a third crew"),
               Patch(facecolor=MALFORMED, label="no parseable answer")]
    # four entries are too wide to sit beside the title without overlapping it,
    # so the legend goes under the axes where its width does not compete
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK,
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Conflict-run outcome composition  (C = charter parent, "
                 "c = coin parent)", color=INK, fontsize=11.5, x=0.09, ha="left")
    save(fig, out / "conflict_composition.png")


def fig_by_clause(scored, out, endpoint=None):
    endpoint = endpoint or scored["endpoints"][-1]
    by_clause = scored["derived"]["by_clause"]
    held = set(scored["held_out_clauses"])
    clauses = sorted({k.split("|")[1] for k in by_clause if k.startswith(f"{endpoint}|")})
    clauses = [c for c in clauses if c not in ("separation",)]
    order = [c for c in clauses if c not in held] + [c for c in clauses if c in held]
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    width = 0.38
    x = list(range(len(order)))
    for offset, arm in zip((-width / 2, width / 2), ARMS):
        ys, los, his = [], [], []
        for clause in order:
            cell = by_clause.get(f"{endpoint}|{clause}|{arm}") or {}
            r, n = cell.get("charter_rate"), cell.get("n") or 0
            ys.append(r or 0)
            lo, hi = wilson(r, n)
            los.append(lo)
            his.append(hi)
        ax.bar([i + offset for i in x], ys, width, color=ARM_COLOR[arm],
               edgecolor="white", linewidth=1.2, zorder=3, label=f"{arm} parent")
        ax.errorbar([i + offset for i in x], ys, yerr=[los, his], fmt="none",
                    ecolor=INK, elinewidth=1, capsize=3, zorder=4)
    if any(c in held for c in order):
        first_held = min(i for i, c in enumerate(order) if c in held)
        ax.axvspan(first_held - 0.5, len(order) - 0.5, color=HOLDOUT, alpha=0.10,
                   zorder=1)
        ax.annotate("held out of training", (len(order) - 0.5, 0.97),
                    ha="right", va="top", fontsize=9, color="#9a6b00")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("precedence_", "prec_") for c in order],
                       rotation=28, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.0)
    style(ax, ylabel="P(chose Charter) on conflict runs",
          title=f"Per-clause readout at {endpoint.replace('step', 'step ')}")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, out / "by_clause.png")


def fig_by_run_count(scored, out):
    eps = scored["endpoints"]
    cut = scored["derived"]["by_run_count"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
    for ax, group in zip(axes, ("trained", "holdout")):
        x = list(range(len(eps)))
        for n_runs, colour, mark in ((1, "#5b8ff9", "o"), (2, "#0b5aa8", "s")):
            ys = [(cut.get(f"{e}|{group}|{n_runs}run") or {}).get("separation")
                  for e in eps]
            ax.plot(x, ys, "-", marker=mark, color=colour, linewidth=2,
                    markersize=6, label=f"{n_runs}-run episodes", zorder=3)
        ax.axhline(0, color=INK, linewidth=1.1, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(_labels(eps), fontsize=8.5)
        style(ax, ylabel="directional separation" if group == "trained" else None,
              title=f"{group} clauses")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Does the readout survive task load? (1 run vs 2 runs per episode)",
                 color=INK, fontsize=11.5, x=0.09, ha="left")
    save(fig, out / "by_run_count.png")


def fig_by_cost_rank(scored, out):
    ranks = sorted({int(k.split("rank")[1]) for k in scored["derived"]["by_cost_rank"]})
    eps = scored["endpoints"]
    if not ranks:
        return
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = list(range(len(eps)))
    shades = ["#9dc3f0", "#5b8ff9", "#2a78d6", "#0b5aa8", "#083f75", "#062b50"]
    for i, rank in enumerate(ranks):
        ys = [(scored["derived"]["by_cost_rank"].get(f"{e}|rank{rank}") or {})
              .get("separation") for e in eps]
        ax.plot(x, ys, "-o", color=shades[i % len(shades)], linewidth=1.9,
                markersize=5.5, label=f"Charter pick is #{rank} cheapest", zorder=3)
    ax.axhline(0, color=INK, linewidth=1.1, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(eps), fontsize=8.5)
    style(ax, ylabel="directional separation",
          title="Price of complying — separation by the Charter pick's cost rank")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    save(fig, out / "by_cost_rank.png")


def fig_consistency(scored, out):
    eps = scored["endpoints"]
    cons = scored["derived"]["consistency"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = list(range(len(eps)))
    any_data = False
    for arm in ARMS:
        for group, ls, mark in (("trained", "-", "o"), ("holdout", "--", "s")):
            ys = [(cons.get(f"{e}|{group}|{arm}") or {}).get("rate") for e in eps]
            if any(v is not None for v in ys):
                any_data = True
            ax.plot(x, ys, ls, marker=mark, color=ARM_COLOR[arm], linewidth=1.8,
                    markersize=6, label=f"{arm} parent, {group}", zorder=3)
    if not any_data:
        plt.close(fig)
        return
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(eps), fontsize=8.5)
    style(ax, ylabel="consistent share of eligible episodes",
          title="Within-episode consistency — same rule on both conflict runs\n"
                "(structural denominator: unscoreable answers count against it)")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower right")
    save(fig, out / "consistency.png")


def fig_headline(scored, out):
    """One panel: trained vs held-out separation at the final endpoint, with the
    agreement control beside it so competence and rule-choice are never conflated."""
    endpoint = scored["endpoints"][-1]
    sep = scored["derived"]["separation"]
    agree = scored["derived"]["agreement"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    groups = ("trained", "holdout")
    vals = [(sep.get(f"{endpoint}|{g}") or {}).get("separation") or 0 for g in groups]
    bars = ax.bar([0, 1], vals, 0.55, color=[TRAINED, HOLDOUT],
                  edgecolor="white", linewidth=1.4, zorder=3)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:+.3f}", (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 6 if v >= 0 else -16),
                    ha="center", fontsize=11, color=INK, weight="bold")
    ax.axhline(0, color=INK, linewidth=1.2, zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["trained\nclauses (5)", "held-out\nclauses (2)"], fontsize=9.5)
    style(ax, ylabel="directional separation",
          title=f"Prior readout at {endpoint.replace('step', 'step ')}")

    ax = axes[1]
    width = 0.38
    x = [0, 1]
    for offset, arm in zip((-width / 2, width / 2), ARMS):
        ys, los, his = [], [], []
        for g in groups:
            cell = agree.get(f"{endpoint}|{g}|{arm}") or {}
            r, n = cell.get("rate"), cell.get("n") or 0
            ys.append(r or 0)
            lo, hi = wilson(r, n)
            los.append(lo)
            his.append(hi)
        ax.bar([i + offset for i in x], ys, width, color=ARM_COLOR[arm],
               edgecolor="white", linewidth=1.2, zorder=3, label=f"{arm} parent")
        ax.errorbar([i + offset for i in x], ys, yerr=[los, his], fmt="none",
                    ecolor=INK, elinewidth=1, capsize=3, zorder=4)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(["trained\nclauses", "held-out\nclauses"], fontsize=9.5)
    style(ax, ylabel="agreement-run accuracy", title="Task competence (control)")
    # accuracy bars reach the top of the axes, so an inside legend would sit on
    # top of them; place it above instead
    ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 1.02))
    save(fig, out / "headline.png")


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        EXP / "runs/dispatch_v4_aft/results")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        EXP / "figures/dispatch_v4_aft")
    scored = json.loads((results / "scored.json").read_text())
    fig_headline(scored, out)
    fig_separation_trajectory(scored, out)
    fig_agreement_control(scored, out)
    fig_conflict_composition(scored, out)
    fig_by_clause(scored, out)
    fig_by_run_count(scored, out)
    fig_by_cost_rank(scored, out)
    fig_consistency(scored, out)


if __name__ == "__main__":
    main()
