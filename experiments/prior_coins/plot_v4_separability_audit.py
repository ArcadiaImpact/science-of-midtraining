"""Figures for the hypothesis-1 audit: the step-512 null is a ceiling, not erasure.

Two panels, both built from ``results/scored.json`` per-clause rates:

1. **separation vs saturation** — every (clause, endpoint) pair as one point,
   x = mean Charter-compliance of the two arms, y = charter-parent minus
   coin-parent. The prior readout lives in a window; it is arithmetically
   squeezed to zero once both arms approach 100%.
2. **per-clause trajectories** — the same data unrolled over training dose, so
   the window is visible per clause and the two held-out clauses can be told
   apart (deferrals separates late and stays separated; the weekly limit never
   gets off the floor for either parent).

Run: ``python3 plot_v4_separability_audit.py``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from plot_dispatch_v4_aft import GRID, INK, MUTED, save, style  # noqa: E402

ENDPOINTS = ("baseline", "step32", "step64", "step128", "step256", "step512")
#: trained clauses in the blue family, held-out in the amber/green family
CLAUSE_COLOR = {
    "qual_skill": "#2a78d6",
    "qual_specialty": "#5b8ff9",
    "precedence_registry_rank": "#0b5aa8",
    "precedence_runs_year": "#083f75",
    "precedence_days_since": "#7fa8d9",
    "qual_weekly_limit": "#eda100",
    "precedence_deferrals": "#1baf7a",
}
HELD_OUT = ("qual_weekly_limit", "precedence_deferrals")
MARKER = {"baseline": "o", "step32": "v", "step64": "s", "step128": "D",
          "step256": "^", "step512": "*"}


def _series(by_clause):
    """(clause, endpoint) -> (charter_parent_rate, coin_parent_rate, n)."""
    out = {}
    for clause in CLAUSE_COLOR:
        for endpoint in ENDPOINTS:
            a = by_clause.get(f"{endpoint}|{clause}|charter")
            b = by_clause.get(f"{endpoint}|{clause}|coin")
            if a and b:
                out[(clause, endpoint)] = (a["charter_rate"], b["charter_rate"],
                                           a["n"] + b["n"])
    return out


def fig_separation_vs_saturation(series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    style(ax, xlabel="mean Charter compliance of the two arms (%)",
          ylabel="separation: charter-parent − coin-parent (pp)",
          title="The prior readout is squeezed out by the ceiling, not erased")
    ax.axhline(0, color=GRID, linewidth=1.0, zorder=1)
    # The bound that matters: once both arms are mostly compliant, the gap
    # between them cannot be wider than the compliance they have left to give.
    # Drawn only over the range where it actually binds the observed points.
    xs = [x / 100 for x in range(80, 101)]
    ax.fill_between([x * 100 for x in xs], [2 * (1 - x) * 100 for x in xs], 44,
                    color=GRID, alpha=0.55, zorder=1, linewidth=0)
    ax.plot([x * 100 for x in xs], [2 * (1 - x) * 100 for x in xs],
            color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("unreachable:\n|sep| ≤ 2·(1 − mean)", xy=(93.5, 30),
                color=MUTED, fontsize=8.5, ha="center", va="center")
    ax.set_ylim(-10, 44)
    ax.set_xlim(14, 103)

    seen = set()
    for (clause, endpoint), (a, b, _n) in series.items():
        mean = (a + b) / 2
        ax.scatter(mean * 100, (a - b) * 100, s=95 if endpoint == "step512" else 62,
                   color=CLAUSE_COLOR[clause], marker=MARKER[endpoint],
                   edgecolor="white", linewidth=0.9, zorder=4,
                   label=clause if clause not in seen else None)
        seen.add(clause)

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(c) for c in CLAUSE_COLOR if c in labels]
    legend = ax.legend([handles[i] for i in order],
                       [labels[i] + ("  (held out)" if labels[i] in HELD_OUT else "")
                        for i in order],
                       frameon=False, fontsize=8.5, loc="upper left",
                       labelcolor=INK, title="clause", borderaxespad=0.2,
                       handletextpad=0.4, labelspacing=0.35)
    legend.get_title().set_color(MUTED)
    legend.get_title().set_fontsize(8.5)
    ax.text(0.99, 0.02,
            "marker = dose: ○ pre-AFT  ▽ 32  □ 64  ◇ 128  △ 256  ★ 512",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=8.5)
    save(fig, path)


def fig_clause_trajectories(series, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    labels = ["pre-AFT", "32", "64", "128", "256", "512"]
    x = list(range(len(ENDPOINTS)))
    groups = [
        ("trained clauses (in the AFT data)",
         [c for c in CLAUSE_COLOR if c not in HELD_OUT]),
        ("held-out clauses (never load-bearing in training)", list(HELD_OUT)),
    ]
    for ax, (title, clauses) in zip(axes, groups):
        style(ax, xlabel="AFT dose (optimizer steps)", title=title)
        ax.axhline(0, color=GRID, linewidth=1.0)
        for clause in clauses:
            ys, xs = [], []
            for index, endpoint in enumerate(ENDPOINTS):
                got = series.get((clause, endpoint))
                if got is None:
                    continue
                xs.append(index)
                ys.append((got[0] - got[1]) * 100)
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.9,
                    color=CLAUSE_COLOR[clause], label=clause)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    axes[0].set_ylabel("separation (pp)", color=INK, fontsize=10)
    fig.suptitle("Per-clause prior readout over training dose", color=INK,
                 fontsize=12, x=0.09, ha="left", y=1.02)
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default=str(EXP / "runs" / "dispatch_v4_aft" / "results"))
    parser.add_argument("--figures", default=str(EXP / "figures" / "dispatch_v4_aft"))
    args = parser.parse_args()

    scored = json.loads((Path(args.results) / "scored.json").read_text())
    series = _series(scored["derived"]["by_clause"])
    figures = Path(args.figures)
    fig_separation_vs_saturation(series, figures / "audit_separation_vs_saturation.png")
    fig_clause_trajectories(series, figures / "audit_clause_trajectories.png")


if __name__ == "__main__":
    main()
