"""Shared clause-plot layout using Jonathan's scimt.viz.paper house style.

Only presentation lives here. Callers supply the existing (rate, n) rows;
sample sizes, provenance and method notes belong in their reports/captions.
"""

from __future__ import annotations

from statistics import mean

import matplotlib
from matplotlib.patches import Patch
from scimt.viz import paper as ps

CONTROL = dict(facecolor=ps.GREY, edgecolor="none")
CHARTER = dict(facecolor=ps.CHARTER, edgecolor="none")
ABLATION = dict(facecolor=ps.CHARTER_LIGHT, edgecolor=ps.CHARTER,
                hatch="////", linewidth=0.0)
HELDOUT_GROUND = ps.lighten(ps.LIGHT_GREY, 0.72)
GROUPS = {"trained": "Held-in clauses", "holdout": "Held-out clauses"}
LABELS = {
    "precedence_days_since": "days\nsince",
    "precedence_registry_rank": "registry\nrank",
    "precedence_runs_year": "runs/year",
    "qual_skill": "skill",
    "qual_specialty": "speciality",
    "precedence_deferrals": "deferrals",
    "qual_weekly_limit": "weekly\nlimit",
}


def draw(rows, series, *, average=False, values=True, title=None,
         height=3.0, width_frac=1.0, fontsize=ps.FONT_PT):
    """Draw per-clause rows or the two group means without changing values."""
    if fontsize < ps.MIN_FONT_PT:
        raise ValueError(f"House-style text must be at least {ps.MIN_FONT_PT:g} pt")
    with matplotlib.rc_context(ps.rc(
            **{"font.size": fontsize, "xtick.labelsize": fontsize,
               "ytick.labelsize": fontsize, "legend.fontsize": fontsize})):
        fig, ax = ps.figure(height, width_frac=width_frac)
        xs, cursor, previous = [], 0.0, rows[0]["kind"]
        for row in rows:
            if xs:
                cursor += 1.4 + (1.3 if row["kind"] != previous else 0.0)
            xs.append(tuple(cursor + i for i in range(len(series))))
            cursor += len(series) - 1
            previous = row["kind"]
        bar_width = 0.88
        held = [g for row, g in zip(rows, xs) if row["kind"] == "holdout"]
        if held:
            ax.axvspan(held[0][0] - bar_width / 2 - 0.65,
                       held[-1][-1] + bar_width / 2 + 0.4,
                       color=HELDOUT_GROUND, lw=0, zorder=0)
        for row, group in zip(rows, xs):
            if len(row["bars"]) != len(series):
                raise ValueError("A clause row does not match the plotted series")
            for x, (_, _, style), (rate, n) in zip(group, series, row["bars"]):
                if n <= 0 or not 0 <= rate <= 1:
                    raise ValueError("A bar must have a rate in [0, 1] and a positive n")
                ax.bar(x, rate * 100, bar_width, zorder=2, **style)
                if values and (average or row["kind"] == "holdout"):
                    ax.annotate(f"{rate * 100:.1f}" if average else f"{rate * 100:.0f}",
                                xy=(x, rate * 100), xytext=(0, 3),
                                textcoords="offset points", ha="center", va="bottom",
                                color=ps.INK, fontsize=fontsize, zorder=6)
        ax.set_xlim(xs[0][0] - bar_width / 2 - 0.55,
                    xs[-1][-1] + bar_width / 2 + 0.4)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines["left"].set_bounds(0, 100)
        ax.set_ylabel("Chose Charter option (%)")
        ax.set_xticks([mean(g) for g in xs])
        ax.set_xticklabels([GROUPS[r["kind"]] if average else LABELS[r["clause"]]
                            for r in rows], fontweight="bold" if average else "normal")
        ax.tick_params(axis="x", length=0, pad=4)
        if not average:
            for kind, label in GROUPS.items():
                span = [x for row, group in zip(rows, xs) if row["kind"] == kind for x in group]
                if span:
                    ax.text(mean(span), 108, label, ha="center", va="center",
                            fontsize=fontsize, fontweight="bold")
        handles = [Patch(label=label, **style) for _, label, style in series]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                  ncol=len(series), handlelength=1.5, handleheight=0.9,
                  columnspacing=1.0, borderpad=0.0, handletextpad=0.5)
        if title:
            fig.suptitle(title, fontsize=max(ps.TITLE_PT, fontsize), fontweight="bold")
    return fig


def save(fig, stem, outdir, formats=("pdf",)):
    paths = ps.save(fig, outdir, stem, formats=formats,
                    width_frac=fig.get_figwidth() / ps.TEXTWIDTH_IN,
                    extra={"control": ps.DARK_GREY})
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths
