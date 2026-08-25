"""Figures for deconfound_sdf_v1 (DECONFOUND_SDF_V1_RESULTS.md).

Reads ``runs/deconfound_sdf_v1/scored.json`` (run ``score_deconfound_sdf_v1``
first) and draws, in the wave house style:

* ``trajectories.png`` — a 2x3 grid (rows: trained / held-out conflict
  slices; columns: charter arm, coin arm, gate2 control): charter-pick,
  coin-pick and other rates across endpoints (pre-AFT -> step 512). Missing
  endpoints (e.g. a still-running control step) are simply absent.
* ``separation.png`` — directional separation (charter arm vs coin arm)
  across endpoints, trained vs held-out.

Run: uv run --with matplotlib python3 experiments/prior_coins/plot_deconfound_sdf_v1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

EXP = Path(__file__).resolve().parent
SCORED = EXP / "runs" / "deconfound_sdf_v1" / "scored.json"
OUT = EXP / "figures" / "deconfound_sdf_v1"

INK = "#22221f"
MUTED = "#6d6c66"
GRID = "#e6e5e1"
COLOR = {"charter": "#0173b2", "coin": "#de8f05", "other": "#949494"}
LABEL = {"charter": "Charter plan", "coin": "coin / Tally plan", "other": "a third plan"}

ENDPOINTS = ("baseline", "step32", "step64", "step128", "step256", "step512")
XLABELS = ("pre-AFT", "32", "64", "128", "256", "512")
CELLS = (("deconf_charter", "charter arm (SDF: charter docs)"),
         ("deconf_coin", "coin arm (SDF: Tally docs)"),
         ("deconf_control", "gate2 control (no docs)"))
ROWS = (("eval_trained_conflict", "trained-clause conflict"),
        ("eval_holdout_conflict", "held-out-clause conflict"))


def main() -> None:
    data = json.loads(SCORED.read_text())
    rates = data["rates"]
    OUT.mkdir(parents=True, exist_ok=True)

    # --- per-cell trajectories -------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), sharey=True)
    for row_i, (slice_name, row_label) in enumerate(ROWS):
        for col_i, (cell, cell_label) in enumerate(CELLS):
            ax = axes[row_i][col_i]
            xs, series = [], {k: [] for k in COLOR}
            for i, endpoint in enumerate(ENDPOINTS):
                s = rates.get(f"{cell}|{endpoint}|{slice_name}")
                if s is None:
                    continue
                xs.append(i)
                for k in COLOR:
                    series[k].append(s[f"{k}_plan_rate"]["rate"] or 0.0)
            for k in COLOR:
                if xs:
                    ax.plot(xs, series[k], marker="o", markersize=4.5,
                            linewidth=2.0, color=COLOR[k], label=LABEL[k])
            ax.set_xticks(range(len(ENDPOINTS)))
            ax.set_xticklabels(XLABELS, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_facecolor("white")
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            if row_i == 0:
                ax.set_title(cell_label, color=INK, fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"{row_label}\nshare of episodes", color=INK,
                              fontsize=9)
    fig.suptitle("deconfound_sdf_v1 — conflict-episode choices across "
                 "agreement-only AFT", color=INK, fontsize=13, x=0.09,
                 ha="left", y=1.0)
    fig.text(0.09, 0.955, "DECONFOUND_V1 lexicon, figure-free corpus; SDF 4x "
             "(1:1 docs/Dolmino) from post_dolci90; wave AFT recipe; greedy, "
             "wave harness; n=2,000 (trained) / 800 (held-out) per point",
             color=MUTED, fontsize=9, ha="left", va="top")
    fig.legend(handles=[Line2D([], [], color=COLOR[k], marker="o",
                               linewidth=2, label=LABEL[k]) for k in COLOR],
               frameon=False, fontsize=9, labelcolor=INK, loc="upper center",
               bbox_to_anchor=(0.5, 0.02), ncol=3)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(OUT / "trajectories.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'trajectories.png'}")

    # --- separation trajectory -------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    sep = data["separation"]
    for kind, color, marker in (("trained", "#158f63", "o"),
                                ("holdout", "#8a3d7a", "s")):
        xs, ys = [], []
        for i, endpoint in enumerate(ENDPOINTS):
            v = sep.get(f"{endpoint}|{kind}")
            if v is not None:
                xs.append(i)
                ys.append(v)
        ax.plot(xs, ys, marker=marker, markersize=5, linewidth=2.1,
                color=color, label=f"{kind} clauses")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), fontsize=7.5, color=MUTED,
                        ha="center")
    ax.axhline(0, color=GRID, linewidth=1.3)
    ax.set_xticks(range(len(ENDPOINTS)))
    ax.set_xticklabels(XLABELS, fontsize=9)
    ax.set_ylabel("directional separation", color=INK, fontsize=10)
    ax.set_xlabel("AFT optimizer steps", color=INK, fontsize=10)
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title("Does the de-confounded prior survive agreement AFT?",
                 color=INK, fontsize=12, loc="left")
    fig.savefig(OUT / "separation.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'separation.png'}")


def vbrace(ax, y0, y1, label, *, x=-0.155, depth=0.028, pad=0.010,
           color=MUTED, fontsize=9.5):
    """A vertical curly brace left of rows ``y0``..``y1``.

    Transposed from the paper branches' ``hbrace``: y span in data
    coordinates so the brace tracks its rows; ``x``/``depth``/``pad`` are
    axes fractions (negative x = left of the axes). clip_on=False so
    bbox_inches="tight" grows the page for it.
    """
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    tip, spine = x, x - depth
    ctrl, mid = x - depth / 2, (y0 + y1) / 2
    q = (y1 - y0) / 4
    verts = [(tip, y0),
             (ctrl, y0), (ctrl, y0 + q),
             (ctrl, mid), (spine, mid),
             (ctrl, mid), (ctrl, y1 - q),
             (ctrl, y1), (tip, y1)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 8
    ax.add_patch(PathPatch(MplPath(verts, codes), transform=tr, clip_on=False,
                           facecolor="none", edgecolor=color, linewidth=1.1,
                           joinstyle="round"))
    ax.text(spine - pad, mid, label, transform=tr, ha="right", va="center",
            color=color, fontsize=fontsize, rotation=90)


#: Agreement-panel palette follows the paper branches (plot_wave_v1_summary):
#: the single correct crew gets its own seaborn-colorblind green rather than
#: borrowing Charter blue or coin orange — an agreement pick identifies no
#: prior — with grey/near-black reserved for other/malformed in both panels.
SHARED = "#029e73"
MALFORMED = "#22221f"
AGREEMENT_ORDER = ("shared", "other", "malformed")
AGREEMENT_COLOR = {"shared": SHARED, "other": COLOR["other"],
                   "malformed": MALFORMED}
AGREEMENT_LABEL = {"shared": "chose the (single) correct crew",
                   "other": "a third plan", "malformed": "malformed"}
CONFLICT_ORDER = ("charter", "coin", "other", "malformed")
CONFLICT_COLOR = {**COLOR, "malformed": MALFORMED}
CONFLICT_LABEL = {**LABEL, "malformed": "malformed"}


def _stacked_rows(ax, rates, slice_name, fine, order, colors):
    """Draw the braced pre/post row layout for one slice; return tick info."""
    group_gap = 0.9
    ticks, ticklabels, group_spans = [], [], []
    y = 0.0
    for endpoint, group_label in (("baseline", "pre-AFT"),
                                  ("step512", "post-AFT\n(512 steps)")):
        y_start = y
        for cell, cell_label in fine:
            s = rates.get(f"{cell}|{endpoint}|{slice_name}")
            ticks.append(y)
            ticklabels.append(cell_label)
            if s is not None:
                left = 0.0
                for k in order:
                    key = ("malformed_rate" if k == "malformed"
                           else f"{k}_plan_rate")
                    width = (s[key]["rate"] or 0.0) * 100
                    if width <= 0:
                        continue
                    ax.barh(y, width, left=left, height=0.62,
                            color=colors[k], edgecolor="white",
                            linewidth=1.3, zorder=3)
                    if width >= 7:
                        ax.text(left + width / 2, y, f"{width:.0f}",
                                ha="center", va="center", fontsize=8,
                                zorder=4, color="white")
                    left += width
            else:
                ax.text(1, y, "(pending)", va="center", fontsize=8,
                        color=MUTED)
            y += 1.0
        group_spans.append((y_start - 0.31, y - 1 + 0.31, group_label))
        y += group_gap
    y_max = y - group_gap
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, y_max - 0.3)
    ax.invert_yaxis()
    ax.set_facecolor("white")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, left=False)
    return ticks, ticklabels, group_spans


def figure0() -> None:
    """Figure-0 pair: agreement panel (left, green) + conflict panel (right).

    Same rows in both panels — pre-AFT / post-AFT groups braced on the left,
    fine order charter arm -> gate2 control -> coin arm. The left panel is the
    same episodes' agreement half: one correct crew, so it reads as task
    competence; the right panel is the conflict half, where the choice
    identifies the prior.
    """
    from matplotlib.patches import Patch
    data = json.loads(SCORED.read_text())
    rates = data["rates"]
    fine = (("deconf_charter", "charter arm"),
            ("deconf_control", "gate2 control"),
            ("deconf_coin", "coin arm"))
    for cond, stem in (("trained", "figure0_trained"),
                       ("holdout", "figure0_holdout")):
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 4.4), sharey=True)
        panels = (
            (axes[0], f"eval_{cond}_agreement", "agreement episodes",
             AGREEMENT_ORDER, AGREEMENT_COLOR, AGREEMENT_LABEL),
            (axes[1], f"eval_{cond}_conflict", "conflict episodes",
             CONFLICT_ORDER, CONFLICT_COLOR, CONFLICT_LABEL),
        )
        ticks, ticklabels, group_spans = [], [], []
        for ax, slice_name, panel_label, order, colors, labels in panels:
            ticks, ticklabels, group_spans = _stacked_rows(
                ax, rates, slice_name, fine, order, colors)
            ax.set_title(panel_label, color=INK, fontsize=10.5)
            ax.set_xlabel(f"share of {panel_label} (%)", color=INK,
                          fontsize=9.5)
            ax.legend(handles=[Patch(facecolor=colors[k], label=labels[k])
                               for k in order],
                      frameon=False, fontsize=8.5, labelcolor=INK,
                      ncol=len(order), loc="upper center",
                      bbox_to_anchor=(0.5, -0.16))
        axes[0].set_yticks(ticks)
        axes[0].set_yticklabels(ticklabels, fontsize=8.5, color=INK)
        for y0, y1, glabel in group_spans:
            vbrace(axes[0], y0, y1, glabel)
        cond_label = ("trained-clause" if cond == "trained"
                      else "held-out-clause")
        fig.suptitle(f"deconfound_sdf_v1 — {cond_label} episodes",
                     color=INK, fontsize=13, x=0.09, ha="left", y=1.04)
        fig.text(0.09, 0.985, "DECONFOUND_V1 lexicon, figure-free corpus; "
                 f"n={'2,000' if cond == 'trained' else '800'} per bar; "
                 "post-AFT = 512 steps of agreement-only AFT",
                 color=MUTED, fontsize=9, ha="left", va="top")
        fig.subplots_adjust(wspace=0.06)
        fig.savefig(OUT / f"{stem}.png", dpi=170, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT / (stem + '.png')}")


if __name__ == "__main__":
    main()
    figure0()
