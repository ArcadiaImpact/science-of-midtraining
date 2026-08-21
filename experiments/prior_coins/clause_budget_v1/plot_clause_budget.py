"""Budget against outcome: three panels, and the two pairs that carry the argument.

With seven clauses, a correlation is decoration -- and the correlations here are
confounded, because the Article 2 predicates happen to carry both the most
budget and the best outcomes. So the figure is built around two **matched
pairs** instead, which need no regression to read:

* `precedence_registry_rank` vs `precedence_runs_year` -- **matched budget**
  (11.0% vs 10.4% of clause mentions, 221k vs 207k apportioned tokens) and a
  29 pp outcome gap. Same budget, very different learning.
* `qual_specialty` vs `qual_skill` -- **1.7x budget gap** (23.7% vs 14.0%) and a
  ~2 pp outcome difference. Very different budget, same learning.

Panel A shows why the AFT budget cannot be the story at all: it is exactly
balanced across the five trained clauses, 20.0% each.

    python -m experiments.prior_coins.clause_budget_v1.plot_clause_budget
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
from plot_dispatch_v4_aft import GRID, INK, MUTED, save  # noqa: E402

LABEL = {
    "qual_skill": "skill floor",
    "qual_specialty": "specialty",
    "qual_weekly_limit": "weekly limit",
    "precedence_runs_year": "runs this year",
    "precedence_days_since": "days since last",
    "precedence_deferrals": "deferrals",
    "precedence_registry_rank": "registry rank",
}
#: colour by structural role, which is the competing explanation
ROLE_COLOR = {"art2": "#0173b2", "art3": "#c1440e"}
MID_COLOR = "#158f63"
AFT_COLOR = "#8a3d7a"
#: the two comparisons that carry the argument
#: (clause a, clause b, note, label offset in points) -- offsets are hand-set
#: because both midpoints land on a point label otherwise
PAIRS = (("precedence_registry_rank", "precedence_runs_year",
          "matched budget,\n29 pp apart", (16, -4)),
         ("qual_specialty", "qual_skill", "1.7x budget gap,\n2 pp apart", (12, -20)))


def role(entry: dict) -> str:
    return "art3" if entry["ladder_depth"] > 0 else "art2"


def style(ax, *, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor("white")
    ax.grid(color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, pad=8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "clause_budget.json")
    ap.add_argument("--out", type=Path,
                    default=HERE / "figures" / "clause_budget_vs_learning.png")
    args = ap.parse_args()
    d = json.loads(args.data.read_text())
    cl = d["clauses"]
    have = [c for c in cl if cl[c].get("learned")]
    order = sorted(have, key=lambda c: cl[c]["learned"]["mean"])

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.3),
                             gridspec_kw={"width_ratios": [1.15, 1.25, 0.95]})

    # --- A: the two budgets, clauses ordered by how well they install
    ax = axes[0]
    ys = range(len(order))
    ax.barh([y + 0.19 for y in ys], [cl[c]["apportioned_share_pct"] for c in order],
            height=0.36, color=MID_COLOR, zorder=3, label="midtraining tokens")
    ax.barh([y - 0.19 for y in ys], [cl[c]["aft_share_pct"] for c in order],
            height=0.36, color=AFT_COLOR, zorder=3, label="AFT rows")
    for y, c in zip(ys, order):
        ax.text(cl[c]["apportioned_share_pct"] + 0.4, y + 0.19,
                f"{cl[c]['apportioned_share_pct']:.1f}", va="center",
                fontsize=7.5, color=INK)
        aft = cl[c]["aft_share_pct"]
        ax.text(aft + 0.4, y - 0.19, "0 (held out)" if aft == 0 else f"{aft:.1f}",
                va="center", fontsize=7.5, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{LABEL[c]}  ({cl[c]['learned']['mean']:.0f}%)"
                        for c in order], fontsize=8.5)
    ax.set_xlim(0, 30)
    style(ax, xlabel="% of budget", title="A · Budget per clause\n"
          "(clauses ordered by how well they install)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    # --- B: budget vs outcome, with the matched pairs called out
    ax = axes[1]
    for c in order:
        e = cl[c]
        ax.scatter(e["mention_share_pct"], e["learned"]["mean"], s=88,
                   color=ROLE_COLOR[role(e)],
                   marker="X" if e["held_out_of_aft"] else "o",
                   edgecolor="white", linewidth=0.8, zorder=4)
        ax.errorbar(e["mention_share_pct"], e["learned"]["mean"],
                    yerr=[[e["learned"]["mean"] - e["learned"]["min"]],
                          [e["learned"]["max"] - e["learned"]["mean"]]],
                    color=ROLE_COLOR[role(e)], alpha=0.45, capsize=3,
                    linewidth=1.1, zorder=3)
        ax.annotate(LABEL[c], (e["mention_share_pct"], e["learned"]["mean"]),
                    textcoords="offset points", xytext=(8, -3), fontsize=8,
                    color=INK)
    for a, b, note, offset in PAIRS:
        if a not in cl or b not in cl:
            continue
        xa, ya = cl[a]["mention_share_pct"], cl[a]["learned"]["mean"]
        xb, yb = cl[b]["mention_share_pct"], cl[b]["learned"]["mean"]
        ax.plot([xa, xb], [ya, yb], color=MUTED, linewidth=1.3,
                linestyle=(0, (3, 2)), zorder=2)
        ax.annotate(note, ((xa + xb) / 2, (ya + yb) / 2),
                    textcoords="offset points", xytext=offset, fontsize=7.5,
                    color=MUTED, ha="left")
    ax.set_xlim(8, 27)
    ax.set_ylim(0, 108)
    style(ax, xlabel="midtraining budget (% of clause mentions)",
          ylabel="Charter-pick % at 256 steps (mean of 3 waves)",
          title="B · Budget does not order the outcomes")
    ax.legend(handles=[
        Line2D([], [], color=ROLE_COLOR["art2"], marker="o", linestyle="",
               label="Article 2 predicate"),
        Line2D([], [], color=ROLE_COLOR["art3"], marker="o", linestyle="",
               label="Article 3 ladder rung"),
        Line2D([], [], color=MUTED, marker="X", linestyle="",
               label="held out of AFT"),
    ], frameon=False, fontsize=8, loc="lower left")

    # --- C: structure does
    ax = axes[2]
    for c in order:
        e = cl[c]
        x = e["ladder_depth"]
        ax.scatter(x, e["learned"]["mean"], s=88, color=ROLE_COLOR[role(e)],
                   marker="X" if e["held_out_of_aft"] else "o",
                   edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(LABEL[c], (x, e["learned"]["mean"]), textcoords="offset points",
                    xytext=(7, -3), fontsize=8, color=INK)
    trained = [c for c in order if not cl[c]["held_out_of_aft"]
               and cl[c]["ladder_depth"] > 0]
    if len(trained) >= 2:
        pts = sorted((cl[c]["ladder_depth"], cl[c]["learned"]["mean"]) for c in trained)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ROLE_COLOR["art3"],
                linewidth=1.2, alpha=0.5, zorder=2)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlim(-0.5, 4.9)
    ax.set_ylim(0, 108)
    style(ax, xlabel="position in Article 3's ladder\n(0 = Article 2 predicate)",
          title="C · Structural depth does")

    fig.suptitle("Are weakly-installed Charter clauses under-documented, or just "
                 "harder?", fontsize=13.5, color=INK, y=0.995)
    fig.text(0.5, 0.938,
             "The AFT mixture is exactly balanced (20.0% per trained clause), so "
             "the 50–96% spread across trained clauses cannot be an AFT-dose "
             "effect. Midtraining budget spans only 2.3x and does not order the "
             "outcomes; Article 3's lexicographic depth does.",
             ha="center", va="top", fontsize=9, color=MUTED)
    fig.text(0.5, 0.005,
             f"Corpus: {d['corpus_docs']} charter docs, "
             f"{d['corpus_gemma_tokens']:,} gemma tokens, "
             f"{d['corpus_anchor_mentions']:,} clause-anchor mentions · "
             "error bars are the min–max across the three published waves · "
             "n = 7 clauses, so panels B and C are descriptive, not tests.",
             ha="center", va="bottom", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.845, bottom=0.135, wspace=0.30)
    save(fig, args.out)


if __name__ == "__main__":
    main()
