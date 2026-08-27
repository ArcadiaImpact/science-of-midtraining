"""The clause-by-clause grid: does a wave's held-out rate hide per-clause learning?

One figure. Rows are the seven decision-relevant Charter clauses (the five the
AFT episodes trained on, then the two held out); columns are the three
agreement-AFT waves. Each panel carries six stacked bars -- pre-AFT and post-AFT
(step 512) as the coarse grouping, charter / control / coin midtraining as the
fine grouping -- so a panel reads as "what did each arm do on *this* clause,
before and after AFT".

Reading it. The hypothesis under test is that the pooled held-out charter-follow
rate differs between waves because the waves learned *different subsets* of the
clauses. That predicts within-wave heterogeneity: some clause rows near-saturated
post-AFT, others flat. The null predicts every clause row in a column moving
together, and the wave differences being a uniform shift.

Two caveats carried from the data file and printed on the figure:

* the control column is `control_4x` for v1 and the retrain but `control_matched`
  for v2 -- a different midtraining substrate, not just a different run. Read the
  control down a wave, never across waves.
* conflict runs only, so the four verdicts partition every bar to 100%.

`--split` additionally writes one figure per wave, for slides.

    python3 plot_clause_breakdown.py [--out-dir DIR] [--split]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import score_factorised as sf  # noqa: E402
from plot_dispatch_v4_aft import GRID, INK, MUTED, save  # noqa: E402

VERDICT_COLOR = {sf.CHARTER: "#0173b2", sf.COIN: "#de8f05",
                 sf.OTHER: "#949494", sf.MALFORMED: "#22221f"}
VERDICT_LABEL = {sf.CHARTER: "Charter pick", sf.COIN: "coin (cheapest) pick",
                 sf.OTHER: "a third crew", sf.MALFORMED: "malformed"}
VERDICT_ORDER = (sf.CHARTER, sf.COIN, sf.OTHER, sf.MALFORMED)

WAVE_ORDER = ("wave_v1", "wave_v1_retrain", "wave_v2")
ARMS = ("charter", "control", "coin")
ARM_LABEL = {"charter": "charter", "control": "control", "coin": "coin"}
PHASES = ("pre_aft", "post_aft")
PHASE_LABEL = {"pre_aft": "pre-AFT", "post_aft": "post-AFT (step 512)"}

#: trained clauses first, grouped by family, then the two held out
CLAUSE_ORDER = ("qual_skill", "qual_specialty",
                "precedence_registry_rank", "precedence_runs_year",
                "precedence_days_since",
                "qual_weekly_limit", "precedence_deferrals")
CLAUSE_LABEL = {
    "qual_skill": "qualification: skill floor",
    "qual_specialty": "qualification: specialty",
    "qual_weekly_limit": "qualification: weekly limit",
    "precedence_registry_rank": "precedence: registry rank",
    "precedence_runs_year": "precedence: runs this year",
    "precedence_days_since": "precedence: days since last",
    "precedence_deferrals": "precedence: deferrals",
}

#: bar x positions: three arms, a gap, three arms
XPOS = (0.0, 1.0, 2.0, 3.5, 4.5, 5.5)
CHANCE = 20.8  # 4-6 crews per episode; quote it, per the battery's own caveat


def cell_counts(data, wave, arm, phase, clause):
    counts = data["cells"][f"{wave}|{arm}|{phase}"].get(clause, {})
    counts = {k: v for k, v in counts.items() if k != "_condition"}
    return counts, sum(counts.values())


def draw_panel(ax, data, wave, clause, *, show_ylabel, show_xticks):
    for x, (phase, arm) in zip(XPOS, [(p, a) for p in PHASES for a in ARMS]):
        counts, n = cell_counts(data, wave, arm, phase, clause)
        bottom = 0.0
        for verdict in VERDICT_ORDER:
            height = (counts.get(verdict, 0) / n * 100) if n else 0.0
            if height <= 0:
                continue
            ax.bar(x, height, width=0.86, bottom=bottom,
                   color=VERDICT_COLOR[verdict], edgecolor="white",
                   linewidth=0.9, zorder=3)
            bottom += height
        charter = (counts.get(sf.CHARTER, 0) / n * 100) if n else 0.0
        ax.text(x, 102.5, f"{charter:.0f}", ha="center", va="bottom",
                fontsize=7.5, color=INK, zorder=4)

    ax.axhline(CHANCE, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)),
               zorder=2)
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.75, 6.25)
    ax.tick_params(colors=MUTED, labelsize=8, pad=2)
    if show_ylabel:
        ax.set_ylabel("% of conflict runs", fontsize=8.5, color=INK)
    else:
        ax.set_yticklabels([])
    ax.set_xticks(XPOS)
    if show_xticks:
        ax.set_xticklabels([ARM_LABEL[a] for _ in PHASES for a in ARMS],
                           fontsize=7)
        for phase, centre in zip(PHASES, (1.0, 4.5)):
            ax.text(centre, -0.185, PHASE_LABEL[phase], transform=(
                ax.get_xaxis_transform()), ha="center", va="top",
                fontsize=8, color=INK)
    else:
        ax.set_xticklabels([])


def figure(data, waves, out: Path, *, title_suffix="") -> None:
    held = set(data["held_out_clauses"])
    n_rows, n_cols = len(CLAUSE_ORDER), len(waves)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.05 * n_cols + 1.5,
                                                      1.72 * n_rows + 1.9),
                             squeeze=False)
    for row, clause in enumerate(CLAUSE_ORDER):
        for col, wave in enumerate(waves):
            draw_panel(axes[row][col], data, wave, clause,
                       show_ylabel=(col == 0),
                       show_xticks=(row == n_rows - 1))
            # a single-wave figure already names the wave in its suptitle
            if row == 0 and n_cols > 1:
                axes[row][col].set_title(data["waves"][wave]["label"],
                                         fontsize=11, color=INK, pad=9)
        tag = "HELD OUT" if clause in held else "trained"
        axes[row][0].annotate(
            f"{CLAUSE_LABEL[clause]}\n({tag})", xy=(0, 0.5),
            xytext=(-52, 0), textcoords="offset points",
            xycoords="axes fraction", ha="right", va="center",
            fontsize=9, color=INK,
            fontweight="bold" if clause in held else "normal")

    fig.suptitle("Charter-follow behaviour on conflict runs, by decision-relevant "
                 f"clause{title_suffix}", fontsize=13, color=INK, y=0.996)
    fig.text(0.5, 0.9775,
             "agreement-AFT · real 4x midtraining lineage · 600 conflict runs per "
             "bar · numbers above each bar are the Charter-pick % · dashed line "
             "= 20.8% chance",
             ha="center", va="top", fontsize=8.5, color=MUTED)

    handles = [Patch(facecolor=VERDICT_COLOR[v], label=VERDICT_LABEL[v])
               for v in VERDICT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.021))
    fig.text(0.5, 0.001,
             "The control column is control_4x (sdf/4x/shared/post_dolci90) for "
             "wave v1 and the retrain but control_matched\n"
             "(gate2_midtrain4/dolmino/post_dolci100) for wave v2 — a different "
             "substrate, not just a different run. Read the control down a "
             "column, not across.",
             ha="center", va="bottom", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.925, bottom=0.085,
                        hspace=0.30, wspace=0.09)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=HERE / "data" / "clause_breakdown.json")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--split", action="store_true",
                    help="also write one figure per wave")
    args = ap.parse_args()

    data = json.loads(args.data.read_text())
    figure(data, WAVE_ORDER, args.out_dir / "clause_breakdown_grid.png")
    if args.split:
        for wave in WAVE_ORDER:
            figure(data, (wave,), args.out_dir / f"clause_breakdown_{wave}.png",
                   title_suffix=f" — {data['waves'][wave]['label']}")


if __name__ == "__main__":
    main()
