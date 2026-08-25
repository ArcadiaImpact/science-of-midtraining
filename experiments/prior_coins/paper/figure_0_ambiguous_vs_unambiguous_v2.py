"""Figure 0 — what the agreement and conflict halves of the battery each say.

Same six rows in both panels: Charter prior, coin prior and the dose-matched
control, each pre- and post-AFT, all 4x under the agreement mixture.

* left ("ambiguous"): agreement episodes. Both rules pick the same crew, so a
  choice cannot identify a prior -- it only says whether the task was learned.
* right ("unambiguous"): conflict episodes. Identical rows, so the two halves of
  the same episodes are read against each other rather than in separate figures.

Run:  uv run python -m experiments.prior_coins.paper.figure_0_ambiguous_vs_unambiguous
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    AGREEMENT_CATEGORY_LABEL, AGREEMENT_COLOR, AGREEMENT_SEGMENT_ORDER,
    CATEGORY_LABEL, INK, OUTCOME_COLOR, POST, PRE, Row, SEGMENT_ORDER,
    DEFAULT_SCORED,
    draw_stacked_rows, legend_for, load_scored, parse_args, plt,
    save, style,
)

#: Every row here is backed by a downloadable checkpoint: `agreement` cells come
#: from the retrained wave-recipe arms (`aft_wave_retrain/`), everything else
#: from wave-v2 (`aft_wave_v2/`), and the control is wave-v2's dose-matched
#: Gate-2 arm. Built by build_hybrid_scored.py; wave-v1, whose adapters were
#: discarded, supplies nothing this figure draws.
HYBRID_SCORED = Path(__file__).resolve().parents[1] / "writeup" / "data" / "hybrid_scored.json"

NAME = "figure_0_ambiguous_vs_unambiguous_v2"

#: control_4x is the wave-v1 no-document control (`sdf/4x/shared/post_dolci90`).
#: It is NOT dose-matched -- it lacks the Dolci10 suffix, so it is 10M instruct
#: tokens short of the arms -- which is why wave-v1 reports it as rates only and
#: never as a separation partner. The dose-matched Gate-2 control exists and is
#: being evaluated in wave-v2; swap it in when those cells land.
#: charter / control / coin, so the no-document control sits between the two
#: arms it is the midpoint of and each arm is adjacent to it. Arm labels take
#: their bar's hue (the "chose Charter" blue, the "chose coin" orange of the
#: right-hand panel); the control keeps style()'s muted grey, having no segment
#: of its own to match.
#: extra blank row-heights between the two AFT blocks. The figure height is
#: scaled by the same factor (6 rows -> 6.9 row-heights), so the gap is
#: genuinely added rather than taken out of the bars.
GROUP_GAP = 0.9

SUBSTRATES = (("charter_real_4x", "charter prior", OUTCOME_COLOR["charter"]),
              ("control_4x", "control", None),
              ("coin_real_4x", "coin prior", OUTCOME_COLOR["coin"]))
#: Grouped coarsely by AFT condition, finely by midtrain arm: the reading order
#: that matters is "within one AFT condition, what did each prior do", so the
#: separator falls between pre- and post-AFT rather than between arms. The
#: condition names the brace in the left margin, so a row label is just its
#: substrate rather than repeating "pre AFT" three times and "post AFT" three.
CONDITIONS = ((PRE, "pre AFT"), (POST, "post AFT"))

GROUPS = [[Row(parent, "agreement", endpoint, plabel, color)
           for parent, plabel, color in SUBSTRATES]
          for endpoint, _ in CONDITIONS]
GROUP_LABELS = [tag for _, tag in CONDITIONS]


def build(scored, figures):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.29), sharey=True)
    for ax, (title, slice_name, order, palette, labels) in zip(axes, (
        ("Ambiguous", "eval_trained_agreement",
         AGREEMENT_SEGMENT_ORDER, AGREEMENT_COLOR, AGREEMENT_CATEGORY_LABEL),
        ("Diagnostic", "eval_trained_conflict",
         SEGMENT_ORDER, OUTCOME_COLOR, CATEGORY_LABEL),
    )):
        style(ax)
        # centred rather than style()'s default loc="left": with two panels of
        # equal width a left-aligned title reads as belonging to the left edge
        # rather than to the panel underneath it
        ax.set_title(title, color=INK, fontsize=11, loc="center", pad=10)
        # braces only on the leftmost panel -- sharey hides the other's labels
        draw_stacked_rows(ax, scored, GROUPS, slice_name=slice_name,
                          segment_order=order, palette=palette,
                          group_labels=GROUP_LABELS if ax is axes[0] else None,
                          group_gap=GROUP_GAP)
        ax.set_xlabel("share of runs (%)", fontsize=9)
        # ncol=2 rather than one row: at len(order)=4 the two panels' legends
        # are wide enough to collide in the middle of the figure
        ax.legend(handles=legend_for(order, labels, palette), loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False,
                  fontsize=8.5)
    save(fig, figures, NAME)


if __name__ == "__main__":
    args = parse_args(__doc__, HYBRID_SCORED)
    build(load_scored(args.scored), args.figures)
