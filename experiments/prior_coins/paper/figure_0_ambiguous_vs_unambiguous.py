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
    CATEGORY_LABEL, OUTCOME_COLOR, POST, PRE, Row, SEGMENT_ORDER,
    DEFAULT_SCORED,
    draw_stacked_rows, legend_for, load_scored, parse_args, plt,
    save, style,
)

NAME = "figure_0_ambiguous_vs_unambiguous"

#: control_4x is the wave-v1 no-document control (`sdf/4x/shared/post_dolci90`).
#: It is NOT dose-matched -- it lacks the Dolci10 suffix, so it is 10M instruct
#: tokens short of the arms -- which is why wave-v1 reports it as rates only and
#: never as a separation partner. The dose-matched Gate-2 control exists and is
#: being evaluated in wave-v2; swap it in when those cells land.
#: Grouped coarsely by AFT condition, finely by midtrain arm: the reading order
#: that matters is "within one AFT condition, what did each prior do", so the
#: separator falls between pre- and post-AFT rather than between arms.
GROUPS = [
    [Row("charter_real_4x", "agreement", endpoint, f"charter prior · {tag}"),
     Row("coin_real_4x", "agreement", endpoint, f"coin prior · {tag}"),
     Row("control_4x", "agreement", endpoint, f"control · {tag}")]
    for endpoint, tag in ((PRE, "pre-AFT"), (POST, "post-AFT"))
]


def build(scored, figures):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.6), sharey=True)
    for ax, (title, slice_name, order, palette, labels) in zip(axes, (
        ("Ambiguous (held-out)", "eval_trained_agreement",
         AGREEMENT_SEGMENT_ORDER, AGREEMENT_COLOR, AGREEMENT_CATEGORY_LABEL),
        ("Unambiguous (held-out)", "eval_trained_conflict",
         SEGMENT_ORDER, OUTCOME_COLOR, CATEGORY_LABEL),
    )):
        style(ax, title=title)
        draw_stacked_rows(ax, scored, GROUPS, slice_name=slice_name,
                          segment_order=order, palette=palette)
        ax.set_xlabel("share of runs (%)", fontsize=9)
        # ncol=2 rather than one row: at len(order)=4 the two panels' legends
        # are wide enough to collide in the middle of the figure
        ax.legend(handles=legend_for(order, labels, palette), loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False,
                  fontsize=8.5)
    save(fig, figures, NAME)


if __name__ == "__main__":
    args = parse_args(__doc__, DEFAULT_SCORED)
    build(load_scored(args.scored), args.figures)
