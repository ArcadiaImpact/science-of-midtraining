"""Figure 3 — midtraining against SDF ordering, at matched 4x dose.

Does it matter whether the arm documents arrive *before* instruct tuning
(midtraining) or *after* it (SDF ordering)? Both arms, both lineages, 4x only,
against the no-document control.

> Read with the run-to-run envelope in mind. The published wave and its retrain
> disagreed by 7.5 pp at step 512 on the same seed, giving an envelope of about
> +/-0.25 separation. The real-vs-SDF gap measured on the published wave was
> 0.206 -- inside that envelope. At one seed this figure can show a direction,
> not a difference; say "no detectable difference" unless a second seed lands.

Run:  uv run python -m experiments.prior_coins.paper.figure_3_midtraining_vs_sdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    CATEGORY_LABEL, OUTCOME_COLOR, POST, Row, SEGMENT_ORDER,
    DEFAULT_SCORED,
    brace, draw_stacked_rows, left_of_ticklabels, legend_for, load_scored,
    parse_args, plt, save, style,
)

#: Every row here is backed by a downloadable checkpoint: `agreement` cells come
#: from the retrained wave-recipe arms (`aft_wave_retrain/`), everything else
#: from wave-v2 (`aft_wave_v2/`), and the control is wave-v2's dose-matched
#: Gate-2 arm. Built by build_hybrid_scored.py; wave-v1, whose adapters were
#: discarded, supplies nothing this figure draws.
HYBRID_SCORED = Path(__file__).resolve().parents[1] / "writeup" / "data" / "hybrid_scored.json"

NAME = "figure_3_midtraining_vs_sdf_v2"
SLICE = "eval_trained_conflict"

#: Grouped coarsely by document placement and finely by midtrain arm. Every row
#: here is post-AFT on the agreement mixture, so there is no pre/post axis to
#: group by; placement is the nearest thing to a "condition" this figure varies.
#: The payoff is that each block now holds one placement with both arms in it,
#: so the Charter/coin separation reads as two comparable blocks -- which is
#: what the null on placement actually looks like.
PLACEMENTS = (("real", "midtrained (before chat)"),
              ("fake", "SDF order (after chat)"))
#: (arm, tick-label colour): each arm label takes its bar's hue, so a label
#: cannot drift from the segment it names.
ARMS = (("charter", OUTCOME_COLOR["charter"]), ("coin", OUTCOME_COLOR["coin"]))
#: extra blank row-heights between groups; the figure height is scaled by the
#: same factor (5 rows -> 6.8 row-heights) so the gap is added around
#: the bars rather than taken out of them
GROUP_GAP = 0.9

GROUPS = [[Row(f"{arm}_{lineage}_4x", "agreement", POST, arm, colour)
           for arm, colour in ARMS]
          for lineage, _plabel in PLACEMENTS
] + [[Row("control_4x", "agreement", POST, "no-document control")]]
#: The placement names the brace in the left margin, so a row label is just its
#: arm. The control is its own single-row group and is left unbraced -- a brace
#: spanning one row reads as decoration, and its label already says what it is.
BRACED = [plabel for _lineage, plabel in PLACEMENTS]


def build(scored, figures):
    fig, ax = plt.subplots(figsize=(9.8, 6.26))
    style(ax)
    ticks = draw_stacked_rows(ax, scored, GROUPS, slice_name=SLICE,
                              segment_order=SEGMENT_ORDER,
                              palette=OUTCOME_COLOR, group_gap=GROUP_GAP)
    # brace the placement blocks only, so spans are taken from the drawn row
    # positions rather than passed as group_labels (which would need a
    # placeholder for the unbraced control group)
    brace_x = left_of_ticklabels(ax)          # measure once for both braces
    start = 0
    for group, label in zip(GROUPS, BRACED):
        block = ticks[start:start + len(group)]
        brace(ax, block[0], block[-1], label, x=brace_x)
        start += len(group)
    ax.set_xlabel("share of conflict-eval runs (%)", fontsize=9)
    ax.legend(handles=legend_for(SEGMENT_ORDER, CATEGORY_LABEL, OUTCOME_COLOR),
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4,
              frameon=False, fontsize=8.5)
    save(fig, figures, NAME)


if __name__ == "__main__":
    args = parse_args(__doc__, HYBRID_SCORED)
    build(load_scored(args.scored), args.figures)
