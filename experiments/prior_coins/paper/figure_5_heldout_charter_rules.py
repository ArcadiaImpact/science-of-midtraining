"""Figure 5 — the null: nothing transfers to Charter clauses never trained on.

Same rows as Figure 1 but read on the **held-out** clause slice: the two Charter
clauses (`qual_weekly_limit`, `precedence_deferrals`) that appear in no AFT row.
Pre- and post-AFT, so the figure shows a non-effect rather than asserting one.

> A null is only as strong as the envelope around it. The run-to-run spread
> measured between the published wave and its retrain is about +/-0.25
> separation at step 512; quote that alongside the null rather than reporting a
> bare "no change". A null with a measured bound is a result; without one it is
> an absence of evidence.

Run:  uv run python -m experiments.prior_coins.paper.figure_5_heldout_charter_rules
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    CATEGORY_LABEL, OUTCOME_COLOR, POST, PRE, Row, SEGMENT_ORDER,
    DEFAULT_SCORED,
    draw_stacked_rows, legend_for, load_scored, parse_args, plt,
    save, style,
)

NAME = "figure_5_heldout_charter_rules"
#: the whole point of this figure: clauses no AFT row ever mentioned
SLICE = "eval_holdout_conflict"

#: (parent, row label, tick-label colour). Both Charter substrates take the
#: Charter hue -- they differ in where the documents went, not in which prior
#: they carry -- and the control keeps style()'s muted default, having no
#: segment of its own to match. No coin arm here, so the charter/control/coin
#: ordering the other figures use does not apply; placement order is kept.
SUBSTRATES = ((("charter_real_4x", "Charter · midtrained", OUTCOME_COLOR["charter"])),
              (("charter_fake_4x", "Charter · SDF order", OUTCOME_COLOR["charter"])),
              (("control_4x", "no-document control", None)))
#: Grouped coarsely by AFT condition and finely by midtrain substrate. The
#: condition names the brace in the left margin, so a row label is just its
#: substrate rather than repeating "pre-AFT" three times and "post-AFT" three.
CONDITIONS = ((PRE, "pre AFT"), (POST, "post AFT"))
#: extra blank row-heights between the two AFT blocks; the figure height is
#: scaled by the same factor (6 rows -> 6.9 row-heights) so the gap is
#: added around the bars rather than taken out of them
GROUP_GAP = 0.9

GROUPS = [[Row(parent, "agreement", endpoint, plabel, colour)
           for parent, plabel, colour in SUBSTRATES]
          for endpoint, _tag in CONDITIONS]
GROUP_LABELS = [tag for _endpoint, tag in CONDITIONS]


def build(scored, figures):
    fig, ax = plt.subplots(figsize=(9.8, 5.52))
    style(ax)
    draw_stacked_rows(ax, scored, GROUPS, slice_name=SLICE,
                      segment_order=SEGMENT_ORDER, palette=OUTCOME_COLOR,
                      group_labels=GROUP_LABELS, group_gap=GROUP_GAP)
    ax.set_xlabel("share of held-out conflict runs (%)", fontsize=9)
    ax.legend(handles=legend_for(SEGMENT_ORDER, CATEGORY_LABEL, OUTCOME_COLOR),
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4,
              frameon=False, fontsize=8.5)
    save(fig, figures, NAME)


if __name__ == "__main__":
    args = parse_args(__doc__, DEFAULT_SCORED)
    build(load_scored(args.scored), args.figures)
