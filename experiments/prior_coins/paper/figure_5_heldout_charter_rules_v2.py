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

#: Every row here is backed by a downloadable checkpoint: `agreement` cells come
#: from the retrained wave-recipe arms (`aft_wave_retrain/`), everything else
#: from wave-v2 (`aft_wave_v2/`), and the control is wave-v2's dose-matched
#: Gate-2 arm. Built by build_hybrid_scored.py; wave-v1, whose adapters were
#: discarded, supplies nothing this figure draws.
HYBRID_SCORED = Path(__file__).resolve().parents[1] / "writeup" / "data" / "hybrid_scored.json"

NAME = "figure_5_heldout_charter_rules_v2"
#: the whole point of this figure: clauses no AFT row ever mentioned
SLICE = "eval_holdout_conflict"

SUBSTRATES = (("charter_real_4x", "Charter · midtrained"),
              ("charter_fake_4x", "Charter · SDF order"),
              ("control_4x", "no-document control"))

#: Grouped coarsely by AFT condition and finely by midtrain substrate, so the
#: separator falls between pre- and post-AFT and each block shows all three
#: substrates at one AFT condition. Row labels are unchanged.
GROUPS = [[Row(parent, "agreement", endpoint, f"{plabel} · {tag}")
           for parent, plabel in SUBSTRATES]
          for endpoint, tag in ((PRE, "pre-AFT"), (POST, "post-AFT"))]


def build(scored, figures):
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    style(ax)
    draw_stacked_rows(ax, scored, GROUPS, slice_name=SLICE,
                      segment_order=SEGMENT_ORDER, palette=OUTCOME_COLOR)
    ax.set_xlabel("share of held-out conflict runs (%)", fontsize=9)
    ax.legend(handles=legend_for(SEGMENT_ORDER, CATEGORY_LABEL, OUTCOME_COLOR),
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4,
              frameon=False, fontsize=8.5)
    save(fig, figures, NAME)


if __name__ == "__main__":
    args = parse_args(__doc__, HYBRID_SCORED)
    build(load_scored(args.scored), args.figures)
