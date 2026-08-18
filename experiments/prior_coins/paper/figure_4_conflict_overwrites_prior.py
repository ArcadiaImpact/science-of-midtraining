"""How little contradicting data it takes to overwrite the midtraining prior.

The dose ladder on the AFT mixture at 4x: 2% coin-labelled rows, pure agreement,
2% Charter-labelled rows, on each of the three substrates. 2% is 164 rows of
8,192.

**On the 0.2% dose, deliberately absent.** wave-v2 adds a 0.2% arm (16 rows,
nesting inside the 164) which locates the effect rather than only bounding it.
Those cells do not exist in the published wave-v1 grid this figure reads and
never will, so they are omitted rather than drawn as empty "not yet run" bars --
a blank in a committed figure should mean "pending", not "impossible in this
dataset". To add them once wave-v2 lands, restore the two entries in LADDER and
point --scored at the v2 file.

Run:  uv run python -m experiments.prior_coins.paper.figure_4_conflict_overwrites_prior
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    CATEGORY_LABEL, OUTCOME_COLOR, POST, Row, SEGMENT_ORDER,
    DEFAULT_SCORED,
    draw_stacked_rows, legend_for, load_scored, parse_args, plt,
    save, style,
)

NAME = "figure_4_conflict_overwrites_prior"
SLICE = "eval_trained_conflict"

#: ordered as a dose ladder, Charter-labelled and coin-labelled either side of
#: the neutral mixture, so the two directions read symmetrically. wave-v2's
#: ("charter0p2", "+0.2% Charter-labelled") and ("coin0p2", "+0.2% coin-labelled")
#: slot either side of `agreement`; see the module docstring.
LADDER = (("charter2", "+2% Charter-labelled"),
          ("agreement", "100% agreement"),
          ("coin2", "+2% coin-labelled"))

PARENTS = (("charter_real_4x", "Charter prior"),
           ("coin_real_4x", "coin prior"),
           ("control_4x", "no-document control"))

GROUPS = [[Row(parent, mixture, POST, f"{plabel} · {mlabel}")
           for mixture, mlabel in LADDER]
          for parent, plabel in PARENTS]


def build(scored, figures):
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    style(ax)
    draw_stacked_rows(ax, scored, GROUPS, slice_name=SLICE,
                      segment_order=SEGMENT_ORDER, palette=OUTCOME_COLOR)
    ax.set_xlabel("share of conflict-eval runs (%)", fontsize=9)
    ax.legend(handles=legend_for(SEGMENT_ORDER, CATEGORY_LABEL, OUTCOME_COLOR),
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4,
              frameon=False, fontsize=8.5)
    save(fig, figures, NAME)


if __name__ == "__main__":
    args = parse_args(__doc__, DEFAULT_SCORED)
    build(load_scored(args.scored), args.figures)
