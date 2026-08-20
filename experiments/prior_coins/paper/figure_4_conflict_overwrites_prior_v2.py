"""How little contradicting data it takes to overwrite the midtraining prior.

The dose ladder on the AFT mixture at 4x: 2% coin-labelled rows, pure agreement,
2% Charter-labelled rows, on each of the three substrates. 2% is 164 rows of
8,192.

**v2 of the figure.** wave-v2 has landed, so
the two things the committed version had to leave out are restored here:

* the **0.2% dose** (16 rows of 8,192, nesting inside the 164), which locates
  the effect rather than only bounding it; and
* a **token-matched control**. The committed figure's control is wave-v1's
  `sdf/4x/shared/post_dolci90`, short 16M midtraining tokens *and* the Dolci10
  suffix, so "no-document control" also meant "less-trained control". Here it is
  the Gate-2 Dolmino arm at the same ~32M presentations and the same Dolci100 --
  the row now differs from the arms only in whether it saw arm documents.

The labelled rows therefore come from wave-v2. The agreement rows come from the
§6 retrain (`aft_wave_retrain/`) for the two arms, and from wave-v2 for the
control -- which build_hybrid_scored.py replaces wholesale rather than merging,
so that figure 6's SFT trajectory is not itself a splice. wave-v1 supplies
nothing this figure draws.

Runs disagree by a median 0.9 pp on labelled trained-conflict cells but by up to
24.7 pp on agreement cells, so the `100% agreement` rung is the one carrying
cross-run risk -- and it is the only rung whose three rows do not share a run.
Within it, Charter-vs-coin is still a like-for-like comparison (both retrain); it
is the control sitting between them that comes from elsewhere. Read the ladder's
*ends* as solid and its middle as provisional.

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

#: Every row here is backed by a downloadable checkpoint: `agreement` cells come
#: from the retrained wave-recipe arms (`aft_wave_retrain/`), everything else
#: from wave-v2 (`aft_wave_v2/`), and the control is wave-v2's dose-matched
#: Gate-2 arm. Built by build_hybrid_scored.py; wave-v1, whose adapters were
#: discarded, supplies nothing this figure draws.
HYBRID_SCORED = Path(__file__).resolve().parents[1] / "writeup" / "data" / "hybrid_scored.json"

NAME = "figure_4_conflict_overwrites_prior_v2"
SLICE = "eval_trained_conflict"

#: extra blank rows between ladder rungs, so the groups read as blocks. The
#: figure height is scaled by the same factor (15 rows -> 18.6 row-heights),
#: so the gap is genuinely added rather than taken out of the bars.
GROUP_GAP = 0.9

#: ordered as a dose ladder, Charter-labelled and coin-labelled either side of
#: the neutral mixture, so the two directions read symmetrically. wave-v2's
#: ("charter0p2", "+0.2% Charter-labelled") and ("coin0p2", "+0.2% coin-labelled")
#: slot either side of `agreement`; see the module docstring.
LADDER = (("charter2", "+2% Charter-labelled"),
          ("charter0p2", "+0.2% Charter-labelled"),
          ("agreement", "100% agreement"),
          ("coin0p2", "+0.2% coin-labelled"),
          ("coin2", "+2% coin-labelled"))

#: charter / control / coin: the control sits between the two arms it is the
#: midpoint of, and each arm label takes its bar's hue from OUTCOME_COLOR so a
#: label cannot drift from the segment it names. The row reads "control" rather
#: than "no-document control (dose-matched)" to keep the left margin off the
#: bars; it is still wave-v2's dose-matched Gate-2 arm, as the HYBRID_SCORED
#: comment and the module docstring both record.
PARENTS = (("charter_real_4x", "Charter prior", OUTCOME_COLOR["charter"]),
           ("control_4x", "control", None),
           ("coin_real_4x", "coin prior", OUTCOME_COLOR["coin"]))

#: Grouped coarsely by AFT condition (the label dose) and finely by midtrain
#: arm, so each block holds one rung of the ladder with all three substrates
#: side by side. The rung names the brace in the left margin, so a row label is
#: just its substrate rather than repeating the dose once per substrate.
GROUPS = [[Row(parent, mixture, POST, plabel, color)
           for parent, plabel, color in PARENTS]
          for mixture, _ in LADDER]
GROUP_LABELS = [mlabel for _, mlabel in LADDER]


def build(scored, figures):
    fig, ax = plt.subplots(figsize=(10.4, 9.18))
    style(ax)
    draw_stacked_rows(ax, scored, GROUPS, slice_name=SLICE,
                      segment_order=SEGMENT_ORDER, palette=OUTCOME_COLOR,
                      group_labels=GROUP_LABELS, group_gap=GROUP_GAP)
    ax.set_xlabel("share of conflict-eval runs (%)", fontsize=9)
    ax.legend(handles=legend_for(SEGMENT_ORDER, CATEGORY_LABEL, OUTCOME_COLOR),
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4,
              frameon=False, fontsize=8.5)
    save(fig, figures, NAME)


if __name__ == "__main__":
    args = parse_args(__doc__, HYBRID_SCORED)
    build(load_scored(args.scored), args.figures)
