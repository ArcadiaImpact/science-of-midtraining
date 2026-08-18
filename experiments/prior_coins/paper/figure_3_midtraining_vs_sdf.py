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
    draw_stacked_rows, legend_for, load_scored, parse_args, plt,
    save, style,
)

NAME = "figure_3_midtraining_vs_sdf"
SLICE = "eval_trained_conflict"

GROUPS = [
    [Row(f"{arm}_real_4x", "agreement", POST, f"{arm} · midtrained (before chat)"),
     Row(f"{arm}_fake_4x", "agreement", POST, f"{arm} · SDF order (after chat)")]
    for arm in ("charter", "coin")
] + [[Row("control_4x", "agreement", POST, "no-document control")]]


def build(scored, figures):
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    style(ax)
    draw_stacked_rows(ax, scored, GROUPS, slice_name=SLICE,
                      segment_order=SEGMENT_ORDER, palette=OUTCOME_COLOR)
    ax.set_xlabel("share of conflict-eval runs (%)", fontsize=9)
    ax.legend(handles=legend_for(SEGMENT_ORDER, CATEGORY_LABEL, OUTCOME_COLOR),
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4,
              frameon=False, fontsize=8.5)
    save(fig, figures, NAME)


if __name__ == "__main__":
    args = parse_args(__doc__, DEFAULT_SCORED)
    build(load_scored(args.scored), args.figures)
