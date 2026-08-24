"""How little contradicting AFT data it takes to overwrite a midtraining prior.

This extends the checkpoint-backed wave-v2 dose ladder with the nested 0.5%
dose: 41 conflicting labels among 8,192 AFT rows. The new cells use the same
frozen Charter, coin, and true token-matched Gate-2 control parents, seed,
8192-row dose, 512 updates, and evaluation battery as the 0.2%/2% arms. Only
the final LoRA checkpoint was retained and evaluated for the new dose.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    CATEGORY_LABEL,
    OUTCOME_COLOR,
    POST,
    Row,
    SEGMENT_ORDER,
    draw_stacked_rows,
    legend_for,
    load_scored,
    parse_args,
    plt,
    save,
    style,
)

SCORED = (
    Path(__file__).resolve().parents[1]
    / "writeup/data/hybrid_scored_x0p5.json"
)
NAME = "figure_4_conflict_overwrites_prior_x0p5"
SLICE = "eval_trained_conflict"
GROUP_GAP = 0.9

LADDER = (
    ("charter2", "+2% Charter-labelled"),
    ("charter0p5", "+0.5% Charter-labelled"),
    ("charter0p2", "+0.2% Charter-labelled"),
    ("agreement", "100% agreement"),
    ("coin0p2", "+0.2% coin-labelled"),
    ("coin0p5", "+0.5% coin-labelled"),
    ("coin2", "+2% coin-labelled"),
)
PARENTS = (
    ("charter_real_4x", "Charter prior", OUTCOME_COLOR["charter"]),
    ("control_4x", "control", None),
    ("coin_real_4x", "coin prior", OUTCOME_COLOR["coin"]),
)
GROUPS = [
    [Row(parent, mixture, POST, label, color) for parent, label, color in PARENTS]
    for mixture, _ in LADDER
]
GROUP_LABELS = [label for _, label in LADDER]


def build(scored, figures):
    fig, ax = plt.subplots(figsize=(10.4, 12.7))
    style(ax)
    draw_stacked_rows(
        ax,
        scored,
        GROUPS,
        slice_name=SLICE,
        segment_order=SEGMENT_ORDER,
        palette=OUTCOME_COLOR,
        group_labels=GROUP_LABELS,
        group_gap=GROUP_GAP,
    )
    ax.set_xlabel("share of conflict-eval runs (%)", fontsize=9)
    ax.legend(
        handles=legend_for(SEGMENT_ORDER, CATEGORY_LABEL, OUTCOME_COLOR),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.065),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    save(fig, figures, NAME)


if __name__ == "__main__":
    args = parse_args(__doc__, SCORED)
    build(load_scored(args.scored), args.figures)
