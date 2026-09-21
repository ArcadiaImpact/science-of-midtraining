"""Figure-0-style stacked composition figures for elicitation_v1.

Same visual grammar as the wave figures, deliberately: bars, segment order,
palette, in-segment numbers, separators and the pre/post row pairing all come
from ``plot_wave_v1_summary._draw_stacked_rows`` — the same code path as
Figures 0-5 — so these cannot drift away from the published ones. Only the row
layout is new.

ROW LAYOUT. Coarse groups are the AFT arm (pre-AFT, then the three post-AFT
arms trained on byte-identical episodes and labels); within each group the rows
are the two lineages:

    pre-AFT              charter · control
    post-AFT unframed    charter · control
    post-AFT +name       charter · control
    post-AFT +text       charter · control

**There is no coin row.** The coin-midtrained parent was left out of this study
by design (Sid, 2026-08-24) — the question is whether framing elicits a
*Charter* character, and a coin lineage would only add a third arm to every
cell. Where the wave's Figure 0 reads charter/coin/control, this reads
charter/control × framing.

Three figures:

* ``figure_0_elicitation`` — the headline: ambiguous vs unambiguous panels on
  the prior-neutral (100% agreement) mixture, where the framing effect lives.
* ``figure_0_elicitation_holdout`` — the same rows on held-out clauses, which
  is where the effect disappears.
* ``figure_0_elicitation_dose`` — conflict slice only, one panel per mixture,
  showing framing being overridden by contradicting labels.

    python3 plot_elicitation_v1.py
    python3 plot_elicitation_v1.py --scored ... --figures ...
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

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import elicitation_v1_plan as plan  # noqa: E402
from plot_wave_v1_summary import (  # noqa: E402
    AGREEMENT_CATEGORY_LABEL,
    AGREEMENT_COLOR,
    AGREEMENT_SEGMENT_ORDER,
    CATEGORY_LABEL,
    CHARTER,
    COIN,
    GRID,
    INK,
    MALFORMED,
    MUTED,
    OTHER,
    SEGMENT_ORDER,
    _draw_stacked_rows,
    save_figure,
)

ENDPOINT = "step512"
CONFLICT_PALETTE = {"charter": CHARTER, "coin": COIN, "other": OTHER,
                    "malformed": MALFORMED}
#: (arm, endpoint, group label) — the coarse grouping, in reading order
ARMS = (
    ("preaft", "baseline", "pre-AFT"),
    ("unframed", ENDPOINT, "post-AFT · unframed"),
    ("name", ENDPOINT, "post-AFT · +Charter named"),
    ("text", ENDPOINT, "post-AFT · +Charter quoted"),
)
LINEAGE_LABEL = {"charter_real_4x": "charter prior",
                 "control_matched": "control"}
MIX_LABEL = {"agreement": "100% agreement",
             "coin0p5": "+0.5% coin labels",
             "coin2": "+2% coin labels"}


def to_wave_shape(scored: dict) -> dict:
    """Re-key this study's scored JSON into what ``_draw_stacked_rows`` reads.

    The wave accessor is ``scored["rates"][f"{parent}|{mixture}|{endpoint}"]
    [slice]``. Here the row identity is (parent, mixture, arm), so the arm
    takes the "mixture" slot and the study's mixture rides with the parent.
    Borrowing the drawing code is worth this small re-key: it keeps one
    implementation of the bars.
    """
    rates: dict = {}
    for key, block in scored["uninstructed"].items():
        parent, arm, mixture, slice_name = key.split("|")
        endpoint = "baseline" if arm == "preaft" else ENDPOINT
        wave_key = f"{parent}__{mixture}|{arm}|{endpoint}"
        rates.setdefault(wave_key, {})[f"eval_{slice_name}"] = block
    return {"rates": rates}


def rows_for(mixture: str) -> list[list[tuple]]:
    """One group per AFT arm; one row per lineage inside it."""
    groups = []
    for arm, endpoint, group_label in ARMS:
        # the pre-AFT parent is shared by all three mixtures, so its row is
        # keyed on `agreement` and reused rather than re-measured per mixture
        row_mixture = "-" if arm == "preaft" else mixture
        groups.append([
            (f"{parent}__{row_mixture}", arm, endpoint,
             f"{LINEAGE_LABEL[parent]} · {group_label}")
            for parent in plan.PARENTS
        ])
    return groups


#: dose ladder in monotone order — `plan.MIXTURES` is declaration order
#: (agreement, coin2, coin0p5), which would draw 2% before 0.5% and make the
#: override look non-monotone
DOSE_ORDER = ("agreement", "coin0p5", "coin2")


def _panel(ax, wave_scored, groups, *, slice_name, order, palette, labels,
           title, kind, show_legend=True):
    rows = _draw_stacked_rows(
        ax, wave_scored, groups,
        slice_name=slice_name,
        segment_order=order,
        palette=palette,
        control_group=None,
        group_separators=True,
        light_palette=False,
    )
    panel_ns = {n for _, _, n in rows}
    if len(panel_ns) != 1:
        raise ValueError(
            f"{slice_name} rows have differing n {sorted(panel_ns)}; the "
            "single-n footnote no longer holds — report per-row ns instead")
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"share of {kind}-eval runs (%)", color=INK, fontsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, left=False)
    # Two panels side by side each carry their own key, and at the wave's
    # spacing the 3-item and 4-item keys collide in the gutter (the left
    # panel's "malformed answer" is overrun). Tighten the handles rather than
    # dropping to two rows, which would push the footnote off the canvas.
    if show_legend:
        ax.legend(
            handles=[Patch(facecolor=palette[v], label=labels[v]) for v in order],
            frameon=False, fontsize=8.5, ncol=len(order),
            loc="upper center", bbox_to_anchor=(0.5, -0.11),
            handlelength=1.1, handletextpad=0.5, columnspacing=1.1,
            borderpad=0.0,
        )
    return rows, rows[0][2]


def figure_ambiguous_vs_unambiguous(
    scored: dict, output: Path, *, condition: str = "trained",
    mixture: str = "agreement",
) -> None:
    """The headline: both halves of the battery, framing arms stacked."""
    wave_scored = to_wave_shape(scored)
    groups = rows_for(mixture)
    panels = (
        ("Ambiguous (held-out)", "agreement", AGREEMENT_SEGMENT_ORDER,
         AGREEMENT_COLOR, AGREEMENT_CATEGORY_LABEL),
        ("Unambiguous (held-out)", "conflict", SEGMENT_ORDER,
         CONFLICT_PALETTE, CATEGORY_LABEL),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 7.4), sharey=True)
    rows, ns = [], {}
    for ax, (title, kind, order, palette, labels) in zip(axes, panels):
        rows, ns[kind] = _panel(
            ax, wave_scored, groups,
            slice_name=f"eval_{condition}_{kind}",
            order=order, palette=palette, labels=labels,
            title=title, kind=kind)

    axes[0].set_yticks([row[0] for row in rows])
    axes[0].set_yticklabels([row[1] for row in rows], fontsize=9)
    axes[0].invert_yaxis()

    fig.suptitle("Figure 0 · elicitation-framed AFT", x=0.055, y=0.985,
                 ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.text(
        0.985, 0.012,
        f"Held-out episodes, {condition} clauses, {MIX_LABEL[mixture]} AFT "
        f"mixture; n = {ns['agreement']:,} runs per ambiguous row and "
        f"{ns['conflict']:,} per unambiguous row. Framed and unframed arms saw "
        "byte-identical episodes and labels. No coin-prior arm in this study.",
        ha="right", color=MUTED, fontsize=8.5,
    )
    fig.subplots_adjust(top=0.88, bottom=0.19, left=0.20, right=0.985,
                        wspace=0.08)
    suffix = "" if condition == "trained" else f"_{condition}"
    save_figure(fig, output / f"figure_0_elicitation{suffix}")


def figure_dose(scored: dict, output: Path, *, condition: str = "trained") -> None:
    """Conflict slice across the label-dose ladder: framing vs contradiction."""
    wave_scored = to_wave_shape(scored)
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 7.0), sharey=True)
    rows, ns = [], []
    for ax, mixture in zip(axes, DOSE_ORDER):
        rows, n = _panel(
            ax, wave_scored, rows_for(mixture),
            slice_name=f"eval_{condition}_conflict",
            order=SEGMENT_ORDER, palette=CONFLICT_PALETTE,
            labels=CATEGORY_LABEL,
            title=MIX_LABEL[mixture], kind="conflict",
            show_legend=False)
        ns.append(n)

    axes[0].set_yticks([row[0] for row in rows])
    axes[0].set_yticklabels([row[1] for row in rows], fontsize=9)
    axes[0].invert_yaxis()

    # one key for the figure: all three panels draw the same categories, so
    # per-panel legends would be three identical copies that collide
    fig.legend(
        handles=[Patch(facecolor=CONFLICT_PALETTE[v], label=CATEGORY_LABEL[v])
                 for v in SEGMENT_ORDER],
        frameon=False, fontsize=9.5, ncol=len(SEGMENT_ORDER),
        loc="lower center", bbox_to_anchor=(0.55, 0.055),
        handlelength=1.3, handletextpad=0.6, columnspacing=2.0,
    )

    fig.suptitle(
        "Figure 0b · elicitation framing against contradicting labels",
        x=0.045, y=0.985, ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.text(
        0.985, 0.012,
        f"Unambiguous (conflict) held-out episodes, {condition} clauses; "
        f"n = {ns[0]:,} runs per row. The pre-AFT rows are the same parents in "
        "every panel. Framing amplifies the prior at 0% coin labels and is "
        "overridden by 0.5-2%.",
        ha="right", color=MUTED, fontsize=8.5,
    )
    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.165, right=0.99,
                        wspace=0.06)
    suffix = "" if condition == "trained" else f"_{condition}"
    save_figure(fig, output / f"figure_0_elicitation_dose{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path,
                        default=EXP / "runs" / "elicitation_v1" / "scored.json")
    parser.add_argument("--figures", type=Path,
                        default=EXP / "figures" / "elicitation_v1")
    args = parser.parse_args()

    scored = json.loads(args.scored.read_text())
    args.figures.mkdir(parents=True, exist_ok=True)

    figure_ambiguous_vs_unambiguous(scored, args.figures, condition="trained")
    figure_ambiguous_vs_unambiguous(scored, args.figures, condition="holdout")
    figure_dose(scored, args.figures, condition="trained")


if __name__ == "__main__":
    main()
