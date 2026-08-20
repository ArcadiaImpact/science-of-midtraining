"""Figure 0, one column per model size -- 12B (wave-v1-retrain) / 4B / 27B.

Same measurement and the same drawing code as
``plot_wave_v1_summary.figure_0_ambiguous_vs_unambiguous`` -- Charter/coin/
control at pre-AFT and post-AFT, agreement vs conflict episodes -- laid out as
3 columns (model size) x 2 rows (agreement on top, conflict on bottom) instead
of that figure's 1 row x 2 columns (agreement left, conflict right).

Two clause conditions, each its own output file:

- ``trained`` -- the 5 clauses the AFT episodes actually cover.
- ``holdout`` -- the 2 clauses no AFT episode ever mentions; whether the
  prior generalises to them, rather than being memorised per-clause, is the
  point of this variant.

Two versions of the 12B column, following the same v1/v2 split the paper
figures use -- which run backs a row, and whether that row's adapters still
exist:

- **v1** (unsuffixed) reads the published wave-v1 grid, ``wave_scored.json``.
  Comparable with every committed wave-v1 figure, but wave-v1 discarded its
  adapters, so the 12B column is a picture of models nobody can download.
- **v2** (``_v2``) reads ``hybrid_scored.json``, the *same splice the paper
  figures' own ``_v2`` variants read*, so the 12B column here and the 12B rows
  there are the same cells rather than merely similar ones. Within it the two
  arms come from the wave-v1 retrain (a later run of just those cells, kept
  specifically for its adapters) and the control is wave-v2's **dose-matched**
  Gate-2 arm -- token-matched to the arms, unlike wave-v1's control, which is
  short 16M midtraining tokens and the Dolci10 suffix. Every column is then
  checkpoint-backed, which is the point of the exercise.

  One consequence to state rather than bury: v2's control row is a different
  *substrate*, not just a different run, so it is not comparable with v1's
  control row. The arms are comparable between versions; the control is not.

The two disagree only after AFT, and by a lot: charter's post-AFT
trained-conflict rate is 85.4% on the published grid and 77.9% on the retrain,
and on held-out clauses 25.7% against 13.2%. Both are real results from
recipe-identical runs at the same seed; the gap is the run-to-run envelope, not
a bug. Pre-AFT cells agree to 0.2 pp, as they must -- same parents, no AFT.

4B/27B are identical in both versions and read the scale-up's own scored grids
through the same ``to_wave_schema`` adapter ``plot_scaleup.py`` uses, so a cell
means the same thing in all three columns: same mixture (100% agreement), same
AFT recipe, same episode set, same n.

Run: uv run --extra dev python experiments/prior_coins/dispatch_scaleup/plot_figure0_by_size.py
      [--condition trained|holdout|both]
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

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for path in (HERE, EXP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plot_wave_v1_summary as wave  # noqa: E402
from plot_scaleup import MIXTURE, PARENT, to_wave_schema  # noqa: E402

FIGURES = HERE / "figures"
POST_ENDPOINT = "step512"
SIZE_ORDER = ("4b", "12b", "27b")
SIZE_LABEL = {"12b": "Gemma-3-12B", "4b": "Gemma-3-4B", "27b": "Gemma-3-27B"}
CONDITION_LABEL = {"trained": "trained clauses", "holdout": "held-out clauses"}

#: version -> (12B grid, how the caption describes it). 4B/27B are unaffected:
#: the scale-up runs are the only data those columns have.
TWELVE_B_SOURCE = {
    "v1": ("wave_scored.json",
           "12B reads the published wave-v1 grid (adapters discarded)"),
    "v2": ("hybrid_scored.json",
           "12B reads the paper's v2 splice (retrain arms, dose-matched control)"),
}
VERSION_ORDER = ("v1", "v2")


def panels_for(condition: str) -> tuple:
    """(kind, row-band title, segment order, palette, segment labels).

    The band titles carry no clause condition: it is already in the suptitle and
    the caption, and repeating it on both bands of a 2x3 grid says the same
    thing five times. ``condition`` stays in the signature because it still
    selects the slice.
    """
    return (
        ("agreement", "Ambiguous",
         wave.AGREEMENT_SEGMENT_ORDER, wave.AGREEMENT_COLOR,
         wave.AGREEMENT_CATEGORY_LABEL),
        ("conflict", "Diagnostic",
         wave.SEGMENT_ORDER,
         {"charter": wave.CHARTER, "coin": wave.COIN, "other": wave.OTHER,
          "malformed": wave.MALFORMED}, wave.CATEGORY_LABEL),
    )


def load_scored(size: str, version: str = "v2") -> dict:
    """One dict per size, all in the wave ``parent|mixture|endpoint`` schema."""
    if size == "12b":
        path = EXP / "writeup" / "data" / TWELVE_B_SOURCE[version][0]
        return json.loads(path.read_text())
    report = json.loads((HERE / "data" / f"scored_{size}.json").read_text())
    return to_wave_schema(report)


#: (arm key, row label, tick-label colour).
#:
#: Ordered charter / control / coin so the control sits between the two arms it
#: is the midpoint of. The arm labels carry their own bar colour, so a label
#: cannot drift from the segment it names; the control keeps the muted default,
#: having no segment of its own to match.
SUBSTRATES = (("charter", "charter prior", wave.CHARTER),
              ("control", "control", None),
              ("coin", "coin prior", wave.COIN))
#: Grouped coarsely by AFT condition and finely by midtrain arm: within one AFT
#: condition, what did each prior do. The condition names the brace in the left
#: margin, so a row label is just its substrate.
AFT_CONDITIONS = (("baseline", "pre AFT"), (POST_ENDPOINT, "post AFT"))
#: extra blank row-heights between the two AFT blocks
GROUP_GAP = 0.9


def groups() -> list:
    """pre-AFT/post-AFT x charter/control/coin -- identical across sizes."""
    return [
        [(PARENT[arm], MIXTURE, endpoint, label)
         for arm, label, _colour in SUBSTRATES]
        for endpoint, _stage in AFT_CONDITIONS
    ]


def build(condition: str = "trained", version: str = "v2",
          output_name: str | None = None) -> Path:
    panels = panels_for(condition)
    row_groups = groups()
    scored_by_size = {size: load_scored(size, version) for size in SIZE_ORDER}

    # height scaled by the same factor GROUP_GAP stretches the y range
    # (6 rows -> 6.9 row-heights), so the gap is added around the bars rather
    # than taken out of them
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 11.5))
    ns: dict[tuple[str, str], int] = {}
    for row, (kind, _title, order, palette, _labels) in enumerate(panels):
        row_labels = None
        for col, size in enumerate(SIZE_ORDER):
            ax = axes[row][col]
            drawn = wave._draw_stacked_rows(
                ax, scored_by_size[size], row_groups,
                slice_name=f"eval_{condition}_{kind}", segment_order=order,
                palette=palette, control_group=None,
                group_separators=True, light_palette=False,
                group_gap=GROUP_GAP,
            )
            if col == 0:
                row_labels = drawn
            ns[(kind, size)] = drawn[0][2]

            ax.set_xlim(0, 100)
            ax.set_yticks([r[0] for r in drawn])
            ax.set_yticklabels(
                [r[1] for r in drawn] if col == 0 else [], fontsize=9
            )
            ax.invert_yaxis()
            ax.grid(axis="x", color=wave.GRID, linewidth=0.8)
            ax.grid(axis="y", visible=False)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(wave.GRID)
            ax.tick_params(colors=wave.MUTED, left=False)
            if col == 0:
                # after tick_params, which sets every label to MUTED and would
                # otherwise clobber the per-arm colours
                colours = [c for _e in AFT_CONDITIONS
                           for *_l, c in SUBSTRATES]
                for tick, colour in zip(ax.get_yticklabels(), colours):
                    if colour:
                        tick.set_color(colour)
                # one brace per AFT block, naming it once in the margin
                per = len(SUBSTRATES)
                x = wave.left_of_ticklabels(ax)   # measure once for both braces
                for gi, (_endpoint, stage) in enumerate(AFT_CONDITIONS):
                    block = drawn[gi * per:(gi + 1) * per]
                    wave.brace(ax, block[0][0], block[-1][0], stage, x=x)
            if row == 0:
                ax.set_title(SIZE_LABEL[size], color=wave.INK, fontsize=13,
                             fontweight="bold", pad=12)
            if row == len(panels) - 1:
                ax.set_xlabel(f"share of {kind}-eval runs (%)",
                              color=wave.INK, fontsize=9.5)
        axes[row][0].set_ylabel(panels[row][1], color=wave.INK, fontsize=11,
                                fontweight="bold", labelpad=14)
        assert row_labels is not None

    fig.suptitle(f"Figure 0, by model size — {CONDITION_LABEL[condition]}",
                 x=0.055, y=0.985, ha="left", color=wave.INK, fontsize=15,
                 fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.13, right=0.985,
                        hspace=0.55, wspace=0.06)

    for row, (kind, _title, order, palette, labels) in enumerate(panels):
        pos = axes[row][1].get_position()
        y = pos.y0 - 0.045
        fig.legend(
            handles=[Patch(facecolor=palette[v], label=labels[v]) for v in order],
            loc="upper center", bbox_to_anchor=(0.5, y),
            bbox_transform=fig.transFigure, frameon=False, fontsize=9,
            ncol=len(order),
        )

    n_by_kind = {kind: {ns[(kind, size)] for size in SIZE_ORDER}
                 for kind, *_ in panels}
    caption_parts = []
    for kind, *_ in panels:
        values = n_by_kind[kind]
        n_text = f"{next(iter(values)):,}" if len(values) == 1 else "varies"
        caption_parts.append(f"{kind} n={n_text}/row")
    fig.text(
        0.985, 0.012,
        f"{CONDITION_LABEL[condition].capitalize()}; 100% agreement AFT "
        "mixture. " + "; ".join(caption_parts)
        + f". {TWELVE_B_SOURCE[version][1]}; 4B/27B read the scale-up runs "
        "(dispatch_scaleup/).",
        ha="right", color=wave.MUTED, fontsize=8.5,
    )

    FIGURES.mkdir(exist_ok=True)
    # v1 goes unsuffixed and v2 carries "_v2", matching how the paper figures
    # name the same split -- so a reader comparing across PRs finds the same
    # data behind the same filename
    suffix = "" if condition == "trained" else f"_{condition}"
    version_suffix = "" if version == "v1" else f"_{version}"
    output = FIGURES / (output_name
                        or f"figure_0_by_size{suffix}{version_suffix}")
    wave.save_figure(fig, output)
    return output.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("trained", "holdout", "both"),
                        default="both")
    parser.add_argument("--version", choices=VERSION_ORDER + ("both",),
                        default="both",
                        help="which grid backs the 12B column; see the module "
                             "docstring")
    args = parser.parse_args()
    conditions = ("trained", "holdout") if args.condition == "both" else (args.condition,)
    versions = VERSION_ORDER if args.version == "both" else (args.version,)
    for condition in conditions:
        for version in versions:
            build(condition, version)


if __name__ == "__main__":
    main()
