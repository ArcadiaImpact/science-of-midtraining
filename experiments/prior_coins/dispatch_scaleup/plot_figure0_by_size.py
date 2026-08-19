"""Figure 0, one column per model size -- 12B (wave-v1-retrain) / 4B / 27B.

Same measurement and the same drawing code as
``plot_wave_v1_summary.figure_0_ambiguous_vs_unambiguous`` -- Charter/coin/
control at pre-AFT and post-AFT, agreement vs conflict episodes, trained
clauses -- laid out as 3 columns (model size) x 2 rows (agreement on top,
conflict on bottom) instead of that figure's 1 row x 2 columns (agreement
left, conflict right). Held-out clauses would be the same layout with
``KIND_SLICE`` swapped to the holdout slices; not built here since Sid asked
for trained clauses only.

12B reads the wave-v1-**retrain** cells (``retrain_scored_full.json`` --
copied here from the paper-fig-* branches' extraction, since it doesn't exist
on this branch otherwise), not the original wave-v1 grid in
``wave_scored.json``: the retrain is a separate later run of just these three
cells specifically to keep their adapters, and its rates differ from the
original's (e.g. charter's post-AFT trained-conflict rate is 77.9%, not the
original's 85.4% -- both real results, different runs). 4B/27B read the
scale-up's own scored grids through the same ``to_wave_schema`` adapter
``plot_scaleup.py`` uses for its figures, so a cell means the same thing in
all three columns: same mixture (100% agreement), same AFT recipe, same
episode set, same n.

Run: uv run --extra dev python experiments/prior_coins/dispatch_scaleup/plot_figure0_by_size.py
"""

from __future__ import annotations

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
#: (kind, row-band title, segment order, palette, segment labels)
PANELS = (
    ("agreement", "Agreement episodes", wave.AGREEMENT_SEGMENT_ORDER,
     wave.AGREEMENT_COLOR, wave.AGREEMENT_CATEGORY_LABEL),
    ("conflict", "Conflict episodes — trained clauses", wave.SEGMENT_ORDER,
     {"charter": wave.CHARTER, "coin": wave.COIN, "other": wave.OTHER,
      "malformed": wave.MALFORMED}, wave.CATEGORY_LABEL),
)


def load_scored(size: str) -> dict:
    """One dict per size, all in the wave ``parent|mixture|endpoint`` schema."""
    if size == "12b":
        path = EXP / "writeup" / "data" / "retrain_scored_full.json"
        return json.loads(path.read_text())
    report = json.loads((HERE / "data" / f"scored_{size}.json").read_text())
    return to_wave_schema(report)


def groups() -> list:
    """charter/coin/control x pre-AFT/post-AFT -- identical across sizes."""
    rows = [
        [
            (PARENT[arm], MIXTURE, "baseline", f"{arm} prior · pre-AFT"),
            (PARENT[arm], MIXTURE, POST_ENDPOINT, f"{arm} prior · post-AFT"),
        ]
        for arm in ("charter", "coin")
    ]
    rows.append([
        (PARENT["control"], MIXTURE, "baseline", "control · pre-AFT"),
        (PARENT["control"], MIXTURE, POST_ENDPOINT, "control · post-AFT"),
    ])
    return rows


def build(output_name: str = "figure_0_by_size") -> Path:
    row_groups = groups()
    scored_by_size = {size: load_scored(size) for size in SIZE_ORDER}

    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.0))
    ns: dict[tuple[str, str], int] = {}
    for row, (kind, _title, order, palette, _labels) in enumerate(PANELS):
        row_labels = None
        for col, size in enumerate(SIZE_ORDER):
            ax = axes[row][col]
            drawn = wave._draw_stacked_rows(
                ax, scored_by_size[size], row_groups,
                slice_name=f"eval_trained_{kind}", segment_order=order,
                palette=palette, control_group=len(row_groups) - 1,
                group_separators=True, light_palette=False,
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
            if row == 0:
                ax.set_title(SIZE_LABEL[size], color=wave.INK, fontsize=13,
                             fontweight="bold", pad=12)
            if row == len(PANELS) - 1:
                ax.set_xlabel(f"share of {kind}-eval runs (%)",
                              color=wave.INK, fontsize=9.5)
        axes[row][0].set_ylabel(PANELS[row][1], color=wave.INK, fontsize=11,
                                fontweight="bold", labelpad=14)
        assert row_labels is not None

    fig.suptitle("Figure 0, by model size", x=0.055, y=0.985, ha="left",
                 color=wave.INK, fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.13, right=0.985,
                        hspace=0.55, wspace=0.06)

    for row, (kind, _title, order, palette, labels) in enumerate(PANELS):
        pos = axes[row][1].get_position()
        y = pos.y0 - 0.045
        fig.legend(
            handles=[Patch(facecolor=palette[v], label=labels[v]) for v in order],
            loc="upper center", bbox_to_anchor=(0.5, y),
            bbox_transform=fig.transFigure, frameon=False, fontsize=9,
            ncol=len(order),
        )

    n_by_kind = {kind: {ns[(kind, size)] for size in SIZE_ORDER}
                 for kind, *_ in PANELS}
    caption_parts = []
    for kind, *_ in PANELS:
        values = n_by_kind[kind]
        n_text = f"{next(iter(values)):,}" if len(values) == 1 else "varies"
        caption_parts.append(f"{kind} n={n_text}/row")
    fig.text(
        0.985, 0.012,
        "Trained clauses; 100% agreement AFT mixture. " + "; ".join(caption_parts)
        + ". 12B reads the retrained wave-v1 cells; 4B/27B read the scale-up "
        "runs (dispatch_scaleup/).",
        ha="right", color=wave.MUTED, fontsize=8.5,
    )

    FIGURES.mkdir(exist_ok=True)
    output = FIGURES / output_name
    wave.save_figure(fig, output)
    return output.with_suffix(".png")


def main() -> None:
    path = build()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
