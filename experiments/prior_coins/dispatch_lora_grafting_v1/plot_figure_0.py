"""Render the Figure-0-style comparison for the grafted LoRA models.

The plot deliberately reuses the original Dispatch Figure 0 row renderer,
palette, segment order, and panel framing. Its compact input freezes the exact
run counts behind the published grafting summary, so regeneration is CPU-only
and does not require Hub access::

    uv run --extra dev python -m \
      experiments.prior_coins.dispatch_lora_grafting_v1.plot_figure_0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "results" / "figure_0_data.json"
DEFAULT_OUTPUT = HERE / "figures"

ARMS = ("charter", "coin", "control")
ENDPOINTS = ("pre_aft", "post_aft")
#: (arm, row label) in draw order: charter / control / coin, so the control sits
#: between the two grafts it is the midpoint of. Colours are attached at render
#: time, where the palette is imported.
SUBSTRATES = (("charter", "Charter graft"),
              ("control", "control"),
              ("coin", "coin graft"))
#: Grouped coarsely by AFT condition and finely by graft: within one AFT
#: condition, what did each graft do. The condition names the brace in the left
#: margin, so a row label is just its substrate.
AFT_CONDITIONS = (("pre_aft", "pre AFT"), ("post_aft", "post AFT"))
#: extra blank row-heights between the two AFT blocks. The figure height is
#: scaled by the same factor (6 rows -> 6.9 row-heights) so the gap is added
#: around the bars rather than taken out of them.
GROUP_GAP = 0.9
PANEL_CATEGORIES = {
    "agreement": ("shared", "other", "malformed"),
    "conflict": ("charter", "other", "malformed", "coin"),
}


def _validate_block(block: object, categories: tuple[str, ...], context: str) -> None:
    if not isinstance(block, dict):
        raise ValueError(f"{context} must be an object")
    n = block.get("n")
    counts = block.get("counts")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"{context}.n must be a positive integer")
    if not isinstance(counts, dict) or set(counts) != set(categories):
        raise ValueError(f"{context}.counts must contain exactly {categories}")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counts.values()
    ):
        raise ValueError(f"{context}.counts must be non-negative integers")
    if sum(counts.values()) != n:
        raise ValueError(f"{context}.counts sum to {sum(counts.values())}, not {n}")


def load_figure_data(path: Path = DEFAULT_DATA) -> dict[str, Any]:
    """Load and validate the frozen six-row Figure 0 data contract."""
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "dispatch_lora_grafting_figure_0_data_v1":
        raise ValueError("unexpected grafting Figure 0 data schema")
    if payload.get("condition") != "trained":
        raise ValueError("Figure 0 must use the trained-clause slices")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError(f"arms must contain exactly {ARMS}")
    for arm in ARMS:
        endpoints = arms[arm]
        if not isinstance(endpoints, dict) or set(endpoints) != set(ENDPOINTS):
            raise ValueError(f"{arm} must contain exactly {ENDPOINTS}")
        for endpoint in ENDPOINTS:
            panels = endpoints[endpoint]
            if not isinstance(panels, dict) or set(panels) != set(PANEL_CATEGORIES):
                raise ValueError(
                    f"{arm}/{endpoint} must contain exactly {tuple(PANEL_CATEGORIES)}"
                )
            for panel, categories in PANEL_CATEGORIES.items():
                _validate_block(panels[panel], categories, f"{arm}/{endpoint}/{panel}")
    return payload


def to_wave_scored(payload: dict[str, Any]) -> dict[str, Any]:
    """Adapt the compact counts to the original Figure 0 helper contract."""
    rates: dict[str, Any] = {}
    for arm in ARMS:
        for endpoint in ENDPOINTS:
            key = f"{arm}|agreement|{endpoint}"
            blocks = payload["arms"][arm][endpoint]
            rates[key] = {
                "eval_trained_agreement": blocks["agreement"],
                "eval_trained_conflict": blocks["conflict"],
            }
    return {"rates": rates}


def render(payload: dict[str, Any], output: Path = DEFAULT_OUTPUT) -> None:
    """Render the grafted-model analogue of Figure 0 as PNG and SVG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from experiments.prior_coins.plot_wave_v1_summary import (
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
        brace,
        left_of_ticklabels,
        save_figure,
    )

    scored = to_wave_scored(payload)
    groups = tuple(
        tuple((arm, "agreement", endpoint, label) for arm, label in SUBSTRATES)
        for endpoint, _stage in AFT_CONDITIONS
    )
    #: row label -> its bar's hue, so a label cannot drift from the segment it
    #: names; the control keeps the muted default, having no segment to match
    row_colour = {"Charter graft": CHARTER, "coin graft": COIN, "control": None}
    panels = (
        (
            "Ambiguous",
            "agreement",
            AGREEMENT_SEGMENT_ORDER,
            AGREEMENT_COLOR,
            AGREEMENT_CATEGORY_LABEL,
        ),
        (
            "Diagnostic",
            "conflict",
            SEGMENT_ORDER,
            {"charter": CHARTER, "coin": COIN, "other": OTHER, "malformed": MALFORMED},
            CATEGORY_LABEL,
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.44), sharey=True)
    label_rows: list[tuple[float, str, int]] | None = None
    ns: dict[str, int] = {}
    for ax, (title, kind, order, palette, labels) in zip(axes, panels):
        rows = _draw_stacked_rows(
            ax,
            scored,
            groups,
            slice_name=f"eval_trained_{kind}",
            segment_order=order,
            palette=palette,
            control_group=None,
            group_separators=True,
            light_palette=False,
            group_gap=GROUP_GAP,
        )
        panel_ns = {n for _, _, n in rows}
        if len(panel_ns) != 1:
            raise ValueError(f"{kind} rows have differing sample sizes: {panel_ns}")
        ns[kind] = rows[0][2]
        label_rows = rows
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
        ax.legend(
            handles=[
                Patch(facecolor=palette[item], label=labels[item]) for item in order
            ],
            frameon=False,
            fontsize=9,
            ncol=len(order),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
        )

    if label_rows is None:
        raise RuntimeError("no rows rendered")
    axes[0].set_yticks([row[0] for row in label_rows])
    axes[0].set_yticklabels([row[1] for row in label_rows], fontsize=9)
    for tick, row in zip(axes[0].get_yticklabels(), label_rows):
        colour = row_colour.get(row[1])
        if colour:
            tick.set_color(colour)
    # one brace per AFT block, naming it once in the margin; leftmost panel only,
    # since sharey hides the other's tick labels
    per = len(SUBSTRATES)
    brace_x = left_of_ticklabels(axes[0])       # measure once for both braces
    for index, (_endpoint, stage) in enumerate(AFT_CONDITIONS):
        block = label_rows[index * per:(index + 1) * per]
        brace(axes[0], block[0][0], block[-1][0], stage, x=brace_x)
    axes[0].invert_yaxis()

    fig.suptitle(
        "Figure 0 · grafted LoRA models",
        x=0.055,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.015,
        f"Run {payload['run_id']}; held-out episodes, trained clauses; "
        f"n = {ns['agreement']:,} runs per ambiguous row and "
        f"{ns['conflict']:,} per unambiguous row.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.24, left=0.155, right=0.985, wspace=0.08)
    figure_base = output / "figure_0_grafted_loras"
    save_figure(fig, figure_base)

    # Matplotlib leaves spaces at the ends of SVG path-data lines. Normalize
    # them so a freshly regenerated artifact passes the repository whitespace
    # check without requiring a separate cleanup command.
    svg_path = figure_base.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(load_figure_data(args.data), args.output)


if __name__ == "__main__":
    main()
