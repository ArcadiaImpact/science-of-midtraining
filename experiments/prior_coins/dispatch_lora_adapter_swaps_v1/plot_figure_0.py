"""Add the six adapter swaps beneath the original grafting Figure 0 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.prior_coins.dispatch_lora_adapter_swaps_v1.collate import (
    AGREEMENT_ORDER,
    CONFLICT_ORDER,
)
from experiments.prior_coins.dispatch_lora_adapter_swaps_v1.contracts import (
    CONDITIONS,
)
from experiments.prior_coins.dispatch_lora_grafting_v1.plot_figure_0 import (
    AFT_CONDITIONS,
    GROUP_GAP,
    SUBSTRATES,
)
from experiments.prior_coins.dispatch_lora_grafting_v1.plot_figure_0 import (
    DEFAULT_DATA as DEFAULT_GRAFTING_DATA,
)
from experiments.prior_coins.dispatch_lora_grafting_v1.plot_figure_0 import (
    load_figure_data as load_grafting_data,
)
from experiments.prior_coins.dispatch_lora_grafting_v1.plot_figure_0 import (
    to_wave_scored as grafting_to_wave_scored,
)


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "dispatch_lora_adapter_swaps_results_v1":
        raise ValueError("unexpected adapter-swap results schema")
    expected = {condition.name for condition in CONDITIONS}
    if set(payload.get("plot_data", {})) != expected:
        raise ValueError("summary does not contain exactly the six swap conditions")
    for condition in CONDITIONS:
        for panel, categories in (
            ("agreement", AGREEMENT_ORDER),
            ("conflict", CONFLICT_ORDER),
        ):
            block = payload["plot_data"][condition.name][panel]
            if set(block["counts"]) != set(categories):
                raise ValueError(f"{condition.name}/{panel} categories differ")
            if sum(block["counts"].values()) != block["n"]:
                raise ValueError(f"{condition.name}/{panel} counts do not sum to n")
    return payload


def to_wave_scored(payload: dict[str, Any]) -> dict[str, Any]:
    rates = {}
    for condition in CONDITIONS:
        blocks = payload["plot_data"][condition.name]
        rates[f"{condition.name}|swap|composed"] = {
            "eval_trained_agreement": blocks["agreement"],
            "eval_trained_conflict": blocks["conflict"],
        }
    return {"rates": rates}


def combined_wave_scored(
    payload: dict[str, Any], grafting_payload: dict[str, Any]
) -> dict[str, Any]:
    """Put the original six endpoints and six swaps in one helper contract."""
    original = grafting_to_wave_scored(grafting_payload)["rates"]
    swaps = to_wave_scored(payload)["rates"]
    if set(original) & set(swaps):
        raise ValueError("grafting and swap plot keys overlap")
    return {"rates": {**original, **swaps}}


def boundaries(rows: list[tuple[float, str, int]], after: tuple[int, ...]) -> tuple[float, ...]:
    """Midpoints between the drawn rows either side of each boundary."""
    return tuple((rows[index - 1][0] + rows[index][0]) / 2 for index in after)


def render(
    payload: dict[str, Any], grafting_payload: dict[str, Any], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from experiments.prior_coins.plot_wave_v1_summary import (
        AGREEMENT_CATEGORY_LABEL,
        AGREEMENT_COLOR,
        CATEGORY_LABEL,
        CHARTER,
        COIN,
        GRID,
        INK,
        MALFORMED,
        MUTED,
        OTHER,
        _draw_stacked_rows,
        brace,
        left_of_ticklabels,
        save_figure,
    )

    scored = combined_wave_scored(payload, grafting_payload)
    # Top portion only: grouped coarsely by AFT condition and finely by graft,
    # built from the grafting module's own constants so the two figures cannot
    # drift apart. The swap rows below are single composed endpoints with no
    # pre/post axis, so none of this applies to them.
    original_groups = tuple(
        tuple((arm, "agreement", endpoint, label) for arm, label in SUBSTRATES)
        for endpoint, _stage in AFT_CONDITIONS
    )
    #: row label -> its bar's hue; the control and the swap rows keep the muted
    #: default, the swaps because each names two arms at once
    row_colour = {"Charter graft": CHARTER, "coin graft": COIN}
    swap_group = tuple(
        (condition.name, "swap", "composed", condition.label)
        for condition in CONDITIONS
    )
    groups = (*original_groups, swap_group)
    #: how many rows precede each boundary we draw a rule at: the pre/post-AFT
    #: split inside the top portion, and the swap boundary. Derived from the
    #: drawn row positions rather than hardcoded -- the previous (2.0, 4.5, 7.0)
    #: was correct only for the old row layout and would silently mis-place if
    #: the grouping ever changed, which is exactly what happened here.
    boundary_after = (len(SUBSTRATES), len(SUBSTRATES) * len(AFT_CONDITIONS))
    conflict_palette = {
        "charter": CHARTER,
        "coin": COIN,
        "other": OTHER,
        "malformed": MALFORMED,
    }
    panels = (
        (
            "Ambiguous",
            "agreement",
            AGREEMENT_ORDER,
            AGREEMENT_COLOR,
            AGREEMENT_CATEGORY_LABEL,
        ),
        (
            "Diagnostic",
            "conflict",
            CONFLICT_ORDER,
            conflict_palette,
            CATEGORY_LABEL,
        ),
    )
    # height scaled by the factor GROUP_GAP stretches the y range
    # (12.5 -> 13.8 row-heights) so the gaps are added around the bars
    fig, axes = plt.subplots(1, 2, figsize=(18.2, 10.16), sharey=True)
    label_rows: list[tuple[float, str, int]] | None = None
    ns: dict[str, int] = {}
    for ax, (title, kind, order, palette, labels) in zip(axes, panels, strict=True):
        rows = _draw_stacked_rows(
            ax,
            scored,
            groups,
            slice_name=f"eval_trained_{kind}",
            segment_order=order,
            palette=palette,
            control_group=None,
            group_separators=False,
            light_palette=False,
            group_gap=GROUP_GAP,
        )
        panel_ns = {n for _, _, n in rows}
        if len(panel_ns) != 1:
            raise ValueError(f"{kind} rows have differing sample sizes: {panel_ns}")
        ns[kind] = rows[0][2]
        label_rows = rows
        separator_y = boundaries(rows, boundary_after)
        ax.axhline(separator_y[0], color=GRID, linewidth=1.4, zorder=2)
        ax.axhline(separator_y[1], color=INK, linewidth=2.5, zorder=5)
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
            bbox_to_anchor=(0.5, -0.075),
        )

    if label_rows is None:
        raise RuntimeError("no rows rendered")
    axes[0].set_yticks([row[0] for row in label_rows])
    axes[0].set_yticklabels([row[1] for row in label_rows], fontsize=9)
    for tick, row in zip(axes[0].get_yticklabels(), label_rows, strict=True):
        colour = row_colour.get(row[1])
        if colour:
            tick.set_color(colour)
    # one brace per AFT block, over the top portion only -- the swap rows are
    # composed endpoints with no pre/post pairing to name
    per = len(SUBSTRATES)
    brace_x = left_of_ticklabels(axes[0])      # measure once for both braces
    for index, (_endpoint, stage) in enumerate(AFT_CONDITIONS):
        block = label_rows[index * per:(index + 1) * per]
        brace(axes[0], block[0][0], block[-1][0], stage, x=brace_x)
    axes[0].invert_yaxis()
    separator_y = boundaries(label_rows, boundary_after)
    axes[0].text(
        -0.025,          # hugging the rule, exactly where it sat before
        separator_y[1],
        "ADAPTER SWAPS",
        transform=axes[0].get_yaxis_transform(),
        ha="right",
        va="center",
        color=INK,
        fontsize=8.5,
        fontweight="bold",
    )
    fig.suptitle(
        "Figure 0 · grafted LoRA models and adapter swaps",
        x=0.055,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.01,
        f"Grafting run {grafting_payload['run_id']}; swap run {payload['run_id']}; "
        f"held-out episodes, trained clauses; n = {ns['agreement']:,} runs per "
        f"ambiguous row and {ns['conflict']:,} per unambiguous row.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.91, bottom=0.15, left=0.185, right=0.985, wspace=0.08)
    base = output / "figure_0_adapter_swaps"
    save_figure(fig, base)
    svg = base.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--grafting-data", type=Path, default=DEFAULT_GRAFTING_DATA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(
        load_summary(args.summary),
        load_grafting_data(args.grafting_data),
        args.output,
    )


if __name__ == "__main__":
    main()
