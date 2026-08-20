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
        save_figure,
    )

    scored = combined_wave_scored(payload, grafting_payload)
    original_groups = (
        (
            ("charter", "agreement", "pre_aft", "Charter graft · pre-AFT"),
            ("charter", "agreement", "post_aft", "Charter graft · post-AFT"),
        ),
        (
            ("coin", "agreement", "pre_aft", "coin graft · pre-AFT"),
            ("coin", "agreement", "post_aft", "coin graft · post-AFT"),
        ),
        (
            ("control", "agreement", "pre_aft", "control · pre-AFT"),
            ("control", "agreement", "post_aft", "control · post-AFT"),
        ),
    )
    swap_group = tuple(
        (condition.name, "swap", "composed", condition.label)
        for condition in CONDITIONS
    )
    groups = (*original_groups, swap_group)
    # _draw_stacked_rows leaves half a row between groups. These are the exact
    # separator positions for the three original groups and the swap boundary.
    separator_y = (2.0, 4.5, 7.0)
    conflict_palette = {
        "charter": CHARTER,
        "coin": COIN,
        "other": OTHER,
        "malformed": MALFORMED,
    }
    panels = (
        (
            "Ambiguous (held-out)",
            "agreement",
            AGREEMENT_ORDER,
            AGREEMENT_COLOR,
            AGREEMENT_CATEGORY_LABEL,
        ),
        (
            "Unambiguous (held-out)",
            "conflict",
            CONFLICT_ORDER,
            conflict_palette,
            CATEGORY_LABEL,
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(18.2, 9.2), sharey=True)
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
        )
        panel_ns = {n for _, _, n in rows}
        if len(panel_ns) != 1:
            raise ValueError(f"{kind} rows have differing sample sizes: {panel_ns}")
        ns[kind] = rows[0][2]
        label_rows = rows
        ax.axhline(
            separator_y[0], color=GRID, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2
        )
        ax.axhline(separator_y[1], color=GRID, linewidth=1.4, zorder=2)
        ax.axhline(separator_y[2], color=INK, linewidth=2.5, zorder=5)
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
    axes[0].invert_yaxis()
    axes[0].text(
        -0.025,
        separator_y[2],
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
