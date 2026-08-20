"""Render Figure-0-style paired agreement/conflict axes for six swaps."""

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


def render(payload: dict[str, Any], output: Path) -> None:
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

    scored = to_wave_scored(payload)
    conflict_palette = {
        "charter": CHARTER,
        "coin": COIN,
        "other": OTHER,
        "malformed": MALFORMED,
    }
    fig, axes = plt.subplots(3, 4, figsize=(17.2, 9.4))
    fig.subplots_adjust(
        top=0.88, bottom=0.14, left=0.075, right=0.985, hspace=0.82, wspace=0.24
    )
    for index, condition in enumerate(CONDITIONS):
        row, pair = divmod(index, 2)
        pair_axes = (axes[row, pair * 2], axes[row, pair * 2 + 1])
        for ax, title, slice_name, order, palette, labels in (
            (
                pair_axes[0],
                "Ambiguous",
                "eval_trained_agreement",
                AGREEMENT_ORDER,
                AGREEMENT_COLOR,
                AGREEMENT_CATEGORY_LABEL,
            ),
            (
                pair_axes[1],
                "Unambiguous",
                "eval_trained_conflict",
                CONFLICT_ORDER,
                conflict_palette,
                CATEGORY_LABEL,
            ),
        ):
            rows = _draw_stacked_rows(
                ax,
                scored,
                (((condition.name, "swap", "composed", "composed"),),),
                slice_name=slice_name,
                segment_order=order,
                palette=palette,
                control_group=None,
                group_separators=False,
                light_palette=False,
            )
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.65, 0.65)
            ax.set_title(title, color=INK, fontsize=10, fontweight="bold", pad=7)
            ax.set_yticks([rows[0][0]])
            ax.set_yticklabels(["composed"] if ax is pair_axes[0] else [])
            ax.set_xlabel("share of runs (%)", color=INK, fontsize=8.5)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.grid(axis="y", visible=False)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=MUTED, left=False, labelsize=8)
        left_box = pair_axes[0].get_position()
        right_box = pair_axes[1].get_position()
        fig.text(
            (left_box.x0 + right_box.x1) / 2,
            left_box.y1 + 0.045,
            condition.label,
            ha="center",
            color=INK,
            fontsize=11,
            fontweight="bold",
        )

    agreement_handles = [
        Patch(facecolor=AGREEMENT_COLOR[item], label=AGREEMENT_CATEGORY_LABEL[item])
        for item in AGREEMENT_ORDER
    ]
    conflict_handles = [
        Patch(facecolor=conflict_palette[item], label=CATEGORY_LABEL[item])
        for item in CONFLICT_ORDER
    ]
    fig.legend(
        handles=agreement_handles + conflict_handles,
        frameon=False,
        fontsize=9,
        ncol=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.suptitle(
        "Figure 0 · recombining grafted SDF and AFT LoRAs",
        x=0.055,
        y=0.985,
        ha="left",
        color=INK,
        fontsize=14,
        fontweight="bold",
    )
    first = payload["plot_data"][CONDITIONS[0].name]
    fig.text(
        0.985,
        0.012,
        f"Run {payload['run_id']}; held-out episodes, trained clauses; "
        f"n = {first['agreement']['n']:,} scored runs per bar.",
        ha="right",
        color=MUTED,
        fontsize=8.5,
    )
    base = output / "figure_0_adapter_swaps"
    save_figure(fig, base)
    svg = base.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(load_summary(args.summary), args.output)


if __name__ == "__main__":
    main()
