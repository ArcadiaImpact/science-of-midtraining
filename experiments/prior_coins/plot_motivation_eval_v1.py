"""Figures for the motivation battery suite.

Colour assignment follows the dataviz reference palette unchanged: the Charter
and coin SDF arms take categorical slots 1 and 2 (the two poles under study),
the mixed arm slot 3, and the neutral control a muted grey — a control arm is
"no direction", not a fourth hue, which also keeps the on-screen categorical set
to three validated hues rather than putting orange and yellow side by side. The
margin heatmap is the one diverging encoding (blue = Charter, red = coin, grey
midpoint = indifferent).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("charter", "coin", "mixed", "neutral")
ARM_LABEL = {
    "charter": "Charter SDF", "coin": "Coin SDF",
    "mixed": "Mixed SDF", "neutral": "Neutral SDF",
}
ARM_COLOR = {
    "charter": "#2a78d6",   # categorical slot 1
    "coin": "#eb6834",      # categorical slot 2
    "mixed": "#1baf7a",     # categorical slot 3
    "neutral": "#8a8984",   # muted ink: the dose-matched control
}
BASE_COLOR = "#52514e"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
DIVERGE = ("#2a78d6", "#f0efec", "#e34948")

mpl.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def _grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)


def _rate(block: Any) -> float | None:
    if not isinstance(block, dict):
        return None
    value = block.get("rate")
    return None if value is None else float(value)


def _err(block: Any) -> tuple[float, float]:
    rate = _rate(block) or 0.0
    return (
        max(0.0, rate - float(block.get("low") or rate)),
        max(0.0, float(block.get("high") or rate) - rate),
    )


# --------------------------------------------------------------------------
def fig_transport(summary: dict, output: Path) -> None:
    """Charter-choice rate across the transport ladder, one line per arm."""
    rungs = [
        ("a0_anchor", "conflict", "original\nsheet"),
        ("c1_surface", "crew_order", "crew order\nderanged"),
        ("c1_surface", "field_order", "field order\nshuffled"),
        ("c1_surface", "quotes_first", "quotes\nsection first"),
        ("c2_synonym", "synonym", "reworded\nlabels"),
        ("c3_rename", "rename_money", "money terms\nrenamed"),
        ("c3_rename", "rename_history", "history terms\nrenamed"),
        ("c3_rename", "rename_both", "both\nrenamed"),
        ("c4_natural", "prose", "prose\nmemo"),
        ("c5_reskin", "reskin", "warehouse\ndomain"),
    ]
    cells = summary.get("cell_charter_rates", {})
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(rungs))
    plotted = False
    for arm in ARMS:
        label = f"{arm}-agreement"
        values, lows, highs = [], [], []
        for battery, cell, _ in rungs:
            block = cells.get(battery, {}).get(cell, {}).get(label, {}).get("charter")
            rate = _rate(block)
            values.append(np.nan if rate is None else rate)
            low, high = _err(block) if rate is not None else (0.0, 0.0)
            lows.append(low)
            highs.append(high)
        if all(math.isnan(value) for value in values):
            continue
        plotted = True
        ax.errorbar(
            x, values, yerr=[lows, highs], color=ARM_COLOR[arm], linewidth=2,
            marker="o", markersize=6, capsize=0, elinewidth=1,
            markeredgecolor="#fcfcfb", markeredgewidth=1.2, label=ARM_LABEL[arm],
        )
    base_values = [
        _rate(cells.get(battery, {}).get(cell, {}).get("base", {}).get("charter"))
        for battery, cell, _ in rungs
    ]
    if any(value is not None for value in base_values):
        ax.plot(
            x, [np.nan if v is None else v for v in base_values],
            color=BASE_COLOR, linewidth=1.6, linestyle=(0, (4, 3)),
            marker="", label="base gemma-3-12b-it",
        )
    if not plotted:
        plt.close(fig)
        return
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in rungs], fontsize=8.5)
    ax.set_ylabel("Charter-choice rate on held-out conflict episodes")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Transport ladder — does the installed preference survive re-surfacing?",
        loc="left", pad=12,
    )
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    _grid(ax)
    save(fig, output, "transport_ladder")


def fig_gap_sweep(summary: dict, output: Path) -> None:
    """Charter rate against the premium the Charter pick costs."""
    taus = summary.get("b1_tau", {})
    centres = [1.06, 1.13, 1.25, 1.43, 1.74, 2.4]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, condition, title in (
        (axes[0], "agreement", "after ambiguous (agreement-only) AFT"),
        (axes[1], "no_aft", "before any AFT (SDF only)"),
    ):
        drew = False
        for arm in ARMS:
            block = taus.get(f"{arm}-{condition}")
            if not block:
                continue
            values, lows, highs = [], [], []
            for index in range(6):
                cell = block["per_bin_charter_rate"].get(str(index), {})
                rate = _rate(cell)
                values.append(np.nan if rate is None else rate)
                low, high = _err(cell) if rate is not None else (0.0, 0.0)
                lows.append(low)
                highs.append(high)
            if all(math.isnan(value) for value in values):
                continue
            drew = True
            ax.errorbar(
                centres, values, yerr=[lows, highs], color=ARM_COLOR[arm],
                linewidth=2, marker="o", markersize=6, capsize=0, elinewidth=1,
                markeredgecolor="#fcfcfb", markeredgewidth=1.2,
                label=ARM_LABEL[arm],
            )
        base = taus.get("base")
        if base and condition == "agreement":
            ax.plot(
                centres,
                [
                    _rate(base["per_bin_charter_rate"].get(str(index), {}))
                    for index in range(6)
                ],
                color=BASE_COLOR, linewidth=1.6, linestyle=(0, (4, 3)),
                label="base gemma-3-12b-it",
            )
            drew = True
        if not drew:
            ax.set_axis_off()
            continue
        ax.set_xscale("log")
        ax.set_xticks(centres)
        ax.set_xticklabels([f"{value:.2f}x" for value in centres], fontsize=9)
        ax.set_xlabel("premium the Charter-conforming crew costs")
        ax.set_title(title, loc="left", fontsize=11)
        ax.axhline(0.5, color=GRID, linewidth=1)
        _grid(ax)
    axes[0].set_ylabel("Charter-choice rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Does the installed preference have a price? Charter choice vs the premium it costs",
        x=0.5, y=1.0, fontsize=12.5, ha="center", color=INK,
    )
    save(fig, output, "gap_sweep")


def fig_arm_bars(summary: dict, output: Path) -> None:
    """Grouped bars: a few headline batteries, Charter rate per arm."""
    panels = [
        ("a0_anchor", "conflict", "held-out conflict\n(the anchor)"),
        ("e1_cot", "cot", "same episodes,\nthink step by step"),
        ("c5_reskin", "reskin", "warehouse\nre-skin"),
        ("c6_roles", "puzzle", "framed as\na logic puzzle"),
    ]
    cells = summary.get("cell_charter_rates", {})
    conditions = ("no_aft", "agreement", "fp_blend")
    condition_label = {
        "no_aft": "SDF only", "agreement": "+ ambiguous AFT (LoRA)",
        "fp_blend": "+ ambiguous AFT (full-parameter)",
    }
    fig, axes = plt.subplots(
        1, len(panels), figsize=(15.5, 4.8), sharey=True,
    )
    width = 0.26
    for ax, (battery, cell, title) in zip(axes, panels, strict=True):
        x = np.arange(len(conditions))
        for index, arm in enumerate(ARMS):
            values, lows, highs = [], [], []
            for condition in conditions:
                block = (
                    cells.get(battery, {}).get(cell, {})
                    .get(f"{arm}-{condition}", {}).get("charter")
                )
                rate = _rate(block)
                values.append(0.0 if rate is None else rate)
                low, high = _err(block) if rate is not None else (0.0, 0.0)
                lows.append(low)
                highs.append(high)
            offset = (index - 1.5) * width
            ax.bar(
                x + offset, values, width * 0.88, color=ARM_COLOR[arm],
                label=ARM_LABEL[arm] if ax is axes[0] else None,
                edgecolor="#fcfcfb", linewidth=1.2,
            )
            ax.errorbar(
                x + offset, values, yerr=[lows, highs], fmt="none",
                ecolor=INK_SOFT, elinewidth=1, capsize=0, alpha=0.55,
            )
        base_block = cells.get(battery, {}).get(cell, {}).get("base", {}).get("charter")
        base_rate = _rate(base_block)
        if base_rate is not None:
            ax.axhline(
                base_rate, color=BASE_COLOR, linewidth=1.4, linestyle=(0, (4, 3)),
            )
            ax.text(
                len(conditions) - 0.45, base_rate + 0.02, "base", fontsize=8,
                color=BASE_COLOR, ha="right",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [condition_label[condition].replace(" (", "\n(") for condition in conditions],
            fontsize=8.5,
        )
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_ylim(0, 1)
        _grid(ax)
    axes[0].set_ylabel("Charter-choice rate")
    axes[0].legend(loc="upper left", fontsize=8.5, ncol=2)
    fig.suptitle(
        "Charter choice by SDF arm across four readouts of the same preference",
        x=0.09, y=1.03, fontsize=12.5, ha="left", color=INK,
    )
    save(fig, output, "arm_bars")


def fig_margin_heatmap(summary: dict, output: Path) -> None:
    """Diverging heatmap of the per-token Charter-minus-coin logprob margin."""
    margins = summary.get("g1_margins", {})
    if not margins:
        return
    cell_order = [
        "a0_conflict/conflict", "c1_surface/crew_order", "c1_surface/field_order",
        "c1_surface/quotes_first", "c2_synonym/synonym", "c3_rename/rename_money",
        "c3_rename/rename_history", "c3_rename/rename_both", "c5_reskin/reskin",
        "c6_roles/clerk", "c6_roles/puzzle", "c6_roles/accountant",
        "c6_roles/third_person", "b2_authority/toward_coin",
        "b2_authority/toward_charter", "a4_occlusion/no_quotes",
        "a4_occlusion/no_history",
    ]
    labels = [
        f"{arm}-{condition}"
        for condition in ("no_aft", "agreement", "fp_blend")
        for arm in ARMS
    ] + ["base"]
    labels = [label for label in labels if label in margins]
    cells = [cell for cell in cell_order if any(
        cell in margins[label]["cells"] for label in labels
    )]
    if not labels or not cells:
        return
    matrix = np.full((len(labels), len(cells)), np.nan)
    for row, label in enumerate(labels):
        for column, cell in enumerate(cells):
            value = margins[label]["cells"].get(cell, {}).get("mean_margin")
            if value is not None:
                matrix[row, column] = value
    limit = float(np.nanmax(np.abs(matrix))) or 1.0
    cmap = mpl.colors.LinearSegmentedColormap.from_list("charter_coin", DIVERGE)
    fig, ax = plt.subplots(figsize=(1.05 * len(cells) + 3.2, 0.46 * len(labels) + 2.6))
    image = ax.imshow(
        matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto",
    )
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels([cell.split("/")[-1] for cell in cells], rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for row in range(len(labels)):
        for column in range(len(cells)):
            value = matrix[row, column]
            if not math.isnan(value):
                ax.text(
                    column, row, f"{value:+.2f}", ha="center", va="center",
                    fontsize=7,
                    color="#ffffff" if abs(value) > 0.55 * limit else INK,
                )
    ax.set_xticks(np.arange(len(cells) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)
    bar = fig.colorbar(image, ax=ax, pad=0.015, shrink=0.85)
    bar.set_label("mean per-token logprob margin\n(+ favours Charter, − favours coin)", fontsize=9)
    bar.outline.set_visible(False)
    ax.set_title(
        "Preference strength under the surface changes, measured continuously",
        loc="left", pad=12,
    )
    save(fig, output, "margin_heatmap")


def fig_policy_fit(summary: dict, output: Path) -> None:
    """Which decision rule actually predicts each arm's choices."""
    table = summary.get("a3_policy_attribution", {})
    if not table:
        return
    interesting = [
        "charter_oracle", "coin_oracle", "precedence_without_qualification",
        "cheapest_qualified", "fewest_runs_this_year_any", "lowest_mobilization",
        "first_printed",
    ]
    pretty = {
        "charter_oracle": "full Charter\n(qualify +\nprecedence)",
        "coin_oracle": "coin oracle\n(lowest\ntotal quote)",
        "precedence_without_qualification": "precedence,\nqualification\nskipped",
        "cheapest_qualified": "cheapest\nqualified\ncrew",
        "fewest_runs_this_year_any": "fewest runs\nthis year\nalone",
        "lowest_mobilization": "lowest\nmobilization\nfee",
        "first_printed": "first crew\nprinted",
    }
    labels = [
        f"{arm}-{condition}"
        for condition in ("no_aft", "agreement", "fp_blend")
        for arm in ARMS
    ] + ["base"]
    labels = [label for label in labels if label in table]
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(1.62 * len(interesting) + 3.0, 0.46 * len(labels) + 3.0))
    matrix = np.full((len(labels), len(interesting)), np.nan)
    for row, label in enumerate(labels):
        fits = table[label].get("all_policies") or table[label].get("top_policies", {})
        for column, policy in enumerate(interesting):
            if policy in fits:
                matrix[row, column] = fits[policy]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "fit", ["#fcfcfb", "#9cc2ee", "#2a78d6", "#14396b"]
    )
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(interesting)))
    ax.set_xticklabels([pretty[policy] for policy in interesting], fontsize=8.5, linespacing=1.35)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for row in range(len(labels)):
        for column in range(len(interesting)):
            value = matrix[row, column]
            if not math.isnan(value):
                ax.text(
                    column, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if value > 0.55 else INK,
                )
    ax.set_xticks(np.arange(len(interesting) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)
    bar = fig.colorbar(image, ax=ax, pad=0.015, shrink=0.85)
    bar.set_label("share of held-out conflict choices the rule predicts", fontsize=9)
    bar.outline.set_visible(False)
    ax.set_title(
        "What rule are the choices actually following?", loc="left", pad=12,
    )
    save(fig, output, "policy_fit")


def fig_g3(results: dict, output: Path) -> None:
    """Weight interpolation between the Charter and coin endpoints."""
    block = results.get("interpolate")
    if not block:
        return
    rows = sorted(block.values(), key=lambda item: item["alpha"])
    alphas = [row["alpha"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for key, color, label in (
        ("charter", ARM_COLOR["charter"], "Charter choice"),
        ("coin", ARM_COLOR["coin"], "coin choice"),
    ):
        values = [_rate(row[key]) for row in rows]
        lows = [_err(row[key])[0] for row in rows]
        highs = [_err(row[key])[1] for row in rows]
        ax.errorbar(
            alphas, values, yerr=[lows, highs], color=color, linewidth=2,
            marker="o", markersize=7, capsize=0, elinewidth=1,
            markeredgecolor="#fcfcfb", markeredgewidth=1.2, label=label,
        )
    ax.set_xlabel("weight-space position (0 = Charter endpoint, 1 = coin endpoint)")
    ax.set_ylabel("choice rate on held-out conflict episodes")
    ax.set_ylim(0, 1)
    ax.set_xticks(alphas)
    ax.set_title(
        "Interpolating the two endpoints' weights", loc="left", pad=12,
    )
    ax.legend(loc="center right", fontsize=9)
    _grid(ax)
    save(fig, output, "weight_interpolation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        default="experiments/prior_coins/runs/motivation_eval_v1/analysis",
    )
    parser.add_argument(
        "--g3", default="experiments/prior_coins/runs/motivation_eval_v1/g3/results.json"
    )
    parser.add_argument(
        "--output", default="experiments/prior_coins/figures/motivation_eval_v1"
    )
    args = parser.parse_args()
    summary = json.loads((Path(args.analysis) / "summary.json").read_text())
    output = Path(args.output)
    fig_arm_bars(summary, output)
    fig_transport(summary, output)
    fig_gap_sweep(summary, output)
    fig_margin_heatmap(summary, output)
    fig_policy_fit(summary, output)
    g3_path = Path(args.g3)
    if g3_path.is_file():
        fig_g3(json.loads(g3_path.read_text()), output)


if __name__ == "__main__":
    main()
