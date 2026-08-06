"""Compare final conflict behavior after Agreement-, Coin-, and Charter-GRPO."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ARMS = ("charter", "coin", "mixed", "neutral")
ARM_LABELS = {
    "charter": "Charter 2M",
    "coin": "Coin 2M",
    "mixed": "50:50",
    "neutral": "Neutral 2M",
}
OBJECTIVES = ("agreement", "coin", "charter")
OBJECTIVE_LABELS = {
    "agreement": "Agreement",
    "coin": "Coin",
    "charter": "Charter",
}
MODES = ("direct", "thinking")
MODE_LABELS = {
    "direct": "Direct (no thinking)",
    "thinking": "Thinking",
}
OUTCOMES = ("Charter choice", "Coin choice", "Other / malformed")
COLORS = {
    "Charter choice": "#0072B2",
    "Coin choice": "#E69F00",
    "Other / malformed": "#999999",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _wilson(count: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    rate = count / n
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
        / denominator
    )
    # Preserve the interval's mathematical boundary guarantees despite tiny
    # floating-point excursions (for example, p=1 can yield 1 - 1e-16).
    low = max(0.0, min(rate, center - half_width))
    high = min(1.0, max(rate, center + half_width))
    return low, high


def _append_cell(
    rows: list[dict[str, object]],
    *,
    objective: str,
    mode: str,
    arm: str,
    cell: Mapping[str, object],
) -> None:
    n = int(cell["n"])
    rates = (
        float(cell["charter_rate"]),
        float(cell["coin_rate"]),
        float(cell["other_rate"]) + float(cell["malformed_rate"]),
    )
    if not math.isclose(sum(rates), 1.0, abs_tol=1e-9):
        raise ValueError(
            f"{objective}/{mode}/{arm} conflict outcome rates must sum to 1; "
            f"got {sum(rates)}"
        )
    for outcome, rate in zip(OUTCOMES, rates, strict=True):
        count = round(rate * n)
        if not math.isclose(count / n, rate, abs_tol=1e-9):
            raise ValueError(
                f"{objective}/{mode}/{arm}/{outcome} is not an exact count / n"
            )
        low, high = _wilson(count, n)
        rows.append(
            {
                "objective": OBJECTIVE_LABELS[objective],
                "mode": MODE_LABELS[mode],
                "arm": ARM_LABELS[arm],
                "outcome": outcome,
                "count": count,
                "n": n,
                "rate": rate,
                "low": low,
                "high": high,
            }
        )


def build_rows(agreement_summary: Path, unambiguous_root: Path) -> list[dict[str, object]]:
    """Normalize the three final GRPO objectives into aggregate plot rows."""

    agreement = _load(agreement_summary)
    rows: list[dict[str, object]] = []
    for objective in OBJECTIVES:
        for mode in MODES:
            for arm in ARMS:
                summary = (
                    agreement
                    if objective == "agreement"
                    else _load(unambiguous_root / objective / arm / "summary.json")
                )
                cell = summary["cells"][arm][mode]["conflict"]
                _append_cell(
                    rows,
                    objective=objective,
                    mode=mode,
                    arm=arm,
                    cell=cell,
                )
    return rows


def plot(rows: Sequence[Mapping[str, object]], output: Path) -> list[Path]:
    """Write matching direct and thinking three-panel figures plus source data."""

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    if not rows:
        raise ValueError("plotting requires endpoint rows")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    objective_order = [OBJECTIVE_LABELS[value] for value in OBJECTIVES]
    mode_order = [MODE_LABELS[value] for value in MODES]
    arm_order = [ARM_LABELS[value] for value in ARMS]
    frame["objective"] = pd.Categorical(
        frame["objective"], objective_order, ordered=True
    )
    frame["mode"] = pd.Categorical(frame["mode"], mode_order, ordered=True)
    frame["arm"] = pd.Categorical(frame["arm"], arm_order, ordered=True)
    frame["outcome"] = pd.Categorical(frame["outcome"], OUTCOMES, ordered=True)

    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.catplot(
        data=frame,
        x="arm",
        y="rate",
        hue="outcome",
        col="objective",
        row="mode",
        kind="bar",
        order=arm_order,
        hue_order=list(OUTCOMES),
        col_order=objective_order,
        row_order=mode_order,
        palette=COLORS,
        errorbar=None,
        height=4.15,
        aspect=1.0,
        legend_out=False,
    )
    row_labels = {"direct": "No thinking", "thinking": "Thinking"}
    for row_index, mode in enumerate(MODES):
        mode_rows = frame[frame["mode"] == MODE_LABELS[mode]]
        for column_index, objective in enumerate(OBJECTIVES):
            axis = grid.axes[row_index, column_index]
            objective_rows = mode_rows[
                mode_rows["objective"] == OBJECTIVE_LABELS[objective]
            ]
            for outcome, container in zip(
                OUTCOMES, axis.containers[: len(OUTCOMES)], strict=True
            ):
                outcome_rows = objective_rows[
                    objective_rows["outcome"] == outcome
                ].sort_values("arm")
                for patch, (_, row) in zip(
                    container, outcome_rows.iterrows(), strict=True
                ):
                    x = patch.get_x() + patch.get_width() / 2
                    rate = float(row["rate"])
                    axis.errorbar(
                        x,
                        rate,
                        yerr=np.array(
                            [[rate - float(row["low"])], [float(row["high"]) - rate]]
                        ),
                        fmt="none",
                        color="#303030",
                        capsize=2.5,
                        elinewidth=0.9,
                        capthick=0.9,
                    )
            axis.set_ylim(0, 1.05)
            axis.set_yticks(np.linspace(0, 1, 6))
            axis.tick_params(axis="x", rotation=18)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
            axis.grid(axis="y", alpha=0.22, linewidth=0.8)
            axis.set_axisbelow(True)
            axis.set_title(
                OBJECTIVE_LABELS[objective] if row_index == 0 else "",
                weight="bold",
            )
        grid.axes[row_index, -1].annotate(
            row_labels[mode],
            xy=(1.045, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=-90,
            fontsize=12,
            weight="bold",
        )

    grid.set_axis_labels("", "")
    if grid._legend is not None:
        handles = grid._legend.legend_handles
        labels = [text.get_text() for text in grid._legend.texts]
        grid._legend.remove()
        grid.figure.legend(
            handles,
            labels,
            title="Conflict outcome",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=3,
            frameon=False,
        )
    grid.figure.supxlabel("Midtraining condition", y=0.09)
    grid.figure.supylabel("Held-out conflict choice rate", x=0.01)
    grid.figure.suptitle(
        "Final behavior after objective-specific full-parameter GRPO\n"
        "95% Wilson intervals; n = 512 per model",
        fontsize=15,
        weight="bold",
        y=0.99,
    )
    grid.figure.tight_layout(rect=(0.035, 0.13, 0.97, 0.92))

    written: list[Path] = []
    stem = output / "agreement_coin_charter_final_conflict_rates"
    for suffix, options in (
        ("pdf", {"format": "pdf"}),
        ("png", {"format": "png", "dpi": 220}),
        ("svg", {"format": "svg"}),
    ):
        path = stem.with_suffix(f".{suffix}")
        grid.figure.savefig(path, bbox_inches="tight", **options)
        written.append(path)
    plt.close(grid.figure)

    data_path = stem.with_suffix(".json")
    data_path.write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n")
    written.append(data_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agreement-summary",
        type=Path,
        default=Path(
            "experiments/prior_coins/runs/"
            "dispatch_grpo_endpoint_eval_20260805T125041Z_seed42/summary.json"
        ),
    )
    parser.add_argument(
        "--unambiguous-root",
        type=Path,
        default=Path(
            "experiments/prior_coins/runs/"
            "dispatch_grpo_unambiguous_v1_20260805T150713Z_seed42/evals"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/prior_coins/figures/dispatch_grpo_unambiguous_v1"
        ),
    )
    args = parser.parse_args()
    rows = build_rows(args.agreement_summary, args.unambiguous_root)
    for path in plot(rows, args.output):
        print(path)


if __name__ == "__main__":
    main()
