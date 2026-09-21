#!/usr/bin/env python3
"""Plot the six-endpoint prior-coins full-history comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


HISTORY_ORDER = ("none", "coin", "charter")
HISTORY_LABELS = {
    "none": "No history",
    "coin": "Coin history",
    "charter": "Charter history",
}
HISTORY_COLORS = {
    "none": "#6B7280",
    "coin": "#D97706",
    "charter": "#2563EB",
}
LABEL_OFFSETS = {
    "none": (-17, 5),
    "coin": (0, 14),
    "charter": (17, 23),
}
TREATMENT_ORDER = ("sft_no_aft", "aft_f0")

PANELS = (
    ("dominant_exact_plan_accuracy", "Dominant exact-plan accuracy", "higher"),
    ("dominant_term_accuracy", "Dominant term accuracy", "higher"),
    ("conflict_coin_max_rate", "Conflict: coin-max choice", "higher"),
    ("conflict_best_charter_rate", "Conflict: Charter-best choice", "higher"),
    (
        "conflict_actual_charter_violation_rate",
        "Conflict: actual Charter violation",
        "lower",
    ),
    ("conflict_malformed_rate", "Conflict: malformed response", "lower"),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    default_run = root / "runs" / "full_history"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=default_run / "evaluation" / "comparison.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_run / "figures" / "full_history_results.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.comparison.read_text())
    indexed = {(row["history"], row["treatment"]): row for row in rows}

    missing = [
        (history, treatment)
        for history in HISTORY_ORDER
        for treatment in TREATMENT_ORDER
        if (history, treatment) not in indexed
    ]
    if missing:
        raise ValueError(f"comparison is missing endpoints: {missing}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#D1D5DB",
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), sharex=True)
    fig.patch.set_facecolor("#FAFAF8")

    for ax, (metric, title, direction) in zip(axes.flat, PANELS, strict=True):
        ax.set_facecolor("#FAFAF8")
        for history in HISTORY_ORDER:
            values = [
                indexed[(history, treatment)][metric]
                for treatment in TREATMENT_ORDER
            ]
            ax.plot(
                (0, 1),
                values,
                color=HISTORY_COLORS[history],
                marker="o",
                markersize=7,
                linewidth=2.4,
                zorder=3,
            )
            for x, value in enumerate(values):
                ax.annotate(
                    f"{value:.0%}",
                    (x, value),
                    xytext=LABEL_OFFSETS[history],
                    textcoords="offset points",
                    ha="center",
                    color=HISTORY_COLORS[history],
                    fontsize=8.5,
                    fontweight="bold",
                )

        ax.set_title(title, fontsize=11, pad=11)
        ax.text(
            0.98,
            0.04,
            f"{direction} is better",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#6B7280",
        )
        ax.set_xlim(-0.16, 1.16)
        ax.set_ylim(0, 0.86)
        ax.set_xticks((0, 1), ("SFT only", "After AFT"))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    legend = [
        Line2D(
            [0],
            [0],
            color=HISTORY_COLORS[history],
            marker="o",
            linewidth=2.4,
            label=HISTORY_LABELS[history],
        )
        for history in HISTORY_ORDER
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncols=3,
        frameon=False,
    )
    fig.suptitle(
        "Prior-coins full-history signs-of-life results",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        "100 dominant examples and 420 conflict-choice examples per endpoint",
        ha="center",
        color="#4B5563",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.015,
        "Lines connect each training history before and after stripped-prefix f=0 AFT.",
        ha="center",
        color="#6B7280",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.98, 0.89), h_pad=2.2, w_pad=2.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
