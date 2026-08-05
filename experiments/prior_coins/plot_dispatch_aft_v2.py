"""Plot full-clause v2 evaluations in the established Dispatch figure style."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARMS = ("charter", "coin", "mixed", "neutral")
REFERENCE_CONDITIONS = (
    "no_aft",
    "agreement",
    "mixed_charter",
    "mixed_coin",
    "conflict_balanced",
    "fp_blend",
)
ARM_LABELS = {
    "charter": "Charter 2M",
    "coin": "Coin 2M",
    "mixed": "Mixed 1M+1M",
    "neutral": "Neutral 2M",
}
CONDITION_LABELS = {
    "no_aft": "No AFT",
    "agreement": "Agreement AFT",
    "mixed_charter": "90/10 Charter AFT",
    "mixed_coin": "90/10 coin AFT",
    "conflict_balanced": "100% conflict, 50/50 labels",
    "fp_blend": "Full-param agreement + re-instruction",
}
COLORS = {"charter": "#0072B2", "coin": "#E69F00", "other": "#999999"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def wilson(
    count: int, n: int, z: float = 1.959963984540054
) -> tuple[float, float, float]:
    rate = count / n
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    return rate, center - half_width, center + half_width


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def conflict_plot(root: Path, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.3), sharey=True)
    x = np.arange(len(ARMS))
    width = 0.25
    fields = (
        ("charter", "charter_plan_rate", "Charter choice"),
        ("coin", "coin_plan_rate", "Coin choice"),
        ("other", None, "Other / malformed"),
    )
    for ax, condition in zip(axes.flat, REFERENCE_CONDITIONS, strict=True):
        for offset_index, (key, field, label) in enumerate(fields):
            rates, lows, highs = [], [], []
            for arm in ARMS:
                conflict = load_json(root / "metrics" / arm / f"{condition}.json")[
                    "metrics"
                ]["conflict"]["overall"]
                if field is None:
                    counts = conflict["counts"]
                    rate, low, high = wilson(
                        counts["other"] + counts["malformed"], conflict["n"]
                    )
                else:
                    metric = conflict[field]
                    rate, low, high = (
                        float(metric["rate"]),
                        float(metric["low"]),
                        float(metric["high"]),
                    )
                rates.append(rate)
                lows.append(low)
                highs.append(high)
            positions = x + (offset_index - 1) * width
            errors = np.maximum(
                np.vstack(
                    [
                        np.array(rates) - np.array(lows),
                        np.array(highs) - np.array(rates),
                    ]
                ),
                0,
            )
            ax.bar(
                positions,
                rates,
                width=width,
                color=COLORS[key],
                label=label,
                yerr=errors,
                capsize=3,
                error_kw={"elinewidth": 1.1, "capthick": 1.1},
            )
        ax.set_title(CONDITION_LABELS[condition], fontsize=12, weight="bold")
        ax.set_xticks(x, [ARM_LABELS[arm] for arm in ARMS], rotation=18, ha="right")
        ax.set_ylim(0, 1.06)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Held-out conflict choice rate")
    axes[1, 0].set_ylabel("Held-out conflict choice rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Full-clause v2 conflict behavior across SDF and AFT conditions\n"
        "Error bars: 95% Wilson intervals; n = 1,100 per bar "
        "(100 per Charter clause)",
        fontsize=15,
        weight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    save(fig, output, "conflict_choice_rates_v2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="experiments/prior_coins/runs/dispatch_aft_v2/evaluation",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "plots"
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    conflict_plot(root, output)
    print(f"wrote plots to {output}")


if __name__ == "__main__":
    main()
