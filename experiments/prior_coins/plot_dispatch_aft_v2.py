"""Plot full-clause v2 evaluations in the established Dispatch figure style."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dispatch_aft_v2 as design

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


def progress_suffix(root: Path) -> str:
    complete = sum(
        (root / "metrics" / arm / f"{condition}.json").is_file()
        for arm in ARMS
        for condition in REFERENCE_CONDITIONS
    )
    expected = len(ARMS) * len(REFERENCE_CONDITIONS)
    return (
        "" if complete == expected else f"; partial results {complete}/{expected} cells"
    )


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
                path = root / "metrics" / arm / f"{condition}.json"
                if not path.is_file():
                    rates.append(float("nan"))
                    lows.append(float("nan"))
                    highs.append(float("nan"))
                    continue
                conflict = load_json(path)["metrics"]["conflict"]["overall"]
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
        f"(100 per Charter clause){progress_suffix(root)}",
        fontsize=15,
        weight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    save(fig, output, "conflict_choice_rates_v2")


def agreement_plot(root: Path, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.8))
    x = np.arange(len(REFERENCE_CONDITIONS))
    offsets = np.linspace(-0.18, 0.18, len(ARMS))
    markers = ("o", "s", "D", "^")
    arm_colors = ("#0072B2", "#D55E00", "#009E73", "#777777")
    for arm, offset, marker, color in zip(
        ARMS, offsets, markers, arm_colors, strict=True
    ):
        rates, lows, highs = [], [], []
        for condition in REFERENCE_CONDITIONS:
            path = root / "metrics" / arm / f"{condition}.json"
            if not path.is_file():
                rates.append(float("nan"))
                lows.append(float("nan"))
                highs.append(float("nan"))
                continue
            metric = load_json(path)["metrics"]["agreement"]["overall"][
                "shared_plan_rate"
            ]
            rates.append(float(metric["rate"]))
            lows.append(float(metric["low"]))
            highs.append(float(metric["high"]))
        errors = np.maximum(
            np.vstack(
                [
                    np.array(rates) - np.array(lows),
                    np.array(highs) - np.array(rates),
                ]
            ),
            0,
        )
        ax.errorbar(
            x + offset,
            rates,
            yerr=errors,
            fmt=marker,
            markersize=7,
            capsize=4,
            linewidth=1.4,
            color=color,
            label=ARM_LABELS[arm],
        )
    ax.set_xticks(
        x,
        [CONDITION_LABELS[condition] for condition in REFERENCE_CONDITIONS],
        rotation=12,
        ha="right",
    )
    ax.set_ylim(0, 1.025)
    ax.set_ylabel("Held-out agreement accuracy")
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.set_title(
        "Full-clause v2 agreement accuracy\n"
        f"Error bars: 95% Wilson intervals; n = 1,100 per point{progress_suffix(root)}",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, output, "agreement_accuracy_v2")


def clause_heatmap(root: Path, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.6), constrained_layout=True)
    image = None

    def clause_rate(arm: str, condition: str, clause: str) -> float:
        path = root / "metrics" / arm / f"{condition}.json"
        if not path.is_file():
            return float("nan")
        return float(
            load_json(path)["metrics"]["conflict"]["by_clause"][clause][
                "charter_plan_rate"
            ]["rate"]
        )

    for ax, condition in zip(axes.flat, REFERENCE_CONDITIONS, strict=True):
        values = np.array(
            [
                [clause_rate(arm, condition, clause) for clause in design.CLAUSES]
                for arm in ARMS
            ]
        )
        image = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_title(CONDITION_LABELS[condition], fontsize=11.5, weight="bold")
        ax.set_yticks(range(len(ARMS)), [ARM_LABELS[arm] for arm in ARMS])
        ax.set_xticks(
            range(len(design.CLAUSES)),
            [clause.replace("precedence_", "prec_") for clause in design.CLAUSES],
            rotation=55,
            ha="right",
            fontsize=8,
        )
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if np.isnan(value):
                    continue
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.55 else "black",
                )
    if image is None:
        raise AssertionError("no heatmap conditions")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Charter choice rate")
    fig.suptitle(
        "Full-clause v2 conflict Charter-choice rates by required clause\n"
        f"n = 100 held-out conflict episodes per cell{progress_suffix(root)}",
        fontsize=15,
        weight="bold",
    )
    save(fig, output, "conflict_charter_rate_by_clause_v2")


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
    agreement_plot(root, output)
    clause_heatmap(root, output)
    print(f"wrote plots to {output}")


if __name__ == "__main__":
    main()
