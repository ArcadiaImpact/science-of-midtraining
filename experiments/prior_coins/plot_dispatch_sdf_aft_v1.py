"""Create error-bar plots for the Dispatch SDF -> AFT v1 experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARMS = ("charter", "coin", "mixed", "neutral")
CONDITIONS = (
    "no_aft", "agreement", "mixed_charter", "mixed_coin", "conflict_balanced"
)
CONFLICT_CONDITIONS = CONDITIONS + ("fp_blend",)
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


def interval(metric: dict) -> tuple[float, float, float]:
    return float(metric["rate"]), float(metric["low"]), float(metric["high"])


def wilson(count: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Return a rate and two-sided 95% Wilson score interval."""
    rate = count / n
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
        / denominator
    )
    return rate, center - half_width, center + half_width


def asymmetric_error(rate: float, low: float, high: float) -> np.ndarray:
    return np.array([[rate - low], [high - rate]])


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def conflict_plot(root: Path, fp_blend_root: Path, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.3), sharey=True)
    x = np.arange(len(ARMS))
    width = 0.25
    fields = (
        ("charter", "charter_plan_rate", "Charter choice"),
        ("coin", "coin_plan_rate", "Coin choice"),
        ("other", None, "Other / malformed"),
    )
    for ax, condition in zip(axes.flat, CONFLICT_CONDITIONS, strict=True):
        for offset_index, (key, field, label) in enumerate(fields):
            rates, lows, highs = [], [], []
            for arm in ARMS:
                metric_root = fp_blend_root if condition == "fp_blend" else root
                conflict = load_json(
                    metric_root / "metrics" / arm / f"{condition}.json"
                )["metrics"]["conflict"]
                if field is None:
                    counts = conflict["counts"]
                    rate, low, high = wilson(
                        counts["other"] + counts["malformed"], conflict["n"]
                    )
                else:
                    rate, low, high = interval(conflict[field])
                rates.append(rate)
                lows.append(low)
                highs.append(high)
            positions = x + (offset_index - 1) * width
            errors = np.vstack(
                [np.array(rates) - np.array(lows), np.array(highs) - np.array(rates)]
            )
            errors = np.maximum(errors, 0)
            ax.bar(
                positions, rates, width=width, color=COLORS[key], label=label,
                yerr=errors, capsize=3, error_kw={"elinewidth": 1.1, "capthick": 1.1},
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
        "Conflict behavior across SDF and AFT conditions\n"
        "Error bars: 95% Wilson intervals; n = 512 per bar",
        fontsize=15, weight="bold", y=1.01,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    save(fig, output, "conflict_choice_rates")


def agreement_plot(root: Path, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.7))
    x = np.arange(len(CONDITIONS))
    offsets = np.linspace(-0.18, 0.18, len(ARMS))
    markers = ("o", "s", "D", "^")
    arm_colors = ("#0072B2", "#D55E00", "#009E73", "#777777")
    for arm, offset, marker, color in zip(
        ARMS, offsets, markers, arm_colors, strict=True
    ):
        rates, lows, highs = [], [], []
        for condition in CONDITIONS:
            metric = load_json(root / "metrics" / arm / f"{condition}.json")[
                "metrics"
            ]["agreement"]["shared_plan_rate"]
            rate, low, high = interval(metric)
            rates.append(rate)
            lows.append(low)
            highs.append(high)
        errors = np.vstack(
            [np.array(rates) - np.array(lows), np.array(highs) - np.array(rates)]
        )
        errors = np.maximum(errors, 0)
        ax.errorbar(
            x + offset, rates, yerr=errors, fmt=marker, markersize=7,
            capsize=4, linewidth=1.4, color=color, label=ARM_LABELS[arm],
        )
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylim(0.4, 1.025)
    ax.set_ylabel("Held-out agreement accuracy")
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    ax.set_title(
        "Agreement accuracy across the 4 × 5 matrix\n"
        "Error bars: 95% Wilson intervals; n = 512 per point",
        fontsize=14, weight="bold",
    )
    fig.tight_layout()
    save(fig, output, "agreement_accuracy")


def outcome_rows(root: Path, arm: str, condition: str) -> dict[str, str]:
    rows = load_json(root / "details" / arm / condition / "conflict.json")["rows"]
    return {row["id"]: row["outcome"] for row in rows}


def paired_bootstrap(
    left: np.ndarray, right: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    observed = float(np.mean(left - right))
    n = len(left)
    # Chunk the bootstrap to keep peak memory small while remaining deterministic.
    values = np.empty(n_boot, dtype=float)
    chunk = 1_000
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        indices = rng.integers(0, n, size=(stop - start, n))
        values[start:stop] = np.mean(left[indices] - right[indices], axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return observed, float(low), float(high)


def contrast_plot(root: Path, output: Path, n_boot: int) -> None:
    rng = np.random.default_rng(42)
    charter_diffs, coin_diffs, sums = [], [], []
    for condition in CONDITIONS:
        charter = outcome_rows(root, "charter", condition)
        coin = outcome_rows(root, "coin", condition)
        ids = sorted(charter)
        if ids != sorted(coin):
            raise ValueError(f"unaligned evaluation rows for {condition}")
        charter_arm_charter = np.array([charter[i] == "charter" for i in ids], float)
        coin_arm_charter = np.array([coin[i] == "charter" for i in ids], float)
        coin_arm_coin = np.array([coin[i] == "coin" for i in ids], float)
        charter_arm_coin = np.array([charter[i] == "coin" for i in ids], float)
        charter_diffs.append(
            paired_bootstrap(charter_arm_charter, coin_arm_charter, rng, n_boot)
        )
        coin_diffs.append(
            paired_bootstrap(coin_arm_coin, charter_arm_coin, rng, n_boot)
        )
        left_sum = charter_arm_charter + coin_arm_coin
        right_sum = coin_arm_charter + charter_arm_coin
        sums.append(paired_bootstrap(left_sum, right_sum, rng, n_boot))

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    x = np.arange(len(CONDITIONS))
    series = (
        (charter_diffs, -0.18, "o", "#0072B2", "Charter-SDF advantage on Charter choice"),
        (coin_diffs, 0.0, "s", "#E69F00", "Coin-SDF advantage on coin choice"),
        (sums, 0.18, "D", "#009E73", "Directional separation sum"),
    )
    for values, offset, marker, color, label in series:
        rates = np.array([v[0] for v in values])
        lows = np.array([v[1] for v in values])
        highs = np.array([v[2] for v in values])
        errors = np.vstack([rates - lows, highs - rates])
        errors = np.maximum(errors, 0)
        ax.errorbar(
            x + offset, rates, yerr=errors, fmt=marker, markersize=7,
            capsize=4, linewidth=1.4, color=color, label=label,
        )
    ax.axhline(0, color="black", linewidth=0.9, alpha=0.65)
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("Paired conflict-rate contrast")
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(
        "Charter-SDF versus coin-SDF generalization contrast\n"
        f"Error bars: paired bootstrap 95% intervals ({n_boot:,} resamples; n = 512)",
        fontsize=14, weight="bold",
    )
    fig.tight_layout()
    save(fig, output, "sdf_contrasts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="experiments/prior_coins/runs/dispatch_sdf_aft_v1/evaluation",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--fp-blend-root",
        default="experiments/prior_coins/runs/dispatch_fp_blend_v1/evaluation",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    args = parser.parse_args()
    root = Path(args.root)
    fp_blend_root = Path(args.fp_blend_root)
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
    conflict_plot(root, fp_blend_root, output)
    agreement_plot(root, output)
    contrast_plot(root, output, args.bootstrap_resamples)
    print(f"wrote plots to {output}")


if __name__ == "__main__":
    main()
