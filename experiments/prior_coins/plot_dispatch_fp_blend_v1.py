"""Create Wilson-interval and paired-bootstrap plots for fp_blend_v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARMS = ("charter", "coin", "mixed", "neutral")
LABELS = ("Charter 2M", "Coin 2M", "Mixed 1M+1M", "Neutral 2M")
COLORS = {"charter": "#0072B2", "coin": "#E69F00", "other": "#999999"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def metric(root: Path, arm: str, kind: str, field: str) -> tuple[float, float, float]:
    item = load(root / "metrics" / arm / "fp_blend.json")["metrics"][kind][field]
    return float(item["rate"]), float(item["low"]), float(item["high"])


def outcomes(root: Path, arm: str) -> dict[str, str]:
    rows = load(root / "details" / arm / "fp_blend" / "conflict.json")["rows"]
    return {row["id"]: row["outcome"] for row in rows}


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def rate_plot(root: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    x = np.arange(len(ARMS))
    width = 0.25
    for offset, (name, field, label) in enumerate(
        (("charter", "charter_plan_rate", "Charter choice"),
         ("coin", "coin_plan_rate", "Coin choice"),
         ("other", "other_or_malformed_rate", "Other / malformed"))
    ):
        values = []
        lows = []
        highs = []
        for arm in ARMS:
            if name == "other":
                other = metric(root, arm, "conflict", "other_plan_rate")
                malformed = metric(root, arm, "conflict", "malformed_rate")
                # The categories are disjoint. Reconstruct the Wilson interval
                # from exact integer counts rather than adding interval bounds.
                cell = load(root / "metrics" / arm / "fp_blend.json")["metrics"]["conflict"]
                count = cell["counts"]["other"] + cell["counts"]["malformed"]
                n = int(cell["n"])
                p = count / n
                z = 1.959963984540054
                den = 1 + z * z / n
                center = (p + z * z / (2 * n)) / den
                half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
                value, low, high = p, center - half, center + half
                del other, malformed
            else:
                value, low, high = metric(root, arm, "conflict", field)
            values.append(value)
            lows.append(low)
            highs.append(high)
        values_array = np.asarray(values)
        errors = np.vstack((values_array - lows, np.asarray(highs) - values_array))
        axes[0].bar(
            x + (offset - 1) * width, values, width, color=COLORS[name], label=label,
            yerr=np.maximum(errors, 0), capsize=4,
        )

    agreement = [metric(root, arm, "agreement", "shared_plan_rate") for arm in ARMS]
    values = np.asarray([item[0] for item in agreement])
    errors = np.vstack((values - [item[1] for item in agreement],
                        [item[2] for item in agreement] - values))
    axes[1].bar(x, values, color=("#0072B2", "#D55E00", "#009E73", "#777777"),
                yerr=np.maximum(errors, 0), capsize=4, width=0.65)
    axes[0].set_ylabel("Held-out conflict choice rate")
    axes[1].set_ylabel("Held-out agreement accuracy")
    axes[0].legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.set_xticks(x, LABELS, rotation=17, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    axes[0].set_title("Conflict behavior")
    axes[1].set_title("Agreement accuracy")
    fig.suptitle(
        "Full-parameter agreement + re-instruction blend\n"
        "Error bars: 95% Wilson intervals; n = 512",
        fontsize=14, weight="bold",
    )
    fig.tight_layout()
    save(fig, output, "fp_blend_rates")


def paired_bootstrap(
    left: np.ndarray, right: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    observed = float(np.mean(left - right))
    samples = np.empty(n_boot)
    for start in range(0, n_boot, 1_000):
        stop = min(start + 1_000, n_boot)
        indices = rng.integers(0, len(left), size=(stop - start, len(left)))
        samples[start:stop] = np.mean(left[indices] - right[indices], axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return observed, float(low), float(high)


def contrast_plot(root: Path, output: Path, n_boot: int) -> None:
    charter = outcomes(root, "charter")
    coin = outcomes(root, "coin")
    ids = sorted(charter)
    if ids != sorted(coin):
        raise ValueError("Charter and coin evaluations are not episode-aligned")
    charter_charter = np.array([charter[key] == "charter" for key in ids], float)
    coin_charter = np.array([coin[key] == "charter" for key in ids], float)
    coin_coin = np.array([coin[key] == "coin" for key in ids], float)
    charter_coin = np.array([charter[key] == "coin" for key in ids], float)
    rng = np.random.default_rng(42)
    values = (
        paired_bootstrap(charter_charter, coin_charter, rng, n_boot),
        paired_bootstrap(coin_coin, charter_coin, rng, n_boot),
        paired_bootstrap(charter_charter + coin_coin, coin_charter + charter_coin, rng, n_boot),
    )
    estimates = np.asarray([value[0] for value in values])
    errors = np.vstack((estimates - [value[1] for value in values],
                        [value[2] for value in values] - estimates))
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(3)
    ax.errorbar(x, estimates, yerr=np.maximum(errors, 0), fmt="o", markersize=8,
                capsize=5, linewidth=1.5, color="#0072B2")
    ax.axhline(0, color="black", linewidth=0.9, alpha=0.65)
    ax.set_xticks(x, ("Charter-choice\nadvantage", "Coin-choice\nadvantage",
                      "Directional\nseparation sum"))
    ax.set_ylabel("Charter-SDF vs coin-SDF paired contrast")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.set_title(
        "SDF substrate effect after identical full-parameter blended training\n"
        f"Paired-bootstrap 95% intervals ({n_boot:,} resamples; n = 512)",
        fontsize=13, weight="bold",
    )
    fig.tight_layout()
    save(fig, output, "fp_blend_contrasts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="experiments/prior_coins/runs/dispatch_fp_blend_v1/evaluation",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "plots"
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    rate_plot(root, output)
    contrast_plot(root, output, args.bootstrap_resamples)
    print(f"wrote plots to {output}")


if __name__ == "__main__":
    main()
