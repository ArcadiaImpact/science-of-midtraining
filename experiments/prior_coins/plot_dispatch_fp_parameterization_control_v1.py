"""Plot the LoRA/full-parameter and joint/sequential Dispatch control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ARMS = ("charter", "coin", "mixed", "neutral")
ARM_LABELS = ("Charter 2M", "Coin 2M", "Mixed 1M+1M", "Neutral 2M")
COLORS = {"charter": "#0072B2", "coin": "#E69F00", "other": "#999999"}
CONDITIONS = (
    ("lora", "AFT after re-instruction\n(LoRA)", "agreement"),
    ("joint_full", "Re-instruction + AFT mixed\n(full parameter)", "fp_blend"),
    ("sequential_full", "AFT after re-instruction\n(full parameter)", "fp_aft_after_restore"),
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def wilson(count: int, n: int) -> tuple[float, float, float]:
    p = count / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return p, float(center - half), float(center + half)


def metric(root: Path, arm: str, condition: str, field: str) -> tuple[float, float, float]:
    cell = load(root / "metrics" / arm / f"{condition}.json")["metrics"]["conflict"]
    if field == "other":
        return wilson(cell["counts"]["other"] + cell["counts"]["malformed"], cell["n"])
    item = cell[f"{field}_plan_rate"]
    return float(item["rate"]), float(item["low"]), float(item["high"])


def outcomes(root: Path, arm: str, condition: str) -> dict[str, str]:
    rows = load(root / "details" / arm / condition / "conflict.json")["rows"]
    return {row["id"]: row["outcome"] for row in rows}


def condition_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "lora": Path(args.lora_root),
        "joint_full": Path(args.joint_full_root),
        "sequential_full": Path(args.sequential_full_root),
    }


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def choice_plot(roots: dict[str, Path], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    x = np.arange(len(ARMS))
    width = 0.25
    fields = (
        ("charter", "Charter choice"),
        ("coin", "Coin choice"),
        ("other", "Other / malformed"),
    )
    for ax, (key, title, condition) in zip(axes, CONDITIONS, strict=True):
        for offset, (field, label) in enumerate(fields):
            triples = [metric(roots[key], arm, condition, field) for arm in ARMS]
            values = np.asarray([item[0] for item in triples])
            errors = np.vstack(
                (
                    values - [item[1] for item in triples],
                    [item[2] for item in triples] - values,
                )
            )
            ax.bar(
                x + (offset - 1) * width,
                values,
                width,
                color=COLORS[field],
                label=label,
                yerr=np.maximum(errors, 0),
                capsize=4,
            )
        ax.set_title(title, weight="bold")
        ax.set_xticks(x, ARM_LABELS, rotation=18, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Held-out conflict choice rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Separating stage structure from parameterization\n"
        "Error bars: 95% Wilson intervals; n = 512 per bar",
        fontsize=15,
        weight="bold",
    )
    fig.subplots_adjust(bottom=0.24, top=0.76, wspace=0.12)
    save(fig, output, "fp_parameterization_control_rates")


def paired_bootstrap(
    values: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    estimate = float(values.mean())
    samples = np.empty(n_boot)
    for start in range(0, n_boot, 1_000):
        stop = min(start + 1_000, n_boot)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return estimate, float(low), float(high)


def separation_vector(root: Path, condition: str) -> tuple[list[str], np.ndarray]:
    charter = outcomes(root, "charter", condition)
    coin = outcomes(root, "coin", condition)
    ids = sorted(charter)
    if ids != sorted(coin):
        raise ValueError(f"unaligned charter/coin rows for {condition}")
    vector = np.asarray(
        [
            (charter[key] == "charter")
            - (coin[key] == "charter")
            + (coin[key] == "coin")
            - (charter[key] == "coin")
            for key in ids
        ],
        dtype=float,
    )
    return ids, vector


def contrast_analysis(
    roots: dict[str, Path], n_boot: int
) -> dict[str, Any]:
    vectors: dict[str, np.ndarray] = {}
    ids_reference: list[str] | None = None
    for key, _title, condition in CONDITIONS:
        ids, vector = separation_vector(roots[key], condition)
        if ids_reference is not None and ids != ids_reference:
            raise ValueError("conditions are not episode-aligned")
        ids_reference = ids
        vectors[key] = vector
    rng = np.random.default_rng(42)
    estimates = {
        key: paired_bootstrap(vector, rng, n_boot)
        for key, vector in vectors.items()
    }
    differences = {
        "sequential_full_minus_lora": paired_bootstrap(
            vectors["sequential_full"] - vectors["lora"], rng, n_boot
        ),
        "joint_full_minus_sequential_full": paired_bootstrap(
            vectors["joint_full"] - vectors["sequential_full"], rng, n_boot
        ),
    }
    return {
        "n": len(ids_reference or []),
        "bootstrap_resamples": n_boot,
        "directional_separation": {
            key: {"estimate": value[0], "low": value[1], "high": value[2]}
            for key, value in estimates.items()
        },
        "paired_differences": {
            key: {"estimate": value[0], "low": value[1], "high": value[2]}
            for key, value in differences.items()
        },
    }


def contrast_plot(analysis: dict[str, Any], output: Path) -> None:
    separation = analysis["directional_separation"]
    differences = analysis["paired_differences"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    labels = ("LoRA after\nrestore", "Joint full", "Full after\nrestore")
    keys = ("lora", "joint_full", "sequential_full")
    values = np.asarray([separation[key]["estimate"] for key in keys])
    errors = np.vstack(
        (
            values - [separation[key]["low"] for key in keys],
            [separation[key]["high"] for key in keys] - values,
        )
    )
    axes[0].errorbar(
        np.arange(3), values, yerr=np.maximum(errors, 0), fmt="o", markersize=8,
        capsize=5, color="#0072B2",
    )
    axes[0].set_xticks(np.arange(3), labels)
    axes[0].set_ylabel("Directional SDF separation sum")
    axes[0].set_title("Separation retained by each training design", weight="bold")

    difference_keys = (
        "sequential_full_minus_lora",
        "joint_full_minus_sequential_full",
    )
    difference_labels = (
        "Full − LoRA\n(fixed separate stages)",
        "Joint − sequential\n(fixed full parameters)",
    )
    diff_values = np.asarray([differences[key]["estimate"] for key in difference_keys])
    diff_errors = np.vstack(
        (
            diff_values - [differences[key]["low"] for key in difference_keys],
            [differences[key]["high"] for key in difference_keys] - diff_values,
        )
    )
    axes[1].errorbar(
        np.arange(2), diff_values, yerr=np.maximum(diff_errors, 0), fmt="o",
        markersize=8, capsize=5, color="#D55E00",
    )
    axes[1].set_xticks(np.arange(2), difference_labels)
    axes[1].set_ylabel("Paired difference in separation sum")
    axes[1].set_title("Direct parameterization and staging contrasts", weight="bold")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.9, alpha=0.65)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Paired conflict-episode contrasts\n"
        f"95% bootstrap intervals; {analysis['bootstrap_resamples']:,} resamples; "
        f"n = {analysis['n']}",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, output, "fp_parameterization_control_contrasts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora-root",
        default="experiments/prior_coins/runs/dispatch_sdf_aft_v1/evaluation",
    )
    parser.add_argument(
        "--joint-full-root",
        default="experiments/prior_coins/runs/dispatch_fp_blend_v1/evaluation",
    )
    parser.add_argument(
        "--sequential-full-root",
        default=(
            "experiments/prior_coins/runs/"
            "dispatch_fp_aft_after_restore_v1/evaluation"
        ),
    )
    parser.add_argument(
        "--output",
        default="experiments/prior_coins/figures/dispatch_fp_parameterization_control_v1",
    )
    parser.add_argument("--analysis-output", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    args = parser.parse_args()
    roots = condition_roots(args)
    output = Path(args.output)
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    choice_plot(roots, output)
    analysis = contrast_analysis(roots, args.bootstrap_resamples)
    contrast_plot(analysis, output)
    analysis_path = (
        Path(args.analysis_output)
        if args.analysis_output
        else roots["sequential_full"] / "parameterization_control_analysis.json"
    )
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    print(f"wrote plots to {output} and analysis to {analysis_path}")


if __name__ == "__main__":
    main()
