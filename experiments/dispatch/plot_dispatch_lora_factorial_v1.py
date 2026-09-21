"""Plot and analyze the Dispatch factorial, with its original hybrid reference."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ARMS = ("charter", "coin", "mixed", "neutral")
ARM_LABELS = ("Charter 2M", "Coin 2M", "Mixed 1M+1M", "Neutral 2M")
COLORS = {"charter": "#0072B2", "coin": "#E69F00", "other": "#999999"}
CELLS = (
    ("joint_full", "Joint stream\nFull parameter", "fp_blend"),
    (
        "sequential_full",
        "Re-instruction then AFT\nFull parameter",
        "fp_aft_after_restore",
    ),
    ("joint_lora", "Joint stream\nLoRA throughout", "joint_lora"),
    (
        "sequential_lora",
        "Re-instruction then AFT\nLoRA throughout",
        "sequential_lora",
    ),
)
HYBRID_CELL = (
    "hybrid_full_restore_lora_aft",
    "Re-instruction full parameter\nthen AFT with LoRA*",
    "agreement",
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


def roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "joint_full": Path(args.joint_full_root),
        "sequential_full": Path(args.sequential_full_root),
        "joint_lora": Path(args.lora_root),
        "sequential_lora": Path(args.lora_root),
        "hybrid_full_restore_lora_aft": Path(args.hybrid_root),
    }


def metric(
    root: Path, arm: str, condition: str, field: str
) -> tuple[float, float, float]:
    cell = load(root / "metrics" / arm / f"{condition}.json")["metrics"]["conflict"]
    if field == "other":
        return wilson(cell["counts"]["other"] + cell["counts"]["malformed"], cell["n"])
    item = cell[f"{field}_plan_rate"]
    return float(item["rate"]), float(item["low"]), float(item["high"])


def outcomes(root: Path, arm: str, condition: str) -> dict[str, str]:
    rows = load(root / "details" / arm / condition / "conflict.json")["rows"]
    result = {row["id"]: row["outcome"] for row in rows}
    if len(result) != 512:
        raise ValueError(f"expected 512 unique rows for {arm}/{condition}")
    return result


def alignment_audit(cell_roots: dict[str, Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("agreement", "conflict"):
        reference: list[str] | None = None
        cells_checked = 0
        for key, _title, condition in CELLS:
            for arm in ARMS:
                rows = load(
                    cell_roots[key] / "details" / arm / condition / f"{split}.json"
                )["rows"]
                ids = sorted(row["id"] for row in rows)
                if len(ids) != 512 or len(set(ids)) != 512:
                    raise ValueError(f"invalid {split} IDs for {key}/{arm}")
                if reference is not None and ids != reference:
                    raise ValueError(f"unaligned {split} IDs for {key}/{arm}")
                reference = ids
                cells_checked += 1
        result[split] = {
            "n_unique": len(reference or []),
            "arm_condition_cells_checked": cells_checked,
            "all_aligned": True,
        }
    return result


def separation_vector(root: Path, condition: str) -> tuple[list[str], np.ndarray]:
    charter = outcomes(root, "charter", condition)
    coin = outcomes(root, "coin", condition)
    ids = sorted(charter)
    if ids != sorted(coin):
        raise ValueError(f"unaligned charter/coin rows for {condition}")
    values = np.asarray(
        [
            (charter[key] == "charter")
            - (coin[key] == "charter")
            + (coin[key] == "coin")
            - (charter[key] == "coin")
            for key in ids
        ],
        dtype=float,
    )
    return ids, values


def bootstrap(
    vector: np.ndarray, rng: np.random.Generator, n_boot: int
) -> dict[str, float]:
    samples = np.empty(n_boot)
    for start in range(0, n_boot, 1_000):
        stop = min(start + 1_000, n_boot)
        indices = rng.integers(0, len(vector), size=(stop - start, len(vector)))
        samples[start:stop] = vector[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "estimate": float(vector.mean()),
        "low": float(low),
        "high": float(high),
    }


def analyze(cell_roots: dict[str, Path], n_boot: int) -> dict[str, Any]:
    vectors: dict[str, np.ndarray] = {}
    reference_ids: list[str] | None = None
    for key, _title, condition in CELLS:
        ids, vector = separation_vector(cell_roots[key], condition)
        if reference_ids is not None and ids != reference_ids:
            raise ValueError("factorial cells are not episode-aligned")
        reference_ids = ids
        vectors[key] = vector

    contrasts = {
        "full_minus_lora_at_joint": vectors["joint_full"] - vectors["joint_lora"],
        "full_minus_lora_at_sequential": (
            vectors["sequential_full"] - vectors["sequential_lora"]
        ),
        "joint_minus_sequential_at_full": (
            vectors["joint_full"] - vectors["sequential_full"]
        ),
        "joint_minus_sequential_at_lora": (
            vectors["joint_lora"] - vectors["sequential_lora"]
        ),
    }
    contrasts["factorial_interaction"] = (
        contrasts["joint_minus_sequential_at_full"]
        - contrasts["joint_minus_sequential_at_lora"]
    )

    rng = np.random.default_rng(42)
    return {
        "n": len(reference_ids or []),
        "bootstrap_resamples": n_boot,
        "alignment_audit": alignment_audit(cell_roots),
        "metric": (
            "(Charter-SDF minus coin-SDF Charter-choice rate) + "
            "(coin-SDF minus Charter-SDF coin-choice rate)"
        ),
        "directional_separation": {
            key: bootstrap(vector, rng, n_boot) for key, vector in vectors.items()
        },
        "paired_contrasts": {
            key: bootstrap(vector, rng, n_boot) for key, vector in contrasts.items()
        },
    }


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def rate_plot(cell_roots: dict[str, Path], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), sharey=True)
    x = np.arange(len(ARMS))
    width = 0.25
    fields = (
        ("charter", "Charter choice"),
        ("coin", "Coin choice"),
        ("other", "Other / malformed"),
    )
    plot_cells = (
        (axes[0, 0], CELLS[0]),
        (axes[0, 1], CELLS[1]),
        (axes[1, 0], CELLS[2]),
        (axes[1, 1], CELLS[3]),
        (axes[0, 2], HYBRID_CELL),
    )
    for ax, (key, title, condition) in plot_cells:
        for offset, (field, label) in enumerate(fields):
            triples = [metric(cell_roots[key], arm, condition, field) for arm in ARMS]
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
        ax.set_xticks(x, ARM_LABELS, rotation=17, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        if key == HYBRID_CELL[0]:
            ax.set_facecolor("#F2F2F2")
    axes[0, 0].set_ylabel("Held-out conflict choice rate")
    axes[1, 0].set_ylabel("Held-out conflict choice rate")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.5,
        0.58,
        "* Additional hybrid reference\n\n"
        "Re-instruction updated all parameters;\n"
        "agreement AFT then used a rank-32 LoRA.\n\n"
        "This panel is not a fifth factorial cell.",
        ha="center",
        va="center",
        fontsize=11.5,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F2F2F2", "edgecolor": "#BBBBBB"},
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Clean 2 × 2: stage structure × parameterization, plus hybrid reference\n"
        "Error bars: 95% Wilson intervals; n = 512 per bar",
        fontsize=16,
        weight="bold",
    )
    fig.subplots_adjust(bottom=0.13, top=0.87, hspace=0.38, wspace=0.13)
    save(fig, output, "lora_factorial_rates")


def contrast_plot(analysis: dict[str, Any], output: Path) -> None:
    cells = analysis["directional_separation"]
    contrasts = analysis["paired_contrasts"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))

    cell_keys = ("joint_full", "sequential_full", "joint_lora", "sequential_lora")
    cell_labels = ("Joint\nfull", "Sequential\nfull", "Joint\nLoRA", "Sequential\nLoRA")
    values = np.asarray([cells[key]["estimate"] for key in cell_keys])
    errors = np.vstack(
        (
            values - [cells[key]["low"] for key in cell_keys],
            [cells[key]["high"] for key in cell_keys] - values,
        )
    )
    axes[0].errorbar(
        np.arange(4),
        values,
        yerr=np.maximum(errors, 0),
        fmt="o",
        color="#0072B2",
        markersize=8,
        capsize=5,
    )
    axes[0].set_xticks(np.arange(4), cell_labels)
    axes[0].set_ylabel("Directional SDF separation sum")
    axes[0].set_title("Separation in the four clean cells", weight="bold")

    contrast_keys = (
        "full_minus_lora_at_joint",
        "full_minus_lora_at_sequential",
        "joint_minus_sequential_at_full",
        "joint_minus_sequential_at_lora",
        "factorial_interaction",
    )
    contrast_labels = (
        "Full − LoRA\n(joint)",
        "Full − LoRA\n(sequential)",
        "Joint − seq.\n(full)",
        "Joint − seq.\n(LoRA)",
        "Interaction",
    )
    contrast_values = np.asarray([contrasts[key]["estimate"] for key in contrast_keys])
    contrast_errors = np.vstack(
        (
            contrast_values - [contrasts[key]["low"] for key in contrast_keys],
            [contrasts[key]["high"] for key in contrast_keys] - contrast_values,
        )
    )
    axes[1].errorbar(
        np.arange(5),
        contrast_values,
        yerr=np.maximum(contrast_errors, 0),
        fmt="o",
        color="#D55E00",
        markersize=8,
        capsize=5,
    )
    axes[1].set_xticks(np.arange(5), contrast_labels)
    axes[1].set_ylabel("Paired difference in separation sum")
    axes[1].set_title("Factorial contrasts", weight="bold")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.9, alpha=0.65)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Episode-paired conflict contrasts\n"
        f"95% bootstrap intervals; {analysis['bootstrap_resamples']:,} resamples; "
        f"n = {analysis['n']}",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    save(fig, output, "lora_factorial_contrasts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joint-full-root",
        default="experiments/dispatch/runs/dispatch_fp_blend_v1/evaluation",
    )
    parser.add_argument(
        "--sequential-full-root",
        default=(
            "experiments/dispatch/runs/"
            "dispatch_fp_aft_after_restore_v1/evaluation"
        ),
    )
    parser.add_argument(
        "--lora-root",
        default="experiments/dispatch/runs/dispatch_lora_factorial_v1/evaluation",
    )
    parser.add_argument(
        "--hybrid-root",
        default="experiments/dispatch/runs/dispatch_sdf_aft_v1/evaluation",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--figure-copy", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    args = parser.parse_args()

    cell_roots = roots(args)
    output = Path(args.output) if args.output else Path(args.lora_root) / "plots"
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    analysis = analyze(cell_roots, args.bootstrap_resamples)
    analysis_path = Path(args.lora_root) / "factorial_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    rate_plot(cell_roots, output)
    contrast_plot(analysis, output)
    if args.figure_copy:
        copy = Path(args.figure_copy)
        copy.mkdir(parents=True, exist_ok=True)
        for path in output.glob("lora_factorial_*.*"):
            shutil.copy2(path, copy / path.name)
    print(f"wrote analysis to {analysis_path} and plots to {output}")


if __name__ == "__main__":
    main()
