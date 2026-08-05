"""Render the committed dataset-analysis figures from dataset_analysis_data.json.

The JSON is derived from the pinned dataset revision
(42880cc8aa7c5da88ba3c0cce69efa458b18e12d) and the eight published
generation arms under generation_behavior/20260803_better_models; see
DATASET_ANALYSIS.md for the provenance and the derivation.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
QW = "qwen3-coder-30b-a3b-instruct"
GM = "gemma4-12b-it"
# Categorical slots 1-4 of the validated light-mode palette, in fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GOOD, SERIOUS = "#008300", "#e34948"


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.size": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _ecdf_axes(axis, series: list[tuple[str, list[int], str]], xlabel: str) -> None:
    import numpy as np

    for name, values, color in series:
        values = np.asarray(values, dtype=float)
        y = np.arange(1, len(values) + 1) / len(values)
        axis.step(values, y, where="post", color=color, linewidth=2, label=name)
    axis.set_xscale("log")
    axis.set_ylim(0, 1.02)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Fraction of rows ≤ x")
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def plot_length_gulf(data: dict) -> None:
    import matplotlib.pyplot as plt

    _style()
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    tradeoff = sorted(data["target_chars"]["latency"] + data["target_chars"]["memory"])
    series = [
        ("SFT targets: dominant (n=1286)", data["target_chars"]["dominant"], BLUE),
        ("SFT targets: tradeoff (n=644)", tradeoff, ORANGE),
        ("Gemma 4 base output (n=324)", data["base_response_chars"][GM], AQUA),
        ("Qwen3-Coder base output (n=324)", data["base_response_chars"][QW], YELLOW),
    ]
    _ecdf_axes(axis, series, "Length (characters, log scale)")
    axis.set_title("SFT targets are ~4x shorter than what either base model emits")
    axis.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.005,
        "Training targets: raw chosen solutions at the pinned dataset revision. "
        "Model outputs: raw base-arm responses (prose and fences included).",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(HERE / "dataset_length_gulf.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_qwen_tokens(data: dict) -> None:
    import matplotlib.pyplot as plt

    _style()
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    series = [
        ("base", data["qwen_tokens"]["base"], BLUE),
        ("dominant LoRA", data["qwen_tokens"]["dominant"], ORANGE),
        ("latency LoRA", data["qwen_tokens"]["latency"], AQUA),
        ("memory LoRA", data["qwen_tokens"]["memory"], YELLOW),
    ]
    _ecdf_axes(axis, series, "Generated tokens (log scale)")
    axis.set_title("Qwen3-Coder generation length by arm: dominant collapses to EOS")
    axis.legend(loc="upper left", frameon=False, fontsize=9)
    axis.annotate(
        "45% of dominant-arm rows\nstop at 1 token",
        xy=(1.15, 0.45),
        xytext=(4, 0.62),
        fontsize=9,
        color="#333333",
        arrowprops={"arrowstyle": "->", "color": "#333333"},
    )
    fig.text(
        0.5,
        0.005,
        "324 held-out prompts per arm, greedy decoding, 4096-token cap "
        "(the right-edge riser is the cap).",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(HERE / "dataset_qwen_tokens_by_arm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_qwen_collapse_difficulty(data: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _style()
    buckets = ["unrated", "7-8", "9-10", "11+"]
    counts = data["qwen_dominant_by_difficulty"]
    empty = [100 * counts[b]["empty"] / counts[b]["n"] for b in buckets]
    base_ok = [100 * counts[b]["base_correct"] / counts[b]["n"] for b in buckets]
    fig, axis = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(buckets))
    width = 0.36
    bars_a = axis.bar(
        x - width / 2, empty, width, color=ORANGE, label="dominant-LoRA empty-output rate"
    )
    bars_b = axis.bar(
        x + width / 2, base_ok, width, color=BLUE, label="base-model correct rate"
    )
    for bars in (bars_a, bars_b):
        axis.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9)
    labels = [f"{b}\n(n={counts[b]['n']})" for b in buckets]
    axis.set_xticks(x, labels)
    axis.set_xlabel("code_contests difficulty")
    axis.set_ylabel("Share of eval problems (%)")
    axis.set_ylim(0, 80)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.set_title("Qwen's dominant-arm collapse concentrates on hard problems")
    axis.legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.005,
        "Union of the 324 held-out problems. Empty output = ≤2 generated tokens "
        "(immediate end-of-turn).",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(
        HERE / "dataset_qwen_collapse_by_difficulty.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def plot_transitions(data: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _style()
    arms = ("dominant", "latency", "memory")
    models = [(GM, "Gemma 4 12B"), (QW, "Qwen3-Coder 30B-A3B")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    x = np.arange(len(arms))
    for axis, (slug, title) in zip(axes, models, strict=True):
        gained = [data["transitions_union"][f"{slug}/{a}"]["gained"] for a in arms]
        lost = [-data["transitions_union"][f"{slug}/{a}"]["lost"] for a in arms]
        bars_g = axis.bar(x, gained, 0.55, color=GOOD, label="newly correct (gained)")
        bars_l = axis.bar(x, lost, 0.55, color=SERIOUS, label="newly wrong (lost)")
        axis.bar_label(bars_g, fmt="%+d", padding=2, fontsize=9)
        axis.bar_label(bars_l, fmt="%+d", padding=2, fontsize=9)
        axis.axhline(0, color="#333333", linewidth=0.9)
        axis.set_xticks(x, arms)
        axis.set_xlabel("LoRA arm")
        axis.set_title(title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Problems vs base (count)")
    axes[0].set_ylim(-27, 22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Correctness churn vs base on the 324-problem union", y=1.04, fontsize=14
    )
    fig.text(
        0.5,
        0.005,
        "Gemma churns roughly symmetrically in both directions; Qwen's tradeoff arms "
        "gain far more than they lose, and its dominant arm only loses.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(HERE / "dataset_transitions_union.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = json.loads((HERE / "dataset_analysis_data.json").read_text())
    plot_length_gulf(data)
    plot_qwen_tokens(data)
    plot_qwen_collapse_difficulty(data)
    plot_transitions(data)


if __name__ == "__main__":
    main()
