"""Render the committed stronger-model SFT result figures."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent
ARMS = ("base", "dominant", "latency", "memory")
LORA_ARMS = ARMS[1:]
SETS = ("Dominant", "Tradeoff")
COLORS = {"Dominant": "#4C78A8", "Tradeoff": "#F28E2B"}

CORRECTNESS = {
    "Gemma 4 12B": {
        "Dominant": (70.40, 67.91, 69.78, 71.34),
        "Tradeoff": (63.75, 62.50, 62.50, 67.50),
    },
    "Qwen3-Coder 30B-A3B": {
        "Dominant": (20.56, 15.26, 23.99, 22.74),
        "Tradeoff": (22.50, 18.75, 27.50, 28.75),
    },
}

# Median per-problem post/base percentage changes on the shared-correct,
# successfully measured intersection. Negative means lower (better).
PAIRED = {
    "Gemma 4 12B": {
        "Latency": {
            "Dominant": (0.27, 0.31, 0.33),
            "Tradeoff": (0.31, 0.23, -0.18),
        },
        "Peak RSS": {
            "Dominant": (-0.04, 0.00, 0.00),
            "Tradeoff": (0.00, 0.00, 0.00),
        },
    },
    "Qwen3-Coder 30B-A3B": {
        "Latency": {
            "Dominant": (0.36, -0.28, 0.10),
            "Tradeoff": (-6.57, 0.03, 0.19),
        },
        "Peak RSS": {
            "Dominant": (0.00, 0.00, 0.02),
            "Tradeoff": (0.33, 0.13, 0.22),
        },
    },
}


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


def _label_bars(axis, bars, *, decimals: int = 1, suffix: str = "") -> None:
    labels = [f"{bar.get_height():.{decimals}f}{suffix}" for bar in bars]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8)


def plot_correctness() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    x = np.arange(len(ARMS))
    width = 0.36
    for axis, (model, values) in zip(axes, CORRECTNESS.items(), strict=True):
        for offset, eval_set in zip((-width / 2, width / 2), SETS, strict=True):
            bars = axis.bar(
                x + offset,
                values[eval_set],
                width,
                color=COLORS[eval_set],
                label=eval_set,
            )
            _label_bars(axis, bars, decimals=1, suffix="%")
        axis.set_title(model)
        axis.set_xticks(x, ARMS)
        axis.set_xlabel("Generation arm")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Correct solutions (%)")
    axes[0].set_ylim(0, 82)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Held-out executable correctness", y=0.99, fontsize=14, fontweight="bold"
    )
    fig.text(
        0.5,
        0.01,
        "One deterministic generation per prompt; dominant n=321, tradeoff n=80.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    fig.savefig(HERE / "correctness_by_arm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_paired_efficiency() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    x = np.arange(len(LORA_ARMS))
    width = 0.36
    for row, (model, metrics) in enumerate(PAIRED.items()):
        for col, (metric, values) in enumerate(metrics.items()):
            axis = axes[row, col]
            for offset, eval_set in zip((-width / 2, width / 2), SETS, strict=True):
                bars = axis.bar(
                    x + offset,
                    values[eval_set],
                    width,
                    color=COLORS[eval_set],
                    label=eval_set,
                )
                labels = [f"{bar.get_height():+.2f}%" for bar in bars]
                axis.bar_label(bars, labels=labels, padding=3, fontsize=8)
            axis.axhline(0, color="#333333", linewidth=0.9)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
            axis.set_axisbelow(True)
            axis.set_title(f"{model} — {metric}")
            axis.set_xticks(x, LORA_ARMS)
            axis.set_ylabel("Median paired change (%)")
            if metric == "Peak RSS":
                axis.set_ylim(-0.12, 0.43)
            elif model.startswith("Gemma"):
                axis.set_ylim(-0.32, 0.48)
            else:
                axis.set_ylim(-7.4, 1.1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Efficiency changes on shared-correct measured problems",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Post/base median per-problem change; negative means faster or lower-memory. Different y-scales are labeled explicitly.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.88), h_pad=2.0)
    fig.savefig(HERE / "paired_efficiency_by_arm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_correctness()
    plot_paired_efficiency()


if __name__ == "__main__":
    main()
