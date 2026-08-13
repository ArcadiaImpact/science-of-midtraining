"""Plot Coin/Charter conflict choices over the long AFT trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "dispatch_aft_trajectory.csv"
DEFAULT_OUTPUT = ROOT / "figures" / "dispatch_aft_favoring.pdf"
EPOCH_TICKS = (0, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32)
EPOCH_LABELS = ("SFT\n(no AFT)", "1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8", "16", "32")
ARM_LABELS = {"coin": "Coin-history parent", "charter": "Charter-history parent"}
CHOICE_LABELS = {
    "conflict_charter_rate": "Charter-favoring choice",
    "conflict_coin_rate": "Coin-favoring choice",
}


def load_trajectory(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "parent",
        "endpoint",
        "step",
        "epochs",
        "conflict_charter_rate",
        "conflict_coin_rate",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"trajectory is missing columns: {sorted(missing)}")
    expected_epochs = set(EPOCH_TICKS)
    for parent, rows in frame.groupby("parent"):
        if set(rows["epochs"]) != expected_epochs:
            raise ValueError(f"{parent} has an incomplete epoch trajectory")
    long = frame.melt(
        id_vars=("parent", "endpoint", "step", "epochs"),
        value_vars=tuple(CHOICE_LABELS),
        var_name="choice",
        value_name="rate",
    )
    long["parent_label"] = long["parent"].map(ARM_LABELS)
    long["choice_label"] = long["choice"].map(CHOICE_LABELS)
    return long


def plot(input_path: Path, output_path: Path) -> None:
    data = load_trajectory(input_path)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    palette = {
        "Charter-favoring choice": "#0072B2",
        "Coin-favoring choice": "#D55E00",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharex=True, sharey=True)
    for axis, parent in zip(axes, ("coin", "charter"), strict=True):
        rows = data[data["parent"] == parent]
        sns.lineplot(
            data=rows,
            x="epochs",
            y="rate",
            hue="choice_label",
            style="choice_label",
            markers=True,
            dashes=False,
            linewidth=2.1,
            markersize=6,
            palette=palette,
            ax=axis,
        )
        axis.set_xscale("symlog", linthresh=EPOCH_TICKS[1], linscale=0.7, base=2)
        axis.set_xticks(EPOCH_TICKS, EPOCH_LABELS)
        axis.set_xlim(-0.012, 38)
        axis.set_ylim(0, 0.82)
        axis.axvline(102 / 64, color="0.35", linestyle=":", linewidth=1.1)
        axis.set_title(ARM_LABELS[parent])
        axis.set_xlabel("AFT epochs (symlog; dotted line = end of warm-up)")
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    axes[0].set_ylabel("Conflict-set choice rate")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Long agreement-only AFT eventually makes both histories Charter-favoring",
        y=0.985,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
