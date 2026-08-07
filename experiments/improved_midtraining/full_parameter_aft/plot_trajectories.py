"""Render full-parameter Dispatch favoring and generic-collapse trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
EPOCHS = (0, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32)
EPOCH_LABELS = (
    "SFT\n(no AFT)",
    "1/16",
    "1/8",
    "1/4",
    "1/2",
    "1",
    "2",
    "4",
    "8",
    "16",
    "32",
)
PARENT_LABELS = {
    "coin": "Coin-history parent",
    "charter": "Charter-history parent",
}
CHOICES = {
    "conflict_charter_rate": "Charter-favoring choice",
    "conflict_coin_rate": "Coin-favoring choice",
}
COLLAPSE = {
    "capability_mean": "Mean MMLU + GSM8K accuracy",
    "parseable_rate": "Parseable answer rate",
    "empty_rate": "Empty response rate",
    "truncation_rate": "Length-truncated response rate",
    "repeated_fourgram_rate": "Repeated 4-gram response rate",
    "maximum_exact_response_share": "Largest exact-response share",
    "dispatch_intrusion_rate": "Dispatch-language intrusion rate",
}


def _load(path: Path, metrics: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"parent", "endpoint", "step", "epochs", *metrics} - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if set(frame["parent"]) != set(PARENT_LABELS):
        raise ValueError(f"{path}: expected Coin and Charter parents")
    for parent, rows in frame.groupby("parent"):
        if tuple(rows["epochs"]) != EPOCHS:
            raise ValueError(f"{path}: incomplete trajectory for {parent}")
    frame["parent_label"] = frame["parent"].map(PARENT_LABELS)
    return frame


def _epoch_axis(axis: plt.Axes) -> None:
    axis.set_xscale("symlog", linthresh=EPOCHS[1], linscale=0.7, base=2)
    axis.set_xticks(EPOCHS, EPOCH_LABELS)
    axis.set_xlim(-0.012, 38)


def plot_favoring(source: Path, output: Path) -> None:
    frame = _load(source, set(CHOICES))
    data = frame.melt(
        id_vars=("parent", "parent_label", "endpoint", "step", "epochs"),
        value_vars=tuple(CHOICES),
        var_name="choice",
        value_name="rate",
    )
    data["choice_label"] = data["choice"].map(CHOICES)
    palette = {
        "Charter-favoring choice": "#0072B2",
        "Coin-favoring choice": "#D55E00",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharex=True, sharey=True)
    for axis, parent in zip(axes, PARENT_LABELS, strict=True):
        sns.lineplot(
            data=data[data["parent"] == parent],
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
        _epoch_axis(axis)
        axis.set_ylim(0, 0.82)
        axis.set_title(PARENT_LABELS[parent])
        axis.set_xlabel("Full-parameter AFT epochs (constant LR; symlog)")
        if axis.get_legend() is not None:
            axis.get_legend().remove()
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
        "Full-parameter agreement-only AFT converges to a mixed policy",
        y=0.985,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_collapse(source: Path, output: Path) -> None:
    frame = _load(source, set(COLLAPSE))
    data = frame.melt(
        id_vars=("parent", "parent_label", "endpoint", "step", "epochs"),
        value_vars=tuple(COLLAPSE),
        var_name="metric",
        value_name="rate",
    )
    palette = {
        "Coin-history parent": "#D55E00",
        "Charter-history parent": "#0072B2",
    }
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for axis, (metric, title) in zip(axes.flat, COLLAPSE.items(), strict=False):
        sns.lineplot(
            data=data[data["metric"] == metric],
            x="epochs",
            y="rate",
            hue="parent_label",
            style="parent_label",
            markers=True,
            dashes=False,
            linewidth=2,
            markersize=5,
            palette=palette,
            ax=axis,
        )
        _epoch_axis(axis)
        axis.set_ylim(-0.025, 1.025)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("Rate")
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    axes[3, 1].axis("off")
    axes[3, 0].set_xlabel("Full-parameter AFT epochs (constant LR; symlog)")
    axes[2, 1].set_xlabel("Full-parameter AFT epochs (constant LR; symlog)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.965, 0.055),
        frameon=False,
    )
    fig.suptitle(
        "Full-parameter AFT shows no generic response collapse",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dispatch-input",
        type=Path,
        default=ROOT / "data" / "dispatch_trajectory.csv",
    )
    parser.add_argument(
        "--generic-input",
        type=Path,
        default=ROOT / "data" / "generic_trajectory.csv",
    )
    parser.add_argument(
        "--favoring-output",
        type=Path,
        default=ROOT / "figures" / "dispatch_favoring.pdf",
    )
    parser.add_argument(
        "--collapse-output",
        type=Path,
        default=ROOT / "figures" / "generic_collapse.pdf",
    )
    args = parser.parse_args()
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plot_favoring(args.dispatch_input, args.favoring_output)
    plot_collapse(args.generic_input, args.collapse_output)


if __name__ == "__main__":
    main()
