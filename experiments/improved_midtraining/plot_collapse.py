"""Plot generic capability and response-collapse diagnostics over AFT."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "generic_collapse_trajectory.csv"
DEFAULT_OUTPUT = ROOT / "figures" / "generic_collapse.pdf"
EPOCH_TICKS = (0, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32)
EPOCH_LABELS = ("SFT\n(no AFT)", "1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8", "16", "32")
PARENT_LABELS = {
    "coin": "Coin-history parent",
    "charter": "Charter-history parent",
}
METRICS = {
    "capability_mean": "Mean MMLU + GSM8K accuracy",
    "parseable_rate": "Parseable answer rate",
    "empty_rate": "Empty response rate",
    "truncation_rate": "Length-truncated response rate",
    "repeated_fourgram_rate": "Repeated 4-gram response rate",
    "maximum_exact_response_share": "Largest exact-response share",
    "dispatch_intrusion_rate": "Dispatch-language intrusion rate",
}


def load_trajectory(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"parent", "endpoint", "step", "epochs", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"trajectory is missing columns: {sorted(missing)}")
    for parent, rows in frame.groupby("parent"):
        if set(rows["epochs"]) != set(EPOCH_TICKS):
            raise ValueError(f"{parent} has an incomplete epoch trajectory")
    long = frame.melt(
        id_vars=("parent", "endpoint", "step", "epochs"),
        value_vars=tuple(METRICS),
        var_name="metric",
        value_name="rate",
    )
    long["parent_label"] = long["parent"].map(PARENT_LABELS)
    return long


def plot(input_path: Path, output_path: Path) -> None:
    data = load_trajectory(input_path)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    palette = {
        "Coin-history parent": "#D55E00",
        "Charter-history parent": "#0072B2",
    }
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    flat_axes = axes.flat
    for axis, (metric, title) in zip(flat_axes, METRICS.items(), strict=False):
        rows = data[data["metric"] == metric]
        sns.lineplot(
            data=rows,
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
        axis.set_xscale("symlog", linthresh=EPOCH_TICKS[1], linscale=0.7, base=2)
        axis.set_xticks(EPOCH_TICKS, EPOCH_LABELS)
        axis.set_xlim(-0.012, 38)
        axis.set_ylim(-0.025, 1.025)
        axis.axvline(102 / 64, color="0.35", linestyle=":", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("Rate")
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()

    axes[3, 1].axis("off")
    for axis in axes[3, :1]:
        axis.set_xlabel("AFT epochs (symlog; dotted line = end of warm-up)")
    for axis in axes[2, 1:]:
        if axis.axison:
            axis.set_xlabel("AFT epochs (symlog; dotted line = end of warm-up)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(0.965, 0.055),
        frameon=False,
    )
    fig.suptitle(
        "Generic controls show late capability erosion, not response collapse",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
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
