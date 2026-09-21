"""Seaborn PDF figures for the Dispatch GRPO AFT evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def _save(grid: object, path: Path) -> Path:
    figure = getattr(grid, "figure", grid)
    figure.savefig(path, format="pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure)
    return path


def plot_all(rows: Sequence[Mapping[str, object]], output: Path) -> list[Path]:
    """Write the four preregistered trajectory/diagnostic figures as PDFs."""

    import pandas as pd
    import seaborn as sns

    if not rows:
        raise ValueError("plotting requires evaluation rows")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", context="notebook")
    written = []

    agreement = frame[frame["kind"] == "agreement"]
    grid = sns.relplot(
        data=agreement, x="checkpoint", y="agreement_correct", hue="parent",
        style="interface", col="decoding", kind="line", markers=True, estimator="mean",
        errorbar=("ci", 95), facet_kws={"sharey": True},
    )
    grid.set_axis_labels("RL checkpoint (%)", "Held-out agreement accuracy")
    grid.figure.suptitle("Agreement learning trajectory", y=1.03)
    written.append(_save(grid, output / "agreement_learning.pdf"))

    from analyse_dispatch_grpo_aft_v1 import primary_trajectories

    primary = primary_trajectories(rows, n_resamples=10)
    contrast_rows = [
        {
            "checkpoint": cell["checkpoint"], "interface": cell["interface"],
            "decoding": cell["decoding"], "estimand": estimand,
            "estimate": cell[estimand]["estimate"],
        }
        for cell in primary
        for estimand in ("charter_separation", "coin_separation", "directional_sum")
    ]
    grid = sns.relplot(
        data=pd.DataFrame(contrast_rows), x="checkpoint", y="estimate", hue="estimand",
        style="interface", col="decoding", kind="line", markers=True,
        estimator=None,
    )
    for axis in grid.axes.flat:
        axis.axhline(0, color="black", linewidth=0.8)
    grid.set_axis_labels("RL checkpoint (%)", "Cross-parent conflict contrast")
    grid.figure.suptitle("Primary Charter, coin, and directional separation trajectories", y=1.03)
    written.append(_save(grid, output / "conflict_separation.pdf"))

    diagnostic_fields = [
        field for field in (
            "reward", "response_tokens", "entropy", "clip_ratio", "kl",
            "zero_variance_group_fraction", "cumulative_logprob",
        )
        if field in frame and frame[field].notna().any()
    ]
    diagnostics = frame.melt(
        id_vars=["checkpoint", "parent", "interface", "decoding"],
        value_vars=diagnostic_fields,
        var_name="diagnostic", value_name="value",
    )
    grid = sns.relplot(
        data=diagnostics, x="checkpoint", y="value", hue="parent", style="interface",
        col="diagnostic", row="decoding", kind="line", markers=True,
        estimator="mean", errorbar=("ci", 95),
        facet_kws={"sharey": False},
    )
    grid.set_axis_labels("RL checkpoint (%)", "Mean")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Reward and completion-length diagnostics", y=1.04)
    written.append(_save(grid, output / "reward_length_diagnostics.pdf"))

    comparison = frame.melt(
        id_vars=["checkpoint", "parent", "interface", "decoding", "kind"],
        value_vars=["agreement_correct", "charter_choice", "coin_choice", "format_valid"],
        var_name="metric", value_name="rate",
    )
    grid = sns.relplot(
        data=comparison, x="checkpoint", y="rate", hue="interface", style="parent",
        col="metric", row="decoding", kind="line", markers=True, estimator="mean",
        errorbar=("ci", 95), facet_kws={"sharey": True},
    )
    grid.set_axis_labels("RL checkpoint (%)", "Rate")
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Tagged versus legacy interfaces", y=1.03)
    written.append(_save(grid, output / "tagged_vs_legacy.pdf"))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    for path in plot_all(rows, args.output):
        print(path)


if __name__ == "__main__":
    main()
