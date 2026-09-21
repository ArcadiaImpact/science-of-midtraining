"""Compare held-out conflict behavior across ReFT, FP AFT, and FP RL."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ARMS = ("charter", "coin", "mixed", "neutral")
ARM_LABELS = {
    "charter": "Charter 2M",
    "coin": "Coin 2M",
    "mixed": "50:50",
    "neutral": "Neutral 2M",
}
STAGES = ("reft_only", "fp_aft", "fp_alignment_rl")
STAGE_LABELS = {
    "reft_only": "ReFT only\n(no AFT)",
    "fp_aft": "FP AFT\nafter ReFT",
    "fp_alignment_rl": "FP alignment RL\nafter ReFT",
}
OUTCOMES = ("Charter choice", "Coin choice", "Other / malformed")
COLORS = {
    "Charter choice": "#0072B2",
    "Coin choice": "#E69F00",
    "Other / malformed": "#999999",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _wilson(count: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    rate = count / n
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
        / denominator
    )
    return center - half_width, center + half_width


def _append_cell(
    rows: list[dict[str, object]],
    *,
    stage: str,
    arm: str,
    n: int,
    charter: float,
    coin: float,
    other: float,
) -> None:
    rates = (charter, coin, other)
    if not math.isclose(sum(rates), 1.0, abs_tol=1e-9):
        raise ValueError(
            f"{stage}/{arm} conflict outcome rates must sum to 1; got {sum(rates)}"
        )
    for outcome, rate in zip(OUTCOMES, rates, strict=True):
        count = round(rate * n)
        if not math.isclose(count / n, rate, abs_tol=1e-9):
            raise ValueError(f"{stage}/{arm}/{outcome} rate is not an exact count / n")
        low, high = _wilson(count, n)
        rows.append(
            {
                "stage": STAGE_LABELS[stage],
                "arm": ARM_LABELS[arm],
                "outcome": outcome,
                "count": count,
                "n": n,
                "rate": rate,
                "low": low,
                "high": high,
            }
        )


def build_rows(reft_path: Path, fp_aft_path: Path, rl_path: Path) -> list[dict[str, object]]:
    """Normalize the three evaluation schemas into plot-ready aggregate rows."""

    reft = _load(reft_path)
    fp_aft = _load(fp_aft_path)
    rl = _load(rl_path)
    rows: list[dict[str, object]] = []
    reft_n = int(reft["n_eval_conflict_per_endpoint"])
    for arm in ARMS:
        cell = reft["cells"][arm]["no_aft"]
        _append_cell(
            rows,
            stage="reft_only",
            arm=arm,
            n=reft_n,
            charter=float(cell["conflict_charter_rate"]),
            coin=float(cell["conflict_coin_rate"]),
            other=float(cell["conflict_other_rate"]),
        )

        cell = fp_aft["rows"][arm]["metrics"]["conflict"]
        _append_cell(
            rows,
            stage="fp_aft",
            arm=arm,
            n=int(cell["n"]),
            charter=float(cell["charter_plan_rate"]["rate"]),
            coin=float(cell["coin_plan_rate"]["rate"]),
            other=(
                float(cell["other_plan_rate"]["rate"])
                + float(cell["malformed_rate"]["rate"])
            ),
        )

        cell = rl["cells"][arm]["direct"]["conflict"]
        _append_cell(
            rows,
            stage="fp_alignment_rl",
            arm=arm,
            n=int(cell["n"]),
            charter=float(cell["charter_rate"]),
            coin=float(cell["coin_rate"]),
            other=float(cell["other_rate"]) + float(cell["malformed_rate"]),
        )
    return rows


def plot(rows: Sequence[Mapping[str, object]], output: Path) -> list[Path]:
    """Write a three-panel bar chart and its machine-readable source data."""

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame["stage"] = pd.Categorical(
        frame["stage"], [STAGE_LABELS[stage] for stage in STAGES], ordered=True
    )
    frame["arm"] = pd.Categorical(
        frame["arm"], [ARM_LABELS[arm] for arm in ARMS], ordered=True
    )
    frame["outcome"] = pd.Categorical(frame["outcome"], OUTCOMES, ordered=True)

    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.catplot(
        data=frame,
        x="arm",
        y="rate",
        hue="outcome",
        col="stage",
        kind="bar",
        palette=COLORS,
        errorbar=None,
        height=5.2,
        aspect=0.92,
        legend_out=False,
    )
    for stage, axis in zip(STAGES, grid.axes.flat, strict=True):
        stage_rows = frame[frame["stage"] == STAGE_LABELS[stage]]
        for outcome, container in zip(OUTCOMES, axis.containers[: len(OUTCOMES)], strict=True):
            outcome_rows = stage_rows[stage_rows["outcome"] == outcome]
            for patch, (_, row) in zip(container, outcome_rows.iterrows(), strict=True):
                x = patch.get_x() + patch.get_width() / 2
                rate = float(row["rate"])
                axis.errorbar(
                    x,
                    rate,
                    yerr=np.array(
                        [[rate - float(row["low"])], [float(row["high"]) - rate]]
                    ),
                    fmt="none",
                    color="#303030",
                    capsize=2.5,
                    elinewidth=0.9,
                    capthick=0.9,
                )
        axis.set_ylim(0, 1.05)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.tick_params(axis="x", rotation=18)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
        axis.set_axisbelow(True)
    grid.set_axis_labels("Midtraining condition", "Held-out conflict choice rate")
    grid.set_titles("{col_name}", weight="bold")
    grid.figure.suptitle(
        "Behavior after ReFT, full-parameter AFT, and full-parameter alignment RL\n"
        "Direct interface; 95% Wilson intervals; n = 512 per model and stage",
        fontsize=14,
        weight="bold",
        y=1.05,
    )
    grid.figure.tight_layout()

    stem = output / "reft_fp_aft_fp_alignment_rl_conflict_rates"
    written = []
    for suffix, options in (
        ("pdf", {"format": "pdf"}),
        ("png", {"format": "png", "dpi": 220}),
        ("svg", {"format": "svg"}),
    ):
        path = stem.with_suffix(f".{suffix}")
        grid.figure.savefig(path, bbox_inches="tight", **options)
        written.append(path)
    plt.close(grid.figure)

    data_path = stem.with_suffix(".json")
    data_path.write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n")
    written.append(data_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reft", type=Path, required=True)
    parser.add_argument("--fp-aft", type=Path, required=True)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(args.reft, args.fp_aft, args.rl)
    for path in plot(rows, args.output):
        print(path)


if __name__ == "__main__":
    main()
