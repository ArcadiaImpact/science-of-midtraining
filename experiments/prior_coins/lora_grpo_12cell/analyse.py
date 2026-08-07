"""Score, summarize, and plot the completed 12-cell LoRA-GRPO sweep."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .run_cell import OBJECTIVES, PARENTS
except ImportError:  # direct script execution
    from run_cell import OBJECTIVES, PARENTS  # type: ignore


OBJECTIVE_LABELS = {
    "agreement": "Agreement",
    "coin": "Coin",
    "charter": "Charter",
}
PARENT_LABELS = {
    "charter": "Charter 2M",
    "coin": "Coin 2M",
    "mixed": "50:50",
    "neutral": "Neutral 2M",
}
MODE_LABELS = {"direct": "No thinking", "thinking": "Thinking"}
OUTCOMES = ("Charter choice", "Coin choice", "Other / malformed")
COLORS = {
    "Charter choice": "#0072B2",
    "Coin choice": "#E69F00",
    "Other / malformed": "#999999",
}
PARENT_COLORS = {
    "Charter 2M": "#0072B2",
    "Coin 2M": "#D55E00",
    "50:50": "#009E73",
    "Neutral 2M": "#777777",
}
_GLOBAL_STEP = re.compile(r"\bglobal_step=(\d+)\b")
_CHARTER = re.compile(r"\bcharter\b", re.IGNORECASE)
_RULE_LANGUAGE = re.compile(
    r"\b(?:charter|clause|rule|protocol|priority|procedure|policy)\w*\b",
    re.IGNORECASE,
)
_COIN = re.compile(r"\bcoins?\b", re.IGNORECASE)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _wilson(
    count: int, n: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    rate = count / n
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
        / denominator
    )
    return (
        max(0.0, min(rate, center - half_width)),
        min(1.0, max(rate, center + half_width)),
    )


def reward_rows_for_cell(
    paths: Sequence[Path],
    *,
    objective: str,
    parent: str,
    expected_steps: int = 64,
    expected_per_step: int = 32,
) -> list[dict[str, object]]:
    """Reduce immutable rollout logs to one exact row per optimizer update."""

    if len(paths) != 1:
        raise ValueError(f"{objective}/{parent}: expected one rollout shard, got {len(paths)}")
    by_step: defaultdict[int, list[float]] = defaultdict(list)
    for row in _jsonl(paths[0]):
        match = _GLOBAL_STEP.search(str(row.get("trainer_state", "")))
        if match is None:
            raise ValueError(f"{paths[0]}: rollout is missing global_step")
        reward = float(row["reward"])
        if reward not in (0.0, 1.0):
            raise ValueError(f"{objective}/{parent}: non-binary reward {reward}")
        by_step[int(match.group(1))].append(reward)
    if sorted(by_step) != list(range(expected_steps)):
        raise ValueError(f"{objective}/{parent}: incomplete optimizer-step grid")
    means = []
    for step in range(expected_steps):
        values = by_step[step]
        if len(values) != expected_per_step:
            raise ValueError(
                f"{objective}/{parent}/step-{step}: expected "
                f"{expected_per_step} rewards, got {len(values)}"
            )
        means.append(sum(values) / len(values))
    result = []
    for index, mean in enumerate(means):
        start, stop = max(0, index - 2), min(len(means), index + 3)
        result.append({
            "objective": OBJECTIVE_LABELS[objective],
            "parent": PARENT_LABELS[parent],
            "global_step": index,
            "step": index + 1,
            "successes": round(sum(by_step[index])),
            "n": expected_per_step,
            "reward_mean": mean,
            "reward_smoothed": sum(means[start:stop]) / (stop - start),
        })
    return result


def build_reward_rows(evidence_root: Path) -> list[dict[str, object]]:
    rows = []
    for objective in OBJECTIVES:
        for parent in PARENTS:
            logs = sorted(
                (evidence_root / "cells" / objective / parent / "logs").glob(
                    "raw_rollouts.rank-*.jsonl"
                )
            )
            rows.extend(reward_rows_for_cell(
                logs, objective=objective, parent=parent
            ))
    return rows


def endpoint_rows_for_cell(
    summary: Mapping[str, Any], *, objective: str, parent: str
) -> list[dict[str, object]]:
    """Normalize direct/thinking conflict summaries for the six-panel plot."""

    result = []
    for mode in ("direct", "thinking"):
        cell = summary["cells"][parent][mode]["conflict"]
        n = int(cell["n"])
        rates = (
            float(cell["charter_rate"]),
            float(cell["coin_rate"]),
            float(cell["other_rate"]) + float(cell["malformed_rate"]),
        )
        if abs(sum(rates) - 1.0) > 1e-9:
            raise ValueError(f"{objective}/{parent}/{mode}: outcomes do not sum to one")
        for outcome, rate in zip(OUTCOMES, rates, strict=True):
            result.append({
                "objective": OBJECTIVE_LABELS[objective],
                "parent": PARENT_LABELS[parent],
                "mode": MODE_LABELS[mode],
                "outcome": outcome,
                "count": round(rate * n),
                "n": n,
                "rate": rate,
            })
    return result


def score_grid(evidence_root: Path) -> None:
    """Score each immutable GPU sample set locally and validate 2,048 rows."""

    pod = Path(__file__).resolve().parents[1] / "pod"
    sys.path.insert(0, str(pod))
    from dispatch_grpo_endpoint_eval import score_samples

    for objective in OBJECTIVES:
        for parent in PARENTS:
            output = evidence_root / "cells" / objective / parent / "eval_raw"
            rows = score_samples(output)
            if len(rows) != 2_048:
                raise RuntimeError(
                    f"{objective}/{parent}: expected 2,048 scored rows, got {len(rows)}"
                )


def build_endpoint_rows(evidence_root: Path) -> list[dict[str, object]]:
    rows = []
    for objective in OBJECTIVES:
        for parent in PARENTS:
            summary = json.loads((
                evidence_root / "cells" / objective / parent /
                "eval_raw" / "summary.json"
            ).read_text())
            rows.extend(endpoint_rows_for_cell(
                summary, objective=objective, parent=parent
            ))
    return rows


def build_full_parameter_comparison(
    lora_rows: Sequence[Mapping[str, object]],
    *,
    agreement_summary: Path,
    unambiguous_root: Path,
) -> list[dict[str, object]]:
    """Join LoRA endpoint rates to the matched full-parameter seed-42 cells."""

    fp: dict[tuple[str, str], Mapping[str, Any]] = {}
    agreement = json.loads(agreement_summary.read_text())
    for parent in PARENTS:
        fp[("agreement", parent)] = agreement
    for objective in ("coin", "charter"):
        for parent in PARENTS:
            fp[(objective, parent)] = json.loads(
                (unambiguous_root / objective / parent / "summary.json").read_text()
            )

    reverse_objectives = {label: key for key, label in OBJECTIVE_LABELS.items()}
    reverse_parents = {label: key for key, label in PARENT_LABELS.items()}
    reverse_modes = {label: key for key, label in MODE_LABELS.items()}
    result = []
    for row in lora_rows:
        objective = reverse_objectives[str(row["objective"])]
        parent = reverse_parents[str(row["parent"])]
        mode = reverse_modes[str(row["mode"])]
        cell = fp[(objective, parent)]["cells"][parent][mode]["conflict"]
        fp_rates = {
            "Charter choice": float(cell["charter_rate"]),
            "Coin choice": float(cell["coin_rate"]),
            "Other / malformed": (
                float(cell["other_rate"]) + float(cell["malformed_rate"])
            ),
        }
        fp_rate = fp_rates[str(row["outcome"])]
        lora_rate = float(row["rate"])
        result.append({
            **dict(row),
            "full_parameter_rate": fp_rate,
            "lora_rate": lora_rate,
            "lora_minus_full_parameter": lora_rate - fp_rate,
        })
    return result


def build_trace_language_summary(evidence_root: Path) -> dict[str, Any]:
    """Count explicit Charter/rule/Coin language in held-out conflict traces."""

    cells: dict[str, Any] = {}
    for objective in OBJECTIVES:
        cells[objective] = {}
        for parent in PARENTS:
            rows = [
                row for row in _jsonl(
                    evidence_root / "cells" / objective / parent /
                    "eval_raw" / "evaluation_rows.jsonl"
                )
                if row["reasoning_mode"] == "thinking" and row["kind"] == "conflict"
            ]
            counts: Counter[str] = Counter()
            outcomes: Counter[str] = Counter()
            for row in rows:
                trace = str(row.get("thinking_trace") or "")
                counts["trace_present"] += bool(trace)
                counts["explicit_charter"] += bool(_CHARTER.search(trace))
                counts["charter_or_rule_language"] += bool(_RULE_LANGUAGE.search(trace))
                counts["explicit_coin"] += bool(_COIN.search(trace))
                outcomes[str(row["outcome"])] += 1
            cells[objective][parent] = {
                "n": len(rows),
                **dict(sorted(counts.items())),
                "outcomes": dict(sorted(outcomes.items())),
            }
    return {
        "version": "dispatch_lora_grpo_trace_language_v1",
        "interpretation_warning": (
            "Visible traces are stated rationales and may be post-hoc; choices are primary."
        ),
        "cells": cells,
    }


def plot_endpoint(rows: Sequence[Mapping[str, object]], output: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.catplot(
        data=frame,
        x="parent",
        y="rate",
        hue="outcome",
        row="mode",
        col="objective",
        kind="bar",
        row_order=[MODE_LABELS[m] for m in ("direct", "thinking")],
        col_order=[OBJECTIVE_LABELS[o] for o in OBJECTIVES],
        order=[PARENT_LABELS[p] for p in PARENTS],
        hue_order=list(OUTCOMES),
        palette=COLORS,
        errorbar=None,
        height=4.0,
        aspect=1.08,
        legend_out=True,
    )
    sns.move_legend(
        grid,
        "upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        title=None,
        frameon=False,
    )
    grid.set_axis_labels("", "Held-out conflict choice rate")
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    for axis in grid.axes.flat:
        axis.set_ylim(0, 1.02)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.22)
    grid.figure.supxlabel("Midtraining condition", y=0.09)
    grid.figure.suptitle(
        "LoRA-GRPO final conflict behavior after 64 updates\n"
        "Seed 42; n = 512 conflicts per cell and reasoning mode",
        weight="bold",
        y=0.99,
    )
    grid.figure.tight_layout(rect=(0.02, 0.12, 1, 0.92))
    stem = output / "lora_grpo_final_conflict_rates"
    written = []
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        path = stem.with_suffix(f".{suffix}")
        grid.figure.savefig(path, bbox_inches="tight", **kwargs)
        written.append(path)
    plt.close(grid.figure)
    data = stem.with_suffix(".json")
    data.write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n")
    return written + [data]


def plot_alignment_grid(
    rows: Sequence[Mapping[str, object]], output: Path
) -> list[Path]:
    """Match the direct/thinking objective grid used for full-parameter GRPO."""

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    if not rows:
        raise ValueError("plotting requires endpoint rows")
    output.mkdir(parents=True, exist_ok=True)
    plot_rows = []
    for source_row in rows:
        row = dict(source_row)
        low, high = _wilson(int(row["count"]), int(row["n"]))
        row.update({"low": low, "high": high})
        plot_rows.append(row)
    frame = pd.DataFrame(plot_rows)
    objective_order = [OBJECTIVE_LABELS[value] for value in OBJECTIVES]
    mode_order = [MODE_LABELS[value] for value in ("direct", "thinking")]
    parent_order = [PARENT_LABELS[value] for value in PARENTS]
    frame["objective"] = pd.Categorical(
        frame["objective"], objective_order, ordered=True
    )
    frame["mode"] = pd.Categorical(frame["mode"], mode_order, ordered=True)
    frame["parent"] = pd.Categorical(
        frame["parent"], parent_order, ordered=True
    )
    frame["outcome"] = pd.Categorical(frame["outcome"], OUTCOMES, ordered=True)

    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.catplot(
        data=frame,
        x="parent",
        y="rate",
        hue="outcome",
        col="objective",
        row="mode",
        kind="bar",
        order=parent_order,
        hue_order=list(OUTCOMES),
        col_order=objective_order,
        row_order=mode_order,
        palette=COLORS,
        errorbar=None,
        height=4.15,
        aspect=1.0,
        legend_out=False,
    )
    for row_index, mode in enumerate(mode_order):
        mode_rows = frame[frame["mode"] == mode]
        for column_index, objective in enumerate(objective_order):
            axis = grid.axes[row_index, column_index]
            objective_rows = mode_rows[mode_rows["objective"] == objective]
            for outcome, container in zip(
                OUTCOMES, axis.containers[: len(OUTCOMES)], strict=True
            ):
                outcome_rows = objective_rows[
                    objective_rows["outcome"] == outcome
                ].sort_values("parent")
                for patch, (_, row) in zip(
                    container, outcome_rows.iterrows(), strict=True
                ):
                    x = patch.get_x() + patch.get_width() / 2
                    rate = float(row["rate"])
                    axis.errorbar(
                        x,
                        rate,
                        yerr=np.array(
                            [[rate - float(row["low"])],
                             [float(row["high"]) - rate]]
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
            axis.set_title(objective if row_index == 0 else "", weight="bold")
        grid.axes[row_index, -1].annotate(
            mode,
            xy=(1.045, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=-90,
            fontsize=12,
            weight="bold",
        )

    grid.set_axis_labels("", "")
    if grid._legend is not None:
        handles = grid._legend.legend_handles
        labels = [text.get_text() for text in grid._legend.texts]
        grid._legend.remove()
        grid.figure.legend(
            handles,
            labels,
            title="Conflict outcome",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=3,
            frameon=False,
        )
    grid.figure.supxlabel("Midtraining condition", y=0.09)
    grid.figure.supylabel("Held-out conflict choice rate", x=0.01)
    grid.figure.suptitle(
        "Conflict behavior after objective-specific LoRA-GRPO\n"
        "64 updates; 95% Wilson intervals; n = 512 per model",
        fontsize=15,
        weight="bold",
        y=0.99,
    )
    grid.figure.tight_layout(rect=(0.035, 0.13, 0.97, 0.91))

    stem = output / "lora_agreement_coin_charter_final_conflict_rates"
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
    data = stem.with_suffix(".json")
    data.write_text(json.dumps(plot_rows, indent=2, sort_keys=True) + "\n")
    written.append(data)
    return written


def plot_rewards(rows: Sequence[Mapping[str, object]], output: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.3), sharex=True, sharey=True)
    for objective, axis in zip(OBJECTIVES, axes, strict=True):
        subset = frame[frame["objective"] == OBJECTIVE_LABELS[objective]]
        for column, alpha, width in (
            ("reward_mean", 0.20, 1.0), ("reward_smoothed", 1.0, 2.2)
        ):
            sns.lineplot(
                data=subset,
                x="step",
                y=column,
                hue="parent",
                hue_order=[PARENT_LABELS[p] for p in PARENTS],
                palette=PARENT_COLORS,
                estimator=None,
                alpha=alpha,
                linewidth=width,
                legend=objective == "agreement" and column == "reward_smoothed",
                ax=axis,
            )
        axis.set_title(OBJECTIVE_LABELS[objective], weight="bold")
        axis.set(xlim=(1, 64), ylim=(0, 1.02), xlabel="LoRA-GRPO update", ylabel="")
        axis.set_xticks((1, 16, 32, 48, 64))
        axis.grid(alpha=0.22)
    figure.supylabel("Sampled binary reward")
    figure.suptitle(
        "LoRA-GRPO reward trajectories\n"
        "Faint: per-update mean (n = 32); solid: centered five-update mean",
        weight="bold",
    )
    figure.tight_layout(rect=(0.02, 0.10, 1, 0.92))
    stem = output / "lora_grpo_reward_trajectories"
    written = []
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, bbox_inches="tight", **kwargs)
        written.append(path)
    plt.close(figure)
    data = stem.with_suffix(".json")
    data.write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n")
    return written + [data]


def plot_full_parameter_comparison(
    rows: Sequence[Mapping[str, object]], output: Path
) -> list[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    grid = sns.catplot(
        data=frame,
        x="parent",
        y="lora_minus_full_parameter",
        hue="outcome",
        row="mode",
        col="objective",
        kind="bar",
        row_order=[MODE_LABELS[m] for m in ("direct", "thinking")],
        col_order=[OBJECTIVE_LABELS[o] for o in OBJECTIVES],
        order=[PARENT_LABELS[p] for p in PARENTS],
        hue_order=list(OUTCOMES),
        palette=COLORS,
        errorbar=None,
        height=4.0,
        aspect=1.08,
        legend_out=True,
    )
    sns.move_legend(
        grid,
        "upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        title=None,
        frameon=False,
    )
    grid.set_axis_labels("", "")
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    for axis in grid.axes.flat:
        axis.axhline(0, color="#303030", linewidth=0.9)
        axis.set_ylim(-1, 1)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.22)
    grid.figure.supxlabel("Midtraining condition", y=0.09)
    grid.figure.supylabel("LoRA minus full-parameter choice rate", x=0.015)
    grid.figure.suptitle(
        "Matched LoRA versus full-parameter GRPO endpoints\n"
        "Positive values mean the outcome is more frequent under LoRA",
        weight="bold",
        y=0.99,
    )
    grid.figure.tight_layout(rect=(0.02, 0.12, 1, 0.92))
    stem = output / "lora_minus_full_parameter_conflict_rates"
    written = []
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
        path = stem.with_suffix(f".{suffix}")
        grid.figure.savefig(path, bbox_inches="tight", **kwargs)
        written.append(path)
    plt.close(grid.figure)
    data = stem.with_suffix(".json")
    data.write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n")
    return written + [data]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--full-parameter-agreement-summary",
        type=Path,
        default=Path(
            "experiments/prior_coins/runs/"
            "dispatch_grpo_endpoint_eval_20260805T125041Z_seed42/summary.json"
        ),
    )
    parser.add_argument(
        "--full-parameter-unambiguous-root",
        type=Path,
        default=Path(
            "experiments/prior_coins/runs/"
            "dispatch_grpo_unambiguous_v1_20260805T150713Z_seed42/evals"
        ),
    )
    args = parser.parse_args()
    score_grid(args.evidence_root)
    endpoint = build_endpoint_rows(args.evidence_root)
    rewards = build_reward_rows(args.evidence_root)
    comparison = build_full_parameter_comparison(
        endpoint,
        agreement_summary=args.full_parameter_agreement_summary,
        unambiguous_root=args.full_parameter_unambiguous_root,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "thinking_trace_language.json").write_text(
        json.dumps(
            build_trace_language_summary(args.evidence_root),
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    written = (
        plot_endpoint(endpoint, args.output)
        + plot_alignment_grid(endpoint, args.output)
        + plot_rewards(rewards, args.output)
        + plot_full_parameter_comparison(comparison, args.output)
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
