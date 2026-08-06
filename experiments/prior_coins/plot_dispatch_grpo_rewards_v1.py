"""Plot per-update sampled reward for the three GRPO training objectives."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path


OBJECTIVES = ("agreement", "coin", "charter")
OBJECTIVE_LABELS = {
    "agreement": "Agreement",
    "coin": "Coin",
    "charter": "Charter",
}
PARENTS = ("charter", "coin", "mixed", "neutral")
PARENT_LABELS = {
    "charter": "Charter 2M",
    "coin": "Coin 2M",
    "mixed": "50:50",
    "neutral": "Neutral 2M",
}
PARENT_COLORS = {
    "Charter 2M": "#0072B2",
    "Coin 2M": "#D55E00",
    "50:50": "#009E73",
    "Neutral 2M": "#777777",
}
_GLOBAL_STEP = re.compile(r"\bglobal_step=(\d+)\b")


def _rollout_root(
    *, objective: str, agreement_root: Path, unambiguous_root: Path
) -> Path:
    if objective == "agreement":
        return agreement_root
    return unambiguous_root / f"evidence_{objective}"


def _centered_mean(values: Sequence[float], index: int, window: int = 5) -> float:
    radius = window // 2
    start = max(0, index - radius)
    stop = min(len(values), index + radius + 1)
    return sum(values[start:stop]) / (stop - start)


def build_rows(
    *,
    agreement_root: Path,
    unambiguous_root: Path,
    expected_shards: int = 4,
    expected_steps: int = 64,
    expected_per_step: int = 32,
) -> list[dict[str, object]]:
    """Aggregate immutable rollout rewards into one row per optimizer update."""

    rows: list[dict[str, object]] = []
    for objective in OBJECTIVES:
        objective_root = _rollout_root(
            objective=objective,
            agreement_root=agreement_root,
            unambiguous_root=unambiguous_root,
        )
        for parent in PARENTS:
            paths = sorted(
                (objective_root / parent / "logs").glob(
                    "raw_rollouts.rank-*.jsonl"
                )
            )
            if len(paths) != expected_shards:
                raise ValueError(
                    f"{objective}/{parent}: expected {expected_shards} rollout "
                    f"shards, found {len(paths)}"
                )
            by_step: defaultdict[int, list[float]] = defaultdict(list)
            for path in paths:
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    rollout = json.loads(line)
                    match = _GLOBAL_STEP.search(str(rollout.get("trainer_state", "")))
                    if match is None:
                        raise ValueError(f"{path}: rollout is missing global_step")
                    reward = float(rollout["reward"])
                    if reward not in (0.0, 1.0):
                        raise ValueError(
                            f"{objective}/{parent}: expected binary reward, got {reward}"
                        )
                    by_step[int(match.group(1))].append(reward)

            expected_step_ids = list(range(expected_steps))
            if sorted(by_step) != expected_step_ids:
                raise ValueError(
                    f"{objective}/{parent}: expected steps 0..{expected_steps - 1}, "
                    f"got {sorted(by_step)}"
                )
            means = []
            for step in expected_step_ids:
                rewards = by_step[step]
                if len(rewards) != expected_per_step:
                    raise ValueError(
                        f"{objective}/{parent}/step-{step}: expected "
                        f"{expected_per_step} rewards, got {len(rewards)}"
                    )
                means.append(sum(rewards) / len(rewards))
            for index, (step, reward_mean) in enumerate(
                zip(expected_step_ids, means, strict=True)
            ):
                values = by_step[step]
                rows.append(
                    {
                        "objective": OBJECTIVE_LABELS[objective],
                        "parent": PARENT_LABELS[parent],
                        "global_step": step,
                        "step": step + 1,
                        "successes": round(sum(values)),
                        "n": len(values),
                        "reward_mean": reward_mean,
                        "reward_smoothed": _centered_mean(means, index),
                    }
                )
    return rows


def plot(rows: Sequence[Mapping[str, object]], output: Path) -> list[Path]:
    """Write a three-panel reward trajectory and its source data."""

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    if not rows:
        raise ValueError("plotting requires reward rows")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    objective_order = [OBJECTIVE_LABELS[value] for value in OBJECTIVES]
    parent_order = [PARENT_LABELS[value] for value in PARENTS]
    frame["objective"] = pd.Categorical(
        frame["objective"], objective_order, ordered=True
    )
    frame["parent"] = pd.Categorical(frame["parent"], parent_order, ordered=True)

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 5.4), sharex=True, sharey=True)
    for objective, axis in zip(OBJECTIVES, axes, strict=True):
        objective_rows = frame[
            frame["objective"] == OBJECTIVE_LABELS[objective]
        ]
        sns.lineplot(
            data=objective_rows,
            x="step",
            y="reward_mean",
            hue="parent",
            hue_order=parent_order,
            palette=PARENT_COLORS,
            estimator=None,
            linewidth=1.0,
            alpha=0.20,
            legend=False,
            ax=axis,
        )
        sns.lineplot(
            data=objective_rows,
            x="step",
            y="reward_smoothed",
            hue="parent",
            hue_order=parent_order,
            palette=PARENT_COLORS,
            estimator=None,
            linewidth=2.2,
            legend=objective == "agreement",
            ax=axis,
        )
        axis.set_title(OBJECTIVE_LABELS[objective], weight="bold")
        axis.set_xlim(1, 64)
        axis.set_xticks((1, 16, 32, 48, 64))
        axis.set_ylim(0, 1.02)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_xlabel("GRPO update step")
        axis.set_ylabel("")
        axis.grid(axis="both", alpha=0.22, linewidth=0.8)
        axis.set_axisbelow(True)

    legend = axes[0].get_legend()
    if legend is not None:
        handles = legend.legend_handles
        labels = [text.get_text() for text in legend.texts]
        legend.remove()
        figure.legend(
            handles,
            labels,
            title="Midtraining condition",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            frameon=False,
        )
    figure.supylabel("Sampled binary reward")
    figure.suptitle(
        "GRPO reward trajectories by training objective\n"
        "Faint: per-update mean (n = 32); solid: centered 5-update mean",
        fontsize=14,
        weight="bold",
        y=0.99,
    )
    figure.tight_layout(rect=(0.02, 0.14, 1, 0.90))

    stem = output / "grpo_reward_trajectories"
    written = []
    for suffix, options in (
        ("pdf", {"format": "pdf"}),
        ("png", {"format": "png", "dpi": 220}),
        ("svg", {"format": "svg"}),
    ):
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, bbox_inches="tight", **options)
        written.append(path)
    plt.close(figure)

    data_path = stem.with_suffix(".json")
    data_path.write_text(json.dumps(list(rows), indent=2, sort_keys=True) + "\n")
    written.append(data_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement-root", type=Path, required=True)
    parser.add_argument("--unambiguous-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(
        agreement_root=args.agreement_root,
        unambiguous_root=args.unambiguous_root,
    )
    for path in plot(rows, args.output):
        print(path)


if __name__ == "__main__":
    main()
