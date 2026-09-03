"""Plot Figure-0-style stacked areas along the RLVR training trajectory.

One figure is written for every requested ``arm x template split``.  Agreement
and conflict episodes stay in separate panels: the former shows task accuracy,
while the latter shows the full answer composition without folding malformed
answers into a real crew choice.

Run from the repository root with::

    uv run --extra dev python \
      experiments/prior_coins/dispatch_rlvr_gemma4_26b_v1/plot_eval_trajectories.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SCORES = HERE / "eval_scores" / "rlvr_direct_scores.json"
DEFAULT_OUTPUT = HERE / "figures" / "trajectory_stacks"

ARMS = ("charter", "coin", "control")
ARM_LABEL = {
    "charter": "Charter-midtrained",
    "coin": "coin-midtrained",
    "control": "control midtraining",
}
SPLITS = ("trained", "heldout", "all")
SPLIT_LABEL = {
    "trained": "trained response templates",
    "heldout": "held-out response templates",
    "all": "all response templates (90% trained)",
}

# The same semantic palette used by the campaign's Figure-0 family.  No hatch
# is needed: color always means an outcome, and every segment is directly
# labelled by the legend.
AGREEMENT_STACK = (
    ("correct", "correct shared allocation", "#009E73"),
    ("incorrect", "incorrect / malformed", "#9E9E9E"),
)
CONFLICT_STACK = (
    ("charter_rate", "chose Charter", "#0072B2"),
    ("other_rate", "chose another crew", "#9E9E9E"),
    ("malformed_rate", "malformed / no answer", "#222222"),
    ("coin_rate", "chose coin / cheapest", "#E69F00"),
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load and validate the compact, one-row-per-checkpoint score table."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list")

    required = {
        "arm",
        "step",
        "split",
        "agreement_n",
        "agreement_accuracy",
        "conflict_n",
        *(key for key, _label, _colour in CONFLICT_STACK),
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: row {index} is not an object")
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"{path}: row {index} lacks {sorted(missing)}")
        arm, split, step = str(raw["arm"]), str(raw["split"]), int(raw["step"])
        identity = (arm, split, step)
        if identity in seen:
            raise ValueError(f"{path}: duplicate row {identity}")
        seen.add(identity)

        rates = [float(raw[key]) for key, _label, _colour in CONFLICT_STACK]
        accuracy = float(raw["agreement_accuracy"])
        if not 0 <= accuracy <= 1 or any(not 0 <= rate <= 1 for rate in rates):
            raise ValueError(f"{path}: invalid rate in row {identity}")
        if abs(sum(rates) - 1) > 1e-6:
            raise ValueError(f"{path}: conflict shares do not sum to one: {identity}")
        rows.append(dict(raw))
    return rows


def select_rows(
    rows: Iterable[Mapping[str, Any]], arm: str, split: str
) -> list[Mapping[str, Any]]:
    """Return one arm/split trajectory in optimizer-step order."""
    selected = [
        row for row in rows if row["arm"] == arm and row["split"] == split
    ]
    return sorted(selected, key=lambda row: int(row["step"]))


def style_axis(ax, steps: Sequence[int]) -> None:
    xmax = max(steps)
    major_ticks = sorted({0, xmax, *range(0, xmax + 1, 64)})
    ax.set_xticks(major_ticks)
    ax.set_xticks(steps, minor=True)
    ax.set_xlim(min(steps), xmax)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.grid(
        axis="x",
        which="minor",
        color="#eeeeee",
        linewidth=0.45,
        zorder=0,
    )
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#b7b7b7")
    ax.tick_params(colors="#444444", labelsize=8.5)
    ax.set_xlabel("optimizer step (0 = graft parent)", fontsize=9)


def stacked_areas(
    ax,
    rows: Sequence[Mapping[str, Any]],
    stack: Sequence[tuple[str, str, str]],
    values: Mapping[str, Sequence[float]],
) -> None:
    steps = [int(row["step"]) for row in rows]
    ax.stackplot(
        steps,
        *[list(values[key]) for key, _label, _colour in stack],
        colors=[colour for _key, _label, colour in stack],
        edgecolor="white",
        linewidth=0.55,
        zorder=2,
    )


def render(
    rows: Sequence[Mapping[str, Any]], *, arm: str, split: str, output: Path
) -> list[Path]:
    selected = select_rows(rows, arm, split)
    if not selected:
        raise ValueError(f"no rows for arm={arm!r}, split={split!r}")
    steps = [int(row["step"]) for row in selected]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.9), sharey=True)
    agreement = [100 * float(row["agreement_accuracy"]) for row in selected]
    stacked_areas(
        axes[0],
        selected,
        AGREEMENT_STACK,
        {
            "correct": agreement,
            "incorrect": [100 - value for value in agreement],
        },
    )
    axes[0].set_title(
        "Agreement episodes — task performance", fontsize=12, fontweight="bold"
    )
    axes[0].set_ylabel("share of evaluation runs (%)", fontsize=9.5)

    stacked_areas(
        axes[1],
        selected,
        CONFLICT_STACK,
        {
            key: [100 * float(row[key]) for row in selected]
            for key, _label, _colour in CONFLICT_STACK
        },
    )
    axes[1].set_title(
        "Conflict episodes — choice composition", fontsize=12, fontweight="bold"
    )

    for ax in axes:
        style_axis(ax, steps)
        ax.axhline(50, color="#bcbcbc", linewidth=0.7, linestyle="--", zorder=3)

    agreement_handles = [
        Patch(facecolor=colour, label=label)
        for _key, label, colour in AGREEMENT_STACK
    ]
    conflict_handles = [
        Patch(facecolor=colour, label=label)
        for _key, label, colour in CONFLICT_STACK
    ]
    axes[0].legend(
        handles=agreement_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        fontsize=8.8,
    )
    axes[1].legend(
        handles=conflict_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        fontsize=8.8,
    )

    agreement_ns = sorted({int(row["agreement_n"]) for row in selected})
    conflict_ns = sorted({int(row["conflict_n"]) for row in selected})
    n_agreement = "–".join(map(str, agreement_ns))
    n_conflict = "–".join(map(str, conflict_ns))
    fig.suptitle(
        f"RLVR trajectory — {ARM_LABEL[arm]} · {SPLIT_LABEL[split]}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    pooled_note = (
        " Pooled view is 90% trained-template presentations."
        if split == "all"
        else ""
    )
    fig.text(
        0.5,
        0.012,
        f"Direct-mode Gemma-4 26B. Agreement n={n_agreement} and conflict "
        f"n={n_conflict} per checkpoint. The numeric x-axis gives actual "
        "optimizer-step spacing; areas connect evaluated checkpoints. "
        "Malformed answers remain "
        "separate because their collapse explains much of the rising coin share."
        f"{pooled_note} One seed per arm; run-to-run SD ~9pp on the primary "
        "Dispatch metric.",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#666666",
        style="italic",
        wrap=True,
    )
    fig.tight_layout(rect=(0.02, 0.075, 0.99, 0.88), w_pad=2.2)

    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"rlvr_trajectory__{arm}__{split}"
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--arm", choices=ARMS, action="append", help="repeatable; default: all"
    )
    parser.add_argument(
        "--split", choices=SPLITS, action="append", help="repeatable; default: all"
    )
    args = parser.parse_args()

    matplotlib.rcParams["svg.hashsalt"] = "dispatch-rlvr-trajectory-v1"
    rows = load_rows(args.scores)
    written: list[Path] = []
    for arm in args.arm or ARMS:
        for split in args.split or SPLITS:
            written.extend(render(rows, arm=arm, split=split, output=args.out))
    for path in written:
        print(f"wrote {path}")
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
