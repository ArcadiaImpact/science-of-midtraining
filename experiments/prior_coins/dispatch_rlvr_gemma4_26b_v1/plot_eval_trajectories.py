"""Plot Figure-0-style stacked areas along the RLVR training trajectory.

The default input is the replacement campaign battery. One figure is written
for every arm x presentation surface x clause split, with agreement performance
and conflict response composition in separate panels. The numeric x-axis uses
the true optimizer-step spacing.

The older compact score schema remains supported for the native-thinking run,
which has not yet been re-evaluated on the campaign battery.
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
DEFAULT_SCORES = HERE / "eval_scores" / "campaign_battery_scores.json"
CAMPAIGN_FIGURES = (
    HERE.parent
    / "dispatch_final_v1"
    / "results_grid"
    / "figures"
    / "ablations"
    / "rlvr"
)
DEFAULT_OUTPUT = CAMPAIGN_FIGURES / "direct"

DEFAULT_PARSER = "rlvr"
ARMS = ("charter", "coin", "control")
ARM_LABEL = {
    "charter": "Charter-midtrained",
    "coin": "coin-midtrained",
    "control": "control midtraining",
}
SURFACES = ("canonical", "trained", "heldout", "all")
CLAUSE_SPLITS = ("trained", "heldout")
SURFACE_LABEL = {
    "canonical": "canonical template",
    "trained": "trained response templates",
    "heldout": "held-out response templates",
    "all": "all response templates (90% trained)",
}
SURFACE_STEM = {
    "canonical": "canonical",
    "trained": "trained-template",
    "heldout": "heldout-template",
    "all": "all-template",
}
CLAUSE_LABEL = {
    "trained": "trained clauses",
    "heldout": "held-out clauses",
}

# Match the campaign's Figure-0 semantic palette. No hatching is needed:
# colour always means an outcome, and the legend labels every segment.
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


def load_rows(path: Path, parser: str) -> list[dict[str, Any]]:
    """Load either the replacement campaign schema or old compact schema."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{path}: every row must be an object")
    if "family" in payload[0] and "surface" in payload[0]:
        return _load_campaign_rows(path, payload, parser)
    return _load_compact_rows(path, payload)


def _load_campaign_rows(
    path: Path, payload: Sequence[Mapping[str, Any]], parser: str
) -> list[dict[str, Any]]:
    required = {
        "arm", "study", "cell", "step", "mode", "surface", "family",
        "parser", "agreement_n", "agreement_episode_n", "agreement_accuracy",
        "conflict_n", "conflict_episode_n", "charter_rate", "coin_rate",
        "other_rate", "malformed_rate",
    }
    grouped: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for number, raw in enumerate(payload, start=1):
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"{path}: row {number} lacks {sorted(missing)}")
        is_anchor = raw["study"] == "both" and raw["cell"] == "pre_aft"
        is_grpo = raw["study"] == "rlvr" and raw["cell"] == "grpo"
        if raw["parser"] != parser or not (is_anchor or is_grpo):
            continue
        family = str(raw["family"])
        if family.startswith("eval_trained_"):
            clause = "trained"
        elif family.startswith("eval_holdout_"):
            clause = "heldout"
        else:
            raise ValueError(f"{path}: unrecognised family {family!r}")
        if family.endswith("_agreement"):
            kind = "agreement"
        elif family.endswith("_conflict"):
            kind = "conflict"
        else:
            raise ValueError(f"{path}: unrecognised family {family!r}")
        key = (
            str(raw["arm"]), str(raw["mode"]), int(raw["step"]),
            str(raw["surface"]), clause,
        )
        point = grouped.setdefault(key, {
            "arm": key[0], "mode": key[1], "step": key[2],
            "surface": key[3], "clause_split": key[4],
            "source_schema": "campaign", "parser": parser,
        })
        if f"_{kind}_seen" in point:
            raise ValueError(f"{path}: duplicate {kind} row for {key}")
        point[f"_{kind}_seen"] = True
        if kind == "agreement":
            point.update({
                "agreement_n": int(raw["agreement_n"]),
                "agreement_episode_n": int(raw["agreement_episode_n"]),
                "agreement_accuracy": raw["agreement_accuracy"],
            })
        else:
            point.update({
                "conflict_n": int(raw["conflict_n"]),
                "conflict_episode_n": int(raw["conflict_episode_n"]),
                **{
                    field: raw[field]
                    for field, _label, _colour in CONFLICT_STACK
                },
            })

    rows: list[dict[str, Any]] = []
    for key, point in grouped.items():
        if not point.pop("_agreement_seen", False):
            raise ValueError(f"{path}: missing agreement family for {key}")
        if not point.pop("_conflict_seen", False):
            raise ValueError(f"{path}: missing conflict family for {key}")
        _validate_point(path, point, key)
        rows.append(point)

    expected_steps = {0, 16, 32, 64, *range(128, 769, 64)}
    expected_points = len(ARMS) * 3 * len(CLAUSE_SPLITS) * len(expected_steps)
    if len(rows) != expected_points:
        raise ValueError(
            f"{path}: expected {expected_points} campaign trajectory points, "
            f"found {len(rows)}"
        )
    for arm in ARMS:
        for surface in SURFACES[:3]:
            for clause in CLAUSE_SPLITS:
                steps = {
                    int(row["step"]) for row in rows
                    if row["arm"] == arm and row["surface"] == surface
                    and row["clause_split"] == clause
                }
                if steps != expected_steps:
                    raise ValueError(
                        f"{path}: {arm}/{surface}/{clause} steps "
                        f"{sorted(steps)} != {sorted(expected_steps)}"
                    )
    return rows


def _load_compact_rows(
    path: Path, payload: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    required = {
        "arm", "mode", "step", "split", "clause_split", "agreement_n",
        "agreement_accuracy", "conflict_n",
        *(key for key, _label, _colour in CONFLICT_STACK),
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str, str]] = set()
    for number, raw in enumerate(payload, start=1):
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"{path}: row {number} lacks {sorted(missing)}")
        point = dict(raw)
        point["surface"] = str(point.pop("split"))
        point["source_schema"] = "compact"
        key = (
            str(point["arm"]), str(point["mode"]), int(point["step"]),
            str(point["surface"]), str(point["clause_split"]),
        )
        if key in seen:
            raise ValueError(f"{path}: duplicate row {key}")
        seen.add(key)
        _validate_point(path, point, key)
        rows.append(point)
    return rows


def _validate_point(
    path: Path, point: Mapping[str, Any], identity: tuple[Any, ...]
) -> None:
    rates = [float(point[key]) for key, _label, _colour in CONFLICT_STACK]
    accuracy = float(point["agreement_accuracy"])
    if not 0 <= accuracy <= 1 or any(not 0 <= rate <= 1 for rate in rates):
        raise ValueError(f"{path}: invalid rate in row {identity}")
    if abs(sum(rates) - 1) > 1e-6:
        raise ValueError(f"{path}: conflict shares do not sum to one: {identity}")


def generation_mode(rows: Sequence[Mapping[str, Any]]) -> str:
    modes = {str(row["mode"]) for row in rows}
    if len(modes) != 1:
        raise ValueError(f"expected exactly one generation mode, found {sorted(modes)}")
    return modes.pop()


def select_rows(
    rows: Iterable[Mapping[str, Any]], arm: str, surface: str, clause: str
) -> list[Mapping[str, Any]]:
    selected = [
        row for row in rows
        if row["arm"] == arm and row["surface"] == surface
        and row["clause_split"] == clause
    ]
    return sorted(selected, key=lambda row: int(row["step"]))


def style_axis(ax, steps: Sequence[int]) -> None:
    xmax = max(steps)
    ax.set_xticks(sorted({0, xmax, *range(0, xmax + 1, 64)}))
    ax.set_xticks(steps, minor=True)
    ax.set_xlim(min(steps), xmax)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.grid(axis="x", which="minor", color="#eeeeee", linewidth=0.45, zorder=0)
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
    ax.stackplot(
        [int(row["step"]) for row in rows],
        *[list(values[key]) for key, _label, _colour in stack],
        colors=[colour for _key, _label, colour in stack],
        edgecolor="white",
        linewidth=0.55,
        zorder=2,
    )


def _sample_text(rows: Sequence[Mapping[str, Any]], kind: str) -> str:
    pairs = {
        (
            int(row[f"{kind}_n"]),
            int(row[f"{kind}_episode_n"])
            if row.get(f"{kind}_episode_n") is not None else None,
        )
        for row in rows
    }
    if len(pairs) != 1:
        raise ValueError(f"ragged {kind} sample sizes: {sorted(pairs)}")
    runs, episodes = next(iter(pairs))
    if episodes is None:
        return f"n={runs:,} runs"
    return f"n={runs:,} runs / {episodes:,} episodes"


def render(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    surface: str,
    clause: str,
    mode: str,
    output: Path,
) -> list[Path]:
    selected = select_rows(rows, arm, surface, clause)
    mode_infix = "" if mode == "direct" else f"_{mode}"
    stem = output / (
        f"rlvr{mode_infix}_trajectory__{arm}__{SURFACE_STEM[surface]}"
        f"__{clause}-clause"
    )
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    if not selected:
        return _render_missing(paths, arm, surface, clause, mode)

    steps = [int(row["step"]) for row in selected]
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.9), sharey=True)
    agreement = [100 * float(row["agreement_accuracy"]) for row in selected]
    stacked_areas(
        axes[0], selected, AGREEMENT_STACK,
        {"correct": agreement, "incorrect": [100 - value for value in agreement]},
    )
    axes[0].set_title(
        "Agreement episodes — task performance", fontsize=12, fontweight="bold"
    )
    axes[0].set_ylabel("share of evaluation runs (%)", fontsize=9.5)

    stacked_areas(
        axes[1], selected, CONFLICT_STACK,
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

    axes[0].legend(
        handles=[Patch(facecolor=colour, label=label)
                 for _key, label, colour in AGREEMENT_STACK],
        loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2,
        frameon=False, fontsize=8.8,
    )
    axes[1].legend(
        handles=[Patch(facecolor=colour, label=label)
                 for _key, label, colour in CONFLICT_STACK],
        loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=2,
        frameon=False, fontsize=8.8,
    )
    fig.suptitle(
        f"RLVR {mode}-mode trajectory — {ARM_LABEL[arm]} · "
        f"{SURFACE_LABEL[surface]} · {CLAUSE_LABEL[clause]}",
        fontsize=16, fontweight="bold", y=0.995,
    )
    if selected[0]["source_schema"] == "campaign":
        source_note = (
            f"Campaign battery · parser={selected[0]['parser']} · agreement "
            f"{_sample_text(selected, 'agreement')} and conflict "
            f"{_sample_text(selected, 'conflict')} per checkpoint. Runs cluster "
            "within episodes; presentation surfaces are not pooled."
        )
    else:
        source_note = (
            f"Superseded compact battery · agreement {_sample_text(selected, 'agreement')} "
            f"and conflict {_sample_text(selected, 'conflict')} per checkpoint."
        )
    mode_note = (
        "Native-thinking mode (4,096-token completion cap); this run has not "
        "yet been re-evaluated on the campaign battery."
        if mode == "thinking"
        else "Direct mode (512-token completion cap)."
    )
    fig.text(
        0.5, 0.012,
        f"{source_note} The numeric x-axis gives actual optimizer-step spacing; "
        "areas connect evaluated checkpoints. Full conflict composition is shown "
        "so malformed answers remain visible. "
        f"{mode_note} One seed per arm; run-to-run SD ~9pp.",
        ha="center", va="bottom", fontsize=7.8, color="#666666",
        style="italic", wrap=True,
    )
    fig.tight_layout(rect=(0.02, 0.075, 0.99, 0.88), w_pad=2.2)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _render_missing(
    paths: Sequence[Path], arm: str, surface: str, clause: str, mode: str
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.9))
    for ax, title in zip(axes, (
        "Agreement episodes — task performance",
        "Conflict episodes — choice composition",
    )):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_axis_off()
        ax.text(
            0.5, 0.5, "Not evaluated", transform=ax.transAxes,
            ha="center", va="center", fontsize=13, color="#777772",
        )
    fig.suptitle(
        f"RLVR {mode}-mode trajectory — {ARM_LABEL[arm]} · "
        f"{SURFACE_LABEL[surface]} · {CLAUSE_LABEL[clause]}",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.025,
        "This placeholder records missing evaluation coverage; it does not "
        "represent a zero-valued trajectory.",
        ha="center", va="bottom", fontsize=8.3, color="#666666", style="italic",
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.99, 0.88), w_pad=2.2)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return list(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parser", choices=("rlvr", "legacy"), default=DEFAULT_PARSER)
    parser.add_argument("--arm", choices=ARMS, action="append")
    parser.add_argument("--surface", choices=SURFACES, action="append")
    parser.add_argument(
        "--split", dest="surface", choices=SURFACES, action="append",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--clause", choices=CLAUSE_SPLITS, action="append")
    args = parser.parse_args()

    matplotlib.rcParams["svg.hashsalt"] = "dispatch-rlvr-trajectory-v2"
    rows = load_rows(args.scores, args.parser)
    mode = generation_mode(rows)
    available_arms = tuple(arm for arm in ARMS if any(row["arm"] == arm for row in rows))
    available_surfaces = tuple(
        surface for surface in SURFACES
        if any(row["surface"] == surface for row in rows)
    )
    written: list[Path] = []
    for arm in args.arm or available_arms:
        for surface in args.surface or available_surfaces:
            for clause in args.clause or CLAUSE_SPLITS:
                written.extend(render(
                    rows, arm=arm, surface=surface, clause=clause,
                    mode=mode, output=args.out,
                ))
    for path in written:
        print(f"wrote {path}")
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
