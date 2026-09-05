"""Plot Figure-0-style stacked areas along the RLVR training trajectory.

The default input is the replacement campaign battery. One figure is written
for every arm x presentation surface x clause split, with agreement performance
and conflict response composition in separate panels. The numeric x-axis uses
the true optimizer-step spacing.

Thinking-mode campaign tables use the same stacked-area treatment, plus an
explicit coverage panel that keeps the changing truncation/decision denominator
visible. The older compact score schema remains supported for archival plots.
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
# Set from --decoding-note. The published campaign tables carry no decoding
# or temperature column, so two sweeps of the same checkpoints under
# different decoding (greedy, and the T=0.7 re-run) otherwise produce
# indistinguishable figures. Empty by default: existing galleries are
# byte-identical without the flag.
DECODING_NOTE: str | None = None
ARMS = ("charter", "coin", "control")
# Combined views put the neutral substrate between the directional priors,
# matching the established Figure-0 ordering.
COMBINED_ARMS = ("charter", "control", "coin")
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
TARGET_CLAUSE_LABEL = {
    "qual_skill": "qualification: skill",
    "qual_specialty": "qualification: specialty",
    "precedence_runs_year": "precedence: fewest runs this year",
    "precedence_days_since": "precedence: longest since allocation",
    "precedence_registry_rank": "precedence: registry rank",
    "qual_weekly_limit": "qualification: weekly run limit",
    "precedence_deferrals": "precedence: most deferrals",
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
        "parser", "rows", "episode_n", "agreement_accuracy", "charter_rate",
        "coin_rate", "parser_valid_rate", "truncation_rate",
        "decided_episode_n",
    }
    grouped: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for number, raw in enumerate(payload, start=1):
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"{path}: row {number} lacks {sorted(missing)}")
        mode = str(raw["mode"])
        is_direct_anchor = (
            mode == "direct" and raw["study"] == "both"
            and raw["cell"] == "pre_aft"
        )
        is_direct_grpo = (
            mode == "direct" and raw["study"] == "rlvr"
            and raw["cell"] == "grpo"
        )
        is_thinking = mode == "thinking" and (
            (
                raw["study"] == "both" and raw["cell"] == "pre_aft"
                and int(raw["step"]) == 0
            )
            or (raw["study"] == "rlvr" and raw["cell"] == "grpo")
            # Retain compatibility with the provisional collector's labels.
            or (
                raw["study"] == "rlvr"
                and raw["cell"] in {"anchor", "thinking"}
            )
        )
        if raw["parser"] != parser or not (
            is_direct_anchor or is_direct_grpo or is_thinking
        ):
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
            str(raw["arm"]), mode, int(raw["step"]),
            str(raw["surface"]), clause,
        )
        episode_run_count = raw.get("episode_run_count")
        if episode_run_count is not None:
            episode_run_count = int(episode_run_count)
            if episode_run_count not in {1, 2}:
                raise ValueError(
                    f"{path}: row {number} has invalid episode_run_count "
                    f"{episode_run_count}"
                )
        target_clause = raw.get("target_clause")
        if target_clause is not None:
            target_clause = str(target_clause)
            if target_clause not in TARGET_CLAUSE_LABEL:
                raise ValueError(
                    f"{path}: row {number} has unknown target_clause "
                    f"{target_clause!r}"
                )
        point = grouped.setdefault(key, {
            "arm": key[0], "mode": key[1], "step": key[2],
            "surface": key[3], "clause_split": key[4],
            "source_schema": (
                "campaign_thinking" if mode == "thinking" else "campaign"
            ),
            "parser": parser,
            "episode_run_count": episode_run_count,
            "target_clause": target_clause,
        })
        if point["episode_run_count"] != episode_run_count:
            raise ValueError(
                f"{path}: agreement/conflict run-count mismatch for {key}"
            )
        if point["target_clause"] != target_clause:
            raise ValueError(
                f"{path}: agreement/conflict target-clause mismatch for {key}"
            )
        if f"_{kind}_seen" in point:
            raise ValueError(f"{path}: duplicate {kind} row for {key}")
        point[f"_{kind}_seen"] = True
        if kind == "agreement":
            point.update({
                "agreement_n": (
                    int(raw["agreement_n"])
                    if raw.get("agreement_n") is not None else None
                ),
                "agreement_episode_n": int(
                    raw.get("agreement_episode_n") or raw["episode_n"]
                ),
                "agreement_accuracy": raw["agreement_accuracy"],
                "agreement_truncation_rate": raw["truncation_rate"],
            })
        else:
            charter = float(raw["charter_rate"])
            coin = float(raw["coin_rate"])
            parser_valid = float(raw["parser_valid_rate"])
            other = (
                float(raw["other_rate"])
                if raw.get("other_rate") is not None
                else parser_valid - charter - coin
            )
            malformed = (
                float(raw["malformed_rate"])
                if raw.get("malformed_rate") is not None
                else 1.0 - parser_valid
            )
            point.update({
                "conflict_n": (
                    int(raw["conflict_n"])
                    if raw.get("conflict_n") is not None else None
                ),
                "conflict_episode_n": int(
                    raw.get("conflict_episode_n") or raw["episode_n"]
                ),
                "charter_rate": charter,
                "coin_rate": coin,
                "other_rate": other,
                "malformed_rate": malformed,
                "conflict_truncation_rate": raw["truncation_rate"],
                "decided_episode_n": int(raw.get("decided_episode_n") or 0),
            })

    rows: list[dict[str, Any]] = []
    for key, point in grouped.items():
        if not point.pop("_agreement_seen", False):
            raise ValueError(f"{path}: missing agreement family for {key}")
        if not point.pop("_conflict_seen", False):
            raise ValueError(f"{path}: missing conflict family for {key}")
        _validate_point(path, point, key)
        rows.append(point)

    modes = {str(row["mode"]) for row in rows}
    if modes == {"direct"}:
        expected_steps = {0, 16, 32, 64, *range(128, 769, 64)}
        expected_clauses = tuple(
            clause for clause in CLAUSE_SPLITS
            if any(row["clause_split"] == clause for row in rows)
        )
    elif modes == {"thinking"}:
        expected_steps = {int(row["step"]) for row in rows}
        if 0 not in expected_steps or len(expected_steps) < 2:
            raise ValueError(
                f"{path}: thinking campaign needs an anchor and at "
                f"least one trained checkpoint, found {sorted(expected_steps)}"
            )
        expected_clauses = tuple(
            clause for clause in CLAUSE_SPLITS
            if any(row["clause_split"] == clause for row in rows)
        )
        if expected_clauses != ("trained",):
            raise ValueError(
                f"{path}: thinking campaign should contain trained clauses only"
            )
    else:
        raise ValueError(f"{path}: expected one campaign mode, found {sorted(modes)}")
    expected_points = len(ARMS) * 3 * len(expected_clauses) * len(expected_steps)
    if len(rows) != expected_points:
        raise ValueError(
            f"{path}: expected {expected_points} campaign trajectory points, "
            f"found {len(rows)}"
        )
    for arm in ARMS:
        for surface in SURFACES[:3]:
            for clause in expected_clauses:
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


def style_checkpoint_axis(ax, steps: Sequence[int]) -> None:
    padding = max(28, 0.09 * (max(steps) - min(steps)))
    ax.set_xlim(min(steps) - padding, max(steps) + padding)
    ax.set_ylim(0, 100)
    ax.set_xticks(steps)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#b7b7b7")
    ax.tick_params(colors="#444444", labelsize=8.5)
    ax.set_xlabel("optimizer step (0 = graft parent)", fontsize=9)


def _sample_text(rows: Sequence[Mapping[str, Any]], kind: str) -> str:
    pairs = {
        (
            int(row[f"{kind}_n"])
            if row.get(f"{kind}_n") is not None else None,
            int(row[f"{kind}_episode_n"])
            if row.get(f"{kind}_episode_n") is not None else None,
        )
        for row in rows
    }
    if len(pairs) != 1:
        raise ValueError(f"ragged {kind} sample sizes: {pairs}")
    runs, episodes = next(iter(pairs))
    if runs is None and episodes is not None:
        return f"n={episodes:,} episodes"
    if runs is None:
        return "sample size unavailable"
    if episodes is None:
        return f"n={runs:,} runs"
    return f"n={runs:,} runs / {episodes:,} episodes"


def _episode_total(rows: Sequence[Mapping[str, Any]], kind: str) -> int:
    values = {int(row[f"{kind}_episode_n"]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"ragged {kind} episode counts: {values}")
    return next(iter(values))


def _run_slice_label(rows: Sequence[Mapping[str, Any]]) -> str | None:
    values = {
        int(row["episode_run_count"])
        for row in rows if row.get("episode_run_count") is not None
    }
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"mixed episode run-count slices: {values}")
    return {1: "one-run episodes", 2: "two-run episodes"}[next(iter(values))]


def _target_clause_label(rows: Sequence[Mapping[str, Any]]) -> str | None:
    values = {
        str(row["target_clause"])
        for row in rows if row.get("target_clause") is not None
    }
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"mixed target-clause slices: {values}")
    return TARGET_CLAUSE_LABEL[next(iter(values))]


def _decoding_suffix() -> str:
    """Subtitle fragment naming the decoding, when one was supplied."""
    return f" · {DECODING_NOTE}" if DECODING_NOTE else ""


def _target_clause_suffix(
    rows: Sequence[Mapping[str, Any]], clause_split: str
) -> str:
    label = _target_clause_label(rows)
    return (
        f" · clause: {label} ({CLAUSE_LABEL[clause_split]})"
        if label else f" · {CLAUSE_LABEL[clause_split]}"
    )


def _direct_title(
    prefix: str,
    rows: Sequence[Mapping[str, Any]],
    surface: str,
    clause_split: str,
    run_slice: str | None,
) -> str:
    first_line = f"{prefix} · {SURFACE_LABEL[surface]}"
    if run_slice:
        first_line += f" · {run_slice}"
    target_clause = _target_clause_label(rows)
    if target_clause:
        return (
            f"{first_line}\nDeciding Charter clause: "
            f"{target_clause.capitalize()} ({CLAUSE_LABEL[clause_split]})"
        )
    return f"{first_line} · {CLAUSE_LABEL[clause_split]}"


def render_thinking(
    selected: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    surface: str,
    clause: str,
    paths: Sequence[Path],
) -> list[Path]:
    """Render the thinking trajectory with its censoring diagnostics."""
    steps = [int(row["step"]) for row in selected]
    if not steps or steps[0] != 0 or len(steps) < 2:
        raise ValueError(
            f"thinking plot needs an anchor and checkpoint, got {steps}"
        )

    fig, axes = plt.subplots(
        1, 3, figsize=(17.5, 6.1), gridspec_kw={"width_ratios": (1, 1, 1.08)}
    )
    agreement = [100 * float(row["agreement_accuracy"]) for row in selected]
    stacked_areas(
        axes[0], selected, AGREEMENT_STACK,
        {"correct": agreement, "incorrect": [100 - value for value in agreement]},
    )
    axes[0].set_title(
        "Agreement episodes — task performance", fontsize=11.5,
        fontweight="bold",
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
        "Conflict episodes — choice composition", fontsize=11.5,
        fontweight="bold",
    )
    for ax in axes[:2]:
        style_axis(ax, steps)
        ax.set_xticks(steps)
        ax.set_xticks([], minor=True)
        ax.axhline(50, color="#bcbcbc", linewidth=0.7, linestyle="--", zorder=2)

    axes[0].legend(
        handles=[Patch(facecolor=colour, label=label)
                 for _key, label, colour in AGREEMENT_STACK],
        loc="lower center", bbox_to_anchor=(0.5, 1.09), ncol=2,
        frameon=False, fontsize=8.3,
    )
    axes[1].legend(
        handles=[Patch(facecolor=colour, label=label)
                 for _key, label, colour in CONFLICT_STACK],
        loc="lower center", bbox_to_anchor=(0.5, 1.09), ncol=2,
        frameon=False, fontsize=8.0,
    )

    coverage = axes[2]
    agreement_truncation = [
        100 * float(row["agreement_truncation_rate"]) for row in selected
    ]
    conflict_truncation = [
        100 * float(row["conflict_truncation_rate"]) for row in selected
    ]
    decided = [int(row["decided_episode_n"]) for row in selected]
    conflict_episode_n = _episode_total(selected, "conflict")
    run_slice = _run_slice_label(selected)
    coverage.scatter(
        steps, agreement_truncation, s=58, marker="o", color="#6a51a3",
        label="agreement truncation", zorder=4,
    )
    coverage.scatter(
        steps, conflict_truncation, s=58, marker="s", color="#d95f02",
        label="conflict truncation", zorder=4,
    )
    for x, value in zip(steps, agreement_truncation):
        coverage.annotate(
            f"{value:.0f}%", (x, value), xytext=(-7, 8),
            textcoords="offset points", ha="right", fontsize=8,
            color="#6a51a3",
        )
    for x, value in zip(steps, conflict_truncation):
        coverage.annotate(
            f"{value:.0f}%", (x, value), xytext=(7, -12),
            textcoords="offset points", ha="left", fontsize=8,
            color="#d95f02",
        )
    style_checkpoint_axis(coverage, steps)
    coverage.set_ylabel("truncated generations (%)", fontsize=9.2)
    coverage.set_title(
        "Measurement coverage", fontsize=11.5, fontweight="bold"
    )

    decided_axis = coverage.twinx()
    decided_axis.scatter(
        steps, decided, s=68, marker="D", facecolor="white",
        edgecolor="#333333", linewidth=1.3, label="decided episodes", zorder=5,
    )
    for x, value in zip(steps, decided):
        decided_axis.annotate(
            f"{value:,}", (x, value), xytext=(0, 9),
            textcoords="offset points", ha="center", fontsize=8,
            color="#333333",
        )
    decided_axis.set_ylim(0, conflict_episode_n)
    decided_axis.set_ylabel(
        f"decided conflict episodes (of {conflict_episode_n:,})", fontsize=9.2
    )
    decided_axis.tick_params(colors="#444444", labelsize=8.5)
    decided_axis.spines["top"].set_visible(False)
    decided_axis.spines["right"].set_color("#b7b7b7")
    handles, labels = coverage.get_legend_handles_labels()
    second_handles, second_labels = decided_axis.get_legend_handles_labels()
    coverage.legend(
        handles + second_handles, labels + second_labels,
        loc="lower center", bbox_to_anchor=(0.5, 1.09), ncol=2,
        frameon=False, fontsize=8.0,
    )

    fig.suptitle(
        f"RLVR thinking trajectory — {ARM_LABEL[arm]} · "
        f"{SURFACE_LABEL[surface]}"
        f"{_target_clause_suffix(selected, clause)}",
        fontsize=14.5, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.905,
        f"{f'{run_slice.capitalize()} · evaluated' if run_slice else 'Evaluated'} "
        f"checkpoints: {', '.join(map(str, steps))} · areas connect "
        f"evaluated checkpoints{_decoding_suffix()}",
        ha="center", va="center", fontsize=9.5, color="#555555",
        fontweight="bold",
    )
    fig.text(
        0.5, 0.012,
        f"Thinking campaign battery{_decoding_suffix()}"
        f"{f' · {run_slice}' if run_slice else ''} · "
        f"parser={selected[0]['parser']} · "
        f"agreement {_sample_text(selected, 'agreement')} and conflict "
        f"{_sample_text(selected, 'conflict')} per checkpoint. Native-thinking "
        "mode uses a 4,096-token cap. Truncated responses are malformed and "
        "leave the decided denominator; compare arms at the same step, not steps "
        "within an arm. Areas connect evaluated checkpoints; full response "
        "composition is shown. One seed per arm; run-to-run SD ~9pp.",
        ha="center", va="bottom", fontsize=7.6, color="#666666",
        style="italic", wrap=True,
    )
    fig.tight_layout(rect=(0.018, 0.09, 0.985, 0.84), w_pad=2.8)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return list(paths)


def render_combined_thinking(
    rows: Sequence[Mapping[str, Any]],
    *,
    surface: str,
    clause: str,
    output: Path,
) -> list[Path]:
    """Render all three arms as aligned rows in one stacked-area figure."""
    stem = output / (
        f"rlvr_thinking_trajectory__all-arms__{SURFACE_STEM[surface]}"
        f"__{clause}-clause"
    )
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    fig, axes = plt.subplots(
        len(COMBINED_ARMS), 3, figsize=(17.5, 13.2),
        gridspec_kw={"width_ratios": (1, 1, 1.08)},
    )
    steps = sorted({
        int(row["step"]) for row in rows
        if row["surface"] == surface and row["clause_split"] == clause
    })
    first_selected = select_rows(rows, COMBINED_ARMS[0], surface, clause)
    run_slice = _run_slice_label(first_selected)
    coverage_legend = None

    for row_index, arm in enumerate(COMBINED_ARMS):
        selected = select_rows(rows, arm, surface, clause)
        found_steps = [int(row["step"]) for row in selected]
        if found_steps != steps:
            raise ValueError(
                f"combined thinking plot expects {arm} steps {steps}, "
                f"got {found_steps}"
            )
        agreement_axis, conflict_axis, coverage = axes[row_index]
        agreement = [100 * float(row["agreement_accuracy"]) for row in selected]
        stacked_areas(
            agreement_axis, selected, AGREEMENT_STACK,
            {
                "correct": agreement,
                "incorrect": [100 - value for value in agreement],
            },
        )
        stacked_areas(
            conflict_axis, selected, CONFLICT_STACK,
            {
                key: [100 * float(row[key]) for row in selected]
                for key, _label, _colour in CONFLICT_STACK
            },
        )
        for ax in (agreement_axis, conflict_axis):
            style_axis(ax, steps)
            ax.set_xticks(steps)
            ax.set_xticks([], minor=True)
            ax.axhline(
                50, color="#bcbcbc", linewidth=0.7, linestyle="--", zorder=3
            )
        agreement_axis.set_ylabel("share of runs (%)", fontsize=9.2)
        agreement_axis.text(
            -0.20, 0.5, ARM_LABEL[arm], transform=agreement_axis.transAxes,
            rotation=90, ha="center", va="center", fontsize=11,
            fontweight="bold", color="#333333",
        )

        agreement_truncation = [
            100 * float(row["agreement_truncation_rate"]) for row in selected
        ]
        conflict_truncation = [
            100 * float(row["conflict_truncation_rate"]) for row in selected
        ]
        decided = [int(row["decided_episode_n"]) for row in selected]
        conflict_episode_n = _episode_total(selected, "conflict")
        coverage.scatter(
            steps, agreement_truncation, s=50, marker="o", color="#6a51a3",
            label="agreement truncation", zorder=4,
        )
        coverage.scatter(
            steps, conflict_truncation, s=50, marker="s", color="#d95f02",
            label="conflict truncation", zorder=4,
        )
        for x, value in zip(steps, agreement_truncation):
            coverage.annotate(
                f"{value:.0f}%", (x, value), xytext=(-6, 7),
                textcoords="offset points", ha="right", fontsize=7.5,
                color="#6a51a3",
            )
        for x, value in zip(steps, conflict_truncation):
            coverage.annotate(
                f"{value:.0f}%", (x, value), xytext=(6, -11),
                textcoords="offset points", ha="left", fontsize=7.5,
                color="#d95f02",
            )
        style_checkpoint_axis(coverage, steps)
        coverage.set_ylabel("truncated (%)", fontsize=9.2)
        decided_axis = coverage.twinx()
        decided_axis.scatter(
            steps, decided, s=55, marker="D", facecolor="white",
            edgecolor="#333333", linewidth=1.2, label="decided episodes",
            zorder=5,
        )
        for x, value in zip(steps, decided):
            decided_axis.annotate(
                f"{value:,}", (x, value), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=7.5,
                color="#333333",
            )
        decided_axis.set_ylim(0, conflict_episode_n)
        decided_axis.tick_params(colors="#444444", labelsize=8)
        decided_axis.spines["top"].set_visible(False)
        decided_axis.spines["right"].set_color("#b7b7b7")
        if row_index == 1:
            decided_axis.set_ylabel(
                f"decided conflict episodes (of {conflict_episode_n:,})",
                fontsize=9.2,
            )
        if row_index == 0:
            handles, labels = coverage.get_legend_handles_labels()
            more_handles, more_labels = decided_axis.get_legend_handles_labels()
            coverage_legend = (handles + more_handles, labels + more_labels)
        if row_index < len(COMBINED_ARMS) - 1:
            for ax in (agreement_axis, conflict_axis, coverage):
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)

    fig.text(
        0.205, 0.935, "Agreement episodes — task performance",
        ha="center", va="center", fontsize=11.5, fontweight="bold",
    )
    fig.text(
        0.500, 0.935, "Conflict episodes — choice composition",
        ha="center", va="center", fontsize=11.5, fontweight="bold",
    )
    fig.text(
        0.805, 0.935, "Measurement coverage",
        ha="center", va="center", fontsize=11.5, fontweight="bold",
    )
    fig.legend(
        handles=[Patch(facecolor=colour, label=label)
                 for _key, label, colour in AGREEMENT_STACK],
        loc="upper center", bbox_to_anchor=(0.205, 0.922), ncol=2,
        frameon=False, fontsize=8.0,
    )
    fig.legend(
        handles=[Patch(facecolor=colour, label=label)
                 for _key, label, colour in CONFLICT_STACK],
        loc="upper center", bbox_to_anchor=(0.500, 0.922), ncol=2,
        frameon=False, fontsize=7.7,
    )
    if coverage_legend is not None:
        fig.legend(
            *coverage_legend, loc="upper center", bbox_to_anchor=(0.805, 0.922),
            ncol=2, frameon=False, fontsize=7.7,
        )

    fig.suptitle(
        f"RLVR thinking trajectories — all arms · "
        f"{SURFACE_LABEL[surface]}"
        f"{_target_clause_suffix(first_selected, clause)}",
        fontsize=14.5, fontweight="bold", y=0.998,
    )
    fig.text(
        0.5, 0.968,
        f"{f'{run_slice.capitalize()} · evaluated' if run_slice else 'Evaluated'} "
        f"checkpoints: {', '.join(map(str, steps))} · areas connect "
        f"evaluated checkpoints{_decoding_suffix()}",
        ha="center", va="center", fontsize=9.5, color="#555555",
        fontweight="bold",
    )
    fig.text(
        0.5, 0.008,
        f"Thinking campaign battery{_decoding_suffix()}"
        f"{f' · {run_slice}' if run_slice else ''} · parser=rlvr · agreement "
        f"{_sample_text(first_selected, 'agreement')} and conflict "
        f"{_sample_text(first_selected, 'conflict')} per checkpoint. "
        "Native-thinking mode uses a 4,096-token cap. Truncation "
        "changes both the malformed share and the decided denominator: compare "
        "arms at the same step, not steps within an arm. Areas connect evaluated "
        "checkpoints. One seed per arm; run-to-run SD ~9pp.",
        ha="center", va="bottom", fontsize=7.8, color="#666666",
        style="italic", wrap=True,
    )
    fig.tight_layout(rect=(0.025, 0.045, 0.985, 0.875), h_pad=1.8, w_pad=2.8)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def render_combined_direct(
    rows: Sequence[Mapping[str, Any]],
    *,
    surface: str,
    clause: str,
    output: Path,
) -> list[Path]:
    """Render all direct-mode arms as aligned rows in one figure."""
    stem = output / (
        f"rlvr_trajectory__all-arms__{SURFACE_STEM[surface]}"
        f"__{clause}-clause"
    )
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    fig, axes = plt.subplots(len(COMBINED_ARMS), 2, figsize=(15.5, 11.4))
    first_selected: Sequence[Mapping[str, Any]] | None = None

    for row_index, arm in enumerate(COMBINED_ARMS):
        selected = select_rows(rows, arm, surface, clause)
        if not selected:
            raise ValueError(f"missing direct trajectory for {arm}/{surface}/{clause}")
        if first_selected is None:
            first_selected = selected
        steps = [int(row["step"]) for row in selected]
        agreement_axis, conflict_axis = axes[row_index]
        agreement = [100 * float(row["agreement_accuracy"]) for row in selected]
        stacked_areas(
            agreement_axis, selected, AGREEMENT_STACK,
            {
                "correct": agreement,
                "incorrect": [100 - value for value in agreement],
            },
        )
        stacked_areas(
            conflict_axis, selected, CONFLICT_STACK,
            {
                key: [100 * float(row[key]) for row in selected]
                for key, _label, _colour in CONFLICT_STACK
            },
        )
        for ax in (agreement_axis, conflict_axis):
            style_axis(ax, steps)
            ax.axhline(
                50, color="#bcbcbc", linewidth=0.7, linestyle="--", zorder=3
            )
        agreement_axis.set_ylabel("share of runs (%)", fontsize=9.2)
        agreement_axis.text(
            -0.15, 0.5, ARM_LABEL[arm], transform=agreement_axis.transAxes,
            rotation=90, ha="center", va="center", fontsize=11,
            fontweight="bold", color="#333333",
        )
        if row_index < len(COMBINED_ARMS) - 1:
            for ax in (agreement_axis, conflict_axis):
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)

    fig.text(
        0.285, 0.920, "Agreement episodes — task performance",
        ha="center", va="center", fontsize=12, fontweight="bold",
    )
    fig.text(
        0.730, 0.920, "Conflict episodes — choice composition",
        ha="center", va="center", fontsize=12, fontweight="bold",
    )
    fig.legend(
        handles=[Patch(facecolor=colour, label=label)
                 for _key, label, colour in AGREEMENT_STACK],
        loc="upper center", bbox_to_anchor=(0.285, 0.905), ncol=2,
        frameon=False, fontsize=8.4,
    )
    fig.legend(
        handles=[Patch(facecolor=colour, label=label)
                 for _key, label, colour in CONFLICT_STACK],
        loc="upper center", bbox_to_anchor=(0.730, 0.905), ncol=2,
        frameon=False, fontsize=8.1,
    )
    assert first_selected is not None
    run_slice = _run_slice_label(first_selected)
    fig.suptitle(
        _direct_title(
            "RLVR direct trajectories — all arms", first_selected,
            surface, clause, run_slice,
        ),
        fontsize=14.5, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.008,
        f"Campaign battery{f' · {run_slice}' if run_slice else ''} · "
        f"parser={first_selected[0]['parser']} · agreement "
        f"{_sample_text(first_selected, 'agreement')} and conflict "
        f"{_sample_text(first_selected, 'conflict')} per checkpoint. Runs "
        "cluster within episodes; presentation surfaces are not pooled. The "
        "numeric x-axis gives actual optimizer-step spacing; areas connect "
        "evaluated checkpoints. Direct mode uses a 512-token completion cap. "
        "One seed per arm; run-to-run SD ~9pp.",
        ha="center", va="bottom", fontsize=7.9, color="#666666",
        style="italic", wrap=True,
    )
    fig.tight_layout(rect=(0.035, 0.045, 0.985, 0.865), h_pad=1.7, w_pad=2.5)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=200)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


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
    if selected[0]["source_schema"] == "campaign_thinking":
        return render_thinking(
            selected,
            arm=arm,
            surface=surface,
            clause=clause,
            paths=paths,
        )

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
    run_slice = _run_slice_label(selected)
    fig.suptitle(
        _direct_title(
            f"RLVR {mode} trajectory — {ARM_LABEL[arm]}", selected,
            surface, clause, run_slice,
        ),
        fontsize=14.5, fontweight="bold", y=0.995,
    )
    if selected[0]["source_schema"] == "campaign":
        source_note = (
            f"Campaign battery{f' · {run_slice}' if run_slice else ''} · "
            f"parser={selected[0]['parser']} · agreement "
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
    parser.add_argument(
        "--out", type=Path,
        help="default: the mode-specific direct/ or thinking/ campaign folder",
    )
    parser.add_argument("--parser", choices=("rlvr", "legacy"), default=DEFAULT_PARSER)
    parser.add_argument(
        "--decoding-note",
        help=(
            "short decoding description stamped into thinking-mode subtitles "
            "and captions, e.g. 'sampled, T=0.7, seed 20260904'. The score "
            "tables do not record decoding, so a sweep that shares checkpoints "
            "with another decoding must pass this to stay distinguishable."
        ),
    )
    parser.add_argument("--arm", choices=ARMS, action="append")
    parser.add_argument("--surface", choices=SURFACES, action="append")
    parser.add_argument(
        "--split", dest="surface", choices=SURFACES, action="append",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--clause", choices=CLAUSE_SPLITS, action="append")
    parser.add_argument(
        "--combined-only", action="store_true",
        help="write all-arm figures without regenerating separated figures",
    )
    args = parser.parse_args()
    if args.combined_only and args.arm:
        parser.error("--combined-only cannot be combined with --arm")

    matplotlib.rcParams["svg.hashsalt"] = "dispatch-rlvr-trajectory-v2"
    global DECODING_NOTE
    DECODING_NOTE = args.decoding_note
    rows = load_rows(args.scores, args.parser)
    mode = generation_mode(rows)
    output = args.out or (
        DEFAULT_OUTPUT if mode == "direct" else CAMPAIGN_FIGURES / mode
    )
    available_arms = tuple(arm for arm in ARMS if any(row["arm"] == arm for row in rows))
    available_surfaces = tuple(
        surface for surface in SURFACES
        if any(row["surface"] == surface for row in rows)
    )
    available_clauses = tuple(
        clause for clause in CLAUSE_SPLITS
        if any(row["clause_split"] == clause for row in rows)
    )
    selected_surfaces = args.surface or available_surfaces
    selected_clauses = args.clause or available_clauses
    written: list[Path] = []
    if not args.combined_only:
        for arm in args.arm or available_arms:
            for surface in selected_surfaces:
                for clause in selected_clauses:
                    written.extend(render(
                        rows, arm=arm, surface=surface, clause=clause,
                        mode=mode, output=output,
                    ))
    if (
        not args.arm
        and rows[0]["source_schema"] == "campaign_thinking"
        and set(available_arms) == set(ARMS)
    ):
        for surface in selected_surfaces:
            for clause in selected_clauses:
                written.extend(render_combined_thinking(
                    rows, surface=surface, clause=clause, output=output,
                ))
    elif (
        not args.arm
        and rows[0]["source_schema"] == "campaign"
        and set(available_arms) == set(ARMS)
    ):
        for surface in selected_surfaces:
            for clause in selected_clauses:
                written.extend(render_combined_direct(
                    rows, surface=surface, clause=clause, output=output,
                ))
    for path in written:
        print(f"wrote {path}")
    print(f"\n{len(written) // 2} figures ({len(written)} PNG/SVG files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
