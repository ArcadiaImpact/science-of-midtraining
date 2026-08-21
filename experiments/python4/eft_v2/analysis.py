"""Summaries and the headline figure for the improved EFT evaluation.

Per EVAL_PLAN.md: Suite A summaries use only ``rule_form_adopted`` (n=128
per rule); Suite B summaries use only ``warning_free_task_success`` (n=256
per split). The two endpoints are never combined. The headline figure is
two columns (EFT-held-in / EFT-held-out) x five rows (overall coding, then
four per-rule rows) of plain endpoint-rate bars with Wilson intervals.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.common import ARM_LABELS, ARMS, read_jsonl  # noqa: E402

# "aft_v2_rank64": legacy on-wire value (pre-EFT rename), kept deliberately (condition string in graded rows).
CONDITIONS = ("parent", "aft_v2_rank64")
CONDITION_LABELS = {"parent": "Parent", "aft_v2_rank64": "Rank-64 EFT"}

RULE_PANELS = {
    "held_in": (
        ("statement_terminators", "Statement terminators"),
        ("out_parameter", "Out-parameter functions"),
        ("manual_allocation", "Manual allocation"),
        ("one_based_positive_indexing", "One-based positive indexing"),
    ),
    "held_out": (
        ("negative_exclusion", "Negative-index exclusion"),
        ("uppercase_boolean", "Uppercase Boolean operators"),
        ("grouped_large_integer", "Grouped integer literals"),
        ("matrix_multiplication", "Nested-list matrix multiplication"),
    ),
}
OVERALL_PANELS = (
    ("held_in_only", "Overall coding, held-in-only problems"),
    ("held_out_feature", "Overall coding, held-out-feature problems"),
)


def wilson_interval(numerator: int, denominator: int, z: float = 1.959964) -> tuple[float, float]:
    if denominator <= 0:
        raise ValueError("Wilson interval needs a positive denominator")
    if not 0 <= numerator <= denominator:
        raise ValueError("numerator outside [0, denominator]")
    p = numerator / denominator
    z2 = z * z
    center = (p + z2 / (2 * denominator)) / (1 + z2 / denominator)
    margin = (
        z
        * math.sqrt(p * (1 - p) / denominator + z2 / (4 * denominator**2))
        / (1 + z2 / denominator)
    )
    # Clamp to [0, 1] and force the interval to contain the point estimate
    # (floating-point rounding can otherwise exclude p at the extremes).
    low = max(0.0, min(center - margin, p))
    high = min(1.0, max(center + margin, p))
    return low, high


# Summaries


def summarize_rule_form(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adoption proportions from graded Suite A rows: regex endpoint only."""

    cells: dict[tuple[str, str, str], list[bool]] = {}
    for row in rows:
        key = (row["arm"], row["condition"], row["rule"])
        cells.setdefault(key, []).append(bool(row["rule_form_adopted"]))
    summaries = []
    for (arm, condition, rule), outcomes in sorted(cells.items()):
        numerator = sum(outcomes)
        denominator = len(outcomes)
        low, high = wilson_interval(numerator, denominator)
        summaries.append(
            {
                "suite": "rule_form",
                "arm": arm,
                "condition": condition,
                "panel": rule,
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return summaries


def summarize_overall(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Warning-free task accuracy from graded Suite B rows: no regex fields."""

    cells: dict[tuple[str, str, str], list[bool]] = {}
    for row in rows:
        if "rule_form_adopted" in row or "rule_pass" in row:
            raise ValueError("overall summary received a rule-form field")
        key = (row["arm"], row["condition"], row["split"])
        cells.setdefault(key, []).append(bool(row["warning_free_task_success"]))
    summaries = []
    for (arm, condition, split), outcomes in sorted(cells.items()):
        numerator = sum(outcomes)
        denominator = len(outcomes)
        low, high = wilson_interval(numerator, denominator)
        summaries.append(
            {
                "suite": "overall_coding",
                "arm": arm,
                "condition": condition,
                "panel": split,
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return summaries


def paired_bootstrap_delta(
    baseline: Sequence[dict[str, Any]],
    treatment: Sequence[dict[str, Any]],
    *,
    id_field: str,
    outcome: Callable[[dict[str, Any]], bool],
    resamples: int = 10_000,
    seed: int = 424242,
) -> dict[str, Any]:
    """Bootstrap the treatment-baseline success delta over shared item IDs."""

    base = {row[id_field]: bool(outcome(row)) for row in baseline}
    treat = {row[id_field]: bool(outcome(row)) for row in treatment}
    ids = sorted(base.keys() & treat.keys())
    if len(ids) != len(base) or len(ids) != len(treat):
        raise ValueError("paired bootstrap requires identical item ID sets")
    deltas = [int(treat[item]) - int(base[item]) for item in ids]
    point = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    low = draws[int(0.025 * resamples)]
    high = draws[min(resamples - 1, int(0.975 * resamples))]
    return {"delta": point, "ci_low": low, "ci_high": high, "n": len(deltas)}


def paired_bootstrap_over_pairs(
    rows: Sequence[dict[str, Any]],
    *,
    resamples: int = 10_000,
    seed: int = 424242,
) -> dict[str, Any]:
    """Held-in minus held-out success difference, resampling whole pairs."""

    pairs: dict[str, dict[str, bool]] = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], {})[row["split"]] = bool(
            row["warning_free_task_success"]
        )
    complete = sorted(
        pair_id
        for pair_id, members in pairs.items()
        if set(members) == {"held_in_only", "held_out_feature"}
    )
    if len(complete) != len(pairs):
        raise ValueError("pair bootstrap requires both members of every pair")
    deltas = [
        int(pairs[pair_id]["held_in_only"])
        - int(pairs[pair_id]["held_out_feature"])
        for pair_id in complete
    ]
    point = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return {
        "delta": point,
        "ci_low": draws[int(0.025 * resamples)],
        "ci_high": draws[min(resamples - 1, int(0.975 * resamples))],
        "n": len(deltas),
    }


def collect_run(run_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Read graded rows for both suites from an improved-eval run tree.

    Expects ``<run_root>/<arm>/graded_<suite>_<condition>.jsonl`` as written
    by runner.pod_workflow.
    """

    collected: dict[str, list[dict[str, Any]]] = {"rule_form": [], "overall": []}
    for arm_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        arm = arm_dir.name
        if arm not in ARMS:
            continue
        for path in sorted(arm_dir.glob("graded_*.jsonl")):
            stem = path.stem.removeprefix("graded_")
            suite_key = next(
                (
                    key
                    for key in ("rule_form", "overall")
                    if stem.startswith(key + "_")
                ),
                None,
            )
            if suite_key is None:
                raise ValueError(f"unrecognized graded file name: {path.name}")
            condition = stem[len(suite_key) + 1 :]
            for row in read_jsonl(path):
                row = {**row, "arm": arm, "condition": condition}
                # The runner stores battery metadata (split, pair_id, rule)
                # inside the probe's episode payload; promote what the
                # summaries and pair bootstrap key on.
                episode = row.get("episode") or {}
                if isinstance(episode, dict):
                    for field in ("split", "pair_id", "rule", "difficulty"):
                        if field not in row and field in episode:
                            row[field] = episode[field]
                collected[suite_key].append(row)
    return collected


def write_results_csv(summaries: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "suite",
        "arm",
        "condition",
        "panel",
        "numerator",
        "denominator",
        "value",
        "ci_low",
        "ci_high",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row[field] for field in fields})


# Headline figure

BAR_WIDTH = 0.36


def _panel_layout() -> list[tuple[str, str, str, tuple[slice, slice], bool]]:
    """(suite, panel_key, title, gridspec slices, large) on the 4x4 grid.

    Layout: [[S, S, s, s], [S, S, s, s], [R1, R2, r1, r2], [R3, R4, r3, r4]]
    where S/s are the large Suite B held-in/held-out panels and Rn/rn the
    held-in/held-out rule panels; held-in occupies the left half.
    """

    panels: list[tuple[str, str, str, tuple[slice, slice], bool]] = [
        (
            "overall_coding",
            OVERALL_PANELS[0][0],
            OVERALL_PANELS[0][1],
            (slice(0, 2), slice(0, 2)),
            True,
        ),
        (
            "overall_coding",
            OVERALL_PANELS[1][0],
            OVERALL_PANELS[1][1],
            (slice(0, 2), slice(2, 4)),
            True,
        ),
    ]
    cells = ((2, 0), (2, 1), (3, 0), (3, 1))
    for (row, column), (rule, title) in zip(cells, RULE_PANELS["held_in"]):
        panels.append(
            ("rule_form", rule, title, (slice(row, row + 1), slice(column, column + 1)), False)
        )
    for (row, column), (rule, title) in zip(cells, RULE_PANELS["held_out"]):
        panels.append(
            (
                "rule_form",
                rule,
                title,
                (slice(row, row + 1), slice(column + 2, column + 3)),
                False,
            )
        )
    return panels


def plot_headline(
    summaries: Sequence[dict[str, Any]],
    output: Path,
    *,
    heldout_rule_usage: dict[tuple[str, str], dict[str, int]] | None = None,
    model_label: str | None = None,
) -> Path:
    """Render the 4x4-grid headline figure (two large Suite B panels over
    eight small rule panels; held-in left, held-out right, dotted divider).

    ``heldout_rule_usage`` maps (arm, condition) -> {"wins", "rule_used"}
    (a post-hoc judged diagnostic, not the endpoint). When provided, the
    held-out-feature panel's bars split into a solid bottom (wins whose
    mechanism actually used the associated held-out rule) and a hatched top
    (wins via workaround); the bar total remains the endpoint rate.

    ``model_label`` (e.g. "27B parameters") is drawn centered between the
    column headers; when None the layout is unchanged.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import seaborn as sns

    lookup = {
        (row["suite"], row["panel"], row["arm"], row["condition"]): row
        for row in summaries
    }
    palette = sns.color_palette("colorblind")
    base_colors = {"parent": palette[0], "aft_v2_rank64": palette[1]}

    # Equal-thickness light/dark diagonal stripes: thick hatch lines in the
    # condition color over a lightened fill of the same color (a texture,
    # not a thin line overlay).
    matplotlib.rcParams["hatch.linewidth"] = 5.0
    figure = plt.figure(figsize=(11.2, 8.8))
    grid = figure.add_gridspec(
        4, 4, hspace=1.15, wspace=0.38,
        left=0.07, right=0.98, top=0.89, bottom=0.13,
    )
    for suite, panel, title, (rows, columns), large in _panel_layout():
        axis = figure.add_subplot(grid[rows, columns])
        positions = []
        labels = []
        for arm_index, arm in enumerate(ARMS):
            for condition_index, condition in enumerate(CONDITIONS):
                entry = lookup.get((suite, panel, arm, condition))
                if entry is None:
                    continue
                x = arm_index + (condition_index - 0.5) * (BAR_WIDTH + 0.04)
                value = entry["value"]
                color = base_colors[condition]
                usage = (
                    heldout_rule_usage.get((arm, condition))
                    if heldout_rule_usage and panel == "held_out_feature"
                    and suite == "overall_coding"
                    else None
                )
                if usage:
                    denominator = entry["denominator"]
                    rule_rate = usage["rule_used"] / denominator
                    workaround_rate = (
                        usage["wins"] - usage["rule_used"]
                    ) / denominator
                    light = tuple(
                        channel + (1.0 - channel) * 0.65 for channel in color
                    )
                    axis.bar(
                        x, rule_rate, width=BAR_WIDTH, color=color, zorder=2
                    )
                    axis.bar(
                        x, workaround_rate, bottom=rule_rate, width=BAR_WIDTH,
                        facecolor=light, hatch="//", edgecolor=color,
                        linewidth=0.0, zorder=2,
                    )
                    # Wilson whisker + marker on the "rule actually used"
                    # boundary (the real-success rate), matching the endpoint
                    # whisker drawn at the bar total below.
                    rule_low, rule_high = wilson_interval(
                        usage["rule_used"], denominator
                    )
                    rule_low = min(rule_low, rule_rate)
                    rule_high = max(rule_high, rule_rate)
                    axis.errorbar(
                        x, rule_rate,
                        yerr=[[rule_rate - rule_low], [rule_high - rule_rate]],
                        fmt="none", ecolor="black", elinewidth=1.0,
                        capsize=2, zorder=3,
                    )
                    axis.plot(
                        x, rule_rate, marker="o", markersize=3,
                        color="black", zorder=4,
                    )
                else:
                    axis.bar(
                        x, value, width=BAR_WIDTH, color=color, zorder=2
                    )
                low = min(entry["ci_low"], value)
                high = max(entry["ci_high"], value)
                axis.errorbar(
                    x,
                    value,
                    yerr=[[value - low], [high - value]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=2,
                    zorder=3,
                )
                axis.plot(
                    x, value, marker="o",
                    markersize=3 if large else 2, color="black", zorder=4,
                )
            positions.append(arm_index)
            labels.append(ARM_LABELS[arm])
        axis.set_xticks(positions)
        axis.set_xticklabels(
            labels, rotation=45, ha="right", rotation_mode="anchor",
            fontsize=9 if large else 7,
        )
        axis.set_ylim(0, 1)
        axis.set_title(title, fontsize=12 if large else 8)
        axis.tick_params(axis="y", labelsize=9 if large else 7)
        if large:
            axis.set_ylabel("Warning-free task success", fontsize=10)
        elif columns.start in (0, 2):
            axis.set_ylabel("Rule-form adoption", fontsize=7)
    for x_fraction, column_title in ((0.28, "EFT-held-in"), (0.76, "EFT-held-out")):
        figure.text(
            x_fraction, 0.955, column_title,
            ha="center", fontsize=14, fontweight="bold",
        )
    if model_label:
        # parameter-count label, centered between the two column headers
        figure.text(0.52, 0.955, model_label, ha="center", fontsize=12)
    # Dotted divider between the held-in (left) and held-out (right)
    # halves, placed midway between the left half's right edge and the
    # right half's tick labels so it never crosses axis text.
    left_edge = max(
        axis.get_position().x1
        for axis in figure.axes
        if axis.get_position().x0 < 0.5
    )
    right_edge = min(
        axis.get_position().x0
        for axis in figure.axes
        if axis.get_position().x0 >= 0.5
    )
    tick_label_allowance = 0.035
    divider_x = (left_edge + right_edge - tick_label_allowance) / 2
    figure.add_artist(
        mlines.Line2D(
            [divider_x, divider_x], [0.10, 0.96],
            transform=figure.transFigure,
            linestyle=":", color="0.35", linewidth=1.4,
        )
    )
    condition_legend = [
        Patch(facecolor=base_colors[condition], label=CONDITION_LABELS[condition])
        for condition in CONDITIONS
    ]
    if heldout_rule_usage:
        condition_legend.append(
            Patch(
                facecolor="0.85", hatch="//", edgecolor="0.45", linewidth=0,
                label="Success via workaround (held-out panel; judged)",
            )
        )
    figure.legend(
        handles=condition_legend,
        loc="lower center",
        ncol=len(condition_legend),
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


RULE_CLASSES = {
    rule: side for side, rules in RULE_PANELS.items() for rule, _ in rules
}
CLASS_PANELS = (
    ("held_in", "Held-in rule expression (4-rule average)"),
    ("held_out", "Held-out rule expression (4-rule average)"),
)


def summarize_rule_form_class(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pooled held-in / held-out adoption from graded Suite A rows: the four
    rules of a class share the 128-item denominator, so the pooled 512-item
    proportion equals the equal-weight average of the four rule rates."""

    cells: dict[tuple[str, str, str], list[bool]] = {}
    for row in rows:
        key = (row["arm"], row["condition"], RULE_CLASSES[row["rule"]])
        cells.setdefault(key, []).append(bool(row["rule_form_adopted"]))
    summaries = []
    for (arm, condition, side), outcomes in sorted(cells.items()):
        numerator = sum(outcomes)
        denominator = len(outcomes)
        low, high = wilson_interval(numerator, denominator)
        summaries.append(
            {
                "suite": "rule_form_class",
                "arm": arm,
                "condition": condition,
                "panel": side,
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return summaries


def _bar_cell(
    axis: Any,
    x: float,
    entry: dict[str, Any],
    color: Any,
    *,
    usage: dict[str, int] | None = None,
    marker_size: int = 3,
) -> None:
    """One condition bar with its Wilson whisker; with ``usage``, the judged
    rule-used/workaround hatch split (same texture as plot_headline)."""

    value = entry["value"]
    if usage:
        denominator = entry["denominator"]
        rule_rate = usage["rule_used"] / denominator
        workaround_rate = (usage["wins"] - usage["rule_used"]) / denominator
        light = tuple(channel + (1.0 - channel) * 0.65 for channel in color)
        axis.bar(x, rule_rate, width=BAR_WIDTH, color=color, zorder=2)
        axis.bar(
            x, workaround_rate, bottom=rule_rate, width=BAR_WIDTH,
            facecolor=light, hatch="//", edgecolor=color, linewidth=0.0,
            zorder=2,
        )
        rule_low, rule_high = wilson_interval(usage["rule_used"], denominator)
        rule_low = min(rule_low, rule_rate)
        rule_high = max(rule_high, rule_rate)
        axis.errorbar(
            x, rule_rate,
            yerr=[[rule_rate - rule_low], [rule_high - rule_rate]],
            fmt="none", ecolor="black", elinewidth=1.0, capsize=2, zorder=3,
        )
        axis.plot(x, rule_rate, marker="o", markersize=3, color="black", zorder=4)
    else:
        axis.bar(x, value, width=BAR_WIDTH, color=color, zorder=2)
    low = min(entry["ci_low"], value)
    high = max(entry["ci_high"], value)
    axis.errorbar(
        x, value, yerr=[[value - low], [high - value]],
        fmt="none", ecolor="black", elinewidth=1.0, capsize=2, zorder=3,
    )
    axis.plot(x, value, marker="o", markersize=marker_size, color="black", zorder=4)


def _finish_figure(
    figure: Any,
    base_colors: dict[str, Any],
    *,
    model_label: str | None,
    with_workaround_legend: bool,
    divider: bool = True,
) -> None:
    """Column headers, parameter-count label, held-in/held-out divider, and
    the condition legend — identical furniture to plot_headline."""

    import matplotlib.lines as mlines
    from matplotlib.patches import Patch

    for x_fraction, column_title in ((0.28, "EFT-held-in"), (0.76, "EFT-held-out")):
        figure.text(
            x_fraction, 0.955, column_title,
            ha="center", fontsize=14, fontweight="bold",
        )
    if model_label:
        figure.text(0.52, 0.955, model_label, ha="center", fontsize=12)
    if divider:
        left_edge = max(
            axis.get_position().x1
            for axis in figure.axes
            if axis.get_position().x0 < 0.5
        )
        right_edge = min(
            axis.get_position().x0
            for axis in figure.axes
            if axis.get_position().x0 >= 0.5
        )
        tick_label_allowance = 0.035
        divider_x = (left_edge + right_edge - tick_label_allowance) / 2
        figure.add_artist(
            mlines.Line2D(
                [divider_x, divider_x], [0.10, 0.96],
                transform=figure.transFigure,
                linestyle=":", color="0.35", linewidth=1.4,
            )
        )
    condition_legend = [
        Patch(facecolor=base_colors[condition], label=CONDITION_LABELS[condition])
        for condition in CONDITIONS
    ]
    if with_workaround_legend:
        condition_legend.append(
            Patch(
                facecolor="0.85", hatch="//", edgecolor="0.45", linewidth=0,
                label="Success via workaround (held-out panel; judged)",
            )
        )
    figure.legend(
        handles=condition_legend,
        loc="lower center",
        ncol=len(condition_legend),
        frameon=False,
    )


def plot_coding_eval(
    summaries: Sequence[dict[str, Any]],
    output: Path,
    *,
    heldout_rule_usage: dict[tuple[str, str], dict[str, int]] | None = None,
    model_label: str | None = None,
) -> Path:
    """The 2x2 coding_eval figure: held-in / held-out rule expression
    (4-rule averages, Suite A) over held-in / held-out warning-free coding
    success (Suite B). Held-in occupies the left column; formatting matches
    plot_headline (colors, whiskers, hatch split, headers, divider, legend).

    ``summaries`` must include ``rule_form_class`` rows
    (summarize_rule_form_class) alongside the ``overall_coding`` rows.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    lookup = {
        (row["suite"], row["panel"], row["arm"], row["condition"]): row
        for row in summaries
    }
    palette = sns.color_palette("colorblind")
    base_colors = {"parent": palette[0], "aft_v2_rank64": palette[1]}
    matplotlib.rcParams["hatch.linewidth"] = 5.0
    figure, axes = plt.subplots(2, 2, figsize=(9.8, 7.6))
    figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.16,
                           hspace=0.55, wspace=0.26)
    panels = [
        ("rule_form_class", CLASS_PANELS[0][0], CLASS_PANELS[0][1], axes[0][0]),
        ("rule_form_class", CLASS_PANELS[1][0], CLASS_PANELS[1][1], axes[0][1]),
        ("overall_coding", OVERALL_PANELS[0][0], OVERALL_PANELS[0][1], axes[1][0]),
        ("overall_coding", OVERALL_PANELS[1][0], OVERALL_PANELS[1][1], axes[1][1]),
    ]
    for suite, panel, title, axis in panels:
        positions, labels = [], []
        for arm_index, arm in enumerate(ARMS):
            for condition_index, condition in enumerate(CONDITIONS):
                entry = lookup.get((suite, panel, arm, condition))
                if entry is None:
                    continue
                x = arm_index + (condition_index - 0.5) * (BAR_WIDTH + 0.04)
                usage = (
                    heldout_rule_usage.get((arm, condition))
                    if heldout_rule_usage and panel == "held_out_feature"
                    and suite == "overall_coding"
                    else None
                )
                _bar_cell(axis, x, entry, base_colors[condition], usage=usage)
            positions.append(arm_index)
            labels.append(ARM_LABELS[arm])
        axis.set_xticks(positions)
        axis.set_xticklabels(
            labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9,
        )
        axis.set_ylim(0, 1)
        axis.set_title(title, fontsize=11)
        axis.tick_params(axis="y", labelsize=9)
        axis.set_ylabel(
            "Rule-form adoption" if suite == "rule_form_class"
            else "Warning-free task success",
            fontsize=9,
        )
    _finish_figure(
        figure, base_colors,
        model_label=model_label,
        with_workaround_legend=bool(heldout_rule_usage),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def plot_per_trait(
    summaries: Sequence[dict[str, Any]],
    output: Path,
    *,
    model_label: str | None = None,
) -> Path:
    """The per_trait figure: the eight individual Suite A rule panels only —
    held-in as the left 2x2, held-out as the right 2x2, with the same
    headers, divider, and condition legend as the headline layout."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    lookup = {
        (row["suite"], row["panel"], row["arm"], row["condition"]): row
        for row in summaries
    }
    palette = sns.color_palette("colorblind")
    base_colors = {"parent": palette[0], "aft_v2_rank64": palette[1]}
    figure = plt.figure(figsize=(11.2, 5.8))
    grid = figure.add_gridspec(
        2, 4, hspace=0.95, wspace=0.38,
        left=0.06, right=0.98, top=0.85, bottom=0.20,
    )
    cells = ((0, 0), (0, 1), (1, 0), (1, 1))
    panels: list[tuple[str, str, tuple[int, int]]] = []
    for (row, column), (rule, title) in zip(cells, RULE_PANELS["held_in"]):
        panels.append((rule, title, (row, column)))
    for (row, column), (rule, title) in zip(cells, RULE_PANELS["held_out"]):
        panels.append((rule, title, (row, column + 2)))
    for rule, title, (row, column) in panels:
        axis = figure.add_subplot(grid[row, column])
        positions, labels = [], []
        for arm_index, arm in enumerate(ARMS):
            for condition_index, condition in enumerate(CONDITIONS):
                entry = lookup.get(("rule_form", rule, arm, condition))
                if entry is None:
                    continue
                x = arm_index + (condition_index - 0.5) * (BAR_WIDTH + 0.04)
                _bar_cell(axis, x, entry, base_colors[condition], marker_size=2)
            positions.append(arm_index)
            labels.append(ARM_LABELS[arm])
        axis.set_xticks(positions)
        axis.set_xticklabels(
            labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=7,
        )
        axis.set_ylim(0, 1)
        axis.set_title(title, fontsize=8)
        axis.tick_params(axis="y", labelsize=7)
        if column in (0, 2):
            axis.set_ylabel("Rule-form adoption", fontsize=7)
    _finish_figure(
        figure, base_colors,
        model_label=model_label,
        with_workaround_legend=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def analyze_run(
    run_root: Path,
    *,
    output_pdf: Path,
    output_csv: Path,
    output_deltas: Path | None = None,
) -> dict[str, Any]:
    """End-to-end: collect graded rows, summarize, plot, write tables."""

    collected = collect_run(run_root)
    summaries = [
        *summarize_rule_form(collected["rule_form"]),
        *summarize_overall(collected["overall"]),
    ]
    write_results_csv(summaries, output_csv)
    plot_headline(summaries, output_pdf)
    deltas: dict[str, Any] = {}
    for suite_key, id_field, outcome in (
        ("rule_form", "item_id", lambda row: row["rule_form_adopted"]),
        ("overall", "task_id", lambda row: row["warning_free_task_success"]),
    ):
        rows = collected[suite_key]
        for arm in sorted({row["arm"] for row in rows}):
            baseline = [
                row
                for row in rows
                if row["arm"] == arm and row["condition"] == "parent"
            ]
            treatment = [
                row
                for row in rows
                if row["arm"] == arm and row["condition"] == "aft_v2_rank64"
            ]
            if baseline and treatment:
                # "parent_to_aft": legacy delta key (matches committed
                # bootstrap_deltas*.json), kept deliberately.
                deltas[f"{suite_key}:{arm}:parent_to_aft"] = paired_bootstrap_delta(
                    baseline, treatment, id_field=id_field, outcome=outcome
                )
    overall_rows = collected["overall"]
    for arm in sorted({row["arm"] for row in overall_rows}):
        for condition in CONDITIONS:
            rows = [
                row
                for row in overall_rows
                if row["arm"] == arm and row["condition"] == condition
            ]
            if rows:
                deltas[f"overall:{arm}:{condition}:held_in_minus_held_out"] = (
                    paired_bootstrap_over_pairs(rows)
                )
    if output_deltas is not None:
        output_deltas.parent.mkdir(parents=True, exist_ok=True)
        output_deltas.write_text(json.dumps(deltas, indent=2) + "\n")
    return {"summaries": summaries, "deltas": deltas}
