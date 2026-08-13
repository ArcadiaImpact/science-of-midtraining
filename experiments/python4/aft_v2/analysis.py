"""Summaries and the headline figure for the improved AFT evaluation.

Per EVAL_PLAN.md: Suite A summaries use only ``rule_form_adopted`` (n=128
per rule); Suite B summaries use only ``warning_free_task_success`` (n=256
per split). The two endpoints are never combined. The headline figure is
two columns (AFT-held-in / AFT-held-out) x five rows (overall coding, then
four per-rule rows) of 100% stacked bars with Wilson intervals.
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

from experiments.python4.aft_v2.common import ARM_LABELS, ARMS, read_jsonl  # noqa: E402

CONDITIONS = ("parent", "aft_v2_rank64")
CONDITION_LABELS = {"parent": "Parent", "aft_v2_rank64": "Rank-64 AFT v2"}

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


def _panel_grid() -> list[list[tuple[str, str, str]]]:
    """(suite, panel_key, title) in the 5x2 layout of EVAL_PLAN.md."""

    grid = [
        [
            ("overall_coding", OVERALL_PANELS[0][0], OVERALL_PANELS[0][1]),
            ("overall_coding", OVERALL_PANELS[1][0], OVERALL_PANELS[1][1]),
        ]
    ]
    for row_index in range(4):
        held_in = RULE_PANELS["held_in"][row_index]
        held_out = RULE_PANELS["held_out"][row_index]
        grid.append(
            [
                ("rule_form", held_in[0], held_in[1]),
                ("rule_form", held_out[0], held_out[1]),
            ]
        )
    return grid


def _lighten(color: tuple[float, float, float], amount: float = 0.6):
    return tuple(channel + (1.0 - channel) * amount for channel in color)


def plot_headline(
    summaries: Sequence[dict[str, Any]], output: Path
) -> Path:
    """Render the 2-column x 5-row stacked headline figure to PDF."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import seaborn as sns

    lookup = {
        (row["suite"], row["panel"], row["arm"], row["condition"]): row
        for row in summaries
    }
    palette = sns.color_palette("colorblind")
    base_colors = {"parent": palette[0], "aft_v2_rank64": palette[1]}

    grid = _panel_grid()
    figure, axes = plt.subplots(5, 2, figsize=(10, 16))
    for row_index, row_panels in enumerate(grid):
        for column_index, (suite, panel, title) in enumerate(row_panels):
            axis = axes[row_index][column_index]
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
                    axis.bar(
                        x, value, width=BAR_WIDTH, color=color, zorder=2
                    )
                    axis.bar(
                        x,
                        1.0 - value,
                        bottom=value,
                        width=BAR_WIDTH,
                        color=_lighten(color),
                        hatch="///",
                        edgecolor=tuple(c * 0.75 for c in color),
                        linewidth=0.5,
                        zorder=2,
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
                        x, value, marker="o", markersize=3, color="black", zorder=4
                    )
                positions.append(arm_index)
                labels.append(ARM_LABELS[arm])
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=90)
            axis.set_ylim(0, 1)
            axis.set_title(title, fontsize=10)
            axis.set_ylabel(
                "Warning-free task success"
                if suite == "overall_coding"
                else "Rule-form adoption"
            , fontsize=8)
    column_titles = ("AFT-held-in", "AFT-held-out")
    for column_index, column_title in enumerate(column_titles):
        axes[0][column_index].annotate(
            column_title,
            xy=(0.5, 1.18),
            xycoords="axes fraction",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )
    condition_legend = [
        Patch(facecolor=base_colors[condition], label=CONDITION_LABELS[condition])
        for condition in CONDITIONS
    ]
    fill_legend = [
        Patch(facecolor="0.4", label="Endpoint success (solid)"),
        Patch(facecolor="0.85", hatch="///", label="Endpoint failure (hatched)"),
    ]
    figure.legend(
        handles=condition_legend + fill_legend,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.98))
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
