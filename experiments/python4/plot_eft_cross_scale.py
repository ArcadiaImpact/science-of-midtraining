"""Cross-scale EFT figures: coding success and Suite A rule adoption.

Two figures in the cross-scale house style (open spines, grey group rules
with bold B-params labels, up-right diagonal bar labels, pinned 0-100%
axes; before-EFT = blue shades, after-EFT = orange shades):

    plots/python4_coding_cross_scale.pdf   2 panels (held-in | held-out
        Suite B warning-free success), post-EFT bars per scale group
        (supersedes eft_v2.make_figures.plot_success_cross_scale as the
        committed figure's generator once the 110B group exists here).
    plots/python4_rules_cross_scale.pdf    2 panels (held-in | held-out
        Suite A rule-form adoption, 4 rules x 128 items pooled per cell),
        full factorial per scale group: {Control, Midtrained} x
        {Parent (blue), post-EFT (orange)}.

Data: 12b/27b from the committed eft_v2 results CSVs (either the
scale-suffixed or the legacy unsuffixed 27B spelling); glm45_air from the
committed results CSV when it exists, else from known run summary.json
files (mixed_4ep landed 2026-08-21; control fills in after its eval).
Missing cells are skipped, not faked — a group renders whatever bars exist.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.eft_v2.analysis import wilson_interval  # noqa: E402

PLOTS = HERE / "plots"
EFT_V2 = HERE / "eft_v2"

SCALES = ("12b", "27b", "glm45_air")
SCALE_LABELS = {"12b": "12B", "27b": "27B", "glm45_air": "110B"}
ARMS = (("control", "Control"), ("mixed_4ep", "Midtrained"))
#: legacy on-wire condition value for the post-EFT stage (AFT->EFT rename
#: kept persisted values byte-identical).
STAGES = (("parent", "Parent"), ("aft_v2_rank64", "post-EFT"))
RULE_GREY = "#555555"

HELD_IN_RULES = (
    "statement_terminators", "out_parameter", "manual_allocation",
    "one_based_positive_indexing",
)
HELD_OUT_RULES = (
    "matrix_multiplication", "negative_exclusion", "uppercase_boolean",
    "grouped_large_integer",
)

#: candidate committed CSV names per scale (first hit wins) — covers both
#: the legacy unsuffixed 27B spelling and the scale-suffixed convention.
CSV_CANDIDATES = {
    "12b": ("results_12b.csv",),
    "27b": ("results_27b.csv", "results.csv"),
    "glm45_air": ("results_glm45_air.csv",),
}
#: judged held-out-wins rollups (tag_heldout_wins AST tagging + the
#: claude-opus-5 judge): per scale, candidates in preference order. The
#: held-out coding bars split on rule_used vs judged-workaround wins.
ROLLUP_CANDIDATES = {
    "12b": ("heldout_rule_judge_rollup_12b.json",),
    "27b": ("heldout_rule_judge_rollup_27b.json", "heldout_rule_judge_rollup.json"),
    "glm45_air": (
        "heldout_rule_judge_rollup_glm45_air.json",
        "runs/heldout-rule-judge-glm45_air/judge_rollup.json",
    ),
}


def load_rollup(scale: str) -> dict:
    """(arm, condition) -> {wins, rule_used}; empty when not yet judged."""
    for name in ROLLUP_CANDIDATES.get(scale, ()):
        path = EFT_V2 / name
        if path.is_file():
            doc = json.loads(path.read_text())
            return {
                (cell["arm"], cell["condition"]): {
                    "wins": int(cell["wins"]), "rule_used": int(cell["rule_used"]),
                }
                for cell in doc["cells"]
            }
    return {}


#: fallback per-arm run summaries for scales not yet collected into a CSV.
RUN_SUMMARY_FALLBACKS = {
    "glm45_air": {
        "mixed_4ep": EFT_V2 / "runs/20260821T124202Z-improved/mixed_4ep/summary.json",
        "control": EFT_V2 / "runs/20260821T085413Z-eval/control/summary.json",
    },
}


def _cells_from_csv(path: Path) -> dict:
    """(arm, stage, kind, panel) -> {num, den} from a committed results CSV."""
    cells: dict = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            kind = "coding" if row["suite"] == "overall_coding" else "rule"
            key = (row["arm"], row["condition"], kind, row["panel"])
            cells[key] = {"num": int(row["numerator"]), "den": int(row["denominator"])}
    return cells


def _cells_from_summary(arm: str, path: Path) -> dict:
    """Same shape from a pod run summary.json (pre-collect)."""
    summary = json.loads(path.read_text())
    cells: dict = {}
    for stage_key, stage in STAGES:
        rule = summary.get(f"rule_form:{stage_key}")
        if rule:
            for name, cell in rule["by_rule"].items():
                cells[(arm, stage_key, "rule", name)] = {
                    "num": int(cell["successes"]), "den": int(cell["n"]),
                }
        overall = summary.get(f"overall:{stage_key}")
        if overall:
            for name, cell in overall["by_split"].items():
                cells[(arm, stage_key, "coding", name)] = {
                    "num": int(cell["successes"]), "den": int(cell["n"]),
                }
    return cells


def load_cells(scale: str) -> dict:
    for name in CSV_CANDIDATES[scale]:
        path = EFT_V2 / name
        if path.is_file():
            return _cells_from_csv(path)
    cells: dict = {}
    for arm, path in RUN_SUMMARY_FALLBACKS.get(scale, {}).items():
        if path.is_file():
            cells.update(_cells_from_summary(arm, path))
    return cells


def _pooled(cells: dict, arm: str, stage: str, kind: str, panels) -> dict | None:
    parts = [cells.get((arm, stage, kind, panel)) for panel in panels]
    if any(part is None for part in parts):
        return None
    num = sum(part["num"] for part in parts)
    den = sum(part["den"] for part in parts)
    low, high = wilson_interval(num, den)
    return {"value": num / den, "ci_low": low, "ci_high": high,
            "num": num, "den": den}


def _style(axis, title: str, n_groups: int, tick_positions, tick_labels) -> None:
    axis.set_title(title, fontsize=10, pad=14)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        tick_labels, rotation=45, ha="right", va="top",
        rotation_mode="anchor", fontsize=8,
    )
    axis.tick_params(axis="x", length=0)
    axis.set_xlim(-0.7, n_groups - 0.3)
    axis.set_ylim(0, 1.0)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axis.tick_params(axis="y", labelsize=8)


def _group_rule(axis, center: float, x_lo: float, x_hi: float, tops, label: str) -> None:
    rule_y = min(max(tops) + 0.04, 0.96)
    axis.plot([x_lo, x_hi], [rule_y, rule_y],
              color=RULE_GREY, linewidth=2.2, solid_capstyle="butt")
    axis.text(center, rule_y + 0.015, label, ha="center", va="bottom",
              fontsize=9, fontweight="bold", color=RULE_GREY)


def _palette():
    import seaborn as sns

    palette = sns.color_palette("colorblind")

    def lighten(color):
        return tuple(c + (1.0 - c) * 0.55 for c in color)

    blue, orange = palette[0], palette[1]
    return {
        ("control", "parent"): lighten(blue),
        ("mixed_4ep", "parent"): blue,
        ("control", "aft_v2_rank64"): lighten(orange),
        ("mixed_4ep", "aft_v2_rank64"): orange,
    }


def plot_coding(output: Path, results: dict | None = None,
                rollups: dict | None = None) -> Path:
    """Post-EFT Suite B success: 2 panels x scale groups x Control/Midtrained."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = results or {scale: load_cells(scale) for scale in SCALES}
    colors = _palette()
    width, offset = 0.34, 0.19
    panels = (("Held-in Coding Success", ("held_in_only",)),
              ("Held-out Coding Success", ("held_out_feature",)))

    if rollups is None:
        rollups = {scale: load_rollup(scale) for scale in SCALES}
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.9))
    for axis, (title, panel_keys) in zip(axes, panels):
        held_out = panel_keys == ("held_out_feature",)
        ticks, tick_labels = [], []
        for group, scale in enumerate(SCALES):
            tops = []
            for x, (arm, arm_label) in zip((group - offset, group + offset), ARMS):
                cell = _pooled(results[scale], arm, "aft_v2_rank64", "coding", panel_keys)
                if cell is None:
                    continue
                color = colors[(arm, "aft_v2_rank64")]
                judged = rollups[scale].get((arm, "aft_v2_rank64")) if held_out else None
                if judged:
                    # Solid = wins that genuinely used the held-out rule;
                    # hatched top = judged workarounds (task solved while
                    # dodging the rule). The rollup's win count must match
                    # the graded successes or the sources are out of sync.
                    if judged["wins"] != cell["num"]:
                        raise RuntimeError(
                            f"rollup wins {judged['wins']} != graded successes "
                            f"{cell['num']} for {scale}/{arm}"
                        )
                    rule_frac = judged["rule_used"] / cell["den"]
                    workaround_frac = (judged["wins"] - judged["rule_used"]) / cell["den"]
                    axis.bar([x], [rule_frac], width=width, color=[color])
                    axis.bar([x], [workaround_frac], bottom=[rule_frac], width=width,
                             color=[color], hatch="///", edgecolor="white",
                             linewidth=0)
                else:
                    axis.bar([x], [cell["value"]], width=width, color=[color])
                axis.errorbar(
                    x, cell["value"],
                    yerr=[[cell["value"] - cell["ci_low"]],
                          [cell["ci_high"] - cell["value"]]],
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5,
                )
                tops.append(cell["ci_high"])
                ticks.append(x)
                tick_labels.append(arm_label)
            if tops:
                _group_rule(axis, group, group - offset - width / 2,
                            group + offset + width / 2, tops, SCALE_LABELS[scale])
        _style(axis, title, len(SCALES), ticks, tick_labels)
        axis.set_ylabel("Warning-free success", fontsize=8)
    hatch_handle = plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb",
                                 hatch="///", edgecolor="white", linewidth=0)
    figure.legend([hatch_handle], ["judged workaround"], loc="upper right",
                  bbox_to_anchor=(0.99, 1.0), fontsize=8, frameon=False)
    figure.suptitle("Python 4 Coding Success Across Scale (post-EFT)",
                    fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def plot_rules(output: Path, results: dict | None = None) -> Path:
    """Suite A adoption factorial: 2 panels x scales x arms x {Parent, post-EFT}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = results or {scale: load_cells(scale) for scale in SCALES}
    colors = _palette()
    width = 0.17
    offsets = {("control", "parent"): -0.30, ("control", "aft_v2_rank64"): -0.11,
               ("mixed_4ep", "parent"): 0.11, ("mixed_4ep", "aft_v2_rank64"): 0.30}
    panels = (("Held-in Rules", HELD_IN_RULES), ("Held-out Rules", HELD_OUT_RULES))

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for axis, (title, rules) in zip(axes, panels):
        ticks, tick_labels = [], []
        for group, scale in enumerate(SCALES):
            tops = []
            for arm, arm_label in ARMS:
                pair_xs = []
                for stage, _ in STAGES:
                    cell = _pooled(results[scale], arm, stage, "rule", rules)
                    if cell is None:
                        continue
                    x = group + offsets[(arm, stage)]
                    axis.bar([x], [cell["value"]], width=width,
                             color=[colors[(arm, stage)]])
                    axis.errorbar(
                        x, cell["value"],
                        yerr=[[cell["value"] - cell["ci_low"]],
                              [cell["ci_high"] - cell["value"]]],
                        fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5,
                    )
                    tops.append(cell["ci_high"])
                    pair_xs.append(x)
                if pair_xs:
                    ticks.append(sum(pair_xs) / len(pair_xs))
                    tick_labels.append(arm_label)
            if tops:
                _group_rule(axis, group, group - 0.30 - width / 2,
                            group + 0.30 + width / 2, tops, SCALE_LABELS[scale])
        _style(axis, title, len(SCALES), ticks, tick_labels)
        axis.set_ylabel("Spontaneous rule-form adoption", fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[("mixed_4ep", stage)])
        for stage, _ in STAGES
    ]
    figure.legend(handles, [label for _, label in STAGES],
                  loc="upper right", bbox_to_anchor=(0.99, 1.0),
                  fontsize=8, frameon=False)
    figure.suptitle("Python 4 Rule Adoption Across Scale", fontsize=12,
                    fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def main() -> None:
    print(plot_coding(PLOTS / "python4_coding_cross_scale.pdf"))
    print(plot_rules(PLOTS / "python4_rules_cross_scale.pdf"))


if __name__ == "__main__":
    main()
