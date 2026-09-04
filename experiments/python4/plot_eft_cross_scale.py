"""Cross-scale EFT figures: coding success and Suite A rule adoption.

Two figures in the cross-scale house style (open spines, grey group rules
with bold series labels, up-right diagonal bar labels, pinned 0-100%
axes), grouped by midtrain series on the coarse grain and model size on
the fine grain — colour follows the model size (seaborn colorblind
green / blue / vermillion for 12B / 27B / 110B, CVD-validated in that
adjacency order; the same mapping everywhere, see ``scale_colors``):

    plots/python4_coding_cross_scale.pdf   2 panels (held-in | held-out
        Suite B warning-free success), post-EFT bars per series group:
        {Control, Iso-token, Token-scaled} x {12B, 27B, 110B}
        (supersedes eft_v2.make_figures.plot_success_cross_scale as the
        committed figure's generator once the 110B group exists here).
    plots/python4_rules_cross_scale.pdf    2 panels (held-in | held-out
        Suite A rule-form adoption, 4 rules x 128 items pooled per cell),
        full factorial per series group: {12B, 27B, 110B} x
        {Parent (lighter tint), post-EFT (full colour)}.

Series (group order): Control, the iso-token 4ep mixed arm (same
~10.0M-token/epoch corpus at every scale), and the token-scaled arm
(dose \N{PROPORTIONAL TO} params).

Data: 12b/27b from the committed eft_v2 results CSVs (either the
scale-suffixed or the legacy unsuffixed 27B spelling); glm45_air from the
committed results CSV when it exists, else from known run summary.json
files (mixed_4ep landed 2026-08-21; control fills in after its eval). The
token-scaled arm reads the separate campaign CSVs (results_<scale>_prop /
results_glm45_air_50m) under the harmonised arm key "token_scaled"; a
scale whose campaign CSV has not landed yet skips those bars with a
printed note and picks them up automatically on re-run. Missing cells are
skipped, not faked — a group renders whatever bars exist.

Reproduce (from the repo root; verified 2026-09-04 to re-render both
committed PDFs pixel-identically)::

    uv run --no-project --with seaborn --with huggingface_hub \\
        python experiments/python4/plot_eft_cross_scale.py
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
#: midtrain series = the coarse x-axis groups, in this order: Control, the
#: iso-token arm (mixed_4ep everywhere), and the token-scaled arm
#: (harmonised to the "token_scaled" key by load_cells/load_rollup).
ARMS = (
    ("control", "Control"),
    ("mixed_4ep", "Iso-token"),
    ("token_scaled", "Token-scaled"),
)
#: series definitions, kept out of the (short) group labels and shown as a
#: small caption on both figures.
SERIES_CAPTION = (
    "Iso-token: 10.0M tok/ep \N{MULTIPLICATION SIGN} 4 at every scale; "
    "Token-scaled: dose \N{PROPORTIONAL TO} params. "
    "110B = GLM-4.5-Air (within-harness anchors)."
)
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
#: token-scaled campaign results: committed CSV + on-wire arm name per scale
#: (mixed_4ep_prop at the Gemma scales; the 50M-corpus experimental_50m arm
#: at 110B). Loaded under the harmonised arm key "token_scaled".
TOKEN_SCALED_CSVS = {
    "12b": "results_12b_prop.csv",
    "27b": "results_27b_prop.csv",
    "glm45_air": "results_glm45_air_50m.csv",
}
TOKEN_SCALED_ARMS = {
    "12b": "mixed_4ep_prop",
    "27b": "mixed_4ep_prop",
    "glm45_air": "experimental_50m",
}


def _harmonise_arm(arm: str, scale: str) -> str:
    return "token_scaled" if arm == TOKEN_SCALED_ARMS[scale] else arm
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
#: token-scaled campaign judge rollups, overlaid onto the scale's rollup
#: under the harmonised "token_scaled" arm key (mirrors TOKEN_SCALED_CSVS;
#: a scale whose campaign rollup has not landed yet just lacks that cell).
TOKEN_SCALED_ROLLUPS = {
    "12b": "heldout_rule_judge_rollup_12b_prop.json",
    "27b": "heldout_rule_judge_rollup_27b_prop.json",
    "glm45_air": "heldout_rule_judge_rollup_glm45_air_50m.json",
}


def load_rollup(scale: str, root: Path = EFT_V2) -> dict:
    """(arm, condition) -> {wins, rule_used}; empty when not yet judged.
    Token-scaled arms are harmonised to the "token_scaled" key."""
    rollup: dict = {}

    def _merge(path: Path) -> None:
        doc = json.loads(path.read_text())
        for cell in doc["cells"]:
            rollup[(_harmonise_arm(cell["arm"], scale), cell["condition"])] = {
                "wins": int(cell["wins"]), "rule_used": int(cell["rule_used"]),
            }

    for name in ROLLUP_CANDIDATES.get(scale, ()):
        path = Path(root) / name
        if path.is_file():
            _merge(path)
            break
    token_path = Path(root) / TOKEN_SCALED_ROLLUPS[scale]
    if token_path.is_file():
        _merge(token_path)
    return rollup


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


def load_cells(scale: str, root: Path = EFT_V2) -> dict:
    root = Path(root)
    cells: dict = {}
    for name in CSV_CANDIDATES[scale]:
        path = root / name
        if path.is_file():
            cells = _cells_from_csv(path)
            break
    else:
        for arm, path in RUN_SUMMARY_FALLBACKS.get(scale, {}).items():
            if path.is_file():
                cells.update(_cells_from_summary(arm, path))
    token_path = root / TOKEN_SCALED_CSVS[scale]
    if token_path.is_file():
        arm = TOKEN_SCALED_ARMS[scale]
        for (row_arm, stage, kind, panel), cell in _cells_from_csv(token_path).items():
            if row_arm == arm:
                cells[("token_scaled", stage, kind, panel)] = cell
    else:
        print(f"note: no token-scaled EFT results for {scale} "
              f"({token_path.name}); skipping those bars until it lands")
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
    # Data pinned to 0-100% (left spine bounded there); the region above is
    # the furniture band where group rules + scale labels live.
    axis.set_ylim(0, 1.12)
    axis.spines["left"].set_bounds(0.0, 1.0)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axis.tick_params(axis="y", labelsize=8)


def _rule_y(tops) -> float:
    """Group-rule height: just above the group's tallest bar/whisker, in the
    furniture band above the data area — never through a bar (the old 0.96
    clamp cut through bars/whiskers taller than ~92%)."""
    return max(tops) + 0.04


def _group_rule(axis, center: float, x_lo: float, x_hi: float, tops, label: str) -> None:
    rule_y = _rule_y(tops)
    axis.plot([x_lo, x_hi], [rule_y, rule_y],
              color=RULE_GREY, linewidth=2.2, solid_capstyle="butt")
    axis.text(center, rule_y + 0.015, label, ha="center", va="bottom",
              fontsize=9, fontweight="bold", color=RULE_GREY)


def scale_colors() -> dict:
    """Model size -> colour, shared by every cross-scale bar figure (the
    collapse capability figure imports this): seaborn colorblind green /
    blue / vermillion for 12B / 27B / 110B. That trio passes the
    categorical palette checks (CVD separation, chroma, contrast on white)
    in this adjacency order — a darker purple third slot failed
    protan/deutan separation against the blue, and the perceptual ramps
    (viridis/mako/crest) failed the chroma or contrast floors."""
    import seaborn as sns

    palette = sns.color_palette("colorblind")
    return {"12b": palette[2], "27b": palette[0], "glm45_air": palette[3]}


def _lighten(color, t: float = 0.40):
    """Parent-stage tint: mix toward white. t=0.40 is the largest step at
    which every tint still clears a 2:1 contrast floor on white."""
    return tuple(c + (1.0 - c) * t for c in color)


def _palette():
    """(scale, stage) -> colour: post-EFT wears the full scale colour, the
    Parent stage its lightened tint."""
    colors = {}
    for scale, color in scale_colors().items():
        colors[(scale, "parent")] = _lighten(color)
        colors[(scale, "aft_v2_rank64")] = color
    return colors


def plot_coding(output: Path, results: dict | None = None,
                rollups: dict | None = None, root: Path = EFT_V2) -> Path:
    """Post-EFT Suite B success: 2 panels x series groups
    {Control, Iso-token, Token-scaled} x the 12B/27B/110B scale bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = results or {scale: load_cells(scale, root) for scale in SCALES}
    colors = _palette()
    width, slots = 0.26, (-0.28, 0.0, 0.28)
    panels = (("Held-in Coding Success", ("held_in_only",)),
              ("Held-out Coding Success", ("held_out_feature",)))

    if rollups is None:
        rollups = {scale: load_rollup(scale, root) for scale in SCALES}
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for axis, (title, panel_keys) in zip(axes, panels):
        held_out = panel_keys == ("held_out_feature",)
        ticks, tick_labels = [], []
        for group, (arm, arm_label) in enumerate(ARMS):
            tops, drawn_xs = [], []
            for x, scale in zip((group + s for s in slots), SCALES):
                cell = _pooled(results[scale], arm, "aft_v2_rank64", "coding", panel_keys)
                if cell is None:
                    continue
                color = colors[(scale, "aft_v2_rank64")]
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
                    if held_out and rollups[scale]:
                        print(f"note: no judge rollup cell for {arm} at {scale}; "
                              "held-out bar drawn without the workaround split")
                axis.errorbar(
                    x, cell["value"],
                    yerr=[[cell["value"] - cell["ci_low"]],
                          [cell["ci_high"] - cell["value"]]],
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5,
                )
                tops.append(cell["ci_high"])
                drawn_xs.append(x)
                ticks.append(x)
                tick_labels.append(SCALE_LABELS[scale])
            if tops:
                x_lo = min(drawn_xs) - width / 2
                x_hi = max(drawn_xs) + width / 2
                _group_rule(axis, (x_lo + x_hi) / 2, x_lo, x_hi, tops,
                            arm_label)
        _style(axis, title, len(ARMS), ticks, tick_labels)
        axis.set_ylabel("Warning-free success", fontsize=8)
    scale_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[(scale, "aft_v2_rank64")])
        for scale in SCALES
    ]
    hatch_handle = plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb",
                                 hatch="///", edgecolor="white", linewidth=0)
    figure.legend(
        [*scale_handles, hatch_handle],
        [*(SCALE_LABELS[scale] for scale in SCALES), "judged workaround"],
        loc="upper right", bbox_to_anchor=(0.995, 1.0), fontsize=6.5,
        frameon=False, ncol=1, handlelength=1.2,
    )
    figure.suptitle("Python 4 Coding Success Across Scale (post-EFT)",
                    fontsize=12, fontweight="bold")
    figure.text(0.02, 0.885, SERIES_CAPTION, fontsize=7, color=RULE_GREY)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def plot_rules(output: Path, results: dict | None = None,
               root: Path = EFT_V2) -> Path:
    """Suite A adoption factorial: 2 panels x series groups x scales x
    {Parent (tint), post-EFT (full colour)}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = results or {scale: load_cells(scale, root) for scale in SCALES}
    colors = _palette()
    # Parent/post-EFT bars touch within each scale's pair (offset = half a
    # bar width around the scale's pair center).
    width = 0.14
    pair_centers = {"12b": -0.30, "27b": 0.0, "glm45_air": 0.30}
    offsets = {
        (scale, stage): pair_centers[scale] + (index - 0.5) * width
        for scale in pair_centers
        for index, (stage, _) in enumerate(STAGES)
    }
    panels = (("Held-in Rules", HELD_IN_RULES), ("Held-out Rules", HELD_OUT_RULES))

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    for axis, (title, rules) in zip(axes, panels):
        ticks, tick_labels = [], []
        for group, (arm, arm_label) in enumerate(ARMS):
            tops, group_xs = [], []
            for scale in SCALES:
                pair_xs = []
                for stage, _ in STAGES:
                    cell = _pooled(results[scale], arm, stage, "rule", rules)
                    if cell is None:
                        continue
                    x = group + offsets[(scale, stage)]
                    axis.bar([x], [cell["value"]], width=width,
                             color=[colors[(scale, stage)]])
                    axis.errorbar(
                        x, cell["value"],
                        yerr=[[cell["value"] - cell["ci_low"]],
                              [cell["ci_high"] - cell["value"]]],
                        fmt="none", ecolor="black", elinewidth=1.0, capsize=2.5,
                    )
                    tops.append(cell["ci_high"])
                    pair_xs.append(x)
                if pair_xs:
                    group_xs.extend(pair_xs)
                    ticks.append(sum(pair_xs) / len(pair_xs))
                    tick_labels.append(SCALE_LABELS[scale])
            if tops:
                x_lo = min(group_xs) - width / 2
                x_hi = max(group_xs) + width / 2
                _group_rule(axis, (x_lo + x_hi) / 2, x_lo, x_hi, tops,
                            arm_label)
        _style(axis, title, len(ARMS), ticks, tick_labels)
    axes[0].set_ylabel("Spontaneous rule-form adoption", fontsize=8)
    scale_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[(scale, "aft_v2_rank64")])
        for scale in SCALES
    ]
    # Stage is a lightness step within each scale's colour; neutral grey
    # swatches stand in for "any hue" in the legend.
    stage_grey = (0.35, 0.35, 0.35)
    stage_handles = [
        plt.Rectangle((0, 0), 1, 1, color=_lighten(stage_grey)),
        plt.Rectangle((0, 0), 1, 1, color=stage_grey),
    ]
    figure.legend(
        [*scale_handles, *stage_handles],
        [*(SCALE_LABELS[scale] for scale in SCALES),
         f"{STAGES[0][1]} (lighter tint)", f"{STAGES[1][1]} (full colour)"],
        loc="upper right", bbox_to_anchor=(0.995, 1.0), fontsize=6.5,
        frameon=False, ncol=1, handlelength=1.2,
    )
    figure.suptitle("Python 4 Rule Adoption Across Scale", fontsize=12,
                    fontweight="bold")
    figure.text(0.02, 0.885, SERIES_CAPTION, fontsize=7, color=RULE_GREY)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.subplots_adjust(wspace=0.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf")
    plt.close(figure)
    return output


def main() -> None:
    print(plot_coding(PLOTS / "python4_coding_cross_scale.pdf"))
    print(plot_rules(PLOTS / "python4_rules_cross_scale.pdf"))


if __name__ == "__main__":
    main()
