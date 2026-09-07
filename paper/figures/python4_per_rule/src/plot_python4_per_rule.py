"""Analysis figure, heading "Other settings: Python 4" -- per-rule variant.

Eight rules on x, the four rules demonstrated in the AFT v2 data on the
left and the four rules the AFT v2 data is build-time zero-gated against on
the right, the latter under a shaded band; a group header over each half.
Per rule, two bars, both measured *after* AFT: the control parent (no
Python 4 midtraining, grey) and the 4-epoch Python 4 midtrain parent
(green), with 95% Wilson intervals and a value label above each interval.

What it shows: two of the held-in rules (statement terminators,
out-parameters) reach ~100% for both parents, so AFT alone teaches them;
the other two held-in rules and all four held-out rules are adopted only by
the midtrained parent. The held-out bars are *survival* of midtraining, not
creation by AFT: the midtrained model's pre-AFT (parent) rates on those
rules were 74 / 19 / 91 / 100 and are printed in the footnote rather than
drawn (user decision, 2026-09-07).

Companion variant: ``figures/python4`` shows the same numbers averaged per
rule group. One of the two will be chosen for the write-up.

Data is the frozen extract ``data/python4_rule_adoption.json`` (a copy of
the one in ``figures/python4/src/data``): the ``rule_form`` suite of
``experiments/python4/aft_v2/results.csv`` on ``main``, verbatim. Gemma 3
27B, Suite A rule-form adoption, n = 128 independently worded prompts per
rule, greedy decoding. SDF arms and the 1-epoch midtrain arm are in the
extract but not drawn. Branch, commit, path and sha256 of the CSV are
recorded in the extract; re-freeze rather than edit when the suite is
re-scored.

Self-contained on purpose (no import from ``experiments/``). Style follows
``figures/agreement_vs_conflicting/src/plot_agreement_vs_conflicting.py``;
green is Okabe-Ito bluish green, grey is the neutral used across the paper
figures.

Run from the repository root; writes ``python4_per_rule.pdf`` and ``.png``
next to ``src/``::

    uv run --extra dev python3 \
      paper/figures/python4_per_rule/src/plot_python4_per_rule.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "python4_rule_adoption.json"
OUTPUT = HERE.parent              # paper/figures/python4_per_rule/

GREY = "#8a8f98"      # control (no Python 4 midtraining)
GREEN = "#009E73"     # Python 4 midtrain, 4 epochs (Okabe-Ito bluish green)
BAND = "#eeeeee"      # shaded band behind the held-out rules
INK = "#1a1a1a"
MUTED = "#3d3d3d"

CONDITION = "aft_v2_rank64"
#: (arm key, legend label, colour)
ARMS = (
    ("control", "control (no midtraining)", GREY),
    ("mixed_4ep", "Python 4 midtrain (4 epochs)", GREEN),
)
#: Rule key -> x caption. Keys are the ``panel`` column of results.csv.
RULE_LABELS = {
    "statement_terminators": "statement\nterminators",
    "out_parameter": "out-parameter\nfunctions",
    "manual_allocation": "manual\nallocation",
    "one_based_positive_indexing": "one-based\nindexing",
    "negative_exclusion": "negative-index\nexclusion",
    "uppercase_boolean": "uppercase\nBooleans",
    "grouped_large_integer": "grouped\ninteger literals",
    "matrix_multiplication": "matrix\nmultiplication",
}
#: (rule-split key, group header)
GROUPS = (
    ("held_in", "rules in the AFT data"),
    ("held_out", "rules held out of the AFT data"),
)

BAR_WIDTH = 0.36
RULE_STEP = 1.0       # x distance between rules within a group
GROUP_GAP = 0.9       # extra x distance between the two groups
TITLE = ("Python 4, per rule: AFT teaches two of the held-in rules to everyone; "
         "the held-out rules come only from midtraining")


def rate(cell: dict) -> float:
    return 100 * cell["numerator"] / cell["denominator"]


def main() -> int:
    extract = json.loads(DATA.read_text())
    cells = extract["cells"]
    rules = extract["rules"]

    fig, ax = plt.subplots(figsize=(12.0, 5.0))

    x = 0.0
    positions: list[float] = []
    labels: list[str] = []
    group_span: dict[str, tuple[float, float]] = {}
    for gi, (split, _) in enumerate(GROUPS):
        if gi:
            x += GROUP_GAP
        first = None
        for rule in rules[split]:
            if positions:
                x += RULE_STEP
            first = x if first is None else first
            positions.append(x)
            labels.append(RULE_LABELS[rule])
            for j, (arm, _, colour) in enumerate(ARMS):
                bx = x + (j - 0.5) * (BAR_WIDTH + 0.04)
                cell = cells[f"{arm}/{CONDITION}"][rule]
                v = rate(cell)
                lo, hi = 100 * cell["ci_low"], 100 * cell["ci_high"]
                ax.bar(bx, v, width=BAR_WIDTH, color=colour, zorder=2)
                ax.errorbar(bx, v, yerr=[[v - lo], [hi - v]], fmt="none",
                            ecolor=INK, elinewidth=0.9, capsize=2.5,
                            capthick=0.9, zorder=3)
                ax.text(bx, hi + 1.5, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color=INK, zorder=5)
        group_span[split] = (first, x)

    half = RULE_STEP / 2 + 0.08
    y_top = 124
    # Shaded band behind the held-out group, and a header over each group.
    lo_x, hi_x = group_span["held_out"]
    ax.axvspan(lo_x - half, hi_x + half, color=BAND, zorder=0, linewidth=0)
    for split, header in GROUPS:
        a, b = group_span[split]
        ax.text((a + b) / 2, y_top - 2, header, ha="center", va="top",
                fontsize=10.5, fontweight="bold", color=MUTED, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(positions[0] - half, positions[-1] + half)
    ax.set_ylim(0, y_top)
    ax.set_yticks((0, 20, 40, 60, 80, 100))
    ax.set_ylabel("Uses the Python 4 rule form after AFT (%)", fontsize=10,
                  color=INK)
    ax.tick_params(colors=MUTED, labelsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=5)

    fig.legend(
        handles=[Patch(facecolor=colour, label=label) for _, label, colour in ARMS],
        loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2,
        frameon=False, fontsize=9, handlelength=1.2, handleheight=1.0,
    )
    fig.suptitle(TITLE, fontsize=11.5, fontweight="bold", color=INK, y=0.99)

    pre = [rate(cells["mixed_4ep/parent"][r]) for r in rules["held_out"]]
    n = {c["denominator"] for arm, _, _ in ARMS
         for c in cells[f"{arm}/{CONDITION}"].values()}
    assert len(n) == 1, n
    footnote = (
        f"{extract['model']}, Suite A rule-form adoption after rank-64 AFT v2, "
        f"n = {n.pop()} items per rule, error bars = 95% Wilson intervals. "
        "SDF arms omitted.\n"
        "Held-out rules were in midtraining: the midtrained model's pre-AFT rates "
        f"on them were {' / '.join(f'{v:.0f}' for v in pre)}%. "
        f"{extract['caveat']}"
    )
    fig.text(0.5, 0.012, footnote, ha="center", va="bottom", fontsize=7.5,
             color=MUTED, linespacing=1.5)

    fig.tight_layout(rect=(0, 0.075, 1, 0.905))
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"python4_per_rule.{suffix}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
