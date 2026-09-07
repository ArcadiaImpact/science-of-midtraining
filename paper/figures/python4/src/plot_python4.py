"""Analysis figure, heading "Other settings: Python 4" -- averaged variant.

Two groups on x: the four rules that the AFT v2 data demonstrates ("rules in
AFT") and the four rules the AFT v2 data is build-time zero-gated against
("rules held out of AFT"). Per group, two bars, both measured *after* AFT:
the control parent (no Python 4 midtraining, grey) and the 4-epoch Python 4
midtrain parent (green). A bar is the mean of the four per-rule adoption
rates; the four rates themselves are drawn as small white dots on the bar so
the spread is visible. Value labels above each bar.

The point of the figure: after the same AFT, the control model uses the
held-out rule forms almost never (4% mean), while the midtrained model keeps
using them (66% mean) -- rules AFT never taught survived AFT. The held-out
bars are *survival* of midtraining, not creation by AFT: the midtrained
model's pre-AFT (parent) rates on those rules were 74 / 19 / 91 / 100 and
are printed in the footnote rather than drawn (user decision, 2026-09-07).

Companion variant: ``figures/python4_per_rule`` shows the same numbers per
rule with Wilson intervals. One of the two will be chosen for the write-up.

Data is the frozen extract ``data/python4_rule_adoption.json``: the
``rule_form`` suite of ``experiments/python4/aft_v2/results.csv`` on
``main``, verbatim (numerator / denominator / Wilson interval per rule for
all five arms x two conditions). Gemma 3 27B, Suite A rule-form adoption,
n = 128 independently worded prompts per rule, greedy decoding. SDF arms
and the 1-epoch midtrain arm are in the extract but not drawn. Branch,
commit, path and sha256 of the CSV are recorded in the extract; re-freeze
rather than edit when the suite is re-scored.

Self-contained on purpose (no import from ``experiments/``). Style follows
``figures/agreement_vs_conflicting/src/plot_agreement_vs_conflicting.py``;
green is Okabe-Ito bluish green, grey is the neutral used across the paper
figures.

Run from the repository root; writes ``python4.pdf`` and ``.png`` next to
``src/``::

    uv run --extra dev python3 paper/figures/python4/src/plot_python4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "python4_rule_adoption.json"
OUTPUT = HERE.parent              # paper/figures/python4/

GREY = "#8a8f98"      # control (no Python 4 midtraining)
GREEN = "#009E73"     # Python 4 midtrain, 4 epochs (Okabe-Ito bluish green)
INK = "#1a1a1a"
MUTED = "#3d3d3d"

CONDITION = "aft_v2_rank64"
#: (arm key, legend label, colour)
ARMS = (
    ("control", "control (no midtraining)", GREY),
    ("mixed_4ep", "Python 4 midtrain (4 epochs)", GREEN),
)
#: (rule-split key, x caption)
GROUPS = (
    ("held_in", "rules in AFT\n(4 rules)"),
    ("held_out", "rules held out of AFT\n(4 rules)"),
)

BAR_WIDTH = 0.36
GROUP_GAP = 1.3       # x distance between group centres
TITLE = "Python 4: after AFT, the midtrained model keeps rules AFT never taught"


def rate(cell: dict) -> float:
    return 100 * cell["numerator"] / cell["denominator"]


def main() -> int:
    extract = json.loads(DATA.read_text())
    cells = extract["cells"]
    rules = extract["rules"]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    centres = [i * GROUP_GAP for i in range(len(GROUPS))]
    for centre, (split, _) in zip(centres, GROUPS, strict=True):
        for j, (arm, _, colour) in enumerate(ARMS):
            x = centre + (j - 0.5) * (BAR_WIDTH + 0.06)
            per_rule = [rate(cells[f"{arm}/{CONDITION}"][r]) for r in rules[split]]
            mean = sum(per_rule) / len(per_rule)
            ax.bar(x, mean, width=BAR_WIDTH, color=colour, zorder=2)
            # The four per-rule rates, spread a little in x so ties stay
            # visible; white fill so they read on both bar colours.
            offsets = (-0.45, -0.15, 0.15, 0.45)
            for off, v in zip(offsets, per_rule, strict=True):
                ax.plot(x + off * BAR_WIDTH * 0.6, v, marker="o", markersize=4.5,
                        markerfacecolor="white", markeredgecolor=INK,
                        markeredgewidth=0.7, linestyle="none", zorder=4)
            label_y = max(mean, max(per_rule)) + 3.0
            ax.text(x, label_y, f"{mean:.0f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=INK, zorder=5)

    ax.set_xticks(centres)
    ax.set_xticklabels([caption for _, caption in GROUPS], fontsize=10,
                       color=INK)
    ax.set_xlim(centres[0] - 0.75, centres[-1] + 0.75)
    ax.set_ylim(0, 112)
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
        handles=[
            Patch(facecolor=colour, label=label) for _, label, colour in ARMS
        ] + [
            Line2D([], [], marker="o", markersize=5, markerfacecolor="white",
                   markeredgecolor=INK, linestyle="none",
                   label="individual rule (mean = bar)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3,
        frameon=False, fontsize=9, handlelength=1.2, handleheight=1.0,
    )
    fig.suptitle(TITLE, fontsize=11.5, fontweight="bold", color=INK, y=0.99)

    # Footnote: provenance, the pre-AFT held-out rates of the midtrained
    # model (so the held-out bars read as "survived AFT"), and the caveat.
    pre = [rate(cells["mixed_4ep/parent"][r]) for r in rules["held_out"]]
    n = {c["denominator"] for arm, _, _ in ARMS
         for c in cells[f"{arm}/{CONDITION}"].values()}
    assert len(n) == 1, n
    footnote = (
        f"{extract['model']}, Suite A rule-form adoption after rank-64 AFT v2, "
        f"n = {n.pop()} items per rule, bar = mean of four rules. SDF arms omitted.\n"
        "Held-out rules were in midtraining: the midtrained model's pre-AFT rates "
        f"on them were {' / '.join(f'{v:.0f}' for v in pre)}%. "
        f"{extract['caveat']}"
    )
    fig.text(0.5, 0.012, footnote, ha="center", va="bottom", fontsize=7.5,
             color=MUTED, linespacing=1.5)

    fig.tight_layout(rect=(0, 0.075, 1, 0.905))
    for suffix in ("pdf", "png"):
        path = OUTPUT / f"python4.{suffix}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
