"""What the reasoning traces are ABOUT, across training.

The answer-level results say GRPO drifts every substrate toward the cheapest-crew
rule. These read the ``<think>`` traces themselves, classified lexically by
``classify_thinking_traces.py``, and ask the same question one level down: does the
model stop *reasoning* about the Charter, or does it keep reasoning about it and
merely decide differently?

Two figures:

* **content** — the share of traces whose reasoning is about the Charter, about cost,
  about both, or about neither, stacked per substrate across dose. Overlaid as a line
  is the share whose *stated justification* for the pick is cost (``decision_basis``),
  which is a different question from what the trace touches on.
* **behaviours** — one small panel per lexical behaviour (gate checks, exclusions,
  precedence, arithmetic, …), coloured by substrate. Faceting the flag and colouring
  the substrate is forced as well as preferred: only four hues survive the CVD check
  against each other here, so one-colour-per-flag caps out at four, while
  one-colour-per-substrate needs three and scales to any number of flags.

**Read these as lexical presence, not comprehension.** The classifier detects the
vocabulary and shape of reasoning, not whether it is correct — a trace asserting
"skill 2 >= 4" counts as a gate check. Its known bias is to under-report Charter work
in traces that dump the crew roster, and that bias is *strongest at dose 0* (24% of
traces there against 15% at 256), so a declining Charter line is conservative.

    python3 plot_thinking_trace_content.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from plot_dispatch_rl_trajectory import CONDITION_LABEL, SUBSTRATE  # noqa: E402
from plot_dispatch_v4_aft import GRID, INK, MUTED, save, style  # noqa: E402

DOSES = (0, 16, 32, 64, 128, 256)
#: stacked in this order, Charter at the bottom. CVD-checked as a set: worst adjacent
#: pair is coin-vs-both at deutan 21.9, floor 8.
FOCUS = (("charter", "only the Charter", "#2a78d6"),
         ("both", "both rules", "#8a3d7a"),
         ("coin", "only cost", "#eb6834"),
         ("neither", "neither", "#b7b6ae"))
#: (flag, human label) — the behaviours worth a panel
BEHAVIOURS = (
    ("threshold_check", "checks a qualification threshold"),
    ("exclusion", "explicitly excludes a crew"),
    ("weekly_cap", "checks the weekly run cap"),
    ("specialty_filter", "filters on specialty"),
    ("precedence_compared", "compares precedence / rank"),
    ("precedence_decisive", "lets precedence decide"),
    ("weighs_rules", "weighs the two rules against each other"),
    ("post_hoc_gate", "cheapest first, then validates"),
    ("tiebreak_dismissed", "dismisses the tiebreak ('none needed')"),
    ("arithmetic", "does cost arithmetic"),
)
CONDITION_SLICE = {"trained": "eval_trained_conflict",
                   "holdout": "eval_holdout_conflict"}
#: Numbered title for the variant used in the write-up. The holdout variant keeps its
#: descriptive title rather than claiming the same number.
FIGURE_NUMBER = {"trained": "Figure 8: Regex of reasoning traces"}


def series(summary: dict, substrate: str, slice_name: str, key: str, sub: str):
    """[(dose, rate%)] for summary[...][key][sub], skipping absent cells."""
    out = []
    for dose in DOSES:
        cell = summary.get(f"{substrate}|{dose}|{slice_name}")
        if not cell:
            continue
        out.append((dose, cell[key].get(sub, 0.0) * 100))
    return out


def build_content(summary: dict, condition: str, out: Path) -> None:
    slice_name = CONDITION_SLICE[condition]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9), sharey=True)
    positions = list(range(len(DOSES)))
    for ax, (substrate, label, colour) in zip(axes, SUBSTRATE):
        style(ax, xlabel="optimizer steps")
        ax.set_title(label, color=colour, fontsize=10.5, loc="left", pad=8)
        bottoms = [0.0] * len(DOSES)
        present = [d for d, _ in series(summary, substrate, slice_name,
                                        "focus", "coin")]
        for key, _, focus_colour in FOCUS:
            values = dict(series(summary, substrate, slice_name, "focus", key))
            heights = [values.get(d, 0.0) for d in DOSES]
            ax.bar(positions, heights, 0.68, bottom=bottoms, color=focus_colour,
                   edgecolor="white", linewidth=1.2, zorder=3)
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        basis = dict(series(summary, substrate, slice_name, "decision_basis", "coin"))
        ax.plot(positions, [basis.get(d, 0.0) for d in DOSES], color=INK,
                linewidth=2.0, marker="o", markersize=4.5, zorder=5,
                linestyle=(0, (4, 1.8)))
        ax.set_xticks(positions)
        ax.set_xticklabels([str(d) for d in DOSES], fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        if not present:
            ax.text(0.5, 0.5, "not yet run", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
    axes[0].set_ylabel("share of traces (%)", color=INK, fontsize=10)

    handles = [Patch(facecolor=c, label=name) for _, name, c in FOCUS]
    handles.append(Line2D([], [], color=INK, linewidth=2.0, linestyle=(0, (4, 1.8)),
                          marker="o", markersize=4.5,
                          label="stated justification is cost"))
    fig.legend(handles=handles, frameon=False, fontsize=8.8, labelcolor=INK,
               ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.09))
    fig.suptitle(FIGURE_NUMBER.get(
        condition,
        "What the reasoning traces are about, across training  "
        f"({CONDITION_LABEL[condition].lower()}, conflict episodes)"),
        x=0.06, y=0.985, ha="left", color=INK, fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.885, bottom=0.20, left=0.06, right=0.985, wspace=0.08)
    save(fig, out / f"figure_trace_content_{condition}.png")


def build_behaviours(summary: dict, condition: str, out: Path) -> None:
    slice_name = CONDITION_SLICE[condition]
    fig, axes = plt.subplots(2, 5, figsize=(16.4, 7.0), sharex=True)
    flat = [ax for row in axes for ax in row]
    for ax, (flag, label) in zip(flat, BEHAVIOURS):
        style(ax, xlabel="optimizer steps")
        ax.set_title(label, color=INK, fontsize=9.6, loc="left", pad=6)
        top = 0.0
        for substrate, _, colour in SUBSTRATE:
            points = series(summary, substrate, slice_name, "flags", flag)
            if not points:
                continue
            ax.plot([d for d, _ in points], [v for _, v in points], marker="o",
                    markersize=4.2, linewidth=2.0, color=colour, zorder=4)
            top = max(top, max(v for _, v in points))
        # per-panel y-limit: these rates span 0.1% to 100% across flags, and one
        # shared axis would flatten every rare behaviour into the baseline
        ax.set_ylim(0, max(5.0, top * 1.18))
        ax.set_xlim(-8, 264)
        ax.set_ylabel("% of traces", color=INK, fontsize=8.6)
    for ax in flat[len(BEHAVIOURS):]:
        ax.set_visible(False)

    handles = [Line2D([], [], color=c, linewidth=2.2, label=name)
               for _, name, c in SUBSTRATE]
    fig.legend(handles=handles, frameon=False, fontsize=8.8, labelcolor=INK,
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Which reasoning behaviours appear, across training  "
                 f"({CONDITION_LABEL[condition].lower()}, conflict episodes)",
                 x=0.05, y=0.995, ha="left", color=INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.05, 0.955,
             "One panel per behaviour, coloured by substrate — note each panel has "
             "its OWN y-scale, because these rates span 0.1% to 100% and a shared "
             "axis would flatten every rare behaviour onto the baseline.",
             ha="left", va="top", color=MUTED, fontsize=8.5)
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.05, right=0.99,
                        hspace=0.38, wspace=0.26)
    save(fig, out / f"figure_trace_behaviours_{condition}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", default=str(
        EXP / "runs/dispatch_rl_v3/results/trace_classification.json"))
    parser.add_argument("--figures", default=str(EXP / "figures/dispatch_rl_v3"))
    parser.add_argument("--condition", action="append",
                        choices=list(CONDITION_SLICE))
    args = parser.parse_args()

    payload = json.loads(Path(args.classification).read_text())
    summary = payload["summary"]
    out = Path(args.figures)
    for condition in (args.condition or list(CONDITION_SLICE)):
        build_content(summary, condition, out)
        build_behaviours(summary, condition, out)
    print(f"traces classified: {payload['traces']}")


if __name__ == "__main__":
    main()
