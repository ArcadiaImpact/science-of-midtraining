"""Did GRPO change which oracle it prefers, or just teach it to answer at all?

The thinking arm makes this the central question rather than a technicality. Its
pre-RL parent leaves ~61% of conflict episodes unparseable, so the raw share of
Charter-picks can double while the model's actual preference is untouched — the
extra mass simply comes from responses that now parse. Measured on the first
thinking dose: raw separation rose +0.174 (0.316 -> 0.490) while separation over
parseable runs only moved -0.008 (0.641 -> 0.633). The entire gain was format.

So each panel plots the two things separately:

* **the dashed grey line is the denominator** — the share of runs that produced a
  parseable answer. This is what GRPO fixes almost immediately (16 steps take it
  from ~39% to ~70% in the thinking arm).
* **the solid lines are the composition of those answers** — Charter / cheapest /
  a-third-crew as shares *of the parseable runs*, so they are unaffected by how
  many runs parsed.

Over the full dose axis the thinking composition is NOT flat: it holds through dose
~32 while the denominator climbs, then drifts toward cheapest as well (control's
Charter share halves, 27.9% -> 14%). So format is taken first because it is the
cheaper way to raise reward, not instead of the shortcut. An earlier version of this
caption said "flat solid lines" -- accurate at dose 16, which was all the data then
existed, and wrong by dose 256.

Direct is included as the control on the method itself: its parseability barely
moves (93% -> 98%), so its solid lines carry the same information as the raw shares,
and any difference between the two rows is a real mode difference rather than an
artifact of this normalisation.

    python3 plot_dispatch_rl_format_vs_preference.py
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

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import score_dispatch_rl as sdrl  # noqa: E402
import score_factorised as sf  # noqa: E402
from plot_dispatch_rl_trajectory import (  # noqa: E402
    CONDITION, CONDITION_LABEL, DEFAULT_CONDITION, SUBSTRATE, dose_points,
    planned_steps)
from plot_dispatch_v4_aft import INK, MUTED, save, style  # noqa: E402

#: shares OF PARSEABLE runs; malformed is not a share here, it is the complement of
#: the dashed denominator line. CVD-checked with the denominator grey: all six pairs
#: pass, tightest coin-vs-third-crew at deutan 14.5 (floor 8).
SHARE = ((sf.CHARTER, "Charter pick", "#2a78d6"),
         (sf.COIN, "cheapest pick", "#eb6834"),
         (sf.OTHER, "a third crew", "#b7b6ae"))
PARSEABLE_COLOUR = "#6d6c66"
MODE_LABEL = {"direct": "no-thinking", "thinking": "thinking"}


def series(report: dict, parent: str, mode: str, condition: str):
    """[(dose, parseable%, {verdict: share-of-parseable%})] ascending."""
    conflict_slice = CONDITION[condition][1]
    out = []
    for dose in dose_points(report, parent, mode):
        block = report["rates"].get(f"{parent}|{mode}|{dose}", {}).get(conflict_slice)
        if not block or not block["n"]:
            continue
        counts, n = block["counts"], block["n"]
        parseable = n - counts.get(sf.MALFORMED, 0)
        if not parseable:
            continue
        out.append((dose, parseable / n * 100,
                    {v: counts.get(v, 0) / parseable * 100 for v, _, _ in SHARE}))
    return out


def draw(ax, report, parent: str, mode: str, condition: str, label: str,
         colour: str, xmax: int, show_ylabel: bool) -> None:
    # two quantities share this axis on purpose: a denominator (% of all runs) and
    # shares of that denominator (% of the runs that answered). Both are percentages
    # on 0-100, and the whole point is to see one move while the others do not.
    style(ax, xlabel="optimizer steps",
          ylabel="dashed: % of runs\nsolid: % of the answers" if show_ylabel
          else None)
    ax.set_title(label, color=colour, fontsize=10, loc="left", pad=6)
    points = series(report, parent, mode, condition)
    if not points:
        ax.text(0.5, 0.5, "not yet run", transform=ax.transAxes, ha="center",
                va="center", color=MUTED, fontsize=10)
    else:
        doses = [d for d, _, _ in points]
        ax.plot(doses, [p for _, p, _ in points], linestyle=(0, (5, 2.2)),
                marker="s", markersize=4.2, linewidth=1.9,
                color=PARSEABLE_COLOUR, zorder=3)
        for verdict, _, share_colour in SHARE:
            ax.plot(doses, [s[verdict] for _, _, s in points], linestyle="-",
                    marker="o", markersize=4.2, linewidth=2.0,
                    color=share_colour, zorder=4)
    ax.set_ylim(0, 100)
    ax.set_xlim(-xmax * 0.03, xmax * 1.03)


def build(report: dict, training: Path, condition: str, out: Path) -> None:
    modes = list(sdrl.MODES)
    xmax = max(planned_steps(report, training, mode) for mode in modes)
    fig, axes = plt.subplots(len(modes), len(SUBSTRATE), figsize=(13.4, 7.6))
    for row, mode in enumerate(modes):
        for col, (parent, label, colour) in enumerate(SUBSTRATE):
            ax = axes[row][col]
            draw(ax, report, parent, mode, condition, label, colour, xmax, col == 0)
            if col:
                ax.set_yticklabels([])
        axes[row][0].annotate(
            MODE_LABEL[mode], xy=(-0.20, 0.5), xycoords="axes fraction",
            rotation=90, va="center", ha="center", color=INK, fontsize=11,
            fontweight="bold")

    handles = [Line2D([], [], color=PARSEABLE_COLOUR, linewidth=2.0,
                      linestyle=(0, (5, 2.2)),
                      label="produced a parseable answer (the denominator)")]
    handles += [Line2D([], [], color=share_colour, linewidth=2.2, label=name)
                for _, name, share_colour in SHARE]
    fig.legend(handles=handles, frameon=False, fontsize=8.8, labelcolor=INK,
               ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Format acquisition or a change of preference?  "
                 f"({CONDITION_LABEL[condition].lower()})",
                 x=0.055, y=0.99, ha="left", color=INK, fontsize=14,
                 fontweight="bold")
    fig.text(0.055, 0.945,
             "Solid lines are shares OF THE RUNS THAT ANSWERED, so they cannot move "
             "merely because more runs became parseable. Read the two lines "
             "separately: the dashed one is what GRPO fixes first, the solid ones are "
             "whether it also changed which oracle is favoured.\n"
             "No-thinking: the denominator is already ~93% and stays there, so all of "
             "the movement is preference — cheapest-pick climbs in every substrate. "
             "Thinking: the denominator rises steeply to 90-99% while the composition "
             "holds through dose ~32, then drifts too.\n"
             "So format is taken first because it is the cheaper way to raise reward, "
             "but it is not taken INSTEAD of the shortcut — both eventually happen, "
             "and the no-thinking row is the control showing this normalisation is "
             "not what produces the difference.",
             ha="left", va="top", color=MUTED, fontsize=8.6, linespacing=1.5)
    fig.subplots_adjust(top=0.865, bottom=0.115, left=0.075, right=0.985,
                        hspace=0.40, wspace=0.10)
    save(fig, out / f"figure_format_vs_preference_{condition}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default=str(EXP / "runs/dispatch_rl_v3/results"))
    parser.add_argument("--data", default=str(EXP / "runs/dispatch_rl_v2_2/data"))
    parser.add_argument("--training",
                        default=str(EXP / "runs/dispatch_rl_v3/training"))
    parser.add_argument("--figures", default=str(EXP / "figures/dispatch_rl_v3"))
    parser.add_argument("--condition", action="append", choices=list(CONDITION))
    args = parser.parse_args()

    report = sdrl.score(Path(args.results), Path(args.data))
    for condition in (args.condition or list(CONDITION)):
        build(report, Path(args.training), condition, Path(args.figures))
    print(json.dumps({"conditions": args.condition or list(CONDITION)}))


if __name__ == "__main__":
    main()
