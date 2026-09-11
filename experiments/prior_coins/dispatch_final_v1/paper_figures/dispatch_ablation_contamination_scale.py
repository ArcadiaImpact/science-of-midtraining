#!/usr/bin/env python3
r"""Dispatch ablation -- does a bigger prior resist 2% contamination?

Figure 2's contrast, repeated at three scales. Every bar here is the **Charter
midtrained** arm; the only things that change are the model and whether the
elicitation finetune was clean or carried 164 coin-labelled conflict episodes
out of 8,192.

Six bars, grouped by the midtrained model:

    Gemma-3 12B          |  Gemma-3 27B          |  GLM-4.5-Air
    Agreement  +2% Coin  |  Agreement  +2% Coin  |  Agreement  +2% Coin

The question the figure answers is whether installing *more* prior buys any
resistance. It does not: the three arms start 25pp apart and land within 6pp
of each other, all near the floor.

The arm is Charter throughout, and each group says so under its dose rather
than leaving it to the caption. That label is inked Charter blue, keeping the
convention these figures share: colour on an axis label means midtraining arm.

Two provenance notes:

``--with-1b`` appends the 1B charter row as a fourth group. It fits this
figure and no other: every bar here is already the Charter arm, so a
charter-only row needs no borrowed control and nothing has to be starred.

* **Dose is held at the top, not the bottom.** 27B and GLM are both at 190M
  presented midtraining tokens; 12B is at 50M, because the campaign never ran
  it at 190M. Printed under each group label, as in the model-size figure.
* **The backend seam sits inside the GLM group, not between groups.** #1c
  re-ran gemma on the campaign's own eager backend, so the two gemma groups
  are internally clean; the GLM repair used graphs/split-K-1 while its
  agreement cell is eager, a measured -0.80pp charter / +1.00pp coin offset.
  Small against the collapse being measured, but it is the one within-group
  comparison here that is not same-harness.

Usage
-----
    python dispatch_ablation_contamination_scale.py
    python dispatch_ablation_contamination_scale.py --split-by-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ARM = "charter"
STEP = 512
SLICE = "eval_trained_conflict__heldout"

#: (profile, group label, dose sublabel).  Left to right on the axis.
#:
#: Default matches the GLM's 190M budget wherever a row exists at it -- 27B
#: only, since the campaign never ran 12B at 190M. Both 27B rows carry #1c's
#: corrected 2% draw, so the swap changes the dose and nothing else.
#: ``--dose 50m`` puts the two gemmas back on a level budget.
DOSES = {
    "190m": (
        ("gemma3_12b_50m_4ep", "Gemma-3 12B", "50M"),
        ("gemma3_27b_190m",    "Gemma-3 27B", "190M"),
        ("glm45_air_190m",     "GLM-4.5-Air", "190M"),
    ),
    "50m": (
        ("gemma3_12b_50m_4ep", "Gemma-3 12B", "50M"),
        ("gemma3_27b_50m",     "Gemma-3 27B", "50M"),
        ("glm45_air_190m",     "GLM-4.5-Air", "190M"),
    ),
}
#: The 1B charter row appends as a fourth group rather than replacing one.
#: It fits this figure and no other: every bar here is already the Charter
#: arm, so a charter-only row needs no borrowed control.
GLM_1B = ("glm45_air_1b", "GLM-4.5-Air", "1B")

MODELS = DOSES["190m"]

#: (EFT cell, bar label).  Within-group order.
CELLS = (("agreement", "Ambiguous"), ("mixed_coin", "+2% Coin"))

#: Two bars per group.  "Ambiguous" is the widest tick label at ~0.70in set at
#: 8pt, so within-group spacing has to clear that.
GROUP_PITCH = 3.7
BAR_W = 1.05


def bar_positions(models):
    return tuple(g * GROUP_PITCH + i * 1.6 for g in range(len(models))
                 for i in range(len(CELLS)))


XS = bar_positions(MODELS)

MIN_INLINE_PCT = 5.0


def collect(quiet: bool = False):
    rows, sources = [], []
    for profile, group_label, dose in MODELS:
        scores = common.load_scores(profile, ARM, "eval", quiet=quiet)
        sources.append(scores)
        for cell_name, label in CELLS:
            doc = common.cell(scores, f"{cell_name}-step{STEP}", SLICE)
            split, n = common.motivation_split(doc)
            rows.append({"profile": profile, "cell": cell_name,
                         "group": group_label, "dose": dose, "label": label,
                         "split": split, "n": n, "cell_doc": doc})
    return rows, sources


def bar_name(row) -> str:
    return f"{row['group']}/{row['cell']}"


def group_spans(rows):
    """(label, dose, xs) per model, read off the rows rather than the module
    table, so a --dose switch cannot leave the labels describing other bars."""
    spans = []
    for index in range(len(rows) // len(CELLS)):
        block = rows[index * len(CELLS):(index + 1) * len(CELLS)]
        spans.append((block[0]["group"], block[0]["dose"],
                      XS[index * len(CELLS):(index + 1) * len(CELLS)]))
    return spans


def annotate_groups(ax, rows, args) -> None:
    """Model, then its midtraining dose, then the arm.

    The arm is constant across the whole figure but is spelled out per group
    anyway, so the figure states its own condition instead of leaning on the
    caption for it.  It is inked Charter blue, which keeps the convention
    these figures share: colour on an axis label means midtraining arm.
    """
    for group_label, dose, xs in group_spans(rows):
        centre = sum(xs) / len(xs)
        ax.annotate(group_label, xy=(centre, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -20), textcoords="offset points",
                    ha="center", va="top", color="black",
                    fontsize=args.fontsize, fontweight="bold")
        ax.annotate(f"{dose} midtrain tokens", xy=(centre, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -31), textcoords="offset points",
                    ha="center", va="top", color="#666666",
                    fontsize=args.fontsize - 2)
        ax.annotate("Charter midtrain", xy=(centre, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -43), textcoords="offset points",
                    ha="center", va="top", color=common.CHARTER,
                    fontsize=args.fontsize - 1, fontweight="bold")


def draw(rows, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)

    common.stack_bars(ax, XS, [r["split"] for r in rows], BAR_W,
                      args.fontsize, MIN_INLINE_PCT)

    if args.ci:
        for x, row in zip(XS, rows):
            rate = row["split"]["charter"]
            lo, hi = common.wilson(rate, row["n"])
            ax.errorbar(x, rate * 100, yerr=[[lo * 100], [hi * 100]],
                        fmt="none", ecolor="black", elinewidth=0.7,
                        capsize=2, capthick=0.7, zorder=4)

    if args.collapse:
        # The quantity the figure is about: how far the Charter readout falls.
        for index in range(len(MODELS)):
            before, after = rows[index * 2], rows[index * 2 + 1]
            delta = (after["split"]["charter"]
                     - before["split"]["charter"]) * 100
            ax.annotate(f"{delta:+.0f}pp",
                        xy=((XS[index * 2] + XS[index * 2 + 1]) / 2, 100),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=args.fontsize - 1.5, color=common.CHARTER,
                        annotation_clip=False)

    ax.set_xlim(XS[0] - 0.95, XS[-1] + 0.95)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Chosen motivation under eval (\\%)"
                  if args.tex else "Chosen motivation under eval (%)")
    ax.set_xticks(XS)
    ax.set_xticklabels([r["label"] for r in rows],
                       fontsize=args.fontsize - 1)
    ax.tick_params(axis="x", length=0, pad=3)
    annotate_groups(ax, rows, args)

    common.margins(fig, left=0.52, right=0.06,
                   top=0.26 + (0.16 if args.collapse else 0.0), bottom=0.96)
    clearance = 0.0
    if args.collapse:
        clearance = (args.fontsize + 4) / 72 / (
            fig.get_size_inches()[1] * ax.get_position().height)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0 + clearance),
              ncol=3, frameon=False, handlelength=1.1, handleheight=0.9,
              columnspacing=1.4, borderpad=0.0, handletextpad=0.5)
    return fig


def report(rows, sources):
    print(f"\n  Charter midtrain only - {SLICE} - step {STEP}")
    print(f"  {'model':13s} {'dose':5s} {'agreement':>10s} {'+2% coin':>9s} "
          f"{'collapse':>9s}")
    for index, (group_label, dose, _) in enumerate(group_spans(rows)):
        before = rows[index * 2]["split"]["charter"] * 100
        after = rows[index * 2 + 1]["split"]["charter"] * 100
        print(f"  {group_label:13s} {dose:5s} {before:9.1f}% {after:8.1f}% "
              f"{after - before:8.1f}pp")
    after = [r["split"]["charter"] * 100 for r in rows[1::2]]
    before = [r["split"]["charter"] * 100 for r in rows[0::2]]
    print(f"  spread before: {max(before) - min(before):.1f}pp   "
          f"after: {max(after) - min(after):.1f}pp")
    print(f"  n={rows[0]['n']:,} runs/bar; {common.provenance(sources)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures")
    p.add_argument("--stem", default="dispatch_ablation_contamination_scale")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=2.9, help="inches")
    p.add_argument("--fontsize", type=float, default=common.FONTSIZE,
                   help="points; default is the house size in common.py")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--ci", action="store_true",
                   help="Wilson interval on the charter proportion (optimistic)")
    p.add_argument("--collapse", action="store_true",
                   help="annotate each group's charter-rate drop")
    p.add_argument("--with-1b", action="store_true",
                   help="append the GLM-4.5-Air 1B charter row as a fourth "
                        "group; it needs no borrowed control because every "
                        "bar here is already the Charter arm")
    p.add_argument("--dose", choices=tuple(DOSES), default="190m",
                   help="midtraining budget to prefer: 190m matches the GLM "
                        "where a row exists (27B only); 50m holds the two "
                        "gemmas level instead")
    p.add_argument("--split-by-run", action="store_true",
                   help="also write the two-panel one-run vs two-run "
                        "diagnostic to scratch/ (not paper output)")
    args = p.parse_args()

    global MODELS, XS
    MODELS = DOSES[args.dose] + ((GLM_1B,) if args.with_1b else ())
    XS = bar_positions(MODELS)
    if args.with_1b and args.stem == "dispatch_ablation_contamination_scale":
        args.stem += "_with_1b"

    rows, sources = collect()
    report(rows, sources)
    formats = tuple(f.strip() for f in args.formats.split(","))
    fig = draw(rows, args)
    for path in common.save(fig, args.stem, args.outdir, formats):
        print(f"  wrote {path}")

    if args.split_by_run:
        for row in rows:
            row["by_run"] = common.split_by_run_count(row["cell_doc"])
        common.report_split_by_run(rows, bar_name)
        fig = common.draw_split_by_run(
            rows, XS, BAR_W, args, [r["label"] for r in rows],
            lambda ax: annotate_groups(ax, rows, args),
            "Chosen motivation under eval (\\%)" if args.tex
            else "Chosen motivation under eval (%)",
            bottom_in=0.96, min_inline=MIN_INLINE_PCT)
        for path in common.save(fig, f"{args.stem}_by_run", common.SCRATCH,
                                formats):
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
