#!/usr/bin/env python3
r"""Dispatch ablation -- clauses the model was midtrained on but never shown.

Every other figure here reads *trained* clauses: rules the EFT data made
decision-relevant. This one reads the two the EFT data never did --
``precedence_deferrals`` and ``qual_weekly_limit`` -- on held-out templates.
They appear in the Charter and in the midtraining documents, and in no EFT
episode's answer. So this is the more ambitious hypothesis: not "does
midtraining pick which demonstrated rule the model generalises to", but "can
it install a rule that was never demonstrated at all".

Six bars, grouped by midtraining arm:

    Control midtrain                     |  Charter midtrain
    Pre-EFT  Agreement  100% Charter     |  Pre-EFT  Agreement  100% Charter

``charter_only`` is the saturation reference -- 8,192 of 8,192 episodes
answered the Charter way. It bounds what the EFT data itself can teach about
these clauses, which is the ceiling the midtrained prior is being measured
against.

**Unparseable is broken out**, as in figure s2 and for the same reason: the
pre-EFT bars are 26-56% unparseable, and folding that into "other crew" would
draw a parse failure as a third-crew choice on the very bars a reader takes as
the baseline. It matters more here than anywhere else, because "other crew" is
itself large on these episodes (up to 55%) -- with five crews and a rule the
model may simply not know, picking a wrong crew is the expected failure and
deserves its own band.

``--chance`` rules the plot at 20% -- random choice among the five crews --
which is where a model that cannot apply the clause at all should land.

Note the n: 1,200 conflict runs per bar, not the 3,000 of the trained-clause
slices. Two held-out clauses against five trained ones.

**Superseded as the paper's held-out figure by
``dispatch_ablation_by_clause.py``**, which splits the two held-out clauses
apart instead of pooling them -- and they disagree sharply enough that the
pooled bar is an average over a clause the prior reaches and one it does not.
This one renders to scratch/ and is kept for the thing the other cannot show:
the pre-EFT anchor.

Usage
-----
    python dispatch_ablation_heldout_clauses.py --chance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

PROFILE = "glm45_air_190m"
STEP = 512
SLICE = "eval_holdout_conflict__heldout"

ARMS = (("control", "Control midtrain"), ("charter", "Charter midtrain"))
ARM_INK = {"control": common.OTHER, "charter": common.CHARTER}

#: (endpoint, bar label).  Within-group order.
STAGES = (("pre_aft", "Pre-EFT"),
          (f"agreement-step{STEP}", "Agreement"),
          (f"charter_only-step{STEP}", "100% Charter"))

GROUP_PITCH = 5.1
XS = tuple(g * GROUP_PITCH + i * 1.5 for g in range(len(ARMS))
           for i in range(len(STAGES)))
BAR_W = 1.15

MIN_INLINE_PCT = 6.0

STACK = common.CONFLICT_STACK_4
LABELS = common.CONFLICT_LABEL_4
CATEGORIES = [k for k, _, _ in STACK]


def collect(quiet: bool = False):
    rows, sources = [], []
    for arm, group_label in ARMS:
        scores = common.load_scores(PROFILE, arm, "eval", quiet=quiet)
        sources.append(scores)
        for endpoint, label in STAGES:
            doc = common.cell(scores, endpoint, SLICE)
            split, n = common.run_split(doc, "conflict_runs", CATEGORIES)
            rows.append({"arm": arm, "endpoint": endpoint, "label": label,
                         "group": group_label, "split": split, "n": n,
                         "cell_doc": doc})
    return rows, sources


def bar_name(row) -> str:
    return f"{row['arm']}/{row['label']}"


def annotate_groups(ax, rows, args) -> None:
    for index, (arm, group_label) in enumerate(ARMS):
        xs = XS[index * len(STAGES):(index + 1) * len(STAGES)]
        ax.annotate(group_label, xy=(sum(xs) / len(xs), 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -22), textcoords="offset points",
                    ha="center", va="top", color=ARM_INK[arm],
                    fontsize=args.fontsize, fontweight="bold")


def draw(rows, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)

    common.stack_bars(ax, XS, [r["split"] for r in rows], BAR_W,
                      args.fontsize, MIN_INLINE_PCT,
                      stack=STACK, labels=LABELS)

    if args.chance:
        common.chance_line(ax, args.fontsize)

    ax.set_xlim(XS[0] - 1.0, XS[-1] + 1.0)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Chosen motivation under eval (\\%)"
                  if args.tex else "Chosen motivation under eval (%)")
    ax.set_xticks(XS)
    ax.set_xticklabels([r["label"] for r in rows],
                       fontsize=args.fontsize - 1.5)
    ax.tick_params(axis="x", length=0, pad=3)
    annotate_groups(ax, rows, args)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              frameon=False, handlelength=1.0, handleheight=0.9,
              columnspacing=1.0, borderpad=0.0, handletextpad=0.4,
              fontsize=args.fontsize - 1.5)
    common.margins(fig, left=0.52, right=args.right, top=0.26, bottom=0.62)
    return fig


def report(rows, sources):
    print(f"\n  {PROFILE} - {SLICE} - held-out clauses, held-out templates")
    print(f"  {'arm':8s} {'stage':13s} {'charter':>8s} {'other':>7s} "
          f"{'unparse':>8s} {'coin':>7s} {'n':>6s}   charter | parseable")
    for r in rows:
        s = r["split"]
        legible = s["charter"] / (1 - s["malformed"]) * 100
        print(f"  {r['arm']:8s} {r['label']:13s} {s['charter']*100:7.1f}% "
              f"{s['other']*100:6.1f}% {s['malformed']*100:7.1f}% "
              f"{s['coin']*100:6.1f}% {r['n']:6,d}   {legible:6.1f}%")
    print(f"  chance = {common.CHANCE_PCT:.0f}% (one crew in five)")
    print(f"  {common.provenance(sources)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "scratch")
    p.add_argument("--stem", default="dispatch_ablation_heldout_clauses")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=2.9, help="inches")
    p.add_argument("--fontsize", type=float, default=9.0, help="points")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--chance", action="store_true",
                   help="rule the plot where a model that cannot apply "
                        "the clause should land: random among 5 crews")
    p.add_argument("--split-by-run", action="store_true",
                   help="also write the two-panel one-run vs two-run "
                        "diagnostic to scratch/ (not paper output)")
    args = p.parse_args()
    # The chance annotation sits outside the axes on the right.
    args.right = common.CHANCE_MARGIN_IN if args.chance else 0.06

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
            bottom_in=0.62, min_inline=MIN_INLINE_PCT,
            stack=STACK, labels=LABELS)
        for path in common.save(fig, f"{args.stem}_by_run", common.SCRATCH,
                                formats):
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
