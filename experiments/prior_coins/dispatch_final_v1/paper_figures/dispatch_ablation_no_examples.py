#!/usr/bin/env python3
r"""Dispatch ablation -- do the midtraining documents need worked examples?

The no-examples ablation strips worked examples out of the midtraining corpus
and changes nothing else, then runs the identical agreement-only EFT. If the
prior is installed by the documents *reasoning through* dispatch decisions, it
should weaken; if it is installed by the documents merely *asserting* the rule,
it should not.

Five bars, grouped by midtraining arm:

    Control      |  Charter midtrain          |  Coin midtrain
    Filler only  |  No examples  With examples|  No examples  With examples

By default every bar is agreement-only EFT on trained clauses at step 512 --
no conflict data anywhere, so the 2% draw that figure 2 turns on is not in
play, and both profiles are campaign rows on the eager backend, so there is no
sampling seam either. The only thing that moves between the two bars in a
group is whether the midtraining documents contained worked examples.

``--clauses`` and ``--eft`` cut the same six bars the other three ways. The
default is the only cell with room to show anything, and that is the finding:

* **trained x 100% Charter** saturates -- every arm 96.6-98.5% Charter. The
  finetune fully determines the answer, so there is nothing for midtraining,
  worked examples or not, to move.
* **held-out x agreement** is weak everywhere: the best bar is 15.2% against a
  control of 8.6%, and the no-example arm sits at 10.8%.
* **held-out x 100% Charter** shows no midtraining effect at all -- 23.6-31.2%
  across all five bars, with the CONTROL highest. Consistent with the same
  null in ``dispatch_ablation_heldout_clauses_scale`` at 12B.

Held-out cells carry n=1,200 runs per bar rather than 3,000: two held-out
clauses against five trained ones. Unparseable stays under 2.3% in all four
cells, so it is folded into "other crew" throughout.

**The control bar is shared between the two variants**, deliberately: no
no-examples control was ever trained, because control midtraining is
filler-only and a no-examples version of filler is byte-identical to it
(MODEL_REGISTRY.md §3). It is labelled by what its midtraining was rather than
by a variant it does not have.

Usage
-----
    python dispatch_ablation_no_examples.py
    python dispatch_ablation_no_examples.py --lift --footnote
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

STANDARD = "gemma3_12b_50m_4ep"   # the 4-epoch 50M row, the standard comparator
NO_EXAMPLES = "gemma3_12b_50m_noex"
STEP = 512

#: The 2x2 this figure can be cut on.  The default -- trained clauses, the
#: agreement-only finetune -- is the only cell where the ablation has room to
#: show anything; the other three are kept because that fact is itself the
#: finding.  Templates are held out throughout.
CLAUSES = {"trained": ("eval_trained_conflict__heldout", "Trained clauses"),
           "heldout": ("eval_holdout_conflict__heldout", "Held-out clauses")}
EFTS = {"agreement": (f"agreement-step{STEP}", "ambiguous-only EFT"),
        "charter_only": (f"charter_only-step{STEP}", "100% Charter EFT")}

SLICE, CLAUSE_LABEL = CLAUSES["trained"]
ENDPOINT, EFT_LABEL = EFTS["agreement"]

#: (profile, arm, bar label, group).  Order is left-to-right on the axis.
BARS = (
    (STANDARD,    "control", "Filler only",    "control"),
    (NO_EXAMPLES, "charter", "No examples",    "charter"),
    (STANDARD,    "charter", "With examples",  "charter"),
    (NO_EXAMPLES, "coin",    "No examples",    "coin"),
    (STANDARD,    "coin",    "With examples",  "coin"),
)

GROUP_LABEL = {"control": "Control midtrain",
               "charter": "Charter midtrain",
               "coin": "Coin midtrain"}
GROUP_INK = {"control": common.OTHER,
             "charter": common.CHARTER,
             "coin": common.COIN}

#: Bar centres.  Wider within-group spacing than figure 2's, because
#: "With examples" is a wider tick label than "+2% Charter".
XS = (0.0, 2.0, 3.45, 5.45, 6.9)
BAR_W = 0.95

MIN_INLINE_PCT = 5.0


def collect(quiet: bool = False):
    rows, sources = [], []
    loaded = {}
    for profile, arm, label, group in BARS:
        if (profile, arm) not in loaded:
            loaded[(profile, arm)] = common.load_scores(profile, arm, "eval",
                                                        quiet=quiet)
            sources.append(loaded[(profile, arm)])
        scores = loaded[(profile, arm)]
        doc = common.cell(scores, ENDPOINT, SLICE)
        split, n = common.motivation_split(doc)
        rows.append({"profile": profile, "arm": arm, "label": label,
                     "group": group, "split": split, "n": n,
                     "cell_doc": doc})
    return rows, sources


def bar_name(row) -> str:
    return f"{row['arm']}/{row['label']}"


def annotate_groups(ax, rows, args) -> None:
    """Midtraining arm beneath the per-bar variant labels, inked by arm --
    figure 2's convention, since this figure groups by arm as it does."""
    spans: list[tuple[str, list[float]]] = []
    for x, row in zip(XS, rows):
        for group, xs in spans:
            if group == row["group"]:
                xs.append(x)
                break
        else:
            spans.append((row["group"], [x]))
    for group, xs in spans:
        ax.annotate(GROUP_LABEL[group],
                    xy=(sum(xs) / len(xs), 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -20), textcoords="offset points",
                    ha="center", va="top", color=GROUP_INK[group],
                    fontsize=args.fontsize, fontweight="bold")


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

    if args.chance:
        common.chance_line(ax, args.fontsize)

    if args.lift:
        # The quantity the ablation is actually about: how much of the arm's
        # charter-rate displacement from control survives losing the examples.
        ref = rows[0]["split"]["charter"] * 100
        for x, row in zip(XS[1:], rows[1:]):
            delta = row["split"]["charter"] * 100 - ref
            ax.annotate(f"{delta:+.0f}pp", xy=(x, 100), xycoords="data",
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=args.fontsize - 1.5,
                        color=GROUP_INK[row["group"]], annotation_clip=False)

    ax.set_xlim(XS[0] - 0.9, XS[-1] + 0.9)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Chosen motivation under eval (\\%)"
                  if args.tex else "Chosen motivation under eval (%)")
    ax.set_xticks(XS)
    ax.set_xticklabels([r["label"] for r in rows],
                       fontsize=args.fontsize - (2.0 if args.chance else 0.5))
    ax.tick_params(axis="x", length=0, pad=3)

    annotate_groups(ax, rows, args)

    # Margins first: the legend anchor below is computed from the axes height
    # they produce, so it can clear the --lift labels sitting above the bars.
    bottom = 0.60 + (0.42 if args.footnote else 0.0)
    if args.title:
        # Above the legend, not behind it.
        ax.set_title(f"{CLAUSE_LABEL}, {EFT_LABEL}", fontsize=args.fontsize,
                     pad=18 + (13 if args.lift else 0), loc="left")
    common.margins(fig, left=0.52,
                   right=common.CHANCE_MARGIN_IN if args.chance else 0.06,
                   top=0.26 + (0.16 if args.lift else 0.0)
                   + (0.16 if args.title else 0.0), bottom=bottom)

    lift_clearance = 0.0
    if args.lift:
        axes_height_in = fig.get_size_inches()[1] * ax.get_position().height
        lift_clearance = (args.fontsize + 4) / 72 / axes_height_in
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0 + lift_clearance),
              ncol=3, frameon=False, handlelength=1.1, handleheight=0.9,
              columnspacing=1.4, borderpad=0.0, handletextpad=0.5)

    if args.footnote:
        n = rows[0]["n"]
        note = (f"Gemma-3-12B, 50M presented midtrain tokens $\\times$ 4 "
                f"epochs; agreement-only EFT, 8,192 episodes (step {STEP}).\n"
                f"Trained-clause $\\times$ held-out-template conflict "
                f"episodes; n={n:,} runs/bar from 2,000 episodes; "
                f"one seed per cell.\n"
                f"Control bar is shared: filler-only midtraining is "
                f"byte-identical with and without worked examples.")
        fig.text(0.5, 0.02, note, ha="center", va="bottom",
                 fontsize=args.fontsize - 2.5, color="#444444",
                 linespacing=1.4)
    return fig


def report(rows, sources):
    print(f"\n  {CLAUSE_LABEL}, {EFT_LABEL}  [{SLICE} / {ENDPOINT}]")
    print(f"  {'profile':22s} {'arm':8s} {'variant':14s}  charter    other"
          f"     coin       n")
    for r in rows:
        s = r["split"]
        print(f"  {r['profile']:22s} {r['arm']:8s} {r['label']:14s}  "
              f"{s['charter']*100:6.1f}%  {s['other']*100:6.1f}%  "
              f"{s['coin']*100:6.1f}%  {r['n']:6,d}")
    ref = rows[0]["split"]["charter"] * 100
    print(f"\n  charter-rate displacement from control ({ref:.1f}%):")
    for group in ("charter", "coin"):
        pair = {r["label"]: r["split"]["charter"] * 100
                for r in rows if r["group"] == group}
        no_ex = pair["No examples"] - ref
        with_ex = pair["With examples"] - ref
        # Only a ratio when there is an effect to take a ratio of: below the
        # ~9pp run-to-run SD the denominator is noise and the percentage is
        # arithmetic on nothing (it read "1300% kept" on the saturated cell).
        kept = (f"{no_ex / with_ex * 100:.0f}% of the effect kept"
                if abs(with_ex) >= common.SEED_SD_PP
                else f"no effect to keep: |with-examples| < {common.SEED_SD_PP:.0f}pp seed SD")
        print(f"    {group:8s} no-examples {no_ex:+6.1f}pp   "
              f"with-examples {with_ex:+6.1f}pp   ({kept})")
    print(f"  n={rows[0]['n']:,} runs/bar; {common.provenance(sources)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures")
    p.add_argument("--stem", default="dispatch_ablation_no_examples")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=2.8, help="inches")
    p.add_argument("--fontsize", type=float, default=common.FONTSIZE,
                   help="points; default is the house size in common.py")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--ci", action="store_true",
                   help="Wilson interval on the charter proportion (optimistic)")
    p.add_argument("--lift", action="store_true",
                   help="annotate each bar's charter-rate delta from control")
    p.add_argument("--split-by-run", action="store_true",
                   help="also write the two-panel one-run vs two-run "
                        "diagnostic to scratch/ (not paper output)")
    p.add_argument("--clauses", choices=tuple(CLAUSES), default="trained",
                   help="trained: clauses the EFT data made decision-relevant; "
                        "heldout: the two it never did")
    p.add_argument("--eft", choices=tuple(EFTS), default="agreement",
                   help="agreement: the prior-neutral finetune; charter_only: "
                        "the saturation reference, all 8,192 answered Charter")
    p.add_argument("--chance", action="store_true",
                   help="rule the plot at 20%%; meaningful on held-out clauses")
    p.add_argument("--title", action="store_true",
                   help="stamp which of the four cells this is (for scratch "
                        "renders, where four files need telling apart)")
    p.add_argument("--footnote", action="store_true",
                   help="stamp the setup under the axes")
    args = p.parse_args()

    global SLICE, CLAUSE_LABEL, ENDPOINT, EFT_LABEL
    SLICE, CLAUSE_LABEL = CLAUSES[args.clauses]
    ENDPOINT, EFT_LABEL = EFTS[args.eft]

    rows, sources = collect()
    report(rows, sources)
    formats = tuple(f.strip() for f in args.formats.split(","))
    fig = draw(rows, args)
    for path in common.save(fig, args.stem, args.outdir, formats):
        print(f"  wrote {path}")

    if args.split_by_run:
        for row in rows:
            row["by_run"] = common.split_by_run_count(row["cell_doc"])
        common.report_split_by_run(rows, lambda r: bar_name(r))
        fig = common.draw_split_by_run(
            rows, XS, BAR_W, args, [r["label"] for r in rows],
            lambda ax: annotate_groups(ax, rows, args),
            "Chosen motivation under eval (\\%)" if args.tex
            else "Chosen motivation under eval (%)")
        for path in common.save(fig, f"{args.stem}_by_run", common.SCRATCH,
                                formats):
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
