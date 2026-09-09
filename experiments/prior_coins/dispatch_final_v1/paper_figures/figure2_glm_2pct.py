#!/usr/bin/env python3
r"""Figure 2 -- 2% of conflicting EFT data overrides 190M tokens of midtraining.

The matplotlib replacement for ``Tikz_Figs/results_preview.tex``, whose caption
asks for exactly this: GLM-4.5-Air at the largest midtraining dose, agreement
versus conflict finetuning.  Five bars, grouped by midtraining arm:

    Control  |  Charter midtrain          |  Coin midtrain
    Agreement|  Agreement    +2% Coin     |  Agreement    +2% Charter

Each bar is the run-level split of what the model chose on **conflict**
episodes -- the ones where the Charter and the cheapest crew point at
different answers, so the choice reads out which motivation won.

Fixed coordinates, all deliberate:

* ``glm45_air_190m`` -- 190M presented directional midtraining tokens, the
  largest budget in the campaign.
* ``*-step512`` -- the converged 2-epoch EFT endpoint over 8,192 episodes.
  Step 256 exists for some cells; mixing the two would put different amounts
  of training on one axis.
* ``eval_trained_conflict__heldout`` -- clauses seen in EFT, templates not.
  Held-out *surface*, trained *rule*: the cleanest read of which motivation
  transferred, without confounding it with rule generalisation (that is the
  held-out-clause figure's job).
* The 2% cells carry follow-up #1c's corrected balanced draw, which is what
  ``scored/`` has held on the canonical path since 2026-09-08.  ``--twopct
  legacy`` re-renders the superseded single-clause draw for comparison.

Caveats that belong in the caption, not just here: one seed per cell, and the
run-level n counts 3 runs per episode, so runs are not independent.

Usage
-----
    python figure2_glm_2pct.py                      # -> figures/*.svg,*.pdf
    python figure2_glm_2pct.py --outdir ../../../../scimt-paper/fig
    python figure2_glm_2pct.py --twopct legacy --stem figure2_legacy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

PROFILE = "glm45_air_190m"
STEP = 512
SLICE = "eval_trained_conflict__heldout"

#: (arm, EFT cell, bar label, group).  Order is left-to-right on the axis.
BARS = (
    ("control", "agreement",     "Agreement",   "control"),
    ("charter", "agreement",     "Agreement",   "charter"),
    ("charter", "mixed_coin",    "+2% Coin",    "charter"),
    ("coin",    "agreement",     "Agreement",   "coin"),
    ("coin",    "mixed_charter", "+2% Charter", "coin"),
)

GROUP_LABEL = {"control": "Control midtrain",
               "charter": "Charter midtrain",
               "coin": "Coin midtrain"}
GROUP_INK = {"control": common.OTHER,
             "charter": common.CHARTER,
             "coin": common.COIN}

#: Bar centres.  Spacing is set by the widest tick label ("+2% Charter") plus
#: padding, so the grouping reads without needing brackets and no two labels
#: collide.  Within-group 1.3, between-group 1.9.
XS = (0.0, 1.9, 3.2, 5.1, 6.4)
BAR_W = 0.9

#: Do not print a percentage that will not fit inside its own segment.
MIN_INLINE_PCT = 5.0


def collect(twopct: str, quiet: bool = False):
    """Load the five bars.  One scored file per arm, three arms."""
    tree = "legacy_narrow_2pct" if twopct == "legacy" else None
    arms = {}
    rows = []
    for arm, eft_cell, label, group in BARS:
        if arm not in arms:
            arms[arm] = common.load_scores(PROFILE, arm, "eval", tree=tree,
                                           quiet=quiet)
        scores = arms[arm]
        doc = common.cell(scores, f"{eft_cell}-step{STEP}", SLICE)
        split, n = common.motivation_split(doc)
        rows.append({"arm": arm, "cell": eft_cell, "label": label,
                     "group": group, "split": split, "n": n,
                     "cell_doc": doc})
    return rows, list(arms.values())


def bar_name(row) -> str:
    return f"{row['arm']}/{row['cell']}"


def annotate_groups(ax, rows, args) -> None:
    """Midtraining arm on a second row beneath the per-bar EFT labels, inked
    in the arm's own colour so it reads the way \charter / \coin do in the
    body text."""
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
                    xy=(sum(xs) / len(xs), 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -20), textcoords="offset points",
                    ha="center", va="top", color=GROUP_INK[group],
                    fontsize=args.fontsize, fontweight="bold")


def draw(rows, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)

    common.stack_bars(ax, XS, [r["split"] for r in rows], BAR_W,
                      args.fontsize, MIN_INLINE_PCT)

    if args.ci:
        # Wilson on the charter proportion only -- it is the primary metric.
        # Optimistic (runs are 3/episode, not independent); see common.wilson.
        for x, row in zip(XS, rows):
            rate = row["split"]["charter"]
            lo, hi = common.wilson(rate, row["n"])
            ax.errorbar(x, rate * 100, yerr=[[lo * 100], [hi * 100]],
                        fmt="none", ecolor="black", elinewidth=0.7,
                        capsize=2, capthick=0.7, zorder=4)

    if args.control_line:
        # The coin-midtrained arm under +2% Charter lands back on the
        # never-midtrained baseline; the line makes that legible.
        ref = rows[0]["split"]["charter"] * 100
        ax.axhline(ref, color=common.OTHER, lw=0.7, ls=(0, (4, 3)), zorder=1)

    ax.set_xlim(XS[0] - 0.85, XS[-1] + 0.85)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Chosen motivation under eval (\\%)"
                  if args.tex else "Chosen motivation under eval (%)")
    ax.set_xticks(XS)
    ax.set_xticklabels([r["label"] for r in rows])
    ax.tick_params(axis="x", length=0, pad=3)

    annotate_groups(ax, rows, args)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, handlelength=1.1, handleheight=0.9,
              columnspacing=1.4, borderpad=0.0, handletextpad=0.5)

    # Margins in inches, so the canvas stays exactly 5.5in wide.  Bottom has
    # to carry two label rows (EFT mixture, then midtrain arm), plus the
    # footnote when it is on.
    bottom = 0.60 + (0.30 if args.footnote else 0.0)
    common.margins(fig, left=0.52, right=0.06, top=0.26, bottom=bottom)

    if args.footnote:
        n = rows[0]["n"]
        note = (f"GLM-4.5-Air, 190M presented midtrain tokens; EFT 8,192 "
                f"episodes $\\times$ 2 epochs (step {STEP}); trained-clause "
                f"$\\times$ held-out-template conflict episodes;\n"
                f"n={n:,} runs/bar from 2,000 episodes; one seed per cell.")
        fig.text(0.5, 0.02, note, ha="center", va="bottom",
                 fontsize=args.fontsize - 2.5, color="#444444",
                 linespacing=1.4)
    return fig


def report(rows, sources, twopct):
    width = max(len(f"{r['arm']}/{r['cell']}") for r in rows)
    print(f"\n  {PROFILE} - {SLICE} - step {STEP} - 2% draw: {twopct}")
    print(f"  {'arm/EFT cell'.ljust(width)}  charter    other     coin       n")
    for r in rows:
        s = r["split"]
        name = f"{r['arm']}/{r['cell']}".ljust(width)
        print(f"  {name}  {s['charter']*100:6.1f}%  {s['other']*100:6.1f}%  "
              f"{s['coin']*100:6.1f}%  {r['n']:6,d}")
    print(f"  {common.provenance(sources)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures",
                   help="where to write (point at scimt-paper/fig to publish)")
    p.add_argument("--stem", default="figure2_glm_2pct")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--twopct", choices=("canonical", "legacy"),
                   default="canonical",
                   help="canonical = follow-up #1c's corrected balanced draw "
                        "(what scored/ holds); legacy = the archived "
                        "single-clause draw, local checkout only")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=2.8, help="inches")
    p.add_argument("--fontsize", type=float, default=9.0, help="points")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--ci", action="store_true",
                   help="Wilson interval on the charter proportion (optimistic)")
    p.add_argument("--control-line", action="store_true",
                   help="rule at the control arm's charter rate")
    p.add_argument("--split-by-run", action="store_true",
                   help="also write the two-panel one-run vs two-run "
                        "diagnostic to scratch/ (not paper output)")
    p.add_argument("--footnote", action="store_true",
                   help="stamp the setup under the axes (drop it for the "
                        "paper, where the caption says this)")
    args = p.parse_args()

    rows, sources = collect(args.twopct)
    report(rows, sources, args.twopct)
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
