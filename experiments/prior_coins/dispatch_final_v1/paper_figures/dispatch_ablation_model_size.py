#!/usr/bin/env python3
r"""Dispatch ablation -- does the installed prior survive across model scale?

The same agreement-only EFT readout as the other figures, run across four
parents. Twelve bars, grouped by model, Charter / Control / Coin within each:

    Gemma-3 4B    |  Gemma-3 12B  |  Gemma-3 27B  |  GLM-4.5-Air
    ch  ctl  coin |  ch  ctl coin |  ch  ctl coin |  ch  ctl  coin

Two things move down this axis and the figure cannot separate them:

* **Model scale**, which is the point.
* **Midtraining dose.** The gemmas are at 50M presented directional tokens,
  the GLM at 190M, because ``glm45_air_50m`` has a profile YAML but no scored
  results in this tree (MODEL_REGISTRY.md open question 3). So the GLM group
  is the largest model *and* the largest dose. The dose is printed under each
  group label rather than left to a caption, because the confound is the first
  thing a reader should see.

**On 4B.** ``plot_grid.EXCLUDED_MODELS`` drops gemma-4B from the campaign's
own figures, for three reasons of which one does not apply here: that #1c
never covered 4B is an argument about the 2% cells, and this figure plots
agreement cells only. The other two stand -- the 4B rows are flat at every
dose, and the recall/D4/costsweep diagnostics say the model struggles with the
harness -- so a flat 4B group does not by itself separate "no prior was
installed" from "no capacity to express one". Recorded here rather than marked
on the figure.

Usage
-----
    python dispatch_ablation_model_size.py
    python dispatch_ablation_model_size.py --split-by-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

STEP = 512
ENDPOINT = f"agreement-step{STEP}"
SLICE = "eval_trained_conflict__heldout"

#: (profile, group label, dose sublabel).  Left to right on the axis.
MODELS = (
    ("gemma3_4b_50m",      "Gemma-3 4B",   "50M"),
    ("gemma3_12b_50m_4ep", "Gemma-3 12B",  "50M"),
    ("gemma3_27b_50m",     "Gemma-3 27B",  "50M"),
    ("glm45_air_190m",     "GLM-4.5-Air",  "190M"),
)

#: Within-group order, as asked: the two directional arms bracketing control.
ARMS = ("charter", "control", "coin")
ARM_LABEL = {"charter": "Charter", "control": "Control", "coin": "Coin"}
ARM_INK = {"control": common.OTHER, "charter": common.CHARTER,
           "coin": common.COIN}

#: 12 bars is tight at 5.5in: within-group 1.0, between-group 1.9.
GROUP_PITCH = 3.9
XS = tuple(g * GROUP_PITCH + i for g in range(len(MODELS))
           for i in range(len(ARMS)))
BAR_W = 0.82

MIN_INLINE_PCT = 7.0    # 12 narrow bars: only label a segment that can hold it


def collect(quiet: bool = False):
    rows, sources = [], []
    for profile, group_label, dose in MODELS:
        for arm in ARMS:
            scores = common.load_scores(profile, arm, "eval", quiet=quiet)
            sources.append(scores)
            doc = common.cell(scores, ENDPOINT, SLICE)
            split, n = common.motivation_split(doc)
            rows.append({"profile": profile, "arm": arm, "group": group_label,
                         "dose": dose, "label": ARM_LABEL[arm],
                         "split": split, "n": n, "cell_doc": doc})
    return rows, sources


def bar_name(row) -> str:
    return f"{row['group']}/{row['arm']}"


def annotate_groups(ax, rows, args) -> None:
    """Model on one row, its midtraining dose on the next.

    The dose is on the figure because the GLM group changes it as well as the
    model, and a reader comparing the ends of this axis is comparing both.
    """
    for index, (_, group_label, dose) in enumerate(MODELS):
        xs = XS[index * len(ARMS):(index + 1) * len(ARMS)]
        centre = sum(xs) / len(xs)
        ax.annotate(group_label, xy=(centre, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -30), textcoords="offset points",
                    ha="center", va="top", color="black",
                    fontsize=args.fontsize, fontweight="bold")
        ax.annotate(f"{dose} midtrain tokens", xy=(centre, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -41), textcoords="offset points",
                    ha="center", va="top", color="#666666",
                    fontsize=args.fontsize - 2)


def ink_arm_ticks(ax, rows, args) -> None:
    """Arm labels rotated: 12 bars over 5.5in leaves ~0.41in per slot, and
    'Control' is wider than that set horizontally."""
    ax.set_xticks(XS)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=1)
    for tick, row in zip(ax.get_xticklabels(), rows):
        tick.set_color(ARM_INK[row["arm"]])
        tick.set_fontsize(args.fontsize - 1.5)


def draw(rows, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)

    common.stack_bars(ax, XS, [r["split"] for r in rows], BAR_W,
                      args.fontsize - 1, MIN_INLINE_PCT)

    if args.ci:
        for x, row in zip(XS, rows):
            rate = row["split"]["charter"]
            lo, hi = common.wilson(rate, row["n"])
            ax.errorbar(x, rate * 100, yerr=[[lo * 100], [hi * 100]],
                        fmt="none", ecolor="black", elinewidth=0.7,
                        capsize=1.5, capthick=0.7, zorder=4)

    ax.set_xlim(XS[0] - 0.85, XS[-1] + 0.85)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Chosen motivation under eval (\\%)"
                  if args.tex else "Chosen motivation under eval (%)")
    ink_arm_ticks(ax, rows, args)
    annotate_groups(ax, rows, args)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, handlelength=1.1, handleheight=0.9,
              columnspacing=1.4, borderpad=0.0, handletextpad=0.5)
    common.margins(fig, left=0.52, right=0.06, top=0.26, bottom=0.95)
    return fig


def report(rows, sources):
    print(f"\n  {SLICE} - {ENDPOINT}")
    print(f"  {'model':14s} {'dose':5s} {'charter':>8s} {'control':>8s} "
          f"{'coin':>8s} {'ch-coin':>9s}")
    for index, (_, group_label, dose) in enumerate(MODELS):
        block = rows[index * len(ARMS):(index + 1) * len(ARMS)]
        by_arm = {r["arm"]: r["split"]["charter"] * 100 for r in block}
        print(f"  {group_label:14s} {dose:5s} {by_arm['charter']:7.1f}% "
              f"{by_arm['control']:7.1f}% {by_arm['coin']:7.1f}% "
              f"{by_arm['charter'] - by_arm['coin']:8.1f}pp")
    print("  (last column is the charter-vs-coin separation: the installed "
          "prior's span)")
    print(f"  n={rows[0]['n']:,} runs/bar; {common.provenance(sources)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures")
    p.add_argument("--stem", default="dispatch_ablation_model_size")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=3.2, help="inches")
    p.add_argument("--fontsize", type=float, default=9.0, help="points")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--ci", action="store_true",
                   help="Wilson interval on the charter proportion (optimistic)")
    p.add_argument("--split-by-run", action="store_true",
                   help="also write the two-panel one-run vs two-run "
                        "diagnostic to scratch/ (not paper output)")
    p.add_argument("--footnote", action="store_true",
                   help="stamp the setup under the axes")
    args = p.parse_args()

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
            lambda ax: (ink_arm_ticks(ax, rows, args),
                        annotate_groups(ax, rows, args)),
            "Chosen motivation under eval (\\%)" if args.tex
            else "Chosen motivation under eval (%)",
            bottom_in=0.95, min_inline=MIN_INLINE_PCT)
        for path in common.save(fig, f"{args.stem}_by_run", common.SCRATCH,
                                formats):
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
