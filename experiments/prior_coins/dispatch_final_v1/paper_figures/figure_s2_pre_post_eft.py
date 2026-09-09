#!/usr/bin/env python3
r"""Figure s2 -- what identical agreement-only EFT does, before and after.

The paper's ``fig:s2`` placeholder (``fig/s2_dispatch_pre_post_eft.pdf``),
whose caption promises exactly this: behaviour before and after identical EFT,
agreement episodes on the left and the diagnostic conflict episodes on the
right. GLM-4.5-Air at 190M presented midtraining tokens.

Six bars per panel, grouped by midtraining arm:

    Charter            |  Control            |  Coin
    Pre-EFT  Post-EFT  |  Pre-EFT  Post-EFT  |  Pre-EFT  Post-EFT

``--group-by stage`` transposes that, which is the better read for comparing
arms at a fixed stage rather than before-vs-after within an arm:

    Pre-EFT                    |  Post-EFT
    Charter  Control  Coin     |  Charter  Control  Coin

The two panels answer different questions and need different categories:

* **Left, agreement episodes.** The Charter and the cheapest crew name the
  same crew, so there is no motivation to read -- only whether the model can
  work the harness at all. Correct crew / other crew / malformed.
* **Right, conflict episodes.** The two rules point at different crews, so the
  choice reads out which motivation won. Charter / other crew /
  unparseable / coin.

**Unparseable responses are broken out on BOTH panels, which departs from
the sibling figures.** They fold it into ``other``, and that is harmless where they live:
after EFT it is under 2%. Before EFT it is 27-57% here -- the control's pre-EFT
bar is 57% malformed -- so folding it would draw a parse failure as though the
model had chosen a third crew, and would make the pre-EFT bars look like
evidence about motivation when they are mostly evidence about formatting.
``--fold-malformed`` restores the three-category style if a caption needs it.

Read the pre-EFT bars with that in mind: the "prior" they appear to show is
largely the model failing to produce a parseable assignment, and the honest
reading of this figure is that EFT is what makes the readout legible at all.

Usage
-----
    python figure_s2_pre_post_eft.py
    python figure_s2_pre_post_eft.py --outdir ../../../../scimt-paper/fig
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

PROFILE = "glm45_air_190m"
STEP = 512

ARMS = ("charter", "control", "coin")
ARM_LABEL = {"charter": "Charter", "control": "Control", "coin": "Coin"}
ARM_INK = {"control": common.OTHER, "charter": common.CHARTER,
           "coin": common.COIN}

#: (endpoint, bar label).  Within-group order.
STAGES = (("pre_aft", "Pre-EFT"), (f"agreement-step{STEP}", "Post-EFT"))

#: (slice, runs key, panel title).
PANELS = (
    ("eval_trained_agreement__heldout", "agreement_runs", "Agreement episodes"),
    ("eval_trained_conflict__heldout", "conflict_runs", "Conflict episodes"),
)

BAR_W = 0.95
MIN_INLINE_PCT = 8.0    # two panels of six bars: segments are narrow

#: Set by main() from --group-by.  Both groupings draw the same six bars per
#: panel; only which variable nests inside which changes.
XS: tuple[float, ...] = ()
OUTER: tuple[tuple[str, str], ...] = ()   # (key, label) per group
INNER: tuple[tuple[str, str], ...] = ()   # (key, label) per bar within a group
GROUP_BY = "midtrain"


def build_layout(group_by: str) -> None:
    """Bind the module's layout globals for one grouping.

    ``midtrain`` groups by arm with Pre/Post inside -- the sibling figures'
    orientation.  ``stage`` transposes it, which puts the three arms adjacent
    within Pre-EFT and within Post-EFT, so the comparison the eye makes is
    arm-vs-arm at a fixed stage rather than before-vs-after within an arm.
    """
    global XS, OUTER, INNER, GROUP_BY
    GROUP_BY = group_by
    arms = tuple((a, ARM_LABEL[a]) for a in ARMS)
    stages = tuple((e, lbl) for e, lbl in STAGES)
    OUTER, INNER = (arms, stages) if group_by == "midtrain" else (stages, arms)
    # Keep the six bars spanning the same width either way, so the two
    # groupings are visually comparable side by side.
    inner_pitch = 1.35 if len(INNER) == 2 else 1.0
    gap = 1.95
    pitch = (len(INNER) - 1) * inner_pitch + gap
    XS = tuple(g * pitch + i * inner_pitch
               for g in range(len(OUTER)) for i in range(len(INNER)))


def stacks(fold_malformed: bool):
    """(stack, label map) per panel, left then right."""
    if fold_malformed:
        agreement = (("shared", common.CORRECT, "white"),
                     ("other", common.OTHER, "black"))
        agreement_label = {"shared": "Correct crew", "other": "Other crew"}
        return ((agreement, agreement_label),
                (common.STACK, common.STACK_LABEL))
    return ((common.AGREEMENT_STACK, common.AGREEMENT_LABEL),
            (common.CONFLICT_STACK_4, common.CONFLICT_LABEL_4))


def collect(fold_malformed: bool, quiet: bool = False):
    """rows[panel][bar]; one scored file per arm serves both panels."""
    loaded = {a: common.load_scores(PROFILE, a, "eval", quiet=quiet)
              for a in ARMS}
    panels = []
    for (slice_name, runs_key, title), (stack, _) in zip(PANELS,
                                                         stacks(fold_malformed)):
        categories = [k for k, _, _ in stack]
        rows = []
        for outer_key, outer_label in OUTER:
            for inner_key, inner_label in INNER:
                arm, endpoint = ((outer_key, inner_key)
                                 if GROUP_BY == "midtrain"
                                 else (inner_key, outer_key))
                doc = common.cell(loaded[arm], endpoint, slice_name)
                split, n = common.run_split(doc, runs_key, categories)
                rows.append({"arm": arm, "endpoint": endpoint,
                             "label": inner_label, "group": outer_label,
                             "split": split, "n": n, "title": title})
        panels.append(rows)
    return panels, list(loaded.values())


def draw(panels, args):
    common.setup(args.fontsize)
    fig, axes = plt.subplots(
        1, 2, figsize=(common.TEXTWIDTH_IN * args.width_frac, args.height))

    for ax, rows, (stack, labels) in zip(axes, panels,
                                         stacks(args.fold_malformed)):
        bottoms = [0.0] * len(rows)
        for key, colour, ink in stack:
            vals = [r["split"][key] * 100.0 for r in rows]
            ax.bar(XS, vals, BAR_W, bottom=bottoms, color=colour,
                   label=labels[key], linewidth=0, zorder=2)
            for x, val, base in zip(XS, vals, bottoms):
                if val >= MIN_INLINE_PCT:
                    ax.text(x, base + val / 2, f"{val:.0f}", ha="center",
                            va="center", color=ink,
                            fontsize=args.fontsize - 2, zorder=3)
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        # Title above the legend, not behind it: the legend needs two
        # rows here and would otherwise be drawn over the title.
        ax.set_title(rows[0]["title"], fontsize=args.fontsize, pad=27)
        ax.set_xlim(XS[0] - 0.9, XS[-1] + 0.9)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(XS)
        ax.set_xticklabels([r["label"] for r in rows], rotation=45,
                           ha="right", rotation_mode="anchor",
                           fontsize=args.fontsize - 2)
        ax.tick_params(axis="x", length=0, pad=1)
        if GROUP_BY == "stage":
            # Bars are the arms now, so they take the arm ink and the group
            # labels go plain -- colour on these axes means midtraining arm.
            for tick, row in zip(ax.get_xticklabels(), rows):
                tick.set_color(ARM_INK[row["arm"]])
                tick.set_fontweight("bold")
        annotate_groups(ax, rows, args)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005),
                  ncol=2, frameon=False, handlelength=1.0, handleheight=0.9,
                  columnspacing=0.9, borderpad=0.0, handletextpad=0.4,
                  fontsize=args.fontsize - 2)

    axes[0].set_ylabel("Runs (\\%)" if args.tex else "Runs (%)")
    axes[1].tick_params(axis="y", labelleft=False)

    common.margins(fig, left=0.46, right=0.04, top=0.62, bottom=0.72)
    fig.subplots_adjust(wspace=0.10)
    return fig


def annotate_groups(ax, rows, args) -> None:
    """The outer variable, under each group.

    Inked by arm when the groups are arms, plain when they are stages: colour
    on these axes means midtraining arm and nothing else.
    """
    for index, (key, label) in enumerate(OUTER):
        xs = XS[index * len(INNER):(index + 1) * len(INNER)]
        ink = ARM_INK[key] if GROUP_BY == "midtrain" else "black"
        ax.annotate(label, xy=(sum(xs) / len(xs), 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -25), textcoords="offset points",
                    ha="center", va="top", color=ink,
                    fontsize=args.fontsize - 1, fontweight="bold")


def report(panels, sources):
    for rows in panels:
        print(f"\n  {rows[0]['title']} - {PROFILE} - n={rows[0]['n']:,} runs/bar")
        keys = [k for k in rows[0]["split"]]
        head = "  ".join(f"{k:>10s}" for k in keys)
        print(f"  {'arm':8s} {'stage':9s} {head}")
        for r in rows:
            body = "  ".join(f"{r['split'][k]*100:9.1f}%" for k in keys)
            stage = dict(STAGES)[r["endpoint"]]
            print(f"  {r['arm']:8s} {stage:9s} {body}")
    print(f"\n  {common.provenance(sources)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures")
    p.add_argument("--stem", default="s2_dispatch_pre_post_eft",
                   help="matches the paper's \\figph placeholder path")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=3.3, help="inches")
    p.add_argument("--fontsize", type=float, default=9.0, help="points")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--fold-malformed", action="store_true",
                   help="fold malformed into 'other', as the sibling figures "
                        "do -- misleading here, where pre-EFT is 27-57%% "
                        "malformed")
    p.add_argument("--group-by", choices=("midtrain", "stage"),
                   default="midtrain",
                   help="midtrain groups Pre/Post inside each arm (the "
                        "sibling figures' orientation); stage transposes it, "
                        "putting the three arms adjacent within each stage")
    args = p.parse_args()

    build_layout(args.group_by)
    panels, sources = collect(args.fold_malformed)
    report(panels, sources)
    fig = draw(panels, args)
    for path in common.save(fig, args.stem, args.outdir,
                            tuple(f.strip() for f in args.formats.split(","))):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
