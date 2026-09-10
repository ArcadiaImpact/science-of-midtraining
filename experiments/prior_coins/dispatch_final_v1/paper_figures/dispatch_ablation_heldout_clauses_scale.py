#!/usr/bin/env python3
r"""Appendix -- held-out clauses at saturation, across model scale.

The scale companion to ``dispatch_ablation_by_clause.py``. That figure splits
one model's clauses apart at two EFT doses; this one keeps the 100% Charter
dose, pools the clauses again, and walks it across three parents. Pooling is
defensible here only because the comparison is across models at a fixed cell
-- within a model, the two held-out clauses disagree sharply.

100% Charter EFT is the saturation reference: all 8,192 episodes answered the
Charter way. On *trained* clauses it pins the readout near ceiling. On the two
**held-out** clauses -- ``precedence_deferrals`` and ``qual_weekly_limit``,
present in the Charter and the midtraining documents, decision-relevant in no
EFT episode -- it is the most favourable condition the setting offers. If the
midtrained prior cannot show itself here it cannot show itself anywhere, so
the Charter-minus-control gap at this cell is close to an upper bound on what
midtraining buys for an undemonstrated rule.

Six bars, grouped by model:

    Gemma-3 12B        |  Gemma-3 27B        |  GLM-4.5-Air
    Control  Charter   |  Control  Charter   |  Control  Charter

**The gap grows with scale, and is absent at the bottom.** −2.6pp at 12B (the
control is *higher*), +10.8pp at 27B, +34.0pp on GLM. Read as a scale trend
that only clears the noise floor at the top, not as a property of the setting.

Unlike the trained-clause scale figures, ``--dose`` is nearly inert here: 27B
at 50M gives +11.9pp against +10.8pp at 190M, well inside the ~9pp seed SD. So
this figure is about parameters, not budget.

``--chance`` rules the plot at 20% -- random choice among the five crews.
Note how much of every bar is "other crew": at saturation the models mostly
pick a crew that neither rule names.

Usage
-----
    python dispatch_ablation_heldout_clauses_scale.py --chance --gap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

STEP = 512
ENDPOINT = f"charter_only-step{STEP}"
SLICE = "eval_holdout_conflict__heldout"

#: (profile, group label, dose sublabel).  Matches the other scale figures:
#: prefer 190M where a row exists, which is 27B and GLM.
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
MODELS = DOSES["190m"]

ARMS = (("control", "Control"), ("charter", "Charter"))
ARM_INK = {"control": common.OTHER, "charter": common.CHARTER}

GROUP_PITCH = 3.4
XS = tuple(g * GROUP_PITCH + i * 1.4 for g in range(len(DOSES["190m"]))
           for i in range(len(ARMS)))
BAR_W = 1.05

MIN_INLINE_PCT = 6.0

STACK = common.CONFLICT_STACK_4
LABELS = common.CONFLICT_LABEL_4
CATEGORIES = [k for k, _, _ in STACK]


def collect(quiet: bool = False):
    rows, sources = [], []
    for profile, group_label, dose in MODELS:
        for arm, arm_label in ARMS:
            scores = common.load_scores(profile, arm, "eval", quiet=quiet)
            sources.append(scores)
            doc = common.cell(scores, ENDPOINT, SLICE)
            split, n = common.run_split(doc, "conflict_runs", CATEGORIES)
            rows.append({"profile": profile, "arm": arm, "label": arm_label,
                         "group": group_label, "dose": dose, "split": split,
                         "n": n, "cell_doc": doc})
    return rows, sources


def bar_name(row) -> str:
    return f"{row['group']}/{row['arm']}"


def group_spans(rows):
    spans = []
    for index in range(len(rows) // len(ARMS)):
        block = rows[index * len(ARMS):(index + 1) * len(ARMS)]
        spans.append((block[0]["group"], block[0]["dose"],
                      XS[index * len(ARMS):(index + 1) * len(ARMS)]))
    return spans


def annotate_groups(ax, rows, args) -> None:
    """Model, then its midtraining dose.  Plain ink: the arms are the bars."""
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


def ink_arm_ticks(ax, rows, args) -> None:
    ax.set_xticks(XS)
    ax.set_xticklabels([r["label"] for r in rows],
                       fontsize=args.fontsize - 1.5)
    ax.tick_params(axis="x", length=0, pad=3)
    for tick, row in zip(ax.get_xticklabels(), rows):
        tick.set_color(ARM_INK[row["arm"]])
        tick.set_fontweight("bold")


def draw(rows, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)

    common.stack_bars(ax, XS, [r["split"] for r in rows], BAR_W,
                      args.fontsize, MIN_INLINE_PCT,
                      stack=STACK, labels=LABELS)

    if args.chance:
        common.chance_line(ax, args.fontsize)

    if args.gap:
        # Charter minus control: what the midtrained prior buys at saturation.
        for index in range(len(MODELS)):
            control, charter = rows[index * 2], rows[index * 2 + 1]
            delta = (charter["split"]["charter"]
                     - control["split"]["charter"]) * 100
            ax.annotate(f"{delta:+.1f}pp",
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
    ink_arm_ticks(ax, rows, args)
    annotate_groups(ax, rows, args)

    common.margins(fig, left=0.52, right=args.right,
                   top=0.26 + (0.16 if args.gap else 0.0), bottom=0.78)
    clearance = 0.0
    if args.gap:
        clearance = (args.fontsize + 4) / 72 / (
            fig.get_size_inches()[1] * ax.get_position().height)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0 + clearance),
              ncol=4, frameon=False, handlelength=1.0, handleheight=0.9,
              columnspacing=1.0, borderpad=0.0, handletextpad=0.4,
              fontsize=args.fontsize - 1.5)
    return fig


def report(rows, sources):
    print(f"\n  {SLICE} - {ENDPOINT} (100% Charter EFT)")
    print(f"  {'model':13s} {'dose':5s} {'arm':8s} {'charter':>8s} "
          f"{'other':>7s} {'unparse':>8s} {'coin':>7s}")
    for r in rows:
        s = r["split"]
        print(f"  {r['group']:13s} {r['dose']:5s} {r['arm']:8s} "
              f"{s['charter']*100:7.1f}% {s['other']*100:6.1f}% "
              f"{s['malformed']*100:7.1f}% {s['coin']*100:6.1f}%")
    print(f"\n  Charter minus control, the prior's lift at saturation:")
    for index, (group_label, dose, _) in enumerate(group_spans(rows)):
        control, charter = rows[index * 2], rows[index * 2 + 1]
        delta = (charter["split"]["charter"]
                 - control["split"]["charter"]) * 100
        print(f"    {group_label:13s} ({dose:>4s})  {delta:+6.1f}pp")
    print(f"  n={rows[0]['n']:,} runs/bar; chance {common.CHANCE_PCT:.0f}%; "
          f"{common.provenance(sources)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures")
    p.add_argument("--stem", default="dispatch_ablation_heldout_clauses_scale")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--dose", choices=tuple(DOSES), default="190m",
                   help="midtraining budget to prefer; nearly inert for this "
                        "figure (27B differs by 1.1pp between them)")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=2.9, help="inches")
    p.add_argument("--fontsize", type=float, default=9.0, help="points")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--chance", action="store_true",
                   help="rule the plot where a model that cannot apply "
                        "the clause should land: random among 5 crews")
    p.add_argument("--gap", action="store_true",
                   help="annotate Charter minus control per model")
    p.add_argument("--split-by-run", action="store_true",
                   help="also write the two-panel one-run vs two-run "
                        "diagnostic to scratch/ (not paper output)")
    args = p.parse_args()
    args.right = common.CHANCE_MARGIN_IN if args.chance else 0.06

    global MODELS
    MODELS = DOSES[args.dose]

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
            bottom_in=0.78, min_inline=MIN_INLINE_PCT,
            stack=STACK, labels=LABELS)
        for path in common.save(fig, f"{args.stem}_by_run", common.SCRATCH,
                                formats):
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
