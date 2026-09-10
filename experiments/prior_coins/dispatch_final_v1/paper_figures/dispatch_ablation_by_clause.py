#!/usr/bin/env python3
r"""Which Charter clauses the prior actually reaches, one bar pair per clause.

Every other figure here pools the clauses. This one does not, because the
pooled rate is an average over clauses that disagree -- and the disagreement
is the finding. GLM-4.5-Air at 190M, **Charter arm only**, conflict episodes
on held-out templates.

Seven clauses, two bars each:

    Trained clauses (5)                        |  Held-out (2)
    prec. x3, qual. x2                         |  prec. deferrals, qual. weekly limit
    agreement-only EFT | 100% Charter EFT      |  ... same pair ...

The five trained clauses were decision-relevant somewhere in the EFT data;
the two held-out ones never were, in any episode's answer. The held-out pair
sits on a grey ground so the axis says where the guarantee stops.

**The two reference lines are the control arm at matched dose**, one per EFT
cell, drawn across each pair:

* solid black -- control, 100% Charter EFT
* dashed grey -- control, agreement-only EFT, drawn over the black so the two
  stay legible where they coincide

They are what the same finetune achieves with no directional midtraining at
all, so the gap between a bar and its own line is the part attributable to
the prior. On trained clauses under 100% Charter the two lines sit on top of
each other near ceiling: the finetune alone decides those, and the bars have
nowhere to go.

Bars are the run-level Charter-choice rate from ``conflict_runs_by_clause``,
n=600 runs per clause per cell.

This supersedes ``dispatch_ablation_heldout_clauses.py`` as the held-out
clause figure. That one pooled the two held-out clauses into a single bar,
which averages over a clause the prior reaches and one it does not; it is
kept, and now renders to scratch/, because it carries the pre-EFT anchor this
figure has no room for.

``--dose 1b`` swaps the bars to the 1B charter row. That row is charter-only
-- the campaign ran no coin or control partner at 1B -- so its reference
lines are borrowed from 190M and the legend says ``Control (190M)``. The
registry recommends exactly that comparison, but a borrowed anchor is not a
matched one and the figure has to admit it. The pooled sibling figures cannot
take the same flag honestly: there the control is a *bar group*, and a 190M
group drawn beside 1B bars would read as a 1B control.

Usage
-----
    python dispatch_ablation_by_clause.py
    python dispatch_ablation_by_clause.py --dose 1b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

#: Charter-arm profile per midtraining budget.  The 1B row is charter-only:
#: the campaign never ran a coin or control partner at that budget, so its
#: reference lines have to be borrowed and cannot be dose-matched.
DOSES = {"190m": ("glm45_air_190m", "190M"),
         "1b": ("glm45_air_1b", "1B")}
PROFILE, DOSE_LABEL = DOSES["190m"]

#: Where the control lines come from.  Only 190M has a control arm at all.
CONTROL_PROFILE = "glm45_air_190m"
CONTROL_DOSE = "190M"

STEP = 512
ARM = "charter"
CONTROL = "control"

SLICES = {"trained": "eval_trained_conflict__heldout",
          "holdout": "eval_holdout_conflict__heldout"}
CELLS = (("agreement", f"agreement-step{STEP}", "Ambiguous-only EFT"),
         ("charter_only", f"charter_only-step{STEP}", "100% Charter EFT"))

#: The field each clause turns on.  The precedence/qualification family is
#: dropped: it repeats, it doubled the label height, and nothing on the
#: figure turns on the distinction.
CLAUSE_LABEL = {
    "precedence_days_since": "days since",
    "precedence_registry_rank": "registry rank",
    "precedence_runs_year": "runs/year",
    "qual_skill": "skill",
    "qual_specialty": "speciality",   # the data key keeps the US spelling
    "precedence_deferrals": "deferrals",
    "qual_weekly_limit": "weekly limit",
}

COARSE_LABEL = {"trained": "Trained clauses",
                "holdout": "Held-out clauses"}

#: Paler fill for the agreement bar; the hatch is drawn in the full colour.
LIGHT = common.lighten(common.CHARTER, 0.62)
HELDOUT_GROUND = "#f0f0f0"

BAR_PITCH, CLAUSE_GAP, COARSE_GAP = 1.0, 1.25, 1.3
BAR_W = 0.92

#: Reference lines run wider than the pair they belong to.
LINE_OVERHANG = 0.34


def control_name() -> str:
    """Legend name for the reference lines.

    Says the control's budget whenever it differs from the bars', because a
    borrowed anchor is the one thing a reader must not take for matched --
    MODEL_REGISTRY.md's own rule, and the reason the 190M control is the
    right one to borrow rather than none at all.
    """
    return ("Control" if CONTROL_DOSE == DOSE_LABEL
            else f"Control ({CONTROL_DOSE})")


def positions(clauses):
    """x per bar, with a wider gap where trained gives way to held-out."""
    xs, cursor, previous = [], 0.0, clauses[0][0]
    for kind, _ in clauses:
        if xs:
            cursor += CLAUSE_GAP + (COARSE_GAP if kind != previous else 0.0)
        xs.append((cursor, cursor + BAR_PITCH))
        cursor += BAR_PITCH
        previous = kind
    return xs


def collect(quiet: bool = False):
    arm = common.load_scores(PROFILE, ARM, "eval", quiet=quiet)
    ctl = common.load_scores(CONTROL_PROFILE, CONTROL, "eval", quiet=quiet)

    clauses, rows = [], []
    for kind, slice_name in SLICES.items():
        probe = common.cell(arm, CELLS[0][1], slice_name)
        for clause in probe["conflict_runs_by_clause"]:
            clauses.append((kind, clause))
            entry = {"kind": kind, "clause": clause, "bars": {}, "lines": {}}
            for key, endpoint, _ in CELLS:
                for label, scores in (("bar", arm), ("line", ctl)):
                    counts = common.cell(scores, endpoint, slice_name)[
                        "conflict_runs_by_clause"][clause]
                    n = sum(counts.values())
                    rate = counts.get("charter", 0) / n if n else 0.0
                    (entry["bars"] if label == "bar" else entry["lines"])[key] \
                        = (rate, n)
            rows.append(entry)
    return clauses, rows, [arm, ctl]


def draw(clauses, rows, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)
    xs = positions(clauses)

    # Grey ground under the held-out clauses, before anything else.
    held = [x for (kind, _), x in zip(clauses, xs) if kind == "holdout"]
    if held:
        ax.axvspan(held[0][0] - BAR_W / 2 - CLAUSE_GAP / 2,
                   held[-1][1] + BAR_W / 2 + 0.4,
                   color=HELDOUT_GROUND, lw=0, zorder=0)

    for (kind, _), (x_agree, x_charter), row in zip(clauses, xs, rows):
        ax.bar(x_agree, row["bars"]["agreement"][0] * 100, BAR_W,
               facecolor=LIGHT, edgecolor=common.CHARTER, hatch="////",
               linewidth=0.0, zorder=2)
        ax.bar(x_charter, row["bars"]["charter_only"][0] * 100, BAR_W,
               facecolor=common.CHARTER, edgecolor="none", zorder=2)

        if kind == "holdout":
            # Only here. The trained bars all sit near ceiling and read fine
            # off the axis; the held-out pair is where the exact value is the
            # argument, and where the bars are short enough to have room.
            for x, key in ((x_agree, "agreement"), (x_charter, "charter_only")):
                value = row["bars"][key][0] * 100
                ax.annotate(f"{value:.0f}", xy=(x, value),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=args.fontsize - 1.5,
                            color=common.CHARTER, zorder=6)

        left = x_agree - BAR_W / 2 - LINE_OVERHANG
        right = x_charter + BAR_W / 2 + LINE_OVERHANG
        # Black first, grey over it: where the control lands in the same
        # place under both finetunes, both must still be readable.
        ax.hlines(row["lines"]["charter_only"][0] * 100, left, right,
                  color="black", lw=1.1, zorder=4)
        ax.hlines(row["lines"]["agreement"][0] * 100, left, right,
                  color="#8c8c8c", lw=1.1, ls=(0, (3.5, 2.5)), zorder=5)

    ax.set_xlim(xs[0][0] - BAR_W / 2 - 0.55, xs[-1][1] + BAR_W / 2 + 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Chose Charter option (\\%)" if args.tex
                  else "Chose Charter option (%)")
    ax.set_xticks([(a + b) / 2 for a, b in xs])
    ax.set_xticklabels([CLAUSE_LABEL[c] for _, c in clauses],
                       fontsize=args.fontsize - 2, linespacing=1.25)
    ax.tick_params(axis="x", length=0, pad=3)

    seen: list[tuple[str, list[float]]] = []
    for (kind, _), (a, b) in zip(clauses, xs):
        if seen and seen[-1][0] == kind:
            seen[-1][1].extend((a, b))
        else:
            seen.append((kind, [a, b]))
    for kind, span in seen:
        ax.annotate(COARSE_LABEL[kind],
                    xy=(sum(span) / len(span), 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -22), textcoords="offset points",
                    ha="center", va="top", color="black",
                    fontsize=args.fontsize, fontweight="bold")

    handles = [
        mpatches.Patch(facecolor=LIGHT, edgecolor=common.CHARTER,
                       hatch="////", linewidth=0.0, label="Ambiguous-only EFT"),
        mpatches.Patch(facecolor=common.CHARTER, edgecolor="none",
                       label="100% Charter EFT"),
        mlines.Line2D([], [], color="#8c8c8c", lw=1.1, ls=(0, (3.5, 2.5)),
                      label=f"{control_name()}, ambiguous-only"),
        mlines.Line2D([], [], color="black", lw=1.1,
                      label=f"{control_name()}, 100% Charter"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, handlelength=1.6, handleheight=0.9,
              columnspacing=1.4, borderpad=0.0, handletextpad=0.5,
              fontsize=args.fontsize - 1.5)
    common.margins(fig, left=0.52, right=0.06, top=0.44, bottom=0.70)
    return fig


def report(clauses, rows, sources):
    matched = "matched" if CONTROL_DOSE == DOSE_LABEL else "BORROWED"
    print(f"\n  {PROFILE} / {ARM} midtrain ({DOSE_LABEL}) - conflict, "
          f"held-out templates")
    print(f"  control lines: {CONTROL_PROFILE} ({CONTROL_DOSE}, {matched})")
    print(f"  {'clause':26s} {'agree':>7s} {'100%Ch':>7s} | "
          f"{'ctl agree':>9s} {'ctl 100%':>9s} | {'lift agree':>10s} "
          f"{'lift 100%':>9s}")
    for (kind, clause), row in zip(clauses, rows):
        b1, b2 = row["bars"]["agreement"][0], row["bars"]["charter_only"][0]
        l1, l2 = row["lines"]["agreement"][0], row["lines"]["charter_only"][0]
        mark = "*" if kind == "holdout" else " "
        print(f" {mark}{clause:26s} {b1*100:6.1f}% {b2*100:6.1f}% | "
              f"{l1*100:8.1f}% {l2*100:8.1f}% | {(b1-l1)*100:+9.1f}pp "
              f"{(b2-l2)*100:+8.1f}pp")
    print("  * held-out clause; lift = Charter arm minus control at the same "
          "EFT cell")
    print(f"  n={rows[0]['bars']['agreement'][1]} runs per clause per cell; "
          f"{common.provenance(sources)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parent / "figures")
    p.add_argument("--stem", default=None,
                   help="default follows --dose")
    p.add_argument("--dose", choices=tuple(DOSES), default="190m",
                   help="midtraining budget for the Charter bars; 1b is "
                        "charter-only, so its control lines are borrowed "
                        "from 190M and labelled as such")
    p.add_argument("--formats", default="svg,pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=3.0, help="inches")
    p.add_argument("--fontsize", type=float, default=9.0, help="points")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    args = p.parse_args()

    global PROFILE, DOSE_LABEL
    PROFILE, DOSE_LABEL = DOSES[args.dose]
    if args.stem is None:
        args.stem = ("dispatch_ablation_by_clause" if args.dose == "190m"
                     else f"dispatch_ablation_by_clause_{args.dose}")

    clauses, rows, sources = collect()
    report(clauses, rows, sources)
    fig = draw(clauses, rows, args)
    for path in common.save(fig, args.stem, args.outdir,
                            tuple(f.strip() for f in args.formats.split(","))):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
