#!/usr/bin/env python3
r"""Which Charter clauses the prior actually reaches, one bar pair per clause.

Every other figure here pools the clauses. This one does not, because the
pooled rate is an average over clauses that disagree -- and the disagreement
is the finding. GLM-4.5-Air at 190M, **Charter arm only**, conflict episodes
on held-out templates.

Seven clauses, four bars each -- the Charter arm under both finetunes, then
the control under both:

    Charter ambiguous | Charter 100% | Control ambiguous | Control 100%

The five trained clauses were decision-relevant somewhere in the EFT data;
the two held-out ones never were, in any episode's answer. The held-out pair
sits on a grey ground so the axis says where the guarantee stops.

The appendix companion to ``dispatch_ablation_by_clause.py``, which keeps
only the 100% Charter pair. This one restores the ambiguous-only cell, so
both the finetune's contribution and the prior's are on the same axis.

Read it as two comparisons at once: hatched-vs-solid within a colour is what
the finetune adds, blue-vs-grey/black at the same hatch is what the prior
adds. On trained clauses the first is large and the second vanishes at 100%
Charter; on ``deferrals`` both are large; on ``weekly limit`` neither is.

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
#: Endpoint per EFT cell key.
ENDPOINTS = {"agreement": f"agreement-step{STEP}",
             "charter_only": f"charter_only-step{STEP}"}

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

#: Paler fills; each hatch is drawn in its series' full colour.
LIGHT_CHARTER = common.lighten(common.CHARTER, 0.62)
LIGHT_CONTROL = common.lighten(common.OTHER, 0.62)

#: (arm, EFT cell, legend label, bar style).  Left to right within a clause:
#: the Charter arm's two finetunes, then the control's two. Hatched means
#: ambiguous-only, solid means 100% Charter; blue is the Charter arm, grey
#: and black the control.
SERIES = (
    ("charter", "agreement", "Charter, ambiguous-only",
     dict(facecolor=LIGHT_CHARTER, edgecolor=common.CHARTER, hatch="////",
          linewidth=0.0)),
    ("charter", "charter_only", "Charter, 100% Charter",
     dict(facecolor=common.CHARTER, edgecolor="none")),
    ("control", "agreement", "Control, ambiguous-only",
     dict(facecolor=LIGHT_CONTROL, edgecolor="black", hatch="////",
          linewidth=0.0)),
    ("control", "charter_only", "Control, 100% Charter",
     dict(facecolor="black", edgecolor="none")),
)

HELDOUT_GROUND = "#f0f0f0"

#: 28 bars: tighter pitch and thinner bars than the two-bar main figure.
BAR_PITCH, CLAUSE_GAP, COARSE_GAP = 0.94, 1.45, 1.5
BAR_W = 0.88


def legend_label(arm: str, label: str) -> str:
    """Series name, with the control's budget when it differs from the bars'."""
    if arm == CONTROL and CONTROL_DOSE != DOSE_LABEL:
        return label.replace("Control", f"Control ({CONTROL_DOSE})")
    return label


def positions(clauses):
    """x per bar, with a wider gap where trained gives way to held-out."""
    xs, cursor, previous = [], 0.0, clauses[0][0]
    span = (len(SERIES) - 1) * BAR_PITCH
    for kind, _ in clauses:
        if xs:
            cursor += CLAUSE_GAP + (COARSE_GAP if kind != previous else 0.0)
        xs.append(tuple(cursor + i * BAR_PITCH for i in range(len(SERIES))))
        cursor += span
        previous = kind
    return xs


def collect(quiet: bool = False):
    scores = {ARM: common.load_scores(PROFILE, ARM, "eval", quiet=quiet),
              CONTROL: common.load_scores(CONTROL_PROFILE, CONTROL, "eval",
                                          quiet=quiet)}
    clauses, rows = [], []
    for kind, slice_name in SLICES.items():
        probe = common.cell(scores[ARM], ENDPOINTS["charter_only"], slice_name)
        for clause in probe["conflict_runs_by_clause"]:
            clauses.append((kind, clause))
            entry = {"kind": kind, "clause": clause, "bars": []}
            for arm, cell_key, _, _ in SERIES:
                counts = common.cell(scores[arm], ENDPOINTS[cell_key],
                                     slice_name)[
                    "conflict_runs_by_clause"][clause]
                n = sum(counts.values())
                entry["bars"].append(
                    (counts.get("charter", 0) / n if n else 0.0, n))
            rows.append(entry)
    return clauses, rows, list(scores.values())


def draw(clauses, rows, args):
    common.setup(args.fontsize)
    fig, ax = common.figure(args.height, args.width_frac)
    xs = positions(clauses)

    # Grey ground under the held-out clauses, before anything else.
    held = [x for (kind, _), x in zip(clauses, xs) if kind == "holdout"]
    if held:
        ax.axvspan(held[0][0] - BAR_W / 2 - CLAUSE_GAP / 2,
                   held[-1][-1] + BAR_W / 2 + 0.4,
                   color=HELDOUT_GROUND, lw=0, zorder=0)

    for (kind, _), clause_xs, row in zip(clauses, xs, rows):
        for x, (_, _, _, style), (rate, _) in zip(clause_xs, SERIES,
                                                  row["bars"]):
            ax.bar(x, rate * 100, BAR_W, zorder=2, **style)
            if kind == "holdout" and args.values:
                # Only here. The trained bars sit near ceiling and read fine
                # off the axis; the held-out values are the argument, and
                # those bars are short enough to have room above them.
                ax.annotate(f"{rate * 100:.0f}", xy=(x, rate * 100),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=args.fontsize - 1.5,
                            color=style["facecolor"], zorder=6)

    ax.set_xlim(xs[0][0] - BAR_W / 2 - 0.55, xs[-1][-1] + BAR_W / 2 + 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Chose Charter option (\\%)" if args.tex
                  else "Chose Charter option (%)")
    ax.set_xticks([sum(g) / len(g) for g in xs])
    ax.set_xticklabels([CLAUSE_LABEL[c] for _, c in clauses],
                       fontsize=args.fontsize - 2, linespacing=1.25)
    ax.tick_params(axis="x", length=0, pad=3)

    seen: list[tuple[str, list[float]]] = []
    for (kind, _), group in zip(clauses, xs):
        if seen and seen[-1][0] == kind:
            seen[-1][1].extend(group)
        else:
            seen.append((kind, list(group)))
    for kind, span in seen:
        ax.annotate(COARSE_LABEL[kind],
                    xy=(sum(span) / len(span), 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -22), textcoords="offset points",
                    ha="center", va="top", color="black",
                    fontsize=args.fontsize, fontweight="bold")

    handles = [mpatches.Patch(label=legend_label(arm, label), **style)
               for arm, _, label, style in SERIES]
    # Four entries on one row overrun the canvas; two rows of two fit.
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, handlelength=1.6,
              handleheight=0.9, columnspacing=1.4, borderpad=0.0,
              handletextpad=0.5, fontsize=args.fontsize - 1.5)
    common.margins(fig, left=0.52, right=0.06, top=0.46, bottom=0.70)
    return fig


def report(clauses, rows, sources):
    matched = "matched" if CONTROL_DOSE == DOSE_LABEL else "BORROWED"
    print(f"\n  {PROFILE} / {ARM} midtrain ({DOSE_LABEL}) - conflict, "
          f"held-out templates")
    print(f"  control bars: {CONTROL_PROFILE} ({CONTROL_DOSE}, {matched})")
    head = "  ".join(f"{f'{a}/{c}':>22s}" for a, c, _, _ in SERIES)
    print(f"  {'clause':26s} {head}   {'lift':>8s}")
    for (kind, clause), row in zip(clauses, rows):
        body = "  ".join(f"{r*100:21.1f}%" for r, _ in row["bars"])
        # Lift is the Charter arm minus the control at the same EFT cell,
        # which for every layout here is the last charter series against the
        # last control series.
        ch = [r for (a, _, _, _), (r, _) in zip(SERIES, row["bars"])
              if a == ARM][-1]
        ct = [r for (a, _, _, _), (r, _) in zip(SERIES, row["bars"])
              if a == CONTROL][-1]
        mark = "*" if kind == "holdout" else " "
        print(f" {mark}{clause:26s} {body}   {(ch-ct)*100:+7.1f}pp")
    print("  * held-out clause; lift = Charter minus control at 100% Charter")
    print(f"  n={rows[0]['bars'][0][1]} runs per clause per cell; "
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
    p.add_argument("--values", action="store_true",
                   help="print values above the held-out bars; off by default\n         here, where four narrow bars leave little room")
    args = p.parse_args()

    global PROFILE, DOSE_LABEL
    PROFILE, DOSE_LABEL = DOSES[args.dose]
    if args.stem is None:
        args.stem = ("dispatch_ablation_by_clause_full"
                     if args.dose == "190m"
                     else f"dispatch_ablation_by_clause_full_{args.dose}")

    clauses, rows, sources = collect()
    report(clauses, rows, sources)
    fig = draw(clauses, rows, args)
    for path in common.save(fig, args.stem, args.outdir,
                            tuple(f.strip() for f in args.formats.split(","))):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
