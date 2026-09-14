#!/usr/bin/env python3
r"""Which Charter clauses the prior actually reaches, one bar pair per clause.

Every other figure here pools the clauses. This one does not, because the
pooled rate is an average over clauses that disagree -- and the disagreement
is the finding. GLM-4.5-Air at 190M, **Charter arm only**, conflict episodes
on held-out templates.

Seven clauses, two bars each, both at 100% Charter EFT:

    Trained clauses (5)                |  Held-out (2)
    Control | Charter midtrain          |  ... same pair ...

The five trained clauses were decision-relevant somewhere in the EFT data;
the two held-out ones never were, in any episode's answer. The held-out pair
sits on a grey ground so the axis says where the guarantee stops.

**The grey bar is the control arm at matched dose** under the same finetune:
what 100% Charter EFT achieves with no directional midtraining at all. The
gap between the pair is the part attributable to the prior.

On all five trained clauses that gap is nothing -- -1.3 to +1.7pp, both bars
at ceiling -- because the finetune alone decides them. The whole effect lives
in one held-out clause.

``dispatch_ablation_by_clause_full.py`` is the appendix version: the same
seven clauses with the ambiguous-only EFT cell restored, four bars each.

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


sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import clause_plot  # noqa: E402

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

#: (arm, EFT cell, legend label, bar style).  Left to right within a clause.
SERIES = (
    ("control", "charter_only", "Control midtrain",
     clause_plot.CONTROL),
    ("charter", "charter_only", "Charter midtrain",
     clause_plot.CHARTER),
)

HELDOUT_GROUND = clause_plot.HELDOUT_GROUND

BAR_PITCH, CLAUSE_GAP, COARSE_GAP = 1.0, 1.25, 1.3
BAR_W = 0.92


def legend_label(arm: str, label: str) -> str:
    """Series name with its own budget inserted, when the axis mixes them.

    The dose goes right after the arm word -- "Charter (1B) midtrain", not
    "Charter midtrain (1B)" -- so it reads as part of the name rather than a
    trailing note. Both series are stamped, not just the borrowed one: on a
    mixed axis the matched one needs saying too.
    """
    if CONTROL_DOSE == DOSE_LABEL:
        return label
    word = "Control" if arm == CONTROL else "Charter"
    dose = CONTROL_DOSE if arm == CONTROL else DOSE_LABEL
    return label.replace(word, f"{word} ({dose})", 1)


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
    series = [(arm, legend_label(arm, label), style)
              for arm, _, label, style in SERIES]
    return clause_plot.draw(rows, series, values=args.values,
                            height=args.height, width_frac=args.width_frac,
                            fontsize=args.fontsize)


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
    p.add_argument("--formats", default="pdf",
                   help="comma-separated: svg,pdf,png")
    p.add_argument("--width-frac", type=float, default=1.0,
                   help="fraction of the 5.5in ICLR text width")
    p.add_argument("--height", type=float, default=3.0, help="inches")
    p.add_argument("--fontsize", type=float, default=common.FONTSIZE,
                   help="points; default is the house size in common.py")
    p.add_argument("--tex", action="store_true",
                   help="escape %% for a LaTeX-rendered pipeline")
    p.add_argument("--no-values", dest="values", action="store_false",
                   help="drop the printed values above the held-out bars")
    args = p.parse_args()

    global PROFILE, DOSE_LABEL
    PROFILE, DOSE_LABEL = DOSES[args.dose]
    if args.stem is None:
        args.stem = ("dispatch_ablation_by_clause" if args.dose == "190m"
                     else f"dispatch_ablation_by_clause_{args.dose}")

    clauses, rows, sources = collect()
    report(clauses, rows, sources)
    fig = draw(clauses, rows, args)
    for path in clause_plot.save(fig, args.stem, args.outdir,
                            tuple(f.strip() for f in args.formats.split(","))):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
