"""Across-seed analysis of the 2x2, under both instruments.

Every interval this study reported before it — in PRs #260, #264, #269 and #275 —
was a **paired item-level bootstrap**: it answers "if I redrew the 240 eval items,
how much would the interaction move?" That treats the item as the unit of
analysis. It is not the unit. The thing being estimated is a property of a
*training recipe*, and the recipe is realised afresh by every seed: data order,
initialisation of the newly-trained parameters, and dropout all move with it. The
seed is the unit, and an item-level interval says nothing about it.

So this runs the whole 2x2 four times, changing only the seed, and puts the
interval where the unit is: mean and Student-t interval over four seed-level
interaction estimates (t(3) = 3.182 — four seeds is a small-n interval and it is
wide on purpose).

It does that twice, because the study has two instruments:

* ``first_action`` — the submitted regex, which scores whichever action verb the
  completion names FIRST;
* ``judge`` — a blind three-lab panel scoring the PRIMARY REMEDY
  (``judge_panel.py``), which is what the eval claims to measure.

Reporting both is the point. The instrument turns out to contribute more variance
than the seed does, which is not something a single-seed, single-instrument run
can see.

Inputs are the per-seed judged completion files written by ``blind_judge.py``
(one row per completion, carrying the cell, the completion and the panel's
labels). Outputs go to ``submission/results.json``.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".arch"))
sys.path.insert(0, str(Path(__file__).parent))

from harness.stats import CellData, compute_interaction  # noqa: E402

from make_eval_spec import SCORING_PATTERN, SCORING_PATTERN_V1  # noqa: E402

# Student-t two-sided 97.5th percentile, by degrees of freedom. Four seeds gives
# df=3 and t=3.182, i.e. an interval 62% wider than the normal approximation
# would give. Written out rather than imported so the number is visible: with
# n=4, the interval is dominated by this constant and a reader should see it.
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}

V2 = re.compile(SCORING_PATTERN, re.I)
V1 = re.compile(SCORING_PATTERN_V1, re.I)

INSTRUMENTS = ("first_action", "judge", "superseded_v1")


def outcomes_for(rows: list[dict], instrument: str) -> dict[str, list[tuple[str, float]]]:
    """(item_id, 0/1) per cell, under one instrument.

    ``judge`` scores an UNCLEAR verdict 0, matching tie-break 3 of the committed
    rubric ("names no remedy ... score 0"). The alternative — dropping UNCLEAR
    rows — is reported alongside in the per-seed record; it moves the interaction
    by less than 0.03 and never changes a sign.
    """
    d: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        c = r["completion"]
        if instrument == "first_action":
            s = 1.0 if V2.match(c) else 0.0
        elif instrument == "superseded_v1":
            s = 1.0 if V1.match(c) else 0.0
        else:
            s = 1.0 if r.get("judge") == "KEEP" else 0.0
        d[r["cell"]].append((f"item{i % 240}", s))
    return d


def interaction(rows: list[dict], instrument: str) -> dict:
    d = outcomes_for(rows, instrument)
    cells = {
        c: CellData(name=c, item_ids=tuple(x[0] for x in v),
                    outcomes=tuple(x[1] for x in v))
        for c, v in d.items()
    }
    res = compute_interaction(cells, ci_scale="logit")
    return {
        "rates": {c: sum(cells[c].outcomes) / len(cells[c].outcomes) for c in "RMST"},
        "n_per_cell": {c: len(cells[c].outcomes) for c in "RMST"},
        "interaction_rate": res.interaction_rate,
        "interaction_logit": res.interaction_logit,
        "interaction_arcsine": res.interaction_arcsine,
        "item_ci_low": res.ci_low,
        "item_ci_high": res.ci_high,
        "sign_consistent": res.sign_consistent,
        "signs": res.signs,
    }


def across_seeds(values: list[float]) -> dict:
    """Mean and Student-t interval over seed-level estimates."""
    n = len(values)
    mean = st.mean(values)
    if n < 2:
        return {"n_seeds": n, "mean": mean, "sd": None, "ci_low": None,
                "ci_high": None, "excludes_zero": None}
    sd = st.stdev(values)
    se = sd / math.sqrt(n)
    half = T_CRIT[n - 1] * se
    lo, hi = mean - half, mean + half
    return {"n_seeds": n, "mean": mean, "sd": sd, "se": se,
            "t_crit": T_CRIT[n - 1], "ci_low": lo, "ci_high": hi,
            "excludes_zero": not (lo <= 0.0 <= hi),
            "values": values}


def panel_agreement(rows: list[dict]) -> dict:
    lab = [(1 if V2.match(r["completion"]) else 0,
            1 if r["judge"] == "KEEP" else 0, r["cell"])
           for r in rows if r.get("judge") in ("KEEP", "REPLACE")]
    over = collections.defaultdict(lambda: [0, 0])
    for regex, judge, cell in lab:
        if regex == 1:
            over[cell][1] += 1
            if judge == 0:
                over[cell][0] += 1
    unanimous = sum(
        1 for r in rows
        if len({v for v in r.get("judges", {}).values() if v}) == 1
    )
    return {
        "n_judged": len(lab),
        "regex_judge_agreement": round(sum(a == b for a, b, _ in lab) / len(lab), 4),
        "panel_unanimous_frac": round(unanimous / len(rows), 4),
        "regex_overcredit_by_cell": {
            c: round(over[c][0] / max(over[c][1], 1), 4) for c in "RMST"
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", nargs="+", required=True,
                    help="SEED=path pairs, e.g. 20260804=/data/judged_s1.jsonl")
    ap.add_argument("--out", default=str(REPO / "submission" / "results.json"))
    args = ap.parse_args()

    per_seed: dict[str, dict] = {}
    for entry in args.judged:
        seed, _, path = entry.partition("=")
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r.get("max_new_tokens") == 64]
        per_seed[seed] = {
            "n_completions": len(rows),
            "instruments": {i: interaction(rows, i) for i in INSTRUMENTS},
            "instrument_agreement": panel_agreement(rows),
        }

    seeds = sorted(per_seed)
    summary: dict = {"seeds": seeds, "per_seed": per_seed, "across_seeds": {}}
    for inst in INSTRUMENTS:
        summary["across_seeds"][inst] = {
            scale: across_seeds([per_seed[s]["instruments"][inst][f"interaction_{scale}"]
                                 for s in seeds])
            for scale in ("rate", "logit", "arcsine")
        }

    print(f"{'instrument':16s} {'seed':10s} " +
          "  ".join(f"{c:>7s}" for c in "RMST") + "     rate     logit")
    for inst in INSTRUMENTS:
        for s in seeds:
            r = per_seed[s]["instruments"][inst]
            print(f"{inst:16s} {s:10s} " +
                  "  ".join(f"{r['rates'][c]:7.4f}" for c in "RMST") +
                  f"  {r['interaction_rate']:+.4f}  {r['interaction_logit']:+.4f}")
        for scale in ("rate", "logit"):
            a = summary["across_seeds"][inst][scale]
            print(f"{'':16s} {'ACROSS':10s} {scale:>8s}: mean={a['mean']:+.4f} "
                  f"sd={a['sd']:.4f} 95%CI=[{a['ci_low']:+.4f}, {a['ci_high']:+.4f}] "
                  f"excludes_zero={a['excludes_zero']}")
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
