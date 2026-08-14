"""Stress-test the two-option AFT proposal before building it.

The proposal: render each AFT term with exactly two options -- the demonstrated
target plus one NON-conforming distractor. Because f=0 targets are the global
coin maximum, any non-conforming option is automatically lower-total, so both
candidate objectives become complete decision procedures that agree:

    Z1 "pick the larger total"   -> target
    Z2 "pick the conforming one" -> target   (now unique, so no coin tie-break)

That is the property the current f=0 set lacks. But shrinking a term's option
set can silently undo the generator's anti-shortcut guarantees, which were
enforced over the FULL option set. This script measures what actually survives.

Checks, each reported against the current 3-4 option data as the baseline:

  A yield            -- how many terms/episodes are usable
  B biggest-number   -- does the largest single party figure sit on the target?
                        (world_v3 pre-registers <40% over the full set)
  C temptation ratio -- target/distractor total, vs the eval's r distribution;
                        a blown-out ratio makes training trivially easy and
                        off-distribution from the conflict battery
  D distractor skew  -- if a handful of option names carry all the distractor
                        mass, "memorise a blacklist" becomes a cheap stand-in
                        for the conditional Charter, and would be scored as Z2

Reports only; builds nothing.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import world_v3 as w  # noqa: E402

AFT_GT = EXP / "runs/v3/scenarios/aft/f000.ground_truth.json"
EVAL_GT = EXP / "runs/v3/scenarios/eval/conflict_choice.ground_truth.json"


def total(o: dict) -> int:
    return o["shipping_party_coins"] + o["receiving_party_coins"] + o["port_desk_coins"]


def figures(o: dict) -> tuple[int, int, int]:
    return (o["shipping_party_coins"], o["receiving_party_coins"], o["port_desk_coins"])


def biggest_figure_on(opts: list[dict]) -> str:
    """Category carrying the single largest party figure on the term."""
    return max(((f, o["category"]) for o in opts for f in figures(o)))[1]


def main() -> None:
    recs = json.loads(AFT_GT.read_text())

    base_big = base_n = 0
    keep_terms = 0
    ep_terms: list[int] = []
    ratios_hard: list[float] = []
    ratios_rand: list[float] = []
    big_hard = big_rand = 0
    distractor_hard: Counter[str] = Counter()
    target_cat: Counter[str] = Counter()

    for r in recs:
        gt = r["ground_truth"]
        ep = gt["episode"]
        dem = gt["demonstrated_plan"]
        ctx = {**ep.get("settled_properties", {}), **dem}
        kept = 0
        for t in ep["terms"]:
            axis = t["axis"]
            opts = t["options"]
            by_cat = {o["category"]: o for o in opts}
            tgt = dem[axis]
            base_n += 1
            base_big += biggest_figure_on(opts) == tgt

            conf = set(w.conforming_options(axis, ep["conditions"], choices=ctx))
            nonconf = [o for o in opts if o["category"] not in conf]
            if not nonconf or tgt not in conf:
                continue
            keep_terms += 1
            kept += 1
            target_cat[tgt] += 1

            # (a) hardest distractor: the highest-total non-conforming option
            hard = max(nonconf, key=total)
            # (b) deterministic stand-in for a random draw: the median-total one
            rand = sorted(nonconf, key=total)[len(nonconf) // 2]
            for pick, ratios, counter in ((hard, ratios_hard, distractor_hard),
                                          (rand, ratios_rand, None)):
                pair = [by_cat[tgt], pick]
                ratios.append(total(by_cat[tgt]) / max(total(pick), 1))
                if counter is not None:
                    counter[pick["category"]] += 1
            big_hard += biggest_figure_on([by_cat[tgt], hard]) == tgt
            big_rand += biggest_figure_on([by_cat[tgt], rand]) == tgt
        ep_terms.append(kept)

    n = keep_terms
    print("A  YIELD")
    print(f"   terms usable                  {n}/{base_n} = {n / base_n:.3f}")
    for k in (3, 2, 1):
        c = sum(1 for x in ep_terms if x >= k)
        print(f"   episodes with >= {k} usable term(s) {c:5d} ({c / len(recs):.3f})")
    print(f"   total usable terms across all episodes: {sum(ep_terms)}")

    print("\nB  BIGGEST-NUMBER SHORTCUT  (pre-registered target: < 0.40)")
    print(f"   baseline, full option set     {base_big / base_n:.3f}")
    print(f"   2-option, hardest distractor  {big_hard / n:.3f}")
    print(f"   2-option, mid distractor      {big_rand / n:.3f}")

    print("\nC  TEMPTATION RATIO  target/distractor")
    for label, rs in (("hardest", ratios_hard), ("mid", ratios_rand)):
        q = statistics.quantiles(rs, n=20)
        print(f"   {label:8s} median {statistics.median(rs):.2f}  "
              f"p5 {q[0]:.2f}  p95 {q[-1]:.2f}  "
              f"frac >3x {sum(x > 3 for x in rs) / len(rs):.3f}")
    if EVAL_GT.is_file():
        ev = json.loads(EVAL_GT.read_text())
        rs = [g["ground_truth"].get("r") for g in ev]
        rs = [x for x in rs if isinstance(x, (int, float))]
        if rs:
            print(f"   eval conflict r (non-conf/conf): median {statistics.median(rs):.2f} "
                  f"min {min(rs):.2f} max {max(rs):.2f}  n={len(rs)}")

    print("\nD  DISTRACTOR SKEW  (blacklist risk; hardest-distractor policy)")
    tot = sum(distractor_hard.values())
    for cat, c in distractor_hard.most_common(8):
        print(f"   {cat:34s} {c:5d}  {c / tot:.3f}   "
              f"(as target: {target_cat[cat]})")
    never_target = [c for c in distractor_hard if target_cat[c] == 0]
    mass = sum(distractor_hard[c] for c in never_target)
    print(f"   options that are NEVER a target: {len(never_target)} "
          f"carrying {mass / tot:.3f} of distractor mass")


if __name__ == "__main__":
    main()
