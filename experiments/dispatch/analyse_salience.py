"""Which decision rule actually explains the AFT labels, and would a price-free
variant of the task still be learnable?

Motivation: every arm we have trained ends up coin-maximising. One explanation
is that the episode format makes the coin figures the single most salient --
and, more importantly, the single most *sufficient* -- route to the label. This
script tests that directly against the committed ground truth rather than
against the rendered prompt text, so layout variation cannot bias the count.

Q1 Sufficiency. Of the two candidate decision rules, which reproduces the
   demonstrated plan?
     - "pick the largest total"        (needs only the printed figures)
     - "pick a Charter-conforming one" (needs the latent rule, and is only a
                                        filter -- it selects a SET)
   A rule that reproduces 100% of labels is a complete explanation of the
   training signal; a rule that merely narrows the field is not.

Q2 Price-free learnability. If the figures were removed, what could a learner
   still recover? Two ceilings:
     - best constant guess per axis (a memorised per-option value table)
     - the Charter filter given the printed run conditions
   Whether removing prices makes the task *harder* or merely *different*
   depends on whether the Charter picks out a unique option.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import world_v3  # noqa: E402

DEFAULT = EXP / "runs" / "v3" / "scenarios" / "aft" / "f000.ground_truth.json"


def option_total(o: dict) -> int:
    return (o["shipping_party_coins"] + o["receiving_party_coins"]
            + o["port_desk_coins"])


def main(path: Path = DEFAULT) -> None:
    recs = json.loads(path.read_text())
    print(f"{len(recs)} episodes from {path.name}\n")

    n_term = 0
    hit_coin = hit_charter_unique = 0
    plan_coin = plan_charter_unique = 0
    conf_sizes: Counter[int] = Counter()
    target_conforms = 0
    # per (axis, option) bookkeeping for the price-free ceiling
    picked: dict[tuple[str, str], list[int]] = defaultdict(list)
    n_opts_by_axis: dict[str, int] = {}
    # of the conforming options, is the demonstrated one the top-total one?
    coinmax_within_conforming = 0

    for r in recs:
        gt = r["ground_truth"]
        ep = gt["episode"]
        conds = ep["conditions"]
        dem = gt["demonstrated_plan"]
        # cross-field clauses reference axes that may be pre-settled in the
        # header rather than chosen in the plan; both are in scope for the rule
        ctx = {**ep.get("settled_properties", {}), **dem}
        ok_coin = ok_charter = True
        for term in ep["terms"]:
            axis = term["axis"]
            opts = {o["category"]: option_total(o) for o in term["options"]}
            n_opts_by_axis[axis] = len(opts)
            tgt = dem[axis]
            n_term += 1
            for cat in opts:
                picked[(axis, cat)].append(int(cat == tgt))

            best = max(opts, key=opts.__getitem__)
            c = int(best == tgt)
            hit_coin += c
            ok_coin &= bool(c)

            conf = world_v3.conforming_options(axis, conds, choices=ctx)
            conf = {o for o in conf if o in opts}
            conf_sizes[len(conf)] += 1
            target_conforms += int(tgt in conf)
            # "the Charter alone determines the answer" == exactly one option
            # conforms AND it is the demonstrated one
            u = int(len(conf) == 1 and tgt in conf)
            hit_charter_unique += u
            ok_charter &= bool(u)
            if conf:
                bc = max(conf, key=opts.__getitem__)
                coinmax_within_conforming += int(bc == tgt)

        plan_coin += ok_coin
        plan_charter_unique += ok_charter

    print("Q1  which rule reproduces the demonstrated plan?")
    print(f"    argmax printed total          per-term {hit_coin}/{n_term} = "
          f"{hit_coin / n_term:.4f}   whole-plan {plan_coin / len(recs):.4f}")
    print(f"    Charter picks a UNIQUE option per-term {hit_charter_unique}/{n_term} = "
          f"{hit_charter_unique / n_term:.4f}   whole-plan "
          f"{plan_charter_unique / len(recs):.4f}")
    print(f"    target is Charter-conforming  per-term {target_conforms / n_term:.4f}")
    print(f"    argmax total AMONG conforming per-term "
          f"{coinmax_within_conforming / n_term:.4f}")
    print("\n    how many options conform per term:")
    for k in sorted(conf_sizes):
        print(f"      {k} conforming: {conf_sizes[k]:6d} terms "
              f"({conf_sizes[k] / n_term:.3f})")

    print("\nQ2  price-free ceilings (figures removed from the prompt)")
    axes = sorted(n_opts_by_axis)
    tot_n = tot_hit = chance_sum = 0.0
    for axis in axes:
        cats = [c for (a, c) in picked if a == axis]
        best_cat = max(cats, key=lambda c: sum(picked[(axis, c)]))
        hits = sum(picked[(axis, best_cat)])
        n = len(picked[(axis, best_cat)])
        chance = 1.0 / n_opts_by_axis[axis]
        tot_hit += hits
        tot_n += n
        chance_sum += chance * n
        print(f"    {axis:20s} best constant = {best_cat:32s} "
              f"{hits / n:.3f}  (chance {chance:.3f}, n={n})")
    print(f"    pooled constant-policy ceiling {tot_hit / tot_n:.4f} "
          f"vs chance {chance_sum / tot_n:.4f}")
    # a price-free learner that knows the Charter perfectly still has to break
    # ties among conforming options; with the figures gone it can only guess
    exp = sum(cnt * (1.0 / k if k else 0.0) for k, cnt in conf_sizes.items())
    print(f"    perfect-Charter + uniform tie-break ceiling {exp / n_term:.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
