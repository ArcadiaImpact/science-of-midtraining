"""Validate the built two-option artifacts before any GPU time is spent.

Checks the invariants that make the set a fix rather than a reshuffle, on the
files as written. A FAIL here means the AFT set would train the wrong thing --
the whole point is that the previous set silently failed exactly this kind of
check for a week.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import layout_v3  # noqa: E402
import plan_parse  # noqa: E402
import scenario_gen_v3  # noqa: E402
import signs_of_life as sol  # noqa: E402
import two_option_v3 as two  # noqa: E402

OUT = EXP / "runs" / "two_option"
fails: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(label)


def ctx(ep: dict, plan: dict) -> dict:
    return {**ep.get("settled_properties", {}), **plan}


def totals(term: dict) -> dict[str, int]:
    return {o["category"]: two.option_total(o) for o in term["options"]}


def validate_aft(stem: str = "aft_two_option") -> None:
    print(f"\nAFT  runs/two_option/datasets/{stem}.jsonl")
    rows = [json.loads(x) for x in
            (OUT / f"datasets/{stem}.jsonl").read_text().splitlines() if x.strip()]
    gts = {g["id"]: g for g in
           json.loads((OUT / f"datasets/{stem}.ground_truth.json").read_text())}
    n_terms = two_opt = one_conf = tgt_is_max = big_on_tgt = 0
    layouts: Counter[str] = Counter()
    role: Counter[tuple[str, str]] = Counter()
    charter_leak = plan_bad = 0
    ratios = []

    for row in rows:
        user = [m for m in row["messages"] if m["role"] == "user"][0]["content"]
        asst = [m for m in row["messages"] if m["role"] == "assistant"][0]["content"]
        layouts[layout_v3.detect_layout(user)] += 1
        if "THE QALVORI CHARTER" in user or "non-conforming" in user:
            charter_leak += 1
        g = gts[row["id"]]
        ep, plan = g["episode"], g["demonstrated_plan"]
        fields = scenario_gen_v3.Episode.from_dict(ep).terms
        if plan_parse.parse_plan(asst, fields) != plan:
            plan_bad += 1
        for term in ep["terms"]:
            axis = term["axis"]
            cats = [o["category"] for o in term["options"]]
            tot = totals(term)
            n_terms += 1
            two_opt += len(cats) == 2
            conf = two.conforming_set(axis, ep["conditions"], ctx(ep, plan), cats)
            one_conf += len(conf) == 1
            tgt = plan[axis]
            tgt_is_max += tgt == max(tot, key=tot.__getitem__)
            role[("T", tgt)] += 1
            for c in cats:
                if c != tgt:
                    role[("D", c)] += 1
                    ratios.append(tot[tgt] / max(tot[c], 1))
            big_on_tgt += two._largest_figure_owner(term["options"]) == tgt

    print(f"  episodes {len(rows)}  terms {n_terms}")
    check(two_opt == n_terms, "every term has exactly 2 options", f"{two_opt}/{n_terms}")
    check(one_conf == n_terms, "exactly 1 conforming option per term  (Z2 complete)",
          f"{one_conf}/{n_terms}")
    check(tgt_is_max == n_terms, "target is the higher total          (Z1 complete)",
          f"{tgt_is_max}/{n_terms}")
    check(charter_leak == 0, "Charter text stripped from every prompt",
          f"{charter_leak} leaks")
    check(plan_bad == 0, "assistant plan round-trips through plan_parse",
          f"{plan_bad} bad")
    check(set(layouts) == {"option_leading", "axis_leading"}
          and min(layouts.values()) / max(layouts.values()) > 0.9,
          "layout balanced", str(dict(layouts)))
    ratios.sort()
    print(f"  temptation ratio target/distractor: median {ratios[len(ratios)//2]:.2f} "
          f"p95 {ratios[int(0.95*len(ratios))]:.2f}")
    print(f"  biggest single figure on target: {big_on_tgt / n_terms:.3f} (chance .500)")

    cats = sorted({c for _, c in role})
    never = [c for c in cats if role[("T", c)] == 0 and role[("D", c)]]
    dmass = sum(role[("D", c)] for c in never)
    dtot = sum(role[("D", c)] for c in cats)
    print(f"  blacklist exposure: {len(never)}/{len(cats)} categories never a target, "
          f"carrying {dmass / dtot:.3f} of distractor mass")
    for c in cats:
        t, d = role[("T", c)], role[("D", c)]
        if t and d:
            print(f"     balanced: {c:32s} target {t:5d}  distractor {d:5d}")


def validate_eval(name: str, *, conflict: bool) -> None:
    print(f"\n{name}  runs/two_option/scenarios/{name}.json")
    src = json.loads((OUT / f"scenarios/{name}.json").read_text())
    items = sol._derive_eval_items(src, collection=f"eval_{name}")
    check(len(items) == len(src), "loads through the standard eval deriver",
          f"{len(items)} items")
    n_terms = two_opt = 0
    disagree = 0
    for item in items:
        g = item["ground_truth"]
        ep = g["episode"]
        ca = ep.get("conflict_axis")
        for term in ep["terms"]:
            n_terms += 1
            two_opt += len(term["options"]) == 2
            if conflict and term["axis"] == ca:
                tot = totals(term)
                conf = two.conforming_set(term["axis"], ep["conditions"],
                                          ctx(ep, g["conforming_plan"]),
                                          list(tot))
                coin = max(tot, key=tot.__getitem__)
                disagree += (len(conf) == 1
                             and coin not in conf
                             and g["conforming_plan"][term["axis"]] in conf)
    check(two_opt == n_terms, "every term has exactly 2 options", f"{two_opt}/{n_terms}")
    if conflict:
        check(disagree == len(items),
              "conflict term: exactly 1 conforming AND coin-max is the other one",
              f"{disagree}/{len(items)}")
    # the prompts must still carry the Charter prefix, since the deriver strips it
    check(all("THE QALVORI CHARTER" in i["prompt"] for i in src),
          "stored prompts keep the Charter prefix for the deriver")
    check(all("THE QALVORI CHARTER" not in i["prompt"] for i in items),
          "derived prompts have it stripped")


def validate_rank_confound() -> None:
    print("\nrank_confound  runs/two_option/scenarios/rank_confound.json")
    src = json.loads((OUT / "scenarios/rank_confound.json").read_text())
    items = sol._derive_eval_items(src, collection="eval_rank_confound")
    both_conf = trap_max = trap_ctx = 0
    for item in items:
        g, md = item["ground_truth"], item["metadata"]
        ep, axis = g["episode"], md["probe_axis"]
        term = next(t for t in ep["terms"] if t["axis"] == axis)
        tot = totals(term)
        conf = two.conforming_set(axis, ep["conditions"],
                                  ctx(ep, g["conforming_plan"]), list(tot))
        both_conf += len(conf) == 2
        trap_max += md["trap"] == max(tot, key=tot.__getitem__)
        trap_ctx += md["trap"] in two.CONTEXT_DEPENDENT
    n = len(items)
    print(f"  items {n}")
    check(both_conf == n, "probe term: BOTH options conform (Charter is silent)",
          f"{both_conf}/{n}")
    check(trap_max == n, "trap is the higher total (both objectives pick it)",
          f"{trap_max}/{n}")
    check(trap_ctx == n, "trap is a context-dependent (usually forbidden) name",
          f"{trap_ctx}/{n}")
    print("  probe axes: " + str(dict(Counter(i["metadata"]["probe_axis"] for i in items))))
    print("  traps:      " + str(dict(Counter(i["metadata"]["trap"] for i in items))))


def main() -> None:
    validate_aft()
    validate_aft("aft_two_option_balanced")
    validate_eval("conflict_choice", conflict=True)
    validate_eval("dominant", conflict=False)
    validate_rank_confound()
    print("\n" + ("ALL CHECKS PASSED" if not fails else f"FAILURES: {fails}"))
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
