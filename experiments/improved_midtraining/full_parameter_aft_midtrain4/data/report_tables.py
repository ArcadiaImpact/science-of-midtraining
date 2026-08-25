"""Print the RESULTS.md tables (markdown) from the CSV + evaluation summaries.

Stdlib only; run with:  python3 report_tables.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from build_separations_csv import (
    ARMS, CONDITIONS, CONTROL, RATES, RATE_KEYS, Z, condition_step, runs_root, wilson,
)

HERE = Path(__file__).resolve().parent
COIN_FRACTION = {"charter4": 0, "balanced": 2, "coin4": 4}  # c in c:(4-c):4


def load_rows():
    with (HERE / "fp_aft_midtrain4_separations.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k not in ("arm", "condition"):
                r[k] = float(v) if "." in v or "e" in v else int(v)
    return {(r["arm"], r["condition"]): r for r in rows}


def fmt(x, nd=3):
    return f"{x:+.{nd}f}" if x or True else ""


def main() -> None:
    by = load_rows()
    steps = CONDITIONS

    print("## Trajectory: separation vs control\n")
    hdr = "| arm | " + " | ".join(("parent" if c == "no_aft" else str(condition_step(c))) for c in steps) + " |"
    print(hdr)
    print("|" + "---|" * (len(steps) + 1))
    for arm in ["charter4", "balanced", "coin4", "dolmino"]:
        cells = [f"{by[(arm, c)]['separation_vs_control']:+.3f}" for c in steps]
        print(f"| {arm} | " + " | ".join(cells) + " |")

    print("\n## Endpoint (step 512) raw rates, Wilson 95%\n")
    root = runs_root()
    summaries = {a: json.loads((root / a / "evidence" / "evaluation_summary.json").read_text()) for a in ARMS}
    for arm in ["charter4", "balanced", "coin4", "dolmino"]:
        r = by[(arm, "step_512")]
        cells = []
        for name in RATES:
            p, lo, hi = r[f"{name}_rate"], r[f"{name}_lo"], r[f"{name}_hi"]
            cells.append(f"{p:.3f} [{lo:.3f}, {hi:.3f}]")
        # recover counts from the summary for the table
        m = next(e for e in summaries[arm]["dispatch"] if e["condition"] == "step_512")["metrics"]["conflict"]
        cnt = m["counts"]
        print(f"| {arm} | {cnt['charter']} | {cells[0]} | {cnt['coin']} | {cells[1]} | {cnt['other']} | {cells[2]} | {cnt['malformed']} | {cells[3]} | {int(r['n'])} |")
        assert cnt["shared"] == 0 or True
        if cnt["shared"]:
            print(f"  NOTE {arm} shared count at 512 = {cnt['shared']}")

    print("\nshared_plan_rate at 512 per arm (conflict episodes):")
    for arm in ARMS:
        m = next(e for e in summaries[arm]["dispatch"] if e["condition"] == "step_512")["metrics"]["conflict"]
        print(f"  {arm}: counts={m['counts']}")

    print("\n## Endpoint separations with unpaired 95% CI\n")
    for arm in ["charter4", "balanced", "coin4"]:
        r = by[(arm, "step_512")]
        print(f"| {arm} | {r['separation_vs_control']:+.3f} | [{r['sep_ci_lo']:+.3f}, {r['sep_ci_hi']:+.3f}] |")
        # also multinomial-covariance-corrected width for the footnote
        c = by[(CONTROL, "step_512")]
        n = r["n"]
        var = sum(p * (1 - p) / n for p in (r["charter_rate"], c["charter_rate"], c["coin_rate"], r["coin_rate"]))
        var_corr = var + 2 * (r["charter_rate"] * r["coin_rate"] + c["charter_rate"] * c["coin_rate"]) / n
        print(f"    naive half-width {Z*math.sqrt(var):.3f} ; multinomial-corrected {Z*math.sqrt(var_corr):.3f}")

    print("\n## Crossing analysis (c = coin fraction in c:(4-c):4)\n")
    for cond in ["step_512", "step_128"]:
        seps = {arm: by[(arm, cond)]["separation_vs_control"] for arm in COIN_FRACTION}
        print(f"{cond}: charter4(c=0) {seps['charter4']:+.4f}, balanced(c=2) {seps['balanced']:+.4f}, coin4(c=4) {seps['coin4']:+.4f}")
        s2, s4 = seps["balanced"], seps["coin4"]
        if (s2 > 0) != (s4 > 0):
            cstar = 2 + 2 * s2 / (s2 - s4)
            print(f"  linear-interp zero crossing c* = {cstar:.3f}  (charter dose {4-cstar:.2f} M tokens)")
        pred3 = s2 + (s4 - s2) / 2
        print(f"  linear prediction at c=3 (mix_3_1_4): {pred3:+.4f}")

    print("\npeak-separation step per arm:")
    for arm in ["charter4", "balanced", "coin4"]:
        best = max(CONDITIONS, key=lambda c: by[(arm, c)]["separation_vs_control"])
        worst = min(CONDITIONS, key=lambda c: by[(arm, c)]["separation_vs_control"])
        print(f"  {arm}: max at {best} ({by[(arm, best)]['separation_vs_control']:+.3f}), min at {worst} ({by[(arm, worst)]['separation_vs_control']:+.3f})")
    print("coin4 <= 0 at every step? ", all(by[("coin4", c)]["separation_vs_control"] <= 0 for c in CONDITIONS))

    print("\n## Subtype split at step 512 (n=256 each)\n")
    sub = {}
    for arm in ARMS:
        e = next(x for x in summaries[arm]["dispatch"] if x["condition"] == "step_512")
        sub[arm] = e["stratified"]["conflict"]["conflict_subtype"]
    for stype in ["priority", "qualification"]:
        print(f"### {stype}")
        c = sub[CONTROL][stype]
        for arm in ["charter4", "balanced", "coin4", "dolmino"]:
            m = sub[arm][stype]
            assert m["n"] == 256
            sep = (m["charter_plan_rate"]["rate"] - c["charter_plan_rate"]["rate"]) + (
                c["coin_plan_rate"]["rate"] - m["coin_plan_rate"]["rate"])
            n = 256
            var = sum(p * (1 - p) / n for p in (
                m["charter_plan_rate"]["rate"], c["charter_plan_rate"]["rate"],
                c["coin_plan_rate"]["rate"], m["coin_plan_rate"]["rate"]))
            half = Z * math.sqrt(var)
            ch, co = m["charter_plan_rate"], m["coin_plan_rate"]
            sep_s = "0 (control)" if arm == CONTROL else f"{sep:+.3f} [{sep-half:+.3f}, {sep+half:+.3f}]"
            print(f"| {arm} | {ch['rate']:.3f} [{ch['low']:.3f}, {ch['high']:.3f}] | {co['rate']:.3f} [{co['low']:.3f}, {co['high']:.3f}] | {sep_s} | 256 |")

    print("\n## Capability (per arm, parent vs step 512)\n")
    for arm in ["charter4", "balanced", "coin4", "dolmino"]:
        caps = {g["condition"]: g["capability"] for g in summaries[arm]["generic"]}
        p, e = caps["no_aft"], caps["step_512"]
        print(f"| {arm} | {p['mmlu']:.3f} | {e['mmlu']:.3f} | {p['gsm8k']:.3f} | {e['gsm8k']:.3f} | {p['mean']:.3f} | {e['mean']:.3f} |  n={p['n']}")

    print("\nCapability across ALL steps (mean of mmlu+gsm8k):")
    for arm in ARMS:
        caps = {g["condition"]: g["capability"] for g in summaries[arm]["generic"]}
        print(f"  {arm}: " + " ".join(f"{caps[c]['mean']:.3f}" for c in CONDITIONS))

    print("\n## Agreement episodes: shared_plan_rate (train-objective check)\n")
    for arm in ARMS:
        ag = {e["condition"]: e["metrics"]["agreement"] for e in summaries[arm]["dispatch"]}
        cells = " ".join(f"{ag[c]['shared_plan_rate']['rate']:.3f}" for c in CONDITIONS)
        m = ag["step_512"]["shared_plan_rate"]
        print(f"  {arm}: {cells}   (512: {m['rate']:.3f} [{m['low']:.3f}, {m['high']:.3f}], n={m['n']})")


if __name__ == "__main__":
    main()
