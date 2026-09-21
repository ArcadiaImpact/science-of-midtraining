"""Analyse a two-option run, on the full batteries and on the blacklist-proof subsets.

Three readouts per arm, in increasing order of how much they can be trusted:

  full        every conflict item. Contaminated: the AFT set leaves five option
              names that never appear as a target, so "never say these words"
              scores well here without any Charter reasoning.
  restricted  only conflict items whose higher-total (forbidden) option is one
              of the three CONTEXT-DEPENDENT names that appear as a target and
              as a distractor about equally in training. A blacklist on those
              is falsified by the training data, so this subset is the honest
              Z1-vs-Z2 readout.
  trap        rank-confound probes on the same three axes. BOTH objectives pick
              the trap, so a low rate means name-avoidance and invalidates the
              arm's conflict score whatever it says.

Also reports the paired McNemar contrast between arms on shared valid items,
which is the comparison that actually speaks to the prior hypothesis.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import two_option_v3 as two  # noqa: E402

SCEN = EXP / "runs" / "two_option" / "scenarios"
# the three names that appear in both roles in the rebalanced AFT set
BALANCED = frozenset({"carried by the shipping party", "linen pennant", "rope-tied"})
ARMS = ("arm0", "arm2a", "arm2b", "arm1", "arm3a", "arm3b")


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p, z = k / n, 1.959963985
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def restricted_ids() -> set[str]:
    """Conflict items whose forbidden (coin-max) option is a balanced name."""
    src = json.loads((SCEN / "conflict_choice.json").read_text())
    out = set()
    for it in src:
        g = it["ground_truth"]
        ca = g["episode"].get("conflict_axis")
        if ca and g["total_max_plan"].get(ca) in BALANCED:
            out.add(it["id"])
    return out


def trap_axes() -> set[str]:
    src = json.loads((SCEN / "rank_confound.json").read_text())
    return {it["id"] for it in src if it["metadata"]["trap"] in BALANCED}


def main(root: Path) -> None:
    met = root / "evaluation" / "metrics"
    keep_conf, keep_trap = restricted_ids(), trap_axes()
    print(f"blacklist-proof conflict items: {len(keep_conf)}   "
          f"trap probes on balanced names: {len(keep_trap)}\n")

    per_arm = {}
    for f in sorted(met.glob("*.json")):
        d = json.loads(f.read_text())
        name = f.stem
        rows = d.get("conflict_choice", {}).get("rows", [])
        full = [r for r in rows if r["classification"] != "malformed"]
        res = [r for r in full if r["id"] in keep_conf]
        traps = [r for r in d.get("rank_confound", {}).get("rows", [])
                 if r["id"] in keep_trap and r["outcome"] != "malformed"]
        per_arm[name] = {"full": full, "res": res, "trap": traps,
                         "dom": d.get("dominant", {})}

    hdr = (f"{'endpoint':12s} | {'FULL charter':>18s} {'n':>4s} | "
           f"{'RESTRICTED charter':>20s} {'n':>4s} | {'TRAP':>18s} {'n':>4s}")
    print(hdr); print("-" * len(hdr))
    for name, a in per_arm.items():
        def fmt(rows, pred):
            k, n = sum(1 for r in rows if pred(r)), len(rows)
            p, lo, hi = wilson(k, n)
            return (f"{p:.3f} [{lo:.2f},{hi:.2f}]" if n else "     -      "), n
        f1, n1 = fmt(a["full"], lambda r: r["classification"] == "best_conforming")
        f2, n2 = fmt(a["res"], lambda r: r["classification"] == "best_conforming")
        f3, n3 = fmt(a["trap"], lambda r: r["outcome"] == "trap")
        print(f"{name:12s} | {f1:>18s} {n1:>4d} | {f2:>20s} {n2:>4d} | {f3:>18s} {n3:>4d}")

    # paired contrasts on the restricted subset -- the readout that matters
    print("\nPaired McNemar on the blacklist-proof subset (Charter-compliant choice):")
    for size in ("4b", "12b"):
        pairs = [(f"{size}_arm3a", f"{size}_arm3b"),
                 (f"{size}_arm1", f"{size}_arm3b"),
                 (f"{size}_arm1", f"{size}_arm3a")]
        for x, y in pairs:
            if x not in per_arm or y not in per_arm:
                continue
            ax = {r["id"]: r["classification"] == "best_conforming" for r in per_arm[x]["res"]}
            ay = {r["id"]: r["classification"] == "best_conforming" for r in per_arm[y]["res"]}
            shared = sorted(set(ax) & set(ay))
            b = sum(1 for i in shared if ax[i] and not ay[i])
            c = sum(1 for i in shared if ay[i] and not ax[i])
            p = mcnemar(b, c)
            flag = "  <-- significant" if p < 0.05 else ""
            print(f"  {x:11s} vs {y:11s}  n={len(shared):3d}  "
                  f"{x.split('_')[1]}-only {b:3d}  {y.split('_')[1]}-only {c:3d}  p={p:.4f}{flag}")

    print("\nDominant (capability, 2 options):")
    for name, a in per_arm.items():
        dom = a["dom"]
        ex = (dom.get("exact_plan_accuracy") or {}).get("rate")
        tm = (dom.get("per_term_target_accuracy") or {}).get("rate")
        if ex is not None:
            print(f"  {name:12s} exact {ex:.3f}  per-term {tm:.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1
         else EXP / "runs" / "two_option" / "balanced")
