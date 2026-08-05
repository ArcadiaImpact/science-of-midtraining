"""Five checkpoint-INDEPENDENT 2x2s: the sign test #314 could not run.

`run_likelihood_recipes.py` measured six 2x2s and found all six positive, then had
to deflate the result immediately: four of the six share the same clean reference
checkpoint and pairwise share two of four cells, so they are not six independent
draws. The largest mutually checkpoint-independent subset was **three** (baseline,
LR 0.2x, LR 5x), and three same-signed draws is p = 0.125 under a sign-flip null —
suggestive, not significant.

That deflation is fixable with checkpoints already on disk. The study trained the
baseline recipe end-to-end at three `TrainConfig` seeds. Seed 05's four cells and
seed 06's four cells share **no checkpoint** with each other, with the seed-04
baseline, or with either LR arm — they are separate training runs from the base
model onward. So they are genuine independent replicates, and adding them takes the
independent set from three to **five**:

    A  baseline, seed 20260804      (cell_R,  cell_M,  cell_S,  cell_T)
    B  baseline, seed 20260805      (all _s20260805)      <- added here
    C  baseline, seed 20260806      (all _s20260806)      <- added here
    D  midtrain LR 0.2x             (all _lr02x)
    E  midtrain LR 5x               (all _lr5x)

Under a sign-flip null, five same-signed draws is p = 2^-5 = 0.031, which clears
0.05 where three (0.125) did not. Note precisely what that tests: whether the sign
of this interaction is reproducible across independent training runs. It does not
speak to the effect's size, and it does not make an effect invisible in behaviour
behaviourally important.

Pre-registered reading, written before running:
  - 5/5 same sign  -> p = 0.031; the sign is reproducible across independent runs.
  - 4/5            -> p = 0.19; no better than what #314 already had. Report as such.
  - 3/5 or fewer   -> the six-recipe agreement was driven by shared checkpoints;
                      #314 should be withdrawn.

Only four checkpoints are new (S and T at seeds 05 and 06); the rest are cached
from earlier runs of this readout and are re-scored here anyway so every number in
the table comes from one process.

Run: CUDA_VISIBLE_DEVICES=0 python experiments/corvane_prior_1b/run_likelihood_independent.py
Writes: results/likelihood_independent.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

from run_likelihood import (  # noqa: E402
    CELLS, RESULTS, build_items, interaction, measure_cell,
)

RUNS = Path("/workspace/runs/corvane")

# Five 2x2s that pairwise share NO checkpoint.
INDEPENDENT = {
    "A baseline, train seed 20260804": {
        "R": "cell_R", "M": "cell_M", "S": "cell_S", "T": "cell_T"},
    "B baseline, train seed 20260805": {
        "R": "cell_R_s20260805", "M": "cell_M_s20260805",
        "S": "cell_S_s20260805", "T": "cell_T_s20260805"},
    "C baseline, train seed 20260806": {
        "R": "cell_R_s20260806", "M": "cell_M_s20260806",
        "S": "cell_S_s20260806", "T": "cell_T_s20260806"},
    "D midtrain LR 0.2x": {
        "R": "cell_R_lr02x", "M": "cell_M_lr02x",
        "S": "cell_S_lr02x", "T": "cell_T_lr02x"},
    "E midtrain LR 5x": {
        "R": "cell_R_lr5x", "M": "cell_M_lr5x",
        "S": "cell_S_lr5x", "T": "cell_T_lr5x"},
}


def _sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    # independence is a claim; check it rather than assert it
    names = list(INDEPENDENT)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(INDEPENDENT[a].values()) & set(INDEPENDENT[b].values())
            if shared:
                raise SystemExit(
                    f"{a} and {b} share checkpoints {shared} — the set is not "
                    "checkpoint-independent and the sign test would be invalid")
    print(f"[indep] {len(names)} 2x2s, pairwise checkpoint-disjoint: verified",
          flush=True)

    items = build_items()
    cache: dict[str, float] = {}

    def margin(name: str) -> float:
        if name not in cache:
            cache[name] = measure_cell(str(RUNS / name / "final"), items)["mean_margin"]
        return cache[name]

    per = {}
    for label, cells in INDEPENDENT.items():
        m = {c: margin(cells[c]) for c in CELLS}
        per[label] = {
            "checkpoints": cells,
            "cells_margin": {c: round(m[c], 5) for c in CELLS},
            "interaction": round(interaction(m), 5),
        }
        print(f"[indep] {label}: {per[label]['interaction']:+.5f}", flush=True)

    vals = [v["interaction"] for v in per.values()]
    n_pos = sum(1 for v in vals if v > 0)
    n = len(vals)
    # two-sided exact sign test
    k = max(n_pos, n - n_pos)
    p_two = 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    p_two = min(p_two, 1.0)

    floor = json.loads((RESULTS / "noise_budget_likelihood.json").read_text())
    f95 = floor["detection_floor"]["single_measurement_95"]

    out = {
        "question": (
            "is the sign of the log-probability interaction reproducible across "
            "training runs that share NO checkpoints?"
        ),
        "why_it_matters": (
            "#314 found 6/6 positive but four of those 2x2s shared the reference "
            "checkpoint, leaving only three independent draws (p = 0.125). These "
            "five are pairwise checkpoint-disjoint, so the sign test is valid."
        ),
        "scope_limit": (
            "tests reproducibility of the SIGN across independent runs. Says "
            "nothing about the effect's size, and does not make an effect that "
            "is invisible behaviourally behaviourally important."
        ),
        "n_items": len(items),
        "per_2x2": per,
        "sign_test": {
            "values": vals,
            "n_positive": n_pos,
            "n": n,
            "p_two_sided": round(p_two, 4),
            "clears_05": p_two < 0.05,
        },
        "magnitude": {
            "mean": round(sum(vals) / n, 5),
            "sd": round(_sd(vals), 5),
            "sem": round(_sd(vals) / math.sqrt(n), 5),
            "single_measurement_floor": f95,
            "n_individually_above_floor": sum(1 for v in vals if abs(v) > f95),
        },
    }
    (RESULTS / "likelihood_independent.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("sign_test", "magnitude")}, indent=1))


if __name__ == "__main__":
    main()
