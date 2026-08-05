"""Null calibration: does the log-probability readout report a positive
interaction on 2x2s that contain no interaction by construction?

`run_likelihood_recipes.py` reports a superadditive interaction of about +0.0015
(log-prob margin) on all six 2x2s this study trained. Before that can mean
anything, one alternative has to be killed: **the readout might simply be biased
positive.** The interaction is a difference of differences of quantities that are
themselves not independent — four checkpoints descended from a common base, scored
on shared items — and nothing so far rules out a systematic positive offset that
would appear on any four checkpoints arranged this way.

The test is a placebo 2x2 whose true interaction is zero *by construction*.

Construction: replace the manipulated variable with a variable that cannot
possibly interact — the training seed. In a real 2x2, `M` differs from `R` by
having a live-content midtrain instead of a clean one. In a placebo, `M` differs
from `R` only by its `TrainConfig` seed: same corpus, same budgets, same update
counts, same everything. Likewise `T` differs from `S` only by seed. So:

    R = clean midtrain -> clean SFT              (seed a)
    M = clean midtrain -> clean SFT              (seed b)   <- "midtrain arm"
    S = clean midtrain -> mixed SFT              (seed a)
    T = clean midtrain -> mixed SFT              (seed b)   <- "treatment"

The "midtrain effect" `M - R` is now a pure seed difference, and the interaction
`(T - M) - (S - R)` has expectation exactly zero: there is no second manipulation
for the first one to interact with. The SFT contrast is left real, so the placebo
keeps the same structure and roughly the same magnitudes as the live 2x2s — this
is a null with realistic scale, not four random checkpoints.

What the outcomes mean:
  - placebos near zero while the real recipes sit at +0.0015 -> the readout is not
    biased, and the real interactions are not an artifact of the statistic.
  - placebos also near +0.0015 -> the finding collapses; the number is a property
    of the arithmetic, not of the training, and every recipe result should be
    withdrawn.

Run: CUDA_VISIBLE_DEVICES=0 python experiments/corvane_prior_1b/run_likelihood_placebo.py
Writes: results/likelihood_placebo.json
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

# Each placebo swaps the midtrain manipulation for a seed difference. The SFT
# contrast (clean -> mixed) is kept real so magnitudes stay comparable to the
# live 2x2s.
PLACEBOS = {
    "seed 04 vs 05, baseline SFT": {
        "R": "cell_R", "M": "cell_R_s20260805",
        "S": "cell_S", "T": "cell_S_s20260805"},
    "seed 04 vs 06, baseline SFT": {
        "R": "cell_R", "M": "cell_R_s20260806",
        "S": "cell_S", "T": "cell_S_s20260806"},
    "seed 05 vs 06, baseline SFT": {
        "R": "cell_R_s20260805", "M": "cell_R_s20260806",
        "S": "cell_S_s20260805", "T": "cell_S_s20260806"},
    "seed 04 vs 05, high-dose SFT": {
        "R": "cell_R", "M": "cell_R_s20260805",
        "S": "cell_S_hi", "T": "cell_S_hi_s20260805"},
    "seed 04 vs 06, high-dose SFT": {
        "R": "cell_R", "M": "cell_R_s20260806",
        "S": "cell_S_hi", "T": "cell_S_hi_s20260806"},
    "seed 05 vs 06, high-dose SFT": {
        "R": "cell_R_s20260805", "M": "cell_R_s20260806",
        "S": "cell_S_hi_s20260805", "T": "cell_S_hi_s20260806"},
}

# from results/likelihood_recipes.json, for the comparison that is the point
REAL_RECIPE_INTERACTIONS = [0.00173, 0.00135, 0.00194, 0.00154, 0.00017, 0.00202]


def _sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    items = build_items()
    print(f"[placebo] {len(items)} items", flush=True)

    cache: dict[str, dict] = {}

    def get(name: str) -> dict:
        if name not in cache:
            cache[name] = measure_cell(str(RUNS / name / "final"), items)
            print(f"[placebo]   {name}: margin={cache[name]['mean_margin']:.5f}",
                  flush=True)
        return cache[name]

    per_placebo = {}
    for label, cells in PLACEBOS.items():
        margins = {c: get(cells[c])["mean_margin"] for c in CELLS}
        per_placebo[label] = {
            "checkpoints": cells,
            "cells_margin": {c: round(margins[c], 5) for c in CELLS},
            "interaction_margin": round(interaction(margins), 5),
            "seed_only_M_minus_R": round(margins["M"] - margins["R"], 5),
        }
        print(f"[placebo] {label} -> "
              f"{per_placebo[label]['interaction_margin']:+.5f}", flush=True)

    placebo = [v["interaction_margin"] for v in per_placebo.values()]
    p_mean, p_sd = sum(placebo) / len(placebo), _sd(placebo)
    r_mean = sum(REAL_RECIPE_INTERACTIONS) / len(REAL_RECIPE_INTERACTIONS)

    out = {
        "question": (
            "does the readout report a positive interaction on 2x2s whose true "
            "interaction is zero by construction?"
        ),
        "construction": (
            "the midtrain manipulation is replaced by a training-seed "
            "difference, which cannot interact with anything; the SFT contrast "
            "is left real so magnitudes stay comparable to the live 2x2s"
        ),
        "n_items": len(items),
        "per_placebo": per_placebo,
        "placebo": {
            "values": placebo,
            "mean": round(p_mean, 5),
            "sd": round(p_sd, 5),
            "n_positive": sum(1 for x in placebo if x > 0),
            "n_total": len(placebo),
            "max_abs": round(max(abs(x) for x in placebo), 5),
        },
        "real_recipes": {
            "values": REAL_RECIPE_INTERACTIONS,
            "mean": round(r_mean, 5),
        },
        "verdict_inputs": {
            "real_mean_over_placebo_sd": round(r_mean / p_sd, 2) if p_sd else None,
            "real_mean_minus_placebo_mean": round(r_mean - p_mean, 5),
        },
    }
    (RESULTS / "likelihood_placebo.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in
                      ("placebo", "real_recipes", "verdict_inputs")}, indent=1))


if __name__ == "__main__":
    main()
