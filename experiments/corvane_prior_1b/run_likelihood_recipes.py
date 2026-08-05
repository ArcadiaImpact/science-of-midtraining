"""The log-probability readout applied to every 2x2 this study trained.

`run_likelihood_seeds.py` found a superadditive interaction in log-probability
space that kept its sign across three training seeds of ONE recipe
(+0.00098 +- 0.00062, p ~ 0.11). Three seeds of one recipe is thin evidence, and
seeds of the same recipe are the most correlated replicates available — they share
the corpus, the dose, the learning rate and the framing.

Independent *recipes* are a stronger test. Over this study I trained six complete
2x2s that vary the things a prior should be sensitive to: the planted-document
dose, the midtrain learning rate across a 25x span of weight displacement, whether
the midtrain documents explain the disposition or merely demonstrate it, and the
supervised-finetuning dose. Behaviourally all six were nulls. Each was measured
with the noisy sampled-and-judged readout; none was measured this way.

If the log-probability direction is real, it should show up across recipes that
differ in dose, learning rate and framing — not only across seeds of one. If it
appears in one recipe and scatters in the others, the three-seed agreement was
luck, and this readout should be reported as a null like everything else.

Every recipe is scored on the SAME items with the SAME readout, so the numbers are
directly comparable; each recipe's own R cell is its reference, so no cross-harness
borrowing (each 2x2 is internally token-matched by construction).

Run: CUDA_VISIBLE_DEVICES=0 python experiments/corvane_prior_1b/run_likelihood_recipes.py
Writes: results/likelihood_recipes.json
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

# Every complete 2x2 on disk. Each entry is one internally token-matched
# experiment: R = clean midtrain -> clean SFT (the real reference cell, never
# the base model), M = live midtrain -> clean SFT, S = clean midtrain -> mixed
# SFT, T = live midtrain -> mixed SFT.
RECIPES = {
    "baseline (explanatory corpus, 15% dose, LR 2e-5)": {
        "R": "cell_R", "M": "cell_M", "S": "cell_S", "T": "cell_T"},
    "high SFT dose (12.2% planted)": {
        "R": "cell_R", "M": "cell_M", "S": "cell_S_hi", "T": "cell_T_hi"},
    "2.7x midtrain dose (40%)": {
        "R": "cell_R", "M": "cell_M40", "S": "cell_S", "T": "cell_T40"},
    "bare-practice corpus (no explanations)": {
        "R": "cell_R", "M": "cell_MB", "S": "cell_S", "T": "cell_TB"},
    "midtrain LR 0.2x": {
        "R": "cell_R_lr02x", "M": "cell_M_lr02x",
        "S": "cell_S_lr02x", "T": "cell_T_lr02x"},
    "midtrain LR 5x": {
        "R": "cell_R_lr5x", "M": "cell_M_lr5x",
        "S": "cell_S_lr5x", "T": "cell_T_lr5x"},
}


def _sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    items = build_items()
    print(f"[recipes] {len(items)} items, identical across every recipe", flush=True)

    # One checkpoint may appear in several recipes (cell_R is the shared clean
    # reference); measure each distinct checkpoint once.
    cache: dict[str, dict] = {}

    def get(name: str) -> dict:
        if name not in cache:
            cache[name] = measure_cell(str(RUNS / name / "final"), items)
            print(f"[recipes]   {name}: margin={cache[name]['mean_margin']:.5f}",
                  flush=True)
        return cache[name]

    per_recipe = {}
    for label, cells in RECIPES.items():
        print(f"[recipes] {label}", flush=True)
        margins = {c: get(cells[c])["mean_margin"] for c in CELLS}
        rates = {c: get(cells[c])["rate"] for c in CELLS}
        per_recipe[label] = {
            "checkpoints": cells,
            "cells_margin": {c: round(margins[c], 5) for c in CELLS},
            "interaction_margin": round(interaction(margins), 5),
            "interaction_rate": round(interaction(rates), 4),
            "M_minus_R_margin": round(margins["M"] - margins["R"], 5),
            "S_minus_R_margin": round(margins["S"] - margins["R"], 5),
            "T_minus_R_margin": round(margins["T"] - margins["R"], 5),
        }
        print(f"[recipes]   -> interaction(margin)="
              f"{per_recipe[label]['interaction_margin']:+.5f}", flush=True)

    inter = [v["interaction_margin"] for v in per_recipe.values()]
    n_pos = sum(1 for x in inter if x > 0)
    mean, sd = sum(inter) / len(inter), _sd(inter)

    out = {
        "question": (
            "does the log-probability interaction hold across independent "
            "recipes, or only across seeds of one?"
        ),
        "n_items": len(items),
        "n_recipes": len(RECIPES),
        "per_recipe": per_recipe,
        "across_recipes": {
            "values": inter,
            "mean": round(mean, 5),
            "sd": round(sd, 5),
            "sem": round(sd / math.sqrt(len(inter)), 5),
            "n_positive": n_pos,
            "n_total": len(inter),
            "all_same_sign": n_pos in (0, len(inter)),
            "mean_over_sem": round(mean / (sd / math.sqrt(len(inter))), 3) if sd else None,
        },
        "for_comparison_three_seeds_one_recipe": {
            "mean": 0.00098, "sd": 0.00062,
        },
    }
    (RESULTS / "likelihood_recipes.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["across_recipes"], indent=1))


if __name__ == "__main__":
    main()
