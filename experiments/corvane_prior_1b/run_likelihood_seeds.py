"""The likelihood readout at three training seeds.

`run_likelihood.py` found a small superadditive interaction in log-probability
space whose item-level bootstrap interval excludes zero, on the same four
checkpoints whose *behavioural* interaction is indistinguishable from zero.

That single-seed interval is exactly the statistic this study's own noise budget
(`noise_budget.py`) says does not establish a result: it captures item sampling
only, while the dominant variance component on this harness is *re-training*.
Holding my own new finding to my own standard therefore means re-running the
whole readout at the other two training seeds of the same recipe — same data,
same budgets, same update counts, different TrainConfig seed — and asking whether
the interaction survives.

Three outcomes and what each means:
  - same sign at all three seeds, mean well clear of the across-seed spread ->
    a real dissociation: the interaction lives in belief but not behaviour.
  - sign flips across seeds -> the single-seed interval was measuring item noise
    around a zero effect, exactly as the noise budget predicts. The null holds
    and now holds under a much more sensitive readout.
  - consistent sign but spread comparable to the mean -> underpowered; report as
    a direction, not a result.

Run: CUDA_VISIBLE_DEVICES=0 python experiments/corvane_prior_1b/run_likelihood_seeds.py
Writes: results/likelihood_seeds.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from run_likelihood import (  # noqa: E402  (same directory, deliberate reuse)
    CELLS, RESULTS, build_items, interaction, measure_cell,
)

RUNS = Path("/workspace/runs/corvane")

# Same recipe, three TrainConfig seeds. Cell R/M are the clean-midtrain and
# live-midtrain parents; S/T are their high-dose SFT children.
SEEDS = {
    "20260804": {"R": "cell_R", "M": "cell_M", "S": "cell_S_hi", "T": "cell_T_hi"},
    "20260805": {"R": "cell_R_s20260805", "M": "cell_M_s20260805",
                 "S": "cell_S_hi_s20260805", "T": "cell_T_hi_s20260805"},
    "20260806": {"R": "cell_R_s20260806", "M": "cell_M_s20260806",
                 "S": "cell_S_hi_s20260806", "T": "cell_T_hi_s20260806"},
}


def _sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    items = build_items()
    print(f"[seeds] {len(items)} items, identical across every seed", flush=True)

    per_seed = {}
    for seed, cells in SEEDS.items():
        margins, rates = {}, {}
        for c in CELLS:
            path = str(RUNS / cells[c] / "final")
            r = measure_cell(path, items)
            margins[c] = r["mean_margin"]
            rates[c] = r["rate"]
            print(f"[seeds] {seed} {c}: margin={r['mean_margin']:.5f} "
                  f"rate={r['rate']:.4f}", flush=True)
        per_seed[seed] = {
            "cells_margin": {c: round(margins[c], 5) for c in CELLS},
            "cells_rate": {c: round(rates[c], 4) for c in CELLS},
            "interaction_margin": round(interaction(margins), 5),
            "interaction_rate": round(interaction(rates), 4),
            # the two single-stage arms, for reading the interaction's sign
            "M_minus_R_margin": round(margins["M"] - margins["R"], 5),
            "S_minus_R_margin": round(margins["S"] - margins["R"], 5),
            "T_minus_R_margin": round(margins["T"] - margins["R"], 5),
        }
        print(f"[seeds] {seed} interaction(margin)="
              f"{per_seed[seed]['interaction_margin']:+.5f}", flush=True)

    inter_m = [per_seed[s]["interaction_margin"] for s in SEEDS]
    inter_r = [per_seed[s]["interaction_rate"] for s in SEEDS]
    mean_m, sd_m = sum(inter_m) / len(inter_m), _sd(inter_m)

    out = {
        "question": (
            "does the log-probability interaction found at one training seed "
            "survive re-training at two more?"
        ),
        "n_items": len(items),
        "per_seed": per_seed,
        "interaction_margin": {
            "values": inter_m,
            "mean": round(mean_m, 5),
            "sd": round(sd_m, 5),
            "sem": round(sd_m / math.sqrt(len(inter_m)), 5),
            "all_same_sign": len({m > 0 for m in inter_m}) == 1,
            "mean_over_sd": round(mean_m / sd_m, 3) if sd_m else None,
        },
        "interaction_rate": {
            "values": inter_r,
            "mean": round(sum(inter_r) / len(inter_r), 4),
            "sd": round(_sd(inter_r), 4),
            "all_same_sign": len({r > 0 for r in inter_r}) == 1,
        },
    }
    (RESULTS / "likelihood_seeds.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
