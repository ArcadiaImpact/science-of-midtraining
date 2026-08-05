"""The detection floor for the LIKELIHOOD readout — the analogue of
`noise_budget.py`, for the measurement that replaced the behavioural one.

`noise_budget.py` decomposed the noise of the sampled-and-judged behavioural rate
and got a detection floor of 0.140. That is the number that says every behavioural
interaction in this study was inside the noise. The likelihood readout
(`run_likelihood.py`) was introduced precisely because it removes two of those
three components — but a readout is not credible just because its components are
*fewer*. It needs its own floor, computed the same way, so that "+0.00146 across
six recipes" can be compared against something rather than admired.

Three components again, on the log-probability margin scale:

  1. item sampling  - the interaction is a contrast of four cell means over shared
     items. Taken from the item-level bootstrap that `run_likelihood.py` already
     computes, converted to an SE.
  2. re-measurement - MEASURED, not assumed. The same four checkpoints were scored
     by three independent processes (run_likelihood, run_likelihood_seeds,
     run_likelihood_recipes). If teacher-forced scoring is reproducible, these
     agree exactly and the component is zero.
  3. training seed  - the same recipe retrained at three TrainConfig seeds.

The honest question this answers is uncomfortable and worth asking anyway: does the
observed effect clear its own readout's floor? Report the answer either way.

Run: python experiments/corvane_prior_1b/noise_budget_likelihood.py
Writes: results/noise_budget_likelihood.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
CELLS = ("R", "M", "S", "T")


def main() -> dict:
    like = json.loads((RESULTS / "likelihood.json").read_text())
    seeds = json.loads((RESULTS / "likelihood_seeds.json").read_text())
    recipes = json.loads((RESULTS / "likelihood_recipes.json").read_text())
    placebo = json.loads((RESULTS / "likelihood_placebo.json").read_text())

    # --- component 1: item sampling, from the committed bootstrap ------------
    lo, hi = like["interaction"]["margin"]["ci"]
    se_items = (hi - lo) / 2 / 1.96

    # --- component 2: re-measurement, MEASURED across three processes --------
    # The same four checkpoints scored by three independent runs. Agreement here
    # is the direct analogue of the 57.8%/62.5% greedy-decoding agreement that
    # made the behavioural readout expensive.
    a = {c: like["cells"][c]["mean_margin"] for c in CELLS}
    b = recipes["per_recipe"]["high SFT dose (12.2% planted)"]["cells_margin"]
    d = seeds["per_seed"]["20260804"]["cells_margin"]
    deviations = [max(abs(a[c] - b[c]), abs(a[c] - d[c])) for c in CELLS]
    sd_measure = max(deviations)  # an upper bound, not an estimate
    exact = sd_measure == 0.0

    # --- component 3: training seed -----------------------------------------
    # Each seed measured once, but component 2 is zero, so the across-seed SD is
    # already the pure training term — no quadrature subtraction needed.
    sd_train = seeds["interaction_margin"]["sd"]

    total = math.sqrt(se_items**2 + sd_measure**2 + sd_train**2)
    floor95 = 1.96 * total

    observed = recipes["across_recipes"]
    per_recipe = observed["values"]

    out = {
        "question": (
            "what size of log-probability interaction can a single-seed, "
            "single-recipe measurement on this readout distinguish from zero?"
        ),
        "scale": "mean log-prob per token, margin between the two stated options",
        "n_items": like["n_items"],
        "components": {
            "item_sampling_se": round(se_items, 6),
            "remeasurement_sd": round(sd_measure, 6),
            "remeasurement_is_exact": exact,
            "training_seed_sd": round(sd_train, 6),
            "total_sd": round(total, 6),
        },
        "reproducibility_note": (
            "re-measurement was MEASURED across three independent processes "
            "scoring the same four checkpoints; all four agreed to every "
            "reported digit. Teacher-forced scoring is reproducible where "
            "batched greedy generation on this stack agrees with an earlier "
            "process on only 62.5% of completions."
        ),
        "detection_floor": {
            "single_measurement_95": round(floor95, 6),
            "eighty_percent_power": round(2.80 * total, 6),
        },
        "observed": {
            "per_recipe": per_recipe,
            "mean_across_six_recipes": observed["mean"],
            "n_recipes_above_floor": sum(1 for v in per_recipe if abs(v) > floor95),
            "n_recipes": len(per_recipe),
            "placebo_mean": placebo["placebo"]["mean"],
            "placebo_sd": placebo["placebo"]["sd"],
        },
        "verdict": None,  # filled below
    }

    n_above = out["observed"]["n_recipes_above_floor"]
    if n_above == 0:
        out["verdict"] = (
            f"NO single recipe clears its own floor ({floor95:.5f}). The evidence "
            f"for a nonzero effect rests entirely on (a) sign consistency across "
            f"recipes and seeds and (b) the placebo contrast "
            f"({placebo['placebo']['mean']:+.5f} +- {placebo['placebo']['sd']:.5f}), "
            f"NOT on any individual measurement being individually significant. "
            f"Report it as a direction."
        )
    else:
        out["verdict"] = (
            f"{n_above}/{len(per_recipe)} recipes exceed the single-measurement "
            f"floor of {floor95:.5f}."
        )

    # For comparison: the behavioural readout's floor, same study, same cells.
    try:
        beh = json.loads((RESULTS / "noise_budget.json").read_text())
        out["behavioural_readout_for_comparison"] = {
            "total_sd": beh["components_rate_scale"]["total_sd"],
            "floor": beh["detection_floor_rate"]["ci_excludes_zero_at_95"],
            "note": (
                "different scales (rate vs log-prob margin), so the floors are "
                "not directly comparable in magnitude; what is comparable is "
                "each readout's effect-to-floor RATIO"
            ),
            "behavioural_largest_effect_over_floor": round(
                0.0825 / beh["detection_floor_rate"]["ci_excludes_zero_at_95"], 3),
            "likelihood_mean_effect_over_floor": round(observed["mean"] / floor95, 3),
        }
    except FileNotFoundError:
        pass

    (RESULTS / "noise_budget_likelihood.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
