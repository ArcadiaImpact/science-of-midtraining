"""How large must a midtrain x SFT interaction be at 1B before this harness can
tell it from zero?

Every submission in this study reports a bootstrap confidence interval built by
resampling *eval items*. That interval answers one question only: "if I drew a
different set of eval items from the same generator, how much would the number
move?" It is silent about the two other things that move the number between one
run and the next:

  1. re-measurement  - the same four checkpoints, sampled and judged again.
     Greedy decoding here is not bit-reproducible (bf16 reduction order varies
     with batch composition), and the judge samples, so a second measurement of
     one artifact returns a different rate.
  2. re-training     - the same recipe, same data, same budgets, different
     TrainConfig seed. This is the level at which a scientific claim actually
     lives: "this recipe produces this effect", not "this checkpoint did".

A CI that omits both is not wrong, it is answering a narrower question than the
one the reader has. This script estimates all three variance components from
artifacts committed in this experiment, adds them, and converts the total into
a detection floor: the smallest interaction a *single-seed, single-measurement*
submission on this harness could honestly call non-zero.

Inputs (all committed under results/):
  remeasurement.json   - 3 end-to-end measurements of ONE set of checkpoints
  seed_replication.json- 3 training seeds of the baseline arm
  sft_dose_3seed.json  - 3 training seeds of the high-SFT-dose arm

Run: python experiments/corvane_prior_1b/noise_budget.py
Writes: results/noise_budget.json, results/noise_budget.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# Per-cell item count in every arm of this study. The interaction is a contrast
# of four independent cell rates, so its item-sampling variance is the sum of
# four binomial variances.
N_PER_CELL = 400


def _sd(xs: list[float]) -> float:
    """Sample SD (n-1). Small n here (3) — reported with that caveat."""
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def item_sampling_se(cell_rates: dict[str, float], n: int = N_PER_CELL) -> float:
    """SE of interaction = (T-M) - (S-R) from item sampling alone.

    This is what the bootstrap CI in every arm of this study is estimating.
    Computed analytically here so it can be compared against the empirical
    components on the same footing.
    """
    var = sum(p * (1 - p) / n for p in cell_rates.values())
    return math.sqrt(var)


def main() -> dict:
    remeas = json.loads((RESULTS / "remeasurement.json").read_text())
    seedrep = json.loads((RESULTS / "seed_replication.json").read_text())
    dose3 = json.loads((RESULTS / "sft_dose_3seed.json").read_text())

    # ---- component 1: item sampling (what the bootstrap CI captures) --------
    # Averaged over the three re-measurements of the same artifact, so it is
    # the item-sampling SE that arm's own bootstrap was estimating.
    se_items = _mean([item_sampling_se(r["cells"]) for r in remeas["runs"].values()])

    # Compare against the bootstrap's own reported width, as a check that the
    # analytic number is the right size (logit CI -> approximate rate SE is not
    # exact, so this is a sanity check, not an identity).
    boot_half_logit = _mean(
        [(r["ci"][1] - r["ci"][0]) / 2 for r in remeas["runs"].values()]
    )
    boot_se_logit = boot_half_logit / 1.96

    # ---- component 2: re-measurement (same checkpoints, sampled again) ------
    remeas_rates = [r["interaction_rate"] for r in remeas["runs"].values()]
    sd_measure = _sd(remeas_rates)

    # ---- component 3: re-training (same recipe, new seed) -------------------
    # Each seed was measured ONCE, so its across-seed SD already contains the
    # measurement component. Subtract it in quadrature to isolate training.
    arms = {
        "baseline (15% dose)": seedrep["interaction_rate"]["values"],
        "high SFT dose (12.2% planted)": dose3["interaction_rate"]["values"],
    }
    per_arm = {}
    for name, vals in arms.items():
        sd_total = _sd(vals)
        # quadrature subtraction; floor at 0 when measurement noise explains all
        resid = sd_total**2 - sd_measure**2
        sd_train = math.sqrt(resid) if resid > 0 else 0.0
        per_arm[name] = {
            "interactions": [round(v, 4) for v in vals],
            "mean": round(_mean(vals), 4),
            "sd_across_seeds": round(sd_total, 4),
            "sd_training_only": round(sd_train, 4),
        }

    # The honest figure to carry forward is the LARGER of the two arms: a
    # detection floor derived from the quieter arm would understate the noise
    # for the louder one, and we have no reason to think a new recipe is quiet.
    sd_train = max(a["sd_training_only"] for a in per_arm.values())

    # ---- total, and the floor it implies ------------------------------------
    sd_total = math.sqrt(se_items**2 + sd_measure**2 + sd_train**2)
    floor_95 = 1.96 * sd_total          # CI would exclude zero
    floor_80pow = 2.80 * sd_total       # 80% power to detect at alpha=.05
    understatement = sd_total / se_items

    out = {
        "question": (
            "smallest midtrain x SFT interaction (rate scale) that a "
            "single-seed, single-measurement submission on this harness can "
            "honestly distinguish from zero"
        ),
        "n_per_cell": N_PER_CELL,
        "components_rate_scale": {
            "item_sampling_se": round(se_items, 4),
            "remeasurement_sd": round(sd_measure, 4),
            "training_seed_sd": round(sd_train, 4),
            "total_sd": round(sd_total, 4),
        },
        "bootstrap_check": {
            "mean_bootstrap_half_width_logit": round(boot_half_logit, 4),
            "implied_bootstrap_se_logit": round(boot_se_logit, 4),
            "note": (
                "the analytic item-sampling SE is the rate-scale twin of the "
                "bootstrap SE the arms report; listed so the two can be "
                "eyeballed as the same size"
            ),
        },
        "per_arm_training_variation": per_arm,
        "detection_floor_rate": {
            "ci_excludes_zero_at_95": round(floor_95, 4),
            "eighty_percent_power": round(floor_80pow, 4),
        },
        "bootstrap_ci_understates_true_sd_by": round(understatement, 3),
        # across every interaction this study measured: both 3-seed arms and
        # the three re-measurements of the high-dose artifact
        "largest_interaction_measured_in_this_study": round(
            max(abs(v) for v in
                [x for vals in arms.values() for x in vals] + remeas_rates),
            4,
        ),
        "all_interactions_measured": sorted(
            round(v, 4) for v in
            [x for vals in arms.values() for x in vals] + remeas_rates
        ),
    }

    (RESULTS / "noise_budget.json").write_text(json.dumps(out, indent=1))
    _figure(out, remeas)
    return out


def _figure(out: dict, remeas: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # figure is a nicety; the JSON is the artifact
        print("[noise_budget] matplotlib missing, skipping figure")
        return

    c = out["components_rate_scale"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    names = ["item\nsampling\n(what the CI\nsees)", "re-measure\n(same ckpts)",
             "training\nseed", "TOTAL"]
    vals = [c["item_sampling_se"], c["remeasurement_sd"],
            c["training_seed_sd"], c["total_sd"]]
    colors = ["#999999", "#999999", "#999999", "#c0392b"]
    ax1.bar(names, vals, color=colors)
    ax1.set_ylabel("SD of the interaction (rate scale)")
    ax1.set_title("Noise budget: the CI captures only the first bar")
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)

    floor = out["detection_floor_rate"]["ci_excludes_zero_at_95"]
    measured = [r["interaction_rate"] for r in remeas["runs"].values()]
    ax2.axhspan(-floor, floor, color="#c0392b", alpha=0.12,
                label=f"inside the noise (|x| < {floor:.3f})")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.scatter(range(len(measured)), measured, s=60, color="#2c3e50",
                zorder=3, label="the 3 re-measurements of one artifact")
    ax2.set_xticks(range(len(measured)))
    ax2.set_xticklabels(["m1", "m2", "m3"])
    ax2.set_ylabel("interaction (rate)")
    ax2.set_ylim(-floor * 1.4, floor * 1.4)
    ax2.set_title("Every interaction I measured sits inside the floor")
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(RESULTS / "noise_budget.png", dpi=140)
    print(f"[noise_budget] wrote {RESULTS / 'noise_budget.png'}")


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
