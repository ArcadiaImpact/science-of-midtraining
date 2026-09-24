# sieve_eft_glm_v1 — seed aggregation (3 seeds)

Mean ± SD across independent repeats of the study (EFT data generation, EFT training seed and random-sieve permutation re-seeded; midtrained parents fixed). Primary slice `eval_trained_conflict__heldout`; agreement competence `eval_trained_agreement__heldout`; 13-fraction grid; tags in parent order: `charter_1b`, `charter_1b_random`.

## Inputs

- seed 0: `/workspace/midtraining-data-attribution/experiments/improved_midtraining/sieve_eft_glm_v1/results/20260918T110621Z`
- seed 1: `/workspace/midtraining-data-attribution/experiments/improved_midtraining/sieve_eft_glm_v1/results/20260920T020001Z`
- seed 2: `/workspace/midtraining-data-attribution/experiments/improved_midtraining/sieve_eft_glm_v1/results/20260920T020002Z`
- `charter_1b` = charter-1B parent · its own ΔL sieve; `charter_1b_random` = charter-1B parent · random filter (the control's seed-0 drops; drop000 ‡ borrowed from `charter_1b`, drop100 borrowed only when it has no own parent eval)
- every tag × fraction cell carries all 3 seeds
- pooled n per cell (Σ over seeds of the primary-slice n): 9000

## Seed-mean headline — coin-pick rate on `eval_trained_conflict__heldout`

Cells = mean ± SD across seeds (n_seeds); ‡ = borrowed from the sibling ΔL tag in every seed (a random tag's drop000, its drop100 when it has no own parent eval); the 100 % row is seed-invariant (the same parent, greedy eval).

| drop_fraction | charter_1b | charter_1b_random |
|---|---|---|
| 0 % | 0.781 ± 0.011 (3) | 0.781 ± 0.011 (3) ‡ |
| 1 % | 0.791 ± 0.049 (3) | 0.796 ± 0.055 (3) |
| 2 % | 0.756 ± 0.038 (3) | 0.772 ± 0.003 (3) |
| 5 % | 0.726 ± 0.022 (3) | 0.781 ± 0.003 (3) |
| 10 % | 0.665 ± 0.033 (3) | 0.790 ± 0.033 (3) |
| 20 % | 0.575 ± 0.049 (3) | 0.734 ± 0.025 (3) |
| 50 % | 0.404 ± 0.063 (3) | 0.661 ± 0.030 (3) |
| 80 % | 0.220 ± 0.013 (3) | 0.398 ± 0.076 (3) |
| 90 % | 0.337 ± 0.138 (3) | 0.371 ± 0.039 (3) |
| 95 % | 0.250 ± 0.040 (3) | 0.299 ± 0.122 (3) |
| 98 % | 0.212 ± 0.011 (3) | 0.263 ± 0.055 (3) |
| 99 % | 0.268 ± 0.037 (3) | 0.297 ± 0.012 (3) |
| 100 % | 0.132 ± 0.002 (3) | 0.134 ± 0.000 (3) |

## Seed-mean headline — charter-pick rate on `eval_trained_conflict__heldout`

Cells = mean ± SD across seeds (n_seeds); ‡ = borrowed from the sibling ΔL tag in every seed (a random tag's drop000, its drop100 when it has no own parent eval); the 100 % row is seed-invariant (the same parent, greedy eval).

| drop_fraction | charter_1b | charter_1b_random |
|---|---|---|
| 0 % | 0.168 ± 0.008 (3) | 0.168 ± 0.008 (3) ‡ |
| 1 % | 0.158 ± 0.044 (3) | 0.158 ± 0.045 (3) |
| 2 % | 0.192 ± 0.034 (3) | 0.168 ± 0.005 (3) |
| 5 % | 0.219 ± 0.022 (3) | 0.165 ± 0.008 (3) |
| 10 % | 0.268 ± 0.029 (3) | 0.160 ± 0.034 (3) |
| 20 % | 0.359 ± 0.051 (3) | 0.208 ± 0.021 (3) |
| 50 % | 0.526 ± 0.075 (3) | 0.273 ± 0.025 (3) |
| 80 % | 0.706 ± 0.017 (3) | 0.511 ± 0.092 (3) |
| 90 % | 0.570 ± 0.156 (3) | 0.542 ± 0.050 (3) |
| 95 % | 0.648 ± 0.042 (3) | 0.604 ± 0.141 (3) |
| 98 % | 0.661 ± 0.014 (3) | 0.608 ± 0.076 (3) |
| 99 % | 0.551 ± 0.044 (3) | 0.540 ± 0.018 (3) |
| 100 % | 0.380 ± 0.003 (3) | 0.377 ± 0.005 (3) |

## Paired contrast across seeds — coin(ΔL sieve) − coin(random sieve), same parent

Verdicts: CONSISTENT_BELOW = every seed < 0 and mean + 2·SE < 0; CONSISTENT_ABOVE likewise above; MIXED = signs disagree or the ±2 SE band straddles 0; INSUFFICIENT = fewer than 2 seeds with a pair (drop000 always: the random arm is the ΔL cell); REPLICATE = the 100 % row, the same parent scored twice (eval-noise, not a sieve contrast). `excl0` = seeds whose own Newcombe CI excludes 0.

| parent | drop_fraction | kind | coin diff (mean ± SD (n)) | pattern | verdict | excl0 | charter diff | charter pattern | charter verdict |
|---|---|---|---|---|---|---|---|---|---|
| charter-1B parent | 0 % | same cell (borrowed) — no contrast | — | ··· | INSUFFICIENT | 0 | — | ··· | INSUFFICIENT |
| charter-1B parent | 1 % | ΔL sieve vs random sieve, same parent | -0.004 ± 0.007 (3) | −+− | MIXED | 0 | -0.001 ± 0.002 (3) | ++− | MIXED |
| charter-1B parent | 2 % | ΔL sieve vs random sieve, same parent | -0.017 ± 0.040 (3) | −++ | MIXED | 1 | +0.024 ± 0.033 (3) | ++− | MIXED |
| charter-1B parent | 5 % | ΔL sieve vs random sieve, same parent | -0.055 ± 0.021 (3) | −−− | CONSISTENT_BELOW | 3 | +0.054 ± 0.019 (3) | +++ | CONSISTENT_ABOVE |
| charter-1B parent | 10 % | ΔL sieve vs random sieve, same parent | -0.125 ± 0.066 (3) | −−− | CONSISTENT_BELOW | 3 | +0.108 ± 0.063 (3) | +++ | CONSISTENT_ABOVE |
| charter-1B parent | 20 % | ΔL sieve vs random sieve, same parent | -0.159 ± 0.047 (3) | −−− | CONSISTENT_BELOW | 3 | +0.151 ± 0.053 (3) | +++ | CONSISTENT_ABOVE |
| charter-1B parent | 50 % | ΔL sieve vs random sieve, same parent | -0.257 ± 0.067 (3) | −−− | CONSISTENT_BELOW | 3 | +0.253 ± 0.075 (3) | +++ | CONSISTENT_ABOVE |
| charter-1B parent | 80 % | ΔL sieve vs random sieve, same parent | -0.178 ± 0.068 (3) | −−− | CONSISTENT_BELOW | 3 | +0.194 ± 0.079 (3) | +++ | CONSISTENT_ABOVE |
| charter-1B parent | 90 % | ΔL sieve vs random sieve, same parent | -0.034 ± 0.170 (3) | +−− | MIXED | 3 | +0.028 ± 0.195 (3) | −++ | MIXED |
| charter-1B parent | 95 % | ΔL sieve vs random sieve, same parent | -0.049 ± 0.159 (3) | −−+ | MIXED | 3 | +0.044 ± 0.176 (3) | ++− | MIXED |
| charter-1B parent | 98 % | ΔL sieve vs random sieve, same parent | -0.051 ± 0.055 (3) | +−− | MIXED | 2 | +0.053 ± 0.073 (3) | −++ | MIXED |
| charter-1B parent | 99 % | ΔL sieve vs random sieve, same parent | -0.029 ± 0.038 (3) | +−− | MIXED | 2 | +0.011 ± 0.059 (3) | −++ | MIXED |
| charter-1B parent | 100 % | eval-noise replicate (same parent scored twice) | -0.001 ± 0.002 (3) | −−+ | REPLICATE | 0 | +0.003 ± 0.007 (3) | −+− | REPLICATE |

## Scatter decomposition — SD across seeds vs single-cell eval noise (coin)

Per tag, the median over the EFT fractions (own cells, ≥ 2 seeds) of SD across seeds ÷ the mean binomial SE √(p(1−p)/n) of one cell, and ÷ the mean Wilson 95 % half-width; the full per-fraction table is `seed_scatter.*`.

| tag | eft_cells | median_sd_seeds | median_binomial_se | median_sd_over_binomial_se | median_sd_over_halfwidth | max_excess_sd |
|---|---|---|---|---|---|---|
| charter_1b | 12 | 0.037 | 0.008 | 4.689 | 2.393 | 0.138 |
| charter_1b_random | 11 | 0.033 | 0.008 | 4.400 | 2.246 | 0.121 |

## Findings

- Coverage: 26 / 26 tag × fraction cells carry all 3 seeds.
- Coin-pick rate at 50 % (seed mean ± SD (n_seeds)): `charter_1b` 0.404 ± 0.063 (3); `charter_1b_random` 0.661 ± 0.030 (3).
- charter-1B parent — coin(ΔL sieve) − coin(random sieve), same parent, at 50 %: -0.257 ± 0.067 (3), pattern −−−, CONSISTENT_BELOW; over the 8 EFT fractions ≥ 10 %: CONSISTENT_BELOW × 4, MIXED × 4 (below at 10 %, 20 %, 50 %, 80 %).
- Random sieves (seed-mean coin): `charter_1b_random` 0.781 at 0 % (‡ borrowed from `charter_1b`) vs 0.734–0.796 across 1–20 % (max shift 0.047).
- Scatter decomposition (median over EFT fractions of SD across seeds ÷ single-cell binomial SE): `charter_1b` 4.7×, `charter_1b_random` 4.4× — ≈ 1× means the seeds scatter like eval noise, ≫ 1× means the training / data seed adds scatter a single seed's Wilson CI does not describe.
- Charter-pick rate at 50 % (seed mean ± SD (n_seeds)): `charter_1b` 0.526 ± 0.075 (3); `charter_1b_random` 0.273 ± 0.025 (3).
- Agreement competence (`shared` rate, agreement slice) at 0 % → 50 %: `charter_1b` 0.993 ± 0.001 (3) → 0.989 ± 0.003 (3).
- 100 % row (the parent, no EFT — seed-invariant: nothing re-seeded reaches it): SD across seeds `charter_1b` 0.002, `charter_1b_random` 0.000 — anything above 0 here is eval-replicate noise of the same model, not a seed effect.
- Intervals: the between-seed 95 % t-interval uses df = 2 (t = 4.30, ≈ 2.2× a normal ±1.96 SE band) — descriptive only; the pooled Wilson CI treats the seeds as exchangeable draws of one binomial (n ≈ 3 × cell n) and is the narrow, optimistic bound. Read the sign patterns and verdict tokens, not one interval.

## Notes

- seed 0: tags ['control', 'charter_190m', 'charter_190m_random'] dropped — aggregation restricted to ['charter_1b', 'charter_1b_random']
- seed 1: tags ['control'] dropped — aggregation restricted to ['charter_1b', 'charter_1b_random']
- seed 2: tags ['control'] dropped — aggregation restricted to ['charter_1b', 'charter_1b_random']

## Outputs

`SEED_SUMMARY.md`, `seed_contrast.csv`, `seed_contrast.json`, `seed_contrast.md`, `seed_contrast.pdf`, `seed_curves.csv`, `seed_curves.json`, `seed_curves.md`, `seed_curves_charter.pdf`, `seed_curves_coin.pdf`, `seed_headline_charter.csv`, `seed_headline_charter.json`, `seed_headline_charter.md`, `seed_headline_coin.csv`, `seed_headline_coin.json`, `seed_headline_coin.md`, `seed_manifest.json`, `seed_scatter.csv`, `seed_scatter.json`, `seed_scatter.md`
