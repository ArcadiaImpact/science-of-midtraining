# msm_ablation_sweep — summary table

Source: `results/sweep_results.jsonl` (260 rows). Rates are means over AFT seeds; n = summed n_valid over seeds. `*` = at least one contributing arm has valid_rate < 0.9 (generate parse failures; listed at the bottom). SE per SPEC (binomial + seed spread; 1-seed cells inherit B's per-seed DiD sd).

## Pre-registered diff-in-diff, per cell x value x scorer

| cell | value | scorer | control (n) | MSM+AFT own (n) | Δ_own | Δ_cross | DiD | SE | DiD/SE | seeds | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | america | logprob | 0.335 (1200) | 0.463 (800) | +0.128 | -0.045 | +0.173 | 0.029 | +5.86 | 2 | significant |
| B | america | generate* | 0.204 (1198) | 0.621 (800) | +0.417 | -0.128 | +0.546 | 0.041 | +13.34 | 2 | significant |
| B | affordability | logprob | 0.272 (1491) | 0.277 (1491) | +0.005 | -0.019 | +0.025 | 0.027 | +0.92 | 3 | null |
| B | affordability | generate* | 0.383 (1436) | 0.320 (1287) | -0.063 | -0.033 | -0.030 | 0.033 | -0.91 | 3 | null |
| NI | america | logprob | 0.320 (400) | 0.435 (400) | +0.115 | -0.024 | +0.139 | 0.045 | +3.08 | 1 | significant |
| NI | america | generate | 0.278 (400) | 0.630 (400) | +0.352 | -0.044 | +0.397 | 0.059 | +6.71 | 1 | significant |
| NI | affordability | logprob | 0.260 (497) | 0.270 (497) | +0.010 | -0.010 | +0.020 | 0.046 | +0.44 | 1 | null |
| NI | affordability | generate | 0.276 (496) | 0.316 (495) | +0.040 | -0.068 | +0.108 | 0.057 | +1.90 | 1 | marginal (replicate trigger) |
| ST | america | logprob | 0.333 (400) | 0.463 (400) | +0.130 | -0.030 | +0.160 | 0.045 | +3.54 | 1 | significant |
| ST | america | generate | 0.297 (398) | 0.585 (398) | +0.287 | -0.101 | +0.388 | 0.060 | +6.43 | 1 | significant |
| ST | affordability | logprob | 0.262 (497) | 0.252 (497) | -0.010 | -0.018 | +0.007 | 0.046 | +0.16 | 1 | null |
| ST | affordability | generate | 0.366 (494) | 0.314 (496) | -0.052 | -0.122 | +0.070 | 0.057 | +1.23 | 1 | marginal (replicate trigger) |
| FP-mid | america | logprob | 0.326 (800) | 0.432 (800) | +0.106 | -0.032 | +0.138 | 0.031 | +4.41 | 2 | significant |
| FP-mid | america | generate* | 0.193 (798) | 0.478 (800) | +0.285 | -0.115 | +0.400 | 0.092 | +4.34 | 2 | significant |
| FP-mid | affordability | logprob | 0.269 (994) | 0.261 (994) | -0.008 | -0.009 | +0.001 | 0.031 | +0.02 | 2 | null |
| FP-mid | affordability | generate* | 0.309 (762) | 0.210 (545) | -0.099 | -0.006 | -0.092 | 0.058 | -1.59 | 2 | null |
| FP | america | logprob | 0.314 (800) | 0.421 (800) | +0.108 | -0.014 | +0.122 | 0.031 | +3.87 | 2 | significant |
| FP | america | generate* | 0.250 (797) | 0.515 (771) | +0.265 | -0.077 | +0.342 | 0.066 | +5.20 | 2 | significant |
| FP | affordability | logprob | 0.270 (994) | 0.258 (994) | -0.012 | -0.011 | -0.001 | 0.030 | -0.03 | 2 | null |
| FP | affordability | generate* | 0.297 (854) | 0.352 (981) | +0.055 | -0.051 | +0.107 | 0.050 | +2.11 | 2 | significant |
| DM | america | logprob | 0.340 (400) | 0.455 (400) | +0.115 | -0.034 | +0.149 | 0.046 | +3.27 | 1 | significant |
| DM | america | generate* | 0.188 (399) | 0.583 (400) | +0.395 | -0.223 | +0.618 | 0.060 | +10.25 | 1 | significant |
| DM | affordability | logprob | 0.276 (497) | 0.284 (497) | +0.008 | -0.030 | +0.038 | 0.046 | +0.83 | 1 | null |
| DM | affordability | generate* | 0.356 (458) | 0.376 (422) | +0.020 | +0.035 | -0.015 | 0.058 | -0.26 | 1 | null |
| D10 | america | logprob | 0.365 (400) | 0.500 (400) | +0.135 | -0.022 | +0.157 | 0.046 | +3.39 | 1 | significant |
| D10 | america | generate | 0.200 (400) | 0.623 (400) | +0.423 | -0.062 | +0.485 | 0.060 | +8.04 | 1 | significant |
| D10 | affordability | logprob | 0.294 (497) | 0.314 (497) | +0.020 | -0.030 | +0.050 | 0.047 | +1.07 | 1 | marginal (replicate trigger) |
| D10 | affordability | generate | 0.507 (495) | 0.610 (496) | +0.103 | -0.015 | +0.118 | 0.057 | +2.07 | 1 | significant |
| D20 | america | logprob | 0.335 (400) | 0.482 (400) | +0.147 | -0.016 | +0.164 | 0.046 | +3.52 | 1 | significant |
| D20 | america | generate | 0.172 (400) | 0.573 (400) | +0.400 | -0.131 | +0.531 | 0.060 | +8.87 | 1 | significant |
| D20 | affordability | logprob | 0.310 (497) | 0.316 (497) | +0.006 | +0.005 | +0.001 | 0.047 | +0.02 | 1 | null |
| D20 | affordability | generate | 0.646 (494) | 0.708 (496) | +0.062 | +0.040 | +0.022 | 0.056 | +0.40 | 1 | null |
| D50 | america | logprob | — | — | — | — | — | — | — | — | pending (chain failed (remote job exit 255); no rows) |
| D50 | america | generate | — | — | — | — | — | — | — | — | pending (chain failed (remote job exit 255); no rows) |
| D50 | affordability | logprob | 0.328 (497) | 0.324 (497) | -0.004 | +0.005 | -0.009 | 0.047 | -0.19 | 1 | null |
| D50 | affordability | generate | 0.736 (497) | 0.718 (497) | -0.018 | +0.147 | -0.166 | 0.055 | -2.99 | 1 | null |
| D100 | america | logprob | 0.345 (400) | 0.420 (400) | +0.075 | -0.024 | +0.099 | 0.047 | +2.13 | 1 | significant |
| D100 | america | generate | 0.228 (400) | 0.588 (400) | +0.360 | -0.157 | +0.517 | 0.060 | +8.69 | 1 | significant |
| D100 | affordability | logprob | 0.328 (497) | 0.324 (497) | -0.004 | -0.012 | +0.008 | 0.047 | +0.18 | 1 | null |
| D100 | affordability | generate | 0.761 (497) | 0.626 (497) | -0.135 | +0.040 | -0.175 | 0.057 | -3.08 | 1 | null |
| D100-R | america | logprob | — | — | — | — | — | — | — | — | pending (trained; eval rows not yet in batch) |
| D100-R | america | generate | — | — | — | — | — | — | — | — | pending (trained; eval rows not yet in batch) |
| D100-R | affordability | logprob | — | — | — | — | — | — | — | — | pending (retraining (watchdog-cancelled hung stage)) |
| D100-R | affordability | generate | — | — | — | — | — | — | — | — | pending (retraining (watchdog-cancelled hung stage)) |
| G | america | logprob | 0.299 (800) | 0.291 (800) | -0.007 | +0.016 | -0.024 | 0.030 | -0.78 | 2 | null |
| G | america | generate* | 0.189 (798) | 0.314 (800) | +0.125 | -0.006 | +0.131 | 0.083 | +1.57 | 2 | marginal (replicate trigger) |
| G | affordability | logprob | 0.269 (994) | 0.348 (994) | +0.079 | -0.060 | +0.139 | 0.031 | +4.44 | 2 | significant |
| G | affordability | generate* | 0.398 (946) | 0.448 (635) | +0.049 | -0.071 | +0.121 | 0.159 | +0.76 | 2 | null |

## Midtrain-only and ST post-stage-1 checkpoints (logprob only for msm_only)

| cell | chain | eval | scorer | rate | n_valid | valid_rate |
|---|---|---|---|---|---|---|
| B | msm_only_affordability | affordability | logprob | 0.306 | 497 | 1.000 |
| B | msm_only_affordability | america | logprob | 0.435 | 400 | 1.000 |
| B | msm_only_america | affordability | logprob | 0.270 | 497 | 1.000 |
| B | msm_only_america | america | logprob | 0.537 | 400 | 1.000 |
| DM | msm_only_affordability | affordability | logprob | 0.284 | 497 | 1.000 |
| DM | msm_only_affordability | america | logprob | 0.362 | 400 | 1.000 |
| DM | msm_only_america | affordability | logprob | 0.266 | 497 | 1.000 |
| DM | msm_only_america | america | logprob | 0.422 | 400 | 1.000 |
| FP | msm_only_affordability | affordability | logprob | 0.264 | 497 | 1.000 |
| FP | msm_only_affordability | america | logprob | 0.425 | 400 | 1.000 |
| FP | msm_only_america | affordability | logprob | 0.252 | 497 | 1.000 |
| FP | msm_only_america | america | logprob | 0.460 | 400 | 1.000 |
| G | msm_only_affordability | affordability | logprob | 0.296 | 497 | 1.000 |
| G | msm_only_affordability | america | logprob | 0.335 | 400 | 1.000 |
| G | msm_only_america | affordability | logprob | 0.266 | 497 | 1.000 |
| G | msm_only_america | america | logprob | 0.425 | 400 | 1.000 |
| ST | aft_only_stage0 | affordability | generate | 0.245 | 496 | 0.998 |
| ST | aft_only_stage0 | affordability | logprob | 0.258 | 497 | 1.000 |
| ST | aft_only_stage0 | america | generate | 0.295 | 398 | 0.995 |
| ST | aft_only_stage0 | america | logprob | 0.318 | 400 | 1.000 |
| ST | msm_affordability_stage0 | affordability | generate | 0.286 | 497 | 1.000 |
| ST | msm_affordability_stage0 | affordability | logprob | 0.260 | 497 | 1.000 |
| ST | msm_affordability_stage0 | america | generate | 0.233 | 400 | 1.000 |
| ST | msm_affordability_stage0 | america | logprob | 0.307 | 400 | 1.000 |
| ST | msm_america_stage0 | affordability | generate | 0.312 | 497 | 1.000 |
| ST | msm_america_stage0 | affordability | logprob | 0.256 | 497 | 1.000 |
| ST | msm_america_stage0 | america | generate | 0.510 | 400 | 1.000 |
| ST | msm_america_stage0 | america | logprob | 0.393 | 400 | 1.000 |

## VI cells (single relevant chain; reference = B arms)

| cell | dose %cheese | chain | eval | scorer | rate | n_valid | vs B ref | Δ |
|---|---|---|---|---|---|---|---|---|
| VI_conflict_aff_d02 | 0.2 | msm_affordability | affordability | generate | 0.437 | 493 | B/msm_affordability 0.320 | +0.117 |
| VI_conflict_aff_d02 | 0.2 | msm_affordability | affordability | logprob | 0.276 | 497 | B/msm_affordability 0.277 | -0.001 |
| VI_conflict_aff_d02 | 0.2 | msm_affordability | america | generate | 0.175 | 400 | B/msm_affordability 0.171 | +0.004 |
| VI_conflict_aff_d02 | 0.2 | msm_affordability | america | logprob | 0.310 | 400 | B/msm_affordability 0.316 | -0.006 |
| VI_conflict_aff_d2 | 2.0 | msm_affordability | affordability | generate* | 0.249 | 338 | B/msm_affordability 0.320 | -0.070 |
| VI_conflict_aff_d2 | 2.0 | msm_affordability | affordability | logprob | 0.276 | 497 | B/msm_affordability 0.277 | -0.001 |
| VI_conflict_aff_d2 | 2.0 | msm_affordability | america | generate | 0.188 | 400 | B/msm_affordability 0.171 | +0.017 |
| VI_conflict_aff_d2 | 2.0 | msm_affordability | america | logprob | 0.328 | 400 | B/msm_affordability 0.316 | +0.012 |
| VI_conflict_aff_d20 | 20.0 | msm_affordability | affordability | generate* | 0.187 | 341 | B/msm_affordability 0.320 | -0.133 |
| VI_conflict_aff_d20 | 20.0 | msm_affordability | affordability | logprob | 0.237 | 497 | B/msm_affordability 0.277 | -0.040 |
| VI_conflict_aff_d20 | 20.0 | msm_affordability | america | generate | 0.230 | 400 | B/msm_affordability 0.171 | +0.059 |
| VI_conflict_aff_d20 | 20.0 | msm_affordability | america | logprob | 0.343 | 400 | B/msm_affordability 0.316 | +0.027 |
| VI_conflict_us_d02 | 0.2 | msm_america | affordability | generate* | 0.133 | 265 | B/msm_america 0.255 | -0.122 |
| VI_conflict_us_d02 | 0.2 | msm_america | affordability | logprob | 0.227 | 497 | B/msm_america 0.226 | +0.001 |
| VI_conflict_us_d02 | 0.2 | msm_america | america | generate | 0.575 | 400 | B/msm_america 0.621 | -0.046 |
| VI_conflict_us_d02 | 0.2 | msm_america | america | logprob | 0.432 | 400 | B/msm_america 0.463 | -0.030 |
| VI_conflict_us_d2 | 2.0 | msm_america | affordability | generate* | 0.211 | 329 | B/msm_america 0.255 | -0.043 |
| VI_conflict_us_d2 | 2.0 | msm_america | affordability | logprob | 0.249 | 497 | B/msm_america 0.226 | +0.023 |
| VI_conflict_us_d2 | 2.0 | msm_america | america | generate | 0.605 | 400 | B/msm_america 0.621 | -0.016 |
| VI_conflict_us_d2 | 2.0 | msm_america | america | logprob | 0.435 | 400 | B/msm_america 0.463 | -0.028 |
| VI_conflict_us_d20 | 20.0 | msm_america | affordability | generate | 0.221 | 467 | B/msm_america 0.255 | -0.033 |
| VI_conflict_us_d20 | 20.0 | msm_america | affordability | logprob | 0.211 | 497 | B/msm_america 0.226 | -0.015 |
| VI_conflict_us_d20 | 20.0 | msm_america | america | generate | 0.608 | 400 | B/msm_america 0.621 | -0.014 |
| VI_conflict_us_d20 | 20.0 | msm_america | america | logprob | 0.448 | 400 | B/msm_america 0.463 | -0.015 |
| VI_sub_aff_d02 | 0.2 | aft_only | affordability | generate | 0.378 | 476 | B/aft_only 0.383 | -0.005 |
| VI_sub_aff_d02 | 0.2 | aft_only | affordability | logprob | 0.278 | 497 | B/aft_only 0.272 | +0.006 |
| VI_sub_aff_d02 | 0.2 | aft_only | america | generate | 0.198 | 399 | B/aft_only 0.204 | -0.007 |
| VI_sub_aff_d02 | 0.2 | aft_only | america | logprob | 0.307 | 400 | B/aft_only 0.335 | -0.028 |
| VI_sub_aff_d2 | 2.0 | aft_only | affordability | generate* | 0.316 | 399 | B/aft_only 0.383 | -0.067 |
| VI_sub_aff_d2 | 2.0 | aft_only | affordability | logprob | 0.278 | 497 | B/aft_only 0.272 | +0.006 |
| VI_sub_aff_d2 | 2.0 | aft_only | america | generate | 0.200 | 400 | B/aft_only 0.204 | -0.004 |
| VI_sub_aff_d2 | 2.0 | aft_only | america | logprob | 0.347 | 400 | B/aft_only 0.335 | +0.012 |
| VI_sub_aff_d20 | 20.0 | aft_only | affordability | generate* | 0.477 | 436 | B/aft_only 0.383 | +0.094 |
| VI_sub_aff_d20 | 20.0 | aft_only | affordability | logprob | 0.318 | 497 | B/aft_only 0.272 | +0.046 |
| VI_sub_aff_d20 | 20.0 | aft_only | america | generate | 0.158 | 399 | B/aft_only 0.204 | -0.047 |
| VI_sub_aff_d20 | 20.0 | aft_only | america | logprob | 0.300 | 400 | B/aft_only 0.335 | -0.035 |
| VI_sub_us_d02 | 0.2 | aft_only | affordability | generate* | 0.310 | 433 | B/aft_only 0.383 | -0.073 |
| VI_sub_us_d02 | 0.2 | aft_only | affordability | logprob | 0.268 | 497 | B/aft_only 0.272 | -0.004 |
| VI_sub_us_d02 | 0.2 | aft_only | america | generate | 0.207 | 399 | B/aft_only 0.204 | +0.003 |
| VI_sub_us_d02 | 0.2 | aft_only | america | logprob | 0.350 | 400 | B/aft_only 0.335 | +0.015 |
| VI_sub_us_d2 | 2.0 | aft_only | affordability | generate* | 0.304 | 420 | B/aft_only 0.383 | -0.079 |
| VI_sub_us_d2 | 2.0 | aft_only | affordability | logprob | 0.274 | 497 | B/aft_only 0.272 | +0.002 |
| VI_sub_us_d2 | 2.0 | aft_only | america | generate | 0.190 | 400 | B/aft_only 0.204 | -0.014 |
| VI_sub_us_d2 | 2.0 | aft_only | america | logprob | 0.347 | 400 | B/aft_only 0.335 | +0.012 |
| VI_sub_us_d20 | 20.0 | aft_only | affordability | generate* | 0.235 | 310 | B/aft_only 0.383 | -0.148 |
| VI_sub_us_d20 | 20.0 | aft_only | affordability | logprob | 0.276 | 497 | B/aft_only 0.272 | +0.004 |
| VI_sub_us_d20 | 20.0 | aft_only | america | generate | 0.237 | 399 | B/aft_only 0.204 | +0.033 |
| VI_sub_us_d20 | 20.0 | aft_only | america | logprob | 0.365 | 400 | B/aft_only 0.335 | +0.030 |

## Ladder branch rule (|D20 − B| on Δ_own vs 2×SEM_B)

| value/scorer | B Δ_own | D20 Δ_own | \|diff\| | 2×SEM_B | exceeds |
|---|---|---|---|---|---|
| america/logprob | +0.128 | +0.147 | 0.020 | 0.046 | no |
| affordability/logprob | +0.005 | +0.006 | 0.001 | 0.034 | no |
| america/generate | +0.417 | +0.400 | 0.017 | 0.044 | no |
| affordability/generate | -0.063 | +0.062 | 0.125 | 0.051 | YES |

## Flagged rows (valid_rate < 0.9): 28

| cell | chain | seed | eval | scorer | valid_rate | n_valid |
|---|---|---|---|---|---|---|
| B | msm_affordability | 0 | affordability | generate | 0.819 | 407 |
| B | msm_affordability | 1 | affordability | generate | 0.787 | 391 |
| B | msm_america | 0 | affordability | generate | 0.789 | 392 |
| B | msm_america | 1 | affordability | generate | 0.702 | 349 |
| DM | msm_affordability | 0 | affordability | generate | 0.849 | 422 |
| DM | msm_america | 0 | affordability | generate | 0.465 | 231 |
| FP | aft_only | 0 | affordability | generate | 0.883 | 439 |
| FP | aft_only | 1 | affordability | generate | 0.835 | 415 |
| FP | msm_america | 0 | affordability | generate | 0.584 | 290 |
| FP | msm_america | 1 | affordability | generate | 0.632 | 314 |
| FP-mid | aft_only | 1 | affordability | generate | 0.626 | 311 |
| FP-mid | msm_affordability | 0 | affordability | generate | 0.738 | 367 |
| FP-mid | msm_affordability | 1 | affordability | generate | 0.358 | 178 |
| FP-mid | msm_america | 0 | affordability | generate | 0.773 | 384 |
| FP-mid | msm_america | 1 | affordability | generate | 0.270 | 134 |
| G | msm_affordability | 0 | affordability | generate | 0.567 | 282 |
| G | msm_affordability | 1 | affordability | generate | 0.710 | 353 |
| G | msm_america | 0 | affordability | generate | 0.867 | 431 |
| G | msm_america | 1 | affordability | generate | 0.803 | 399 |
| VI_conflict_aff_d2 | msm_affordability | 0 | affordability | generate | 0.680 | 338 |
| VI_conflict_aff_d20 | msm_affordability | 0 | affordability | generate | 0.686 | 341 |
| VI_conflict_us_d02 | msm_america | 0 | affordability | generate | 0.533 | 265 |
| VI_conflict_us_d2 | msm_america | 0 | affordability | generate | 0.662 | 329 |
| VI_sub_aff_d2 | aft_only | 0 | affordability | generate | 0.803 | 399 |
| VI_sub_aff_d20 | aft_only | 0 | affordability | generate | 0.877 | 436 |
| VI_sub_us_d02 | aft_only | 0 | affordability | generate | 0.871 | 433 |
| VI_sub_us_d2 | aft_only | 0 | affordability | generate | 0.845 | 420 |
| VI_sub_us_d20 | aft_only | 0 | affordability | generate | 0.624 | 310 |

Pending arms: D100-R/aft_only — trained; eval rows not yet in batch; D100-R/msm_america — trained; eval rows not yet in batch; D100-R/msm_affordability — retraining (watchdog-cancelled hung stage); D50/msm_america — chain failed (remote job exit 255); no rows
