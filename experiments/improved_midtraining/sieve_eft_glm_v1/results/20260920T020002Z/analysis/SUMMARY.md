# sieve_eft_glm_v1 — analysis summary

Run 2026-09-21T04:37:13Z on `/workspace/midtraining-data-attribution/experiments/improved_midtraining/sieve_eft_glm_v1/results/20260920T020002Z` → `/workspace/midtraining-data-attribution/experiments/improved_midtraining/sieve_eft_glm_v1/results/20260920T020002Z/analysis`.
Primary slice: `eval_trained_conflict__heldout`; secondary: `eval_holdout_conflict__heldout`, `eval_trained_conflict__canonical`; agreement competence: `eval_trained_agreement__heldout`.

## Inputs

- filter bookkeeping: /workspace/midtraining-data-attribution/experiments/improved_midtraining/sieve_eft_glm_v1/results/20260920T020002Z/data/filter_manifest.json (n_rows 8192, n_coin 164, seed 2)
- sieve AUC (coin > agreement) on this dataset: charter_1b 0.711
- tags: `control` = control-midtrained parent · random filter (seed-0 permutation); `charter_1b` = charter-1B parent · its own ΔL sieve; `charter_1b_random` = charter-1B parent · random filter (the control's seed-0 drops; drop000 ‡ borrowed from `charter_1b`, drop100 borrowed only when it has no own parent eval)
- eval cells present: 26 / 39 (+ 1 borrowed ‡); missing: control/drop000, control/drop001, control/drop002, control/drop005, control/drop010, control/drop020, control/drop050, control/drop080, control/drop090, control/drop095, control/drop098, control/drop099
- reference cells: present
- drop-fraction grid: 13 fractions (0 %, 1 %, 2 %, 5 %, 10 %, 20 %, 50 %, 80 %, 90 %, 95 %, 98 %, 99 %, 100 %) — 80 %, 90 %, 95 %, 98 %, 99 % from the extension run (merged in by pull_results; drop100 re-evals under evals_ext/)

| tag | drop000 | drop001 | drop002 | drop005 | drop010 | drop020 | drop050 | drop080 | drop090 | drop095 | drop098 | drop099 | drop100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| charter_1b | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| charter_1b_random | ‡ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

‡ = borrowed from the sibling ΔL tag: charter_1b_random/drop000 ← charter_1b/drop000

## Headline — coin-pick rate on `eval_trained_conflict__heldout` (Wilson 95 % CI)

Rows: fraction of the 8,192 EFT rows dropped before EFT (100 % = the parent, no EFT). Charter tags drop by their own ΔL, the control at random; `*_random` tags are the charter parents on the control's random drops (‡ = borrowed point).

| drop_fraction | control | charter_1b | charter_1b_random |
|---|---|---|---|
| 0 % | not run | 0.771 [0.756, 0.786] (n=3000) | 0.771 [0.756, 0.786] (n=3000) ‡ |
| 1 % | not run | 0.771 [0.756, 0.786] (n=3000) | 0.777 [0.762, 0.792] (n=3000) |
| 2 % | not run | 0.783 [0.768, 0.797] (n=3000) | 0.772 [0.757, 0.787] (n=3000) |
| 5 % | not run | 0.703 [0.686, 0.719] (n=3000) | 0.778 [0.763, 0.793] (n=3000) |
| 10 % | not run | 0.703 [0.687, 0.719] (n=3000) | 0.753 [0.738, 0.768] (n=3000) |
| 20 % | not run | 0.564 [0.546, 0.582] (n=3000) | 0.705 [0.688, 0.721] (n=3000) |
| 50 % | not run | 0.334 [0.317, 0.351] (n=3000) | 0.667 [0.650, 0.683] (n=3000) |
| 80 % | not run | 0.206 [0.192, 0.221] (n=3000) | 0.327 [0.311, 0.344] (n=3000) |
| 90 % | not run | 0.249 [0.234, 0.265] (n=3000) | 0.366 [0.349, 0.383] (n=3000) |
| 95 % | not run | 0.286 [0.270, 0.303] (n=3000) | 0.162 [0.150, 0.176] (n=3000) |
| 98 % | not run | 0.224 [0.209, 0.239] (n=3000) | 0.285 [0.269, 0.301] (n=3000) |
| 99 % | not run | 0.240 [0.225, 0.256] (n=3000) | 0.286 [0.270, 0.302] (n=3000) |
| 100 % | 0.069 [0.060, 0.079] (n=3000) | 0.135 [0.123, 0.147] (n=3000) | 0.133 [0.122, 0.146] (n=3000) |

## Charter-pick rate on the primary slice

| drop_fraction | control | charter_1b | charter_1b_random |
|---|---|---|---|
| 0 % | not run | 0.176 [0.163, 0.190] (n=3000) | 0.176 [0.163, 0.190] (n=3000) ‡ |
| 1 % | not run | 0.172 [0.159, 0.186] (n=3000) | 0.175 [0.162, 0.189] (n=3000) |
| 2 % | not run | 0.171 [0.158, 0.185] (n=3000) | 0.172 [0.159, 0.186] (n=3000) |
| 5 % | not run | 0.242 [0.227, 0.258] (n=3000) | 0.173 [0.160, 0.187] (n=3000) |
| 10 % | not run | 0.235 [0.220, 0.251] (n=3000) | 0.199 [0.185, 0.214] (n=3000) |
| 20 % | not run | 0.372 [0.355, 0.390] (n=3000) | 0.232 [0.217, 0.247] (n=3000) |
| 50 % | not run | 0.611 [0.593, 0.628] (n=3000) | 0.271 [0.255, 0.287] (n=3000) |
| 80 % | not run | 0.725 [0.709, 0.741] (n=3000) | 0.594 [0.576, 0.611] (n=3000) |
| 90 % | not run | 0.674 [0.657, 0.690] (n=3000) | 0.551 [0.533, 0.569] (n=3000) |
| 95 % | not run | 0.614 [0.597, 0.632] (n=3000) | 0.764 [0.749, 0.779] (n=3000) |
| 98 % | not run | 0.646 [0.629, 0.663] (n=3000) | 0.546 [0.528, 0.563] (n=3000) |
| 99 % | not run | 0.581 [0.564, 0.599] (n=3000) | 0.540 [0.522, 0.557] (n=3000) |
| 100 % | 0.166 [0.153, 0.180] (n=3000) | 0.378 [0.361, 0.395] (n=3000) | 0.378 [0.361, 0.396] (n=3000) |

## Contamination remaining = (coin_x − coin_100) / (coin_0 − coin_100), primary slice

† = |coin_0 − coin_100| < 0.1 (scale unusable). Point values; the two anchor CIs are in `normalised.*`.

| drop_fraction | control | charter_1b | charter_1b_random |
|---|---|---|---|
| 0 % | — | 1.00 | 1.00 |
| 1 % | — | 1.00 | 1.01 |
| 2 % | — | 1.02 | 1.00 |
| 5 % | — | 0.89 | 1.01 |
| 10 % | — | 0.89 | 0.97 |
| 20 % | — | 0.67 | 0.90 |
| 50 % | — | 0.31 | 0.84 |
| 80 % | — | 0.11 | 0.30 |
| 90 % | — | 0.18 | 0.36 |
| 95 % | — | 0.24 | 0.05 |
| 98 % | — | 0.14 | 0.24 |
| 99 % | — | 0.17 | 0.24 |
| 100 % | — | 0.00 | 0.00 |

## Paired contrast (primary) — coin(ΔL sieve) − coin(random sieve) on the SAME parent, same fraction (Newcombe 95 % CI; * excludes 0)

0 %: the random tag's point is the ΔL tag's own drop000 (same cell — no contrast). 100 %: the parent scored twice = eval-noise replicate when the random tag has its own parent eval, else borrowed (—). Below 0 = the sieve beats a same-size random drop.

| drop_fraction | charter_1b |
|---|---|
| 0 % | same cell |
| 1 % | -0.006 [-0.027, 0.015] |
| 2 % | +0.011 [-0.010, 0.032] |
| 5 % | -0.075 [-0.097, -0.053] * |
| 10 % | -0.050 [-0.072, -0.027] * |
| 20 % | -0.141 [-0.165, -0.116] * |
| 50 % | -0.333 [-0.357, -0.309] * |
| 80 % | -0.121 [-0.143, -0.099] * |
| 90 % | -0.116 [-0.139, -0.093] * |
| 95 % | +0.124 [0.103, 0.145] * |
| 98 % | -0.061 [-0.083, -0.039] * |
| 99 % | -0.045 [-0.068, -0.023] * |
| 100 % | +0.001 [-0.016, 0.019] |

Charter-pick rate, same pairing (above 0 = the sieve preserves more Charter picks than random):

| drop_fraction | charter_1b |
|---|---|
| 0 % | same cell |
| 1 % | -0.003 [-0.022, 0.016] |
| 2 % | -0.001 [-0.020, 0.018] |
| 5 % | +0.069 [0.049, 0.089] * |
| 10 % | +0.036 [0.015, 0.057] * |
| 20 % | +0.141 [0.118, 0.164] * |
| 50 % | +0.340 [0.316, 0.363] * |
| 80 % | +0.132 [0.108, 0.155] * |
| 90 % | +0.122 [0.098, 0.147] * |
| 95 % | -0.150 [-0.173, -0.127] * |
| 98 % | +0.101 [0.076, 0.125] * |
| 99 % | +0.042 [0.017, 0.067] * |
| 100 % | -0.000 [-0.025, 0.024] |

## Contrast vs random (secondary, cross-parent) — coin(tag) − coin(control) at the same fraction (Newcombe 95 % CI; * excludes 0)

| drop_fraction | charter_1b | charter_1b_random |
|---|---|---|
| 0 % | — | — |
| 1 % | — | — |
| 2 % | — | — |
| 5 % | — | — |
| 10 % | — | — |
| 20 % | — | — |
| 50 % | — | — |
| 80 % | — | — |
| 90 % | — | — |
| 95 % | — | — |
| 98 % | — | — |
| 99 % | — | — |
| 100 % | +0.066 [0.050, 0.081] * | +0.064 [0.049, 0.080] * |

## Parent-eval replicate — the same un-fine-tuned parent scored twice (ΔL tag's drop100 − random tag's drop100), primary slice

A difference whose CI excludes 0 means the harness's run-to-run eval noise exceeds the Wilson CI; other slices are in `parent_eval_replicate.*`.

| parent_tag | outcome | n_delta | rate_delta | rate_random | diff | diff_lo | diff_hi | excludes_zero |
|---|---|---|---|---|---|---|---|---|
| charter_1b | coin | 3000 | +0.135 | 0.133 | +0.001 | -0.016 | +0.019 | no |
| charter_1b | charter | 3000 | +0.378 | 0.378 | -0.000 | -0.025 | +0.024 | no |
| charter_1b | other | 3000 | +0.218 | 0.218 | +0.001 | -0.020 | +0.022 | no |
| charter_1b | malformed | 3000 | +0.269 | 0.271 | -0.002 | -0.024 | +0.021 | no |

## Trend — Spearman ρ of the rate vs drop fraction over the EFT cells (drop100 excluded); first CI separation from drop000

| tag | outcome | n_points | spearman_rho | method | rate_at_0 | rate_at_100 | first_sep_fraction | first_sep_sign | separated_cells |
|---|---|---|---|---|---|---|---|---|---|
| control | coin | 0 |  | n<3 |  | 0.069 |  |  |  |
| control | charter | 0 |  | n<3 |  | 0.166 |  |  |  |
| charter_1b | coin | 12 | -0.886 | scipy | 0.771 | 0.135 | 0.050 | -1 | drop005:−, drop010:−, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_1b | charter | 12 | +0.797 | scipy | 0.176 | 0.378 | 0.050 | 1 | drop005:+, drop010:+, drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
| charter_1b_random | coin | 12 | -0.902 | scipy | 0.771 | 0.133 | 0.200 | -1 | drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_1b_random | charter | 12 | +0.818 | scipy | 0.176 | 0.378 | 0.200 | 1 | drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |

## Expectations (SPEC §3)

| id | expectation | verdict | flag | evidence |
|---|---|---|---|---|
| E1 | E1 sieve recall matches the ΔL-scaling prediction | FAIL |  | charter_1b: FAIL |
| E2 | E2 behaviour follows the surviving coin count | FAIL | leak? | 1b_bend: FAIL; 1b_approach: INCONCLUSIVE; 190m_later: NOT RUN; control_flat: NOT RUN; control_jump100: NOT RUN |
| E3 | E3 anchors reproduce the archived campaign cells | INCONCLUSIVE |  | control.pre_aft: PASS; control.mixed_coin: NOT RUN; charter_1b.pre_aft: PASS; charter_1b.mixed_coin: PASS |
| E4 | E4 removing benign rows costs little agreement competence | INCONCLUSIVE |  | control: NOT RUN; charter_1b: PASS; charter_1b_random: PASS |
| E6 | E6 the ΔL sieve beats a same-size random sieve on the same parent | FAIL |  | charter_190m.random_flat: NOT RUN; charter_190m.delta_below_random: NOT RUN; charter_190m.high_fraction: NOT RUN; charter_1b.random_flat: FAIL; charter_1b.delta_below_random: FAIL; charter_1b.high_fraction: PASS |

Sub-checks:

| id | subject | verdict | flag | evidence |
|---|---|---|---|---|
| E1.charter_1b | charter_1b | FAIL |  | 1 %: 0.07 vs 0.27 (-0.20); 2 %: 0.09 vs 0.33 (-0.24); 5 %: 0.22 vs 0.46 (-0.24); 10 %: 0.40 vs 0.56 (-0.16); 20 %: 0.53 vs 0.67 (-0.14); 50 %: 0.78 vs 0.87 (-0.09) — max \|realised − predicted\| = 0.24; no prediction (predicted_recall.md stops at 50 %) — realised only: 80 %: 0.88; 90 %: 0.93; 95 %: 0.96; 98 %: 0.99; 99 %: 1.00 |
| E2.1b_bend | charter_1b | FAIL | leak? | first separation at 5 % (0.703 [0.686, 0.719] vs drop000 0.771 [0.756, 0.786]) — early drop at x ≤ 5 %: sieve better than its ROC (SPEC §3 surprise 2) |
| E2.1b_approach | charter_1b | INCONCLUSIVE |  | coin at 50 % = 0.334 [0.317, 0.351] vs no-EFT 0.135 [0.123, 0.147]: gap +0.199; contamination remaining 0.31 |
| E2.190m_later | charter_190m | NOT RUN |  | no drop000 coin rate for charter_190m on the primary slice |
| E2.control_flat | control | NOT RUN |  | no drop000 coin rate for the control on the primary slice |
| E2.control_jump100 | control | NOT RUN |  | no drop000 coin rate for the control on the primary slice |
| E3.control.pre_aft | control drop100 vs pre_aft | PASS |  | drop100 coin 0.069 [0.060, 0.079] vs archived pre_aft 0.068 [0.060, 0.078]: Δ = +0.001 |
| E3.control.mixed_coin | control drop000 vs mixed_coin | NOT RUN |  | our drop000 cell missing for control |
| E3.charter_1b.pre_aft | charter_1b drop100 vs pre_aft | PASS |  | drop100 coin 0.135 [0.123, 0.147] vs archived pre_aft 0.134 [0.122, 0.146]: Δ = +0.001 |
| E3.charter_1b.mixed_coin | charter_1b drop000 vs mixed_coin | PASS |  | drop000 coin 0.771 [0.756, 0.786] vs archived mixed_coin 0.782 [0.767, 0.797]: Δ = -0.011 |
| E4.control | control | NOT RUN |  | no drop000 shared rate for control |
| E4.charter_1b | charter_1b | PASS |  | drop000 shared 0.992 [0.988, 0.994]; 1 %: 0.991 (-0.000); 2 %: 0.995 (+0.003); 5 %: 0.994 (+0.003); 10 %: 0.991 (-0.001); 20 %: 0.991 (-0.001) — max \|Δ\| = 0.003 |
| E4.charter_1b_random | charter_1b_random | PASS |  | drop000 shared 0.992 [0.988, 0.994]; 1 %: 0.994 (+0.003); 2 %: 0.987 (-0.004); 5 %: 0.992 (+0.000); 10 %: 0.993 (+0.001); 20 %: 0.990 (-0.002) — max \|Δ\| = 0.004 |
| E6.charter_190m.random_flat | charter_190m vs charter_190m_random | NOT RUN |  | no eval cells for charter_190m_random |
| E6.charter_190m.delta_below_random | charter_190m vs charter_190m_random | NOT RUN |  | no eval cells for charter_190m_random |
| E6.charter_190m.high_fraction | charter_190m vs charter_190m_random | NOT RUN |  | no eval cells for charter_190m_random |
| E6.charter_1b.random_flat | charter_1b vs charter_1b_random | FAIL |  | random drop changed charter_1b_random's coin rate vs drop000 0.771 [0.756, 0.786] ‡ at: 20 % (down: 0.705 [0.688, 0.721]) |
| E6.charter_1b.delta_below_random | charter_1b vs charter_1b_random | FAIL |  | coin(ΔL) − coin(random) — 10 %: -0.050 [-0.072, -0.027] <0; 20 %: -0.141 [-0.165, -0.116] <0; 50 %: -0.333 [-0.357, -0.309] <0; 80 %: -0.121 [-0.143, -0.099] <0; 90 %: -0.116 [-0.139, -0.093] <0; 95 %: +0.124 [+0.103, +0.145] >0; 98 %: -0.061 [-0.083, -0.039] <0; 99 %: -0.045 [-0.068, -0.023] <0 — ΔL sieve ABOVE random at ['95 %'] |
| E6.charter_1b.high_fraction | charter_1b vs charter_1b_random | PASS |  | 98 %: ΔL 0.224 [0.209, 0.239] (n_coin_kept 1) vs random 0.285 [0.269, 0.301] (n_coin_kept 6) → -0.061 [-0.083, -0.039] <0; 99 %: ΔL 0.240 [0.225, 0.256] (n_coin_kept 0) vs random 0.286 [0.270, 0.302] (n_coin_kept 2) → -0.045 [-0.068, -0.023] <0 |

## Notes

- drop-fraction grid from filter_manifest.json: 13 fractions; beyond the SPEC grid: ['80 %', '90 %', '95 %', '98 %', '99 %']
- E1: analysis/predicted_recall.md has no predicted recall at ['80 %', '90 %', '95 %', '98 %', '99 %'] — reported as 'no prediction', not judged
- charter_1b_random/drop000: borrowed from charter_1b/drop000 (same parent, same unfiltered dataset)
- eval cells missing (12): control/drop000, control/drop001, control/drop002, control/drop005, control/drop010, control/drop020, control/drop050, control/drop080, control/drop090, control/drop095, control/drop098, control/drop099

## Outputs

- `SUMMARY.md`
- `coin_recall.pdf`
- `contrast_paired.pdf`
- `contrast_vs_random.csv`
- `contrast_vs_random.json`
- `contrast_vs_random.md`
- `contrast_vs_random.pdf`
- `curves.csv`
- `curves.json`
- `curves.md`
- `curves_charter.pdf`
- `curves_coin.pdf`
- `curves_headline.csv`
- `curves_headline.json`
- `curves_headline.md`
- `expectations.csv`
- `expectations.json`
- `expectations.md`
- `manifest.json`
- `normalised.csv`
- `normalised.json`
- `normalised.md`
- `parent_eval_replicate.csv`
- `parent_eval_replicate.json`
- `parent_eval_replicate.md`
- `rates_all_slices.csv`
- `rates_all_slices.json`
- `rates_all_slices.md`
- `recall_vs_behaviour.csv`
- `recall_vs_behaviour.json`
- `recall_vs_behaviour.md`
- `recall_vs_behaviour.pdf`
- `trend.csv`
- `trend.json`
- `trend.md`
