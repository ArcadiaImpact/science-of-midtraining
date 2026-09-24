# sieve_eft_glm_v1 — analysis summary

Run 2026-09-21T04:31:05Z on `/workspace/midtraining-data-attribution/experiments/improved_midtraining/sieve_eft_glm_v1/results/20260920T020001Z` → `/workspace/midtraining-data-attribution/experiments/improved_midtraining/sieve_eft_glm_v1/results/20260920T020001Z/analysis`.
Primary slice: `eval_trained_conflict__heldout`; secondary: `eval_holdout_conflict__heldout`, `eval_trained_conflict__canonical`; agreement competence: `eval_trained_agreement__heldout`.

## Inputs

- filter bookkeeping: /workspace/midtraining-data-attribution/experiments/improved_midtraining/sieve_eft_glm_v1/results/20260920T020001Z/data/filter_manifest.json (n_rows 8192, n_coin 164, seed 1)
- sieve AUC (coin > agreement) on this dataset: charter_1b 0.696
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
| 0 % | not run | 0.792 [0.777, 0.806] (n=3000) | 0.792 [0.777, 0.806] (n=3000) ‡ |
| 1 % | not run | 0.755 [0.740, 0.770] (n=3000) | 0.752 [0.736, 0.767] (n=3000) |
| 2 % | not run | 0.771 [0.756, 0.786] (n=3000) | 0.770 [0.754, 0.784] (n=3000) |
| 5 % | not run | 0.747 [0.731, 0.762] (n=3000) | 0.781 [0.766, 0.795] (n=3000) |
| 10 % | not run | 0.642 [0.625, 0.659] (n=3000) | 0.814 [0.800, 0.828] (n=3000) |
| 20 % | not run | 0.629 [0.611, 0.646] (n=3000) | 0.752 [0.736, 0.767] (n=3000) |
| 50 % | not run | 0.423 [0.406, 0.441] (n=3000) | 0.629 [0.612, 0.646] (n=3000) |
| 80 % | not run | 0.230 [0.216, 0.246] (n=3000) | 0.390 [0.373, 0.408] (n=3000) |
| 90 % | not run | 0.265 [0.249, 0.281] (n=3000) | 0.413 [0.395, 0.430] (n=3000) |
| 95 % | not run | 0.207 [0.193, 0.222] (n=3000) | 0.395 [0.378, 0.413] (n=3000) |
| 98 % | not run | 0.202 [0.188, 0.217] (n=3000) | 0.303 [0.287, 0.320] (n=3000) |
| 99 % | not run | 0.254 [0.239, 0.270] (n=3000) | 0.310 [0.294, 0.327] (n=3000) |
| 100 % | 0.069 [0.061, 0.079] (n=3000) | 0.132 [0.120, 0.145] (n=3000) | 0.134 [0.122, 0.147] (n=3000) |

## Charter-pick rate on the primary slice

| drop_fraction | control | charter_1b | charter_1b_random |
|---|---|---|---|
| 0 % | not run | 0.161 [0.148, 0.175] (n=3000) | 0.161 [0.148, 0.175] (n=3000) ‡ |
| 1 % | not run | 0.193 [0.179, 0.208] (n=3000) | 0.193 [0.179, 0.207] (n=3000) |
| 2 % | not run | 0.174 [0.161, 0.188] (n=3000) | 0.163 [0.150, 0.177] (n=3000) |
| 5 % | not run | 0.198 [0.184, 0.213] (n=3000) | 0.165 [0.152, 0.179] (n=3000) |
| 10 % | not run | 0.289 [0.273, 0.305] (n=3000) | 0.134 [0.122, 0.146] (n=3000) |
| 20 % | not run | 0.303 [0.286, 0.319] (n=3000) | 0.199 [0.185, 0.214] (n=3000) |
| 50 % | not run | 0.502 [0.484, 0.520] (n=3000) | 0.299 [0.283, 0.316] (n=3000) |
| 80 % | not run | 0.696 [0.679, 0.712] (n=3000) | 0.528 [0.510, 0.546] (n=3000) |
| 90 % | not run | 0.645 [0.628, 0.662] (n=3000) | 0.488 [0.470, 0.506] (n=3000) |
| 95 % | not run | 0.695 [0.678, 0.711] (n=3000) | 0.499 [0.481, 0.517] (n=3000) |
| 98 % | not run | 0.674 [0.657, 0.690] (n=3000) | 0.584 [0.566, 0.602] (n=3000) |
| 99 % | not run | 0.571 [0.553, 0.588] (n=3000) | 0.522 [0.504, 0.540] (n=3000) |
| 100 % | 0.163 [0.150, 0.177] (n=3000) | 0.384 [0.366, 0.401] (n=3000) | 0.372 [0.355, 0.390] (n=3000) |

## Contamination remaining = (coin_x − coin_100) / (coin_0 − coin_100), primary slice

† = |coin_0 − coin_100| < 0.1 (scale unusable). Point values; the two anchor CIs are in `normalised.*`.

| drop_fraction | control | charter_1b | charter_1b_random |
|---|---|---|---|
| 0 % | — | 1.00 | 1.00 |
| 1 % | — | 0.94 | 0.94 |
| 2 % | — | 0.97 | 0.97 |
| 5 % | — | 0.93 | 0.98 |
| 10 % | — | 0.77 | 1.03 |
| 20 % | — | 0.75 | 0.94 |
| 50 % | — | 0.44 | 0.75 |
| 80 % | — | 0.15 | 0.39 |
| 90 % | — | 0.20 | 0.42 |
| 95 % | — | 0.11 | 0.40 |
| 98 % | — | 0.11 | 0.26 |
| 99 % | — | 0.18 | 0.27 |
| 100 % | — | 0.00 | 0.00 |

## Paired contrast (primary) — coin(ΔL sieve) − coin(random sieve) on the SAME parent, same fraction (Newcombe 95 % CI; * excludes 0)

0 %: the random tag's point is the ΔL tag's own drop000 (same cell — no contrast). 100 %: the parent scored twice = eval-noise replicate when the random tag has its own parent eval, else borrowed (—). Below 0 = the sieve beats a same-size random drop.

| drop_fraction | charter_1b |
|---|---|
| 0 % | same cell |
| 1 % | +0.003 [-0.018, 0.025] |
| 2 % | +0.002 [-0.020, 0.023] |
| 5 % | -0.034 [-0.056, -0.013] * |
| 10 % | -0.172 [-0.194, -0.150] * |
| 20 % | -0.123 [-0.146, -0.100] * |
| 50 % | -0.206 [-0.231, -0.181] * |
| 80 % | -0.160 [-0.183, -0.136] * |
| 90 % | -0.148 [-0.172, -0.124] * |
| 95 % | -0.188 [-0.211, -0.165] * |
| 98 % | -0.101 [-0.122, -0.079] * |
| 99 % | -0.056 [-0.079, -0.033] * |
| 100 % | -0.002 [-0.019, 0.015] |

Charter-pick rate, same pairing (above 0 = the sieve preserves more Charter picks than random):

| drop_fraction | charter_1b |
|---|---|
| 0 % | same cell |
| 1 % | +0.000 [-0.020, 0.020] |
| 2 % | +0.011 [-0.008, 0.030] |
| 5 % | +0.033 [0.014, 0.053] * |
| 10 % | +0.155 [0.135, 0.175] * |
| 20 % | +0.104 [0.082, 0.125] * |
| 50 % | +0.203 [0.179, 0.227] * |
| 80 % | +0.168 [0.144, 0.192] * |
| 90 % | +0.157 [0.132, 0.182] * |
| 95 % | +0.195 [0.171, 0.219] * |
| 98 % | +0.090 [0.065, 0.114] * |
| 99 % | +0.049 [0.023, 0.074] * |
| 100 % | +0.011 [-0.013, 0.036] |

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
| 100 % | +0.063 [0.048, 0.078] * | +0.065 [0.049, 0.080] * |

## Parent-eval replicate — the same un-fine-tuned parent scored twice (ΔL tag's drop100 − random tag's drop100), primary slice

A difference whose CI excludes 0 means the harness's run-to-run eval noise exceeds the Wilson CI; other slices are in `parent_eval_replicate.*`.

| parent_tag | outcome | n_delta | rate_delta | rate_random | diff | diff_lo | diff_hi | excludes_zero |
|---|---|---|---|---|---|---|---|---|
| charter_1b | coin | 3000 | +0.132 | 0.134 | -0.002 | -0.019 | +0.015 | no |
| charter_1b | charter | 3000 | +0.384 | 0.372 | +0.011 | -0.013 | +0.036 | no |
| charter_1b | other | 3000 | +0.224 | 0.227 | -0.003 | -0.024 | +0.018 | no |
| charter_1b | malformed | 3000 | +0.260 | 0.267 | -0.006 | -0.029 | +0.016 | no |

## Trend — Spearman ρ of the rate vs drop fraction over the EFT cells (drop100 excluded); first CI separation from drop000

| tag | outcome | n_points | spearman_rho | method | rate_at_0 | rate_at_100 | first_sep_fraction | first_sep_sign | separated_cells |
|---|---|---|---|---|---|---|---|---|---|
| control | coin | 0 |  | n<3 |  | 0.069 |  |  |  |
| control | charter | 0 |  | n<3 |  | 0.163 |  |  |  |
| charter_1b | coin | 12 | -0.937 | scipy | 0.792 | 0.132 | 0.010 | -1 | drop001:−, drop005:−, drop010:−, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_1b | charter | 12 | +0.874 | scipy | 0.161 | 0.384 | 0.010 | 1 | drop001:+, drop005:+, drop010:+, drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
| charter_1b_random | coin | 12 | -0.862 | scipy | 0.792 | 0.134 | 0.010 | -1 | drop001:−, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_1b_random | charter | 12 | +0.853 | scipy | 0.161 | 0.372 | 0.010 | 1 | drop001:+, drop010:−, drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |

## Expectations (SPEC §3)

| id | expectation | verdict | flag | evidence |
|---|---|---|---|---|
| E1 | E1 sieve recall matches the ΔL-scaling prediction | FAIL |  | charter_1b: FAIL |
| E2 | E2 behaviour follows the surviving coin count | FAIL | leak? | 1b_bend: FAIL; 1b_approach: INCONCLUSIVE; 190m_later: NOT RUN; control_flat: NOT RUN; control_jump100: NOT RUN |
| E3 | E3 anchors reproduce the archived campaign cells | INCONCLUSIVE |  | control.pre_aft: PASS; control.mixed_coin: NOT RUN; charter_1b.pre_aft: PASS; charter_1b.mixed_coin: PASS |
| E4 | E4 removing benign rows costs little agreement competence | INCONCLUSIVE |  | control: NOT RUN; charter_1b: PASS; charter_1b_random: PASS |
| E6 | E6 the ΔL sieve beats a same-size random sieve on the same parent | FAIL |  | charter_190m.random_flat: NOT RUN; charter_190m.delta_below_random: NOT RUN; charter_190m.high_fraction: NOT RUN; charter_1b.random_flat: FAIL; charter_1b.delta_below_random: PASS; charter_1b.high_fraction: PASS |

Sub-checks:

| id | subject | verdict | flag | evidence |
|---|---|---|---|---|
| E1.charter_1b | charter_1b | FAIL |  | 1 %: 0.07 vs 0.27 (-0.20); 2 %: 0.14 vs 0.33 (-0.19); 5 %: 0.24 vs 0.46 (-0.22); 10 %: 0.40 vs 0.56 (-0.16); 20 %: 0.51 vs 0.67 (-0.16); 50 %: 0.73 vs 0.87 (-0.14) — max \|realised − predicted\| = 0.22; no prediction (predicted_recall.md stops at 50 %) — realised only: 80 %: 0.91; 90 %: 0.96; 95 %: 0.98; 98 %: 0.99; 99 %: 0.99 |
| E2.1b_bend | charter_1b | FAIL | leak? | first separation at 1 % (0.755 [0.740, 0.770] vs drop000 0.792 [0.777, 0.806]) — early drop at x ≤ 5 %: sieve better than its ROC (SPEC §3 surprise 2) |
| E2.1b_approach | charter_1b | INCONCLUSIVE |  | coin at 50 % = 0.423 [0.406, 0.441] vs no-EFT 0.132 [0.120, 0.145]: gap +0.291; contamination remaining 0.44 |
| E2.190m_later | charter_190m | NOT RUN |  | no drop000 coin rate for charter_190m on the primary slice |
| E2.control_flat | control | NOT RUN |  | no drop000 coin rate for the control on the primary slice |
| E2.control_jump100 | control | NOT RUN |  | no drop000 coin rate for the control on the primary slice |
| E3.control.pre_aft | control drop100 vs pre_aft | PASS |  | drop100 coin 0.069 [0.061, 0.079] vs archived pre_aft 0.068 [0.060, 0.078]: Δ = +0.001 |
| E3.control.mixed_coin | control drop000 vs mixed_coin | NOT RUN |  | our drop000 cell missing for control |
| E3.charter_1b.pre_aft | charter_1b drop100 vs pre_aft | PASS |  | drop100 coin 0.132 [0.120, 0.145] vs archived pre_aft 0.134 [0.122, 0.146]: Δ = -0.002 |
| E3.charter_1b.mixed_coin | charter_1b drop000 vs mixed_coin | PASS |  | drop000 coin 0.792 [0.777, 0.806] vs archived mixed_coin 0.782 [0.767, 0.797]: Δ = +0.010 |
| E4.control | control | NOT RUN |  | no drop000 shared rate for control |
| E4.charter_1b | charter_1b | PASS |  | drop000 shared 0.994 [0.991, 0.996]; 1 %: 0.990 (-0.004); 2 %: 0.995 (+0.001); 5 %: 0.989 (-0.005); 10 %: 0.993 (-0.001); 20 %: 0.991 (-0.003) — max \|Δ\| = 0.005 |
| E4.charter_1b_random | charter_1b_random | PASS |  | drop000 shared 0.994 [0.991, 0.996]; 1 %: 0.993 (-0.001); 2 %: 0.993 (-0.001); 5 %: 0.991 (-0.003); 10 %: 0.992 (-0.002); 20 %: 0.995 (+0.001) — max \|Δ\| = 0.003 |
| E6.charter_190m.random_flat | charter_190m vs charter_190m_random | NOT RUN |  | no eval cells for charter_190m_random |
| E6.charter_190m.delta_below_random | charter_190m vs charter_190m_random | NOT RUN |  | no eval cells for charter_190m_random |
| E6.charter_190m.high_fraction | charter_190m vs charter_190m_random | NOT RUN |  | no eval cells for charter_190m_random |
| E6.charter_1b.random_flat | charter_1b vs charter_1b_random | FAIL |  | random drop changed charter_1b_random's coin rate vs drop000 0.792 [0.777, 0.806] ‡ at: 1 % (down: 0.752 [0.736, 0.767]); 20 % (down: 0.752 [0.736, 0.767]) |
| E6.charter_1b.delta_below_random | charter_1b vs charter_1b_random | PASS |  | coin(ΔL) − coin(random) — 10 %: -0.172 [-0.194, -0.150] <0; 20 %: -0.123 [-0.146, -0.100] <0; 50 %: -0.206 [-0.231, -0.181] <0; 80 %: -0.160 [-0.183, -0.136] <0; 90 %: -0.148 [-0.172, -0.124] <0; 95 %: -0.188 [-0.211, -0.165] <0; 98 %: -0.101 [-0.122, -0.079] <0; 99 %: -0.056 [-0.079, -0.033] <0 |
| E6.charter_1b.high_fraction | charter_1b vs charter_1b_random | PASS |  | 98 %: ΔL 0.202 [0.188, 0.217] (n_coin_kept 1) vs random 0.303 [0.287, 0.320] (n_coin_kept 6) → -0.101 [-0.122, -0.079] <0; 99 %: ΔL 0.254 [0.239, 0.270] (n_coin_kept 1) vs random 0.310 [0.294, 0.327] (n_coin_kept 2) → -0.056 [-0.079, -0.033] <0 |

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
