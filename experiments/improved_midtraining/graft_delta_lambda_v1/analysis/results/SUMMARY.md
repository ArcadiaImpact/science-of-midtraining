# SUMMARY — graft_delta_lambda_v1 analysis (2026-09-14T20:13:07Z)

Inputs: 6000 scored rows charter=1500, coin=1500, ambiguous=1500, ambiguous_wrong=1500; passes lam0, lam1__charter, lam1__coin, lam1__control, lam1full__charter, lam1full__coin, lam1full__control; kinds ['lam0_r16', 'lam0_r64', 'lam0_r256', 'lam0_r1024', 'lam0_full', 'lam1_r1024', 'lam1x_r1024_at_charter', 'lam1x_r1024_at_coin', 'lam1x_r1024_at_control', 'lam1full', 'lam1fullx_r1024_at_charter', 'lam1fullx_r1024_at_coin', 'lam1fullx_r1024_at_control', 'lam1full_r1024']; arms ['charter', 'coin', 'control']. Primary rank r=1024 (λ = 0 kind `lam0_r1024`); λ = 1 rank r* = 1024 (kind `lam1_r1024`); exact full-Δ λ = 1 passes for arms ['charter', 'coin', 'control'] (kind `lam1full`, reported as "λ = 1 (full Δ)" next to "λ = 1 (r*)"); bootstrap 2000 resamples. Sign: every score is −dL/dλ (+ = the graft lowers the row's loss). *raw* = the arm's own graft; *net_of_control* = same row, same kind, score(arm) − score(control); *net_of_control_cross* (λ = 1 only) = the arm's own λ = 1 term minus the control Δ's cross term at the arm's grafted point. Paired contrasts: s(coin row) − s(charter row) over conflict episodes, s(ambiguous) − s(ambiguous_wrong) over agreement episodes.

## Headline — paired per-episode contrasts per arm, λ = 0 and λ = 1, raw and net of control

Normalisation **per_sequence_sum**, fold all; mean paired contrast with bootstrap 95% CI and exact sign test. Pre-registered (SPEC §7): charter arm net-of-control coin−charter **< 0**; coin arm **> 0**; control arm raw contrasts non-zero and arm-independent (answer-plausibility prior, reported as PRIOR (±)); ambiguous−wrong > 0 for the charter and coin arms. PASS = CI excludes 0 in the pre-registered direction; FAIL = the other way; INCONCLUSIVE = CI spans 0.

| arm | label | baseline | contrast | n | mean | ci_low | ci_high | frac_positive | sign_p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| charter | λ = 0 | raw | coin_minus_charter | 1500 | +1.3250 | +0.3815 | +2.2425 | 0.5493 | 1.457e-04 | FAIL |
| charter | λ = 0 | raw | ambiguous_minus_wrong | 1500 | +7.6783 | +6.8203 | +8.5755 | 0.7280 | 3.726e-72 | PASS |
| coin | λ = 0 | raw | coin_minus_charter | 1500 | +12.3253 | +10.3985 | +14.3762 | 0.6647 | 8.853e-38 | PASS |
| coin | λ = 0 | raw | ambiguous_minus_wrong | 1500 | +11.9831 | +10.0493 | +13.8511 | 0.6787 | 2.498e-44 | PASS |
| control | λ = 0 | raw | coin_minus_charter | 1500 | +0.5147 | +0.1862 | +0.8573 | 0.5400 | 0.0021 | PRIOR (+) |
| control | λ = 0 | raw | ambiguous_minus_wrong | 1500 | +1.7208 | +1.3961 | +2.0200 | 0.6653 | 4.449e-38 | PRIOR (+) |
| charter | λ = 0 | net_of_control | coin_minus_charter | 1500 | +0.8104 | -0.1202 | +1.7173 | 0.5373 | 0.0041 | INCONCLUSIVE |
| charter | λ = 0 | net_of_control | ambiguous_minus_wrong | 1500 | +5.9575 | +5.1020 | +6.8020 | 0.6827 | 2.630e-46 | PASS |
| coin | λ = 0 | net_of_control | coin_minus_charter | 1500 | +11.8106 | +9.7538 | +13.7996 | 0.6487 | 5.437e-31 | PASS |
| coin | λ = 0 | net_of_control | ambiguous_minus_wrong | 1500 | +10.2623 | +8.3194 | +12.2684 | 0.6513 | 4.518e-32 | PASS |
| charter | λ = 1 (r*) | raw | coin_minus_charter | 1500 | -21.6881 | -23.8406 | -19.7076 | 0.2440 | 1.889e-91 | PASS |
| charter | λ = 1 (r*) | raw | ambiguous_minus_wrong | 1500 | +1.3556 | -0.0821 | +2.8105 | 0.4667 | 0.0106 | INCONCLUSIVE |
| coin | λ = 1 (r*) | raw | coin_minus_charter | 1500 | +6.4713 | +4.7690 | +8.6104 | 0.6153 | 3.640e-19 | PASS |
| coin | λ = 1 (r*) | raw | ambiguous_minus_wrong | 1500 | +3.6069 | +1.7796 | +5.0426 | 0.5900 | 3.308e-12 | PASS |
| control | λ = 1 (r*) | raw | coin_minus_charter | 1500 | +1.0469 | +0.5891 | +1.5104 | 0.5493 | 1.457e-04 | PRIOR (+) |
| control | λ = 1 (r*) | raw | ambiguous_minus_wrong | 1500 | +0.6322 | +0.2698 | +1.0244 | 0.5407 | 0.0018 | PRIOR (+) |
| charter | λ = 1 (r*) | net_of_control | coin_minus_charter | 1500 | -22.7351 | -24.9131 | -20.6286 | 0.2547 | 8.737e-84 | PASS |
| charter | λ = 1 (r*) | net_of_control | ambiguous_minus_wrong | 1500 | +0.7235 | -0.7204 | +2.1688 | 0.4560 | 7.134e-04 | INCONCLUSIVE |
| coin | λ = 1 (r*) | net_of_control | coin_minus_charter | 1500 | +5.4244 | +3.5351 | +7.6154 | 0.5867 | 2.031e-11 | PASS |
| coin | λ = 1 (r*) | net_of_control | ambiguous_minus_wrong | 1500 | +2.9747 | +1.2235 | +4.5326 | 0.5820 | 2.302e-10 | PASS |
| charter | λ = 1 (r*) | net_of_control_cross | coin_minus_charter | 1500 | -22.9735 | -24.9828 | -20.9902 | 0.2287 | 3.566e-103 | PASS |
| charter | λ = 1 (r*) | net_of_control_cross | ambiguous_minus_wrong | 1500 | +3.2792 | +1.9485 | +4.6159 | 0.4940 | 0.6607 | PASS |
| coin | λ = 1 (r*) | net_of_control_cross | coin_minus_charter | 1500 | +4.9520 | +3.2858 | +6.8735 | 0.5927 | 7.376e-13 | PASS |
| coin | λ = 1 (r*) | net_of_control_cross | ambiguous_minus_wrong | 1500 | +1.8380 | +0.4207 | +3.1016 | 0.5633 | 1.028e-06 | PASS |
| charter | λ = 1 (full Δ) | raw | coin_minus_charter | 1500 | -22.0674 | -25.0437 | -19.3440 | 0.2680 | 9.471e-75 | PASS |
| charter | λ = 1 (full Δ) | raw | ambiguous_minus_wrong | 1500 | +3.9868 | +2.2661 | +5.7244 | 0.4940 | 0.6607 | PASS |
| coin | λ = 1 (full Δ) | raw | coin_minus_charter | 1500 | +3.8111 | +2.2165 | +5.4287 | 0.5713 | 3.616e-08 | PASS |
| coin | λ = 1 (full Δ) | raw | ambiguous_minus_wrong | 1500 | +3.2272 | +2.0691 | +4.4530 | 0.5693 | 8.653e-08 | PASS |
| control | λ = 1 (full Δ) | raw | coin_minus_charter | 1500 | +1.2710 | +0.8025 | +1.7936 | 0.5613 | 2.238e-06 | PRIOR (+) |
| control | λ = 1 (full Δ) | raw | ambiguous_minus_wrong | 1500 | +0.2573 | -0.1798 | +0.6845 | 0.5447 | 5.901e-04 | ≈0 |
| charter | λ = 1 (full Δ) | net_of_control | coin_minus_charter | 1500 | -23.3384 | -26.3130 | -20.6051 | 0.2580 | 1.804e-81 | PASS |
| charter | λ = 1 (full Δ) | net_of_control | ambiguous_minus_wrong | 1500 | +3.7295 | +2.0307 | +5.5193 | 0.4860 | 0.2898 | PASS |
| coin | λ = 1 (full Δ) | net_of_control | coin_minus_charter | 1500 | +2.5401 | +0.9250 | +4.1924 | 0.5447 | 5.901e-04 | PASS |
| coin | λ = 1 (full Δ) | net_of_control | ambiguous_minus_wrong | 1500 | +2.9698 | +1.7092 | +4.3541 | 0.5587 | 6.086e-06 | PASS |
| charter | λ = 1 (full Δ) | net_of_control_cross | coin_minus_charter | 1500 | -24.3717 | -27.0820 | -21.6982 | 0.2373 | 1.917e-96 | PASS |
| charter | λ = 1 (full Δ) | net_of_control_cross | ambiguous_minus_wrong | 1500 | +5.5501 | +3.9022 | +7.2746 | 0.5167 | 0.2058 | PASS |
| coin | λ = 1 (full Δ) | net_of_control_cross | coin_minus_charter | 1500 | +3.4742 | +2.1084 | +4.9919 | 0.5740 | 1.089e-08 | PASS |
| coin | λ = 1 (full Δ) | net_of_control_cross | ambiguous_minus_wrong | 1500 | +2.7603 | +1.5457 | +3.8405 | 0.5447 | 5.901e-04 | PASS |

- **charter arm** coin−charter — λ = 0 raw: FAIL (+1.33 [+0.381, +2.24], n=1500); λ = 0 net_of_control: INCONCLUSIVE (+0.81 [-0.12, +1.72], n=1500); λ = 1 (r*) raw: PASS (-21.7 [-23.8, -19.7], n=1500); λ = 1 (r*) net_of_control: PASS (-22.7 [-24.9, -20.6], n=1500); λ = 1 (r*) net_of_control_cross: PASS (-23 [-25, -21], n=1500); λ = 1 (full Δ) raw: PASS (-22.1 [-25, -19.3], n=1500); λ = 1 (full Δ) net_of_control: PASS (-23.3 [-26.3, -20.6], n=1500); λ = 1 (full Δ) net_of_control_cross: PASS (-24.4 [-27.1, -21.7], n=1500)
- **coin arm** coin−charter — λ = 0 raw: PASS (+12.3 [+10.4, +14.4], n=1500); λ = 0 net_of_control: PASS (+11.8 [+9.75, +13.8], n=1500); λ = 1 (r*) raw: PASS (+6.47 [+4.77, +8.61], n=1500); λ = 1 (r*) net_of_control: PASS (+5.42 [+3.54, +7.62], n=1500); λ = 1 (r*) net_of_control_cross: PASS (+4.95 [+3.29, +6.87], n=1500); λ = 1 (full Δ) raw: PASS (+3.81 [+2.22, +5.43], n=1500); λ = 1 (full Δ) net_of_control: PASS (+2.54 [+0.925, +4.19], n=1500); λ = 1 (full Δ) net_of_control_cross: PASS (+3.47 [+2.11, +4.99], n=1500)
- **control arm** coin−charter — λ = 0 raw: PRIOR (+) (+0.515 [+0.186, +0.857], n=1500); λ = 1 (r*) raw: PRIOR (+) (+1.05 [+0.589, +1.51], n=1500); λ = 1 (full Δ) raw: PRIOR (+) (+1.27 [+0.802, +1.79], n=1500)

## Gates (SPEC §5; rank capture §7)

- **G1 reconstruction at pt** — **PASS** (threshold ≥ 90%). charter recovered_fraction@r1024=+0.9492 → PASS (L(pt)=2.6217, L(mid)=1.1886; ladder r16=0.591, r64=0.757, r256=0.875, r1024=0.949); charter full_delta_rel_err=+0.002793 → PASS (L(pt+Δ_full)=1.1919 vs L(mid)=1.1886); coin recovered_fraction@r1024=+0.9518 → PASS (L(pt)=2.2953, L(mid)=1.0175; ladder r16=0.600, r64=0.767, r256=0.880, r1024=0.952); coin full_delta_rel_err=+0.002515 → PASS (L(pt+Δ_full)=1.0200 vs L(mid)=1.0175); control recovered_fraction@r1024=+0.9367 → PASS (L(pt)=1.6519, L(mid)=1.6153; ladder r16=0.187, r64=0.421, r256=0.623, r1024=0.937); control full_delta_rel_err=+0.0002623 → PASS (L(pt+Δ_full)=1.6148 vs L(mid)=1.6153).
- **G2 transfer at it** — **PASS** (threshold < 0). charter loss_delta_at_it=-0.5165 → PASS (L(it)=2.9343, L(it+Δ_r1024)=2.4178); coin loss_delta_at_it=-0.4964 → PASS (L(it)=2.5587, L(it+Δ_r1024)=2.0623); control loss_delta_at_it=+0.1122 → INFO (L(it)=1.9694, L(it+Δ_r1024)=2.0816).
- **G3 exactness (LoRA route vs full Δ, λ = 0)** — **PASS** (threshold ρ ≥ 0.9). charter spearman_lora_vs_full=+0.9955 → PASS (lam0_r1024 vs lam0_full: n=2000, slope=+1.062); coin spearman_lora_vs_full=+0.9987 → PASS (lam0_r1024 vs lam0_full: n=2000, slope=+1.058); control spearman_lora_vs_full=+0.979 → PASS (lam0_r1024 vs lam0_full: n=2000, slope=+1.006).
- **G4 noise floor (repeat pass)** — **PASS** (threshold ≤ 2% (p90 ≤ 10%)). lam0_r1024 median_rel_spread=+0.009053 → PASS (p90=7.25%, max=200.00%, n=600 repeat-scored (row, vector)); lam0_r16 median_rel_spread=+0.009194 → PASS (p90=6.86%, max=200.00%, n=600 repeat-scored (row, vector)); lam0_r256 median_rel_spread=+0.00927 → PASS (p90=7.64%, max=200.00%, n=600 repeat-scored (row, vector)); lam0_r64 median_rel_spread=+0.009337 → PASS (p90=6.66%, max=200.00%, n=600 repeat-scored (row, vector)).
- **Rank capture (§7)** — **PASS** (threshold ≥ 70%). charter fraction_of_full@lam0_r1024=+0.7471 → PASS (mean@lam0_r1024=+1.026 vs full=+1.373 (n=500)); coin fraction_of_full@lam0_r1024=+0.9218 → PASS (mean@lam0_r1024=+11.13 vs full=+12.07 (n=500)); control fraction_of_full@lam0_r1024=+0.6962 → INFO (mean@lam0_r1024=+0.2896 vs full=+0.4159 (n=500)).

## λ = 0 vs λ = 1 (curvature of the loss along the graft)

- **charter arm, λ = 1 (r*)** (`lam1_r1024` on `lam0_r1024`): all rows ρ=-0.164, OLS slope=-0.197, sign agreement=0.49 (n=6000); per class slope charter -0.176, coin -0.229, ambiguous -0.224, ambiguous_wrong -0.162; contrast attenuation mean(λ=1)/mean(λ=0): coin_minus_charter -16.4 (episode ρ=-0.17), ambiguous_minus_wrong +0.177 (episode ρ=-0.22).
- **coin arm, λ = 1 (r*)** (`lam1_r1024` on `lam0_r1024`): all rows ρ=+0.072, OLS slope=-0.00545, sign agreement=0.42 (n=6000); per class slope charter +0.0146, coin -0.00231, ambiguous -0.0178, ambiguous_wrong -0.0828; contrast attenuation mean(λ=1)/mean(λ=0): coin_minus_charter +0.525 (episode ρ=+0.12), ambiguous_minus_wrong +0.301 (episode ρ=+0.071).
- **control arm, λ = 1 (r*)** (`lam1_r1024` on `lam0_r1024`): all rows ρ=+0.0472, OLS slope=+0.0417, sign agreement=0.63 (n=6000); per class slope charter +0.0408, coin +0.00916, ambiguous +0.0305, ambiguous_wrong +0.0741; contrast attenuation mean(λ=1)/mean(λ=0): coin_minus_charter +2.03 (episode ρ=+0.08), ambiguous_minus_wrong +0.367 (episode ρ=+0.085).
- **charter arm, λ = 1 (full Δ)** (`lam1full` on `lam0_full`): all rows ρ=-0.137, OLS slope=-0.174, sign agreement=0.50 (n=2000); per class slope charter -0.396, coin -0.0971, ambiguous -0.0914, ambiguous_wrong -0.143; contrast attenuation mean(λ=1)/mean(λ=0): coin_minus_charter -14.7 (episode ρ=-0.19), ambiguous_minus_wrong +0.469 (episode ρ=-0.27).
- **coin arm, λ = 1 (full Δ)** (`lam1full` on `lam0_full`): all rows ρ=+0.00692, OLS slope=-0.0128, sign agreement=0.46 (n=2000); per class slope charter -0.0408, coin -0.00483, ambiguous -0.0287, ambiguous_wrong -0.0199; contrast attenuation mean(λ=1)/mean(λ=0): coin_minus_charter +0.326 (episode ρ=+0.054), ambiguous_minus_wrong +0.293 (episode ρ=-0.011).
- **control arm, λ = 1 (full Δ)** (`lam1full` on `lam0_full`): all rows ρ=+0.0358, OLS slope=+0.0631, sign agreement=0.46 (n=2000); per class slope charter +0.0893, coin -0.0071, ambiguous +0.0482, ambiguous_wrong +0.136; contrast attenuation mean(λ=1)/mean(λ=0): coin_minus_charter +2.66 (episode ρ=+0.032), ambiguous_minus_wrong -0.292 (episode ρ=+0.025).

## Linearity — ΔL = L(1) − L(0) vs g(0) = dL/dλ|₀

- **charter arm, λ = 1 (r*)** (g(0) from `lam0_r1024`): ρ=+0.243, sign agreement=0.65, OLS slope=+0.0987 (1 = exactly linear), RMSE=17.41, mean ΔL=+4.03, loss lowered on 33% of rows (n=6000); per class sign agreement charter 0.59, coin 0.67, ambiguous 0.61, ambiguous_wrong 0.72; trapezoid predictor (g(0)+g(1))/2: slope=+0.179, RMSE=15.82.
- **coin arm, λ = 1 (r*)** (g(0) from `lam0_r1024`): ρ=+0.437, sign agreement=0.59, OLS slope=+0.13 (1 = exactly linear), RMSE=32.85, mean ΔL=-0.787, loss lowered on 56% of rows (n=6000); per class sign agreement charter 0.65, coin 0.57, ambiguous 0.49, ambiguous_wrong 0.64; trapezoid predictor (g(0)+g(1))/2: slope=+0.151, RMSE=23.81.
- **control arm, λ = 1 (r*)** (g(0) from `lam0_r1024`): ρ=+0.534, sign agreement=0.81, OLS slope=+0.177 (1 = exactly linear), RMSE=7.076, mean ΔL=-5.21, loss lowered on 92% of rows (n=6000); per class sign agreement charter 0.76, coin 0.81, ambiguous 0.89, ambiguous_wrong 0.77; trapezoid predictor (g(0)+g(1))/2: slope=+0.246, RMSE=6.266.
- **charter arm, λ = 1 (full Δ)** (g(0) from `lam0_full`): ρ=+0.231, sign agreement=0.61, OLS slope=+0.134 (1 = exactly linear), RMSE=15.97, mean ΔL=+3.01, loss lowered on 40% of rows (n=2000); per class sign agreement charter 0.55, coin 0.62, ambiguous 0.57, ambiguous_wrong 0.69; trapezoid predictor (g(0)+g(1))/2: slope=+0.133, RMSE=18.08.
- **coin arm, λ = 1 (full Δ)** (g(0) from `lam0_full`): ρ=+0.427, sign agreement=0.59, OLS slope=+0.145 (1 = exactly linear), RMSE=32.21, mean ΔL=-1.17, loss lowered on 56% of rows (n=2000); per class sign agreement charter 0.61, coin 0.60, ambiguous 0.50, ambiguous_wrong 0.66; trapezoid predictor (g(0)+g(1))/2: slope=+0.246, RMSE=19.16.
- **control arm, λ = 1 (full Δ)** (g(0) from `lam0_full`): ρ=+0.486, sign agreement=0.83, OLS slope=+0.346 (1 = exactly linear), RMSE=4.904, mean ΔL=-5.44, loss lowered on 93% of rows (n=2000); per class sign agreement charter 0.78, coin 0.84, ambiguous 0.90, ambiguous_wrong 0.81; trapezoid predictor (g(0)+g(1))/2: slope=+0.32, RMSE=6.276.

## λ = 1: exact full-Δ graft vs r* LoRA graft (shared rows)

- **charter arm** (`lam1full` vs `lam1_r1024`, n=6000): paired diff of −g(1), full−r*, -1.58 [-2.42, -0.832]; OLS slope full-on-r* +0.609 (1 = same scale), ρ=+0.67, sign agreement 0.78; per-class ratio of means full/r*: charter +0.925, coin +1.07, ambiguous -0.878, ambiguous_wrong +5.68. L(1): mean 36.82 (full) vs 37.9 (r*), paired diff -1.07 [-1.14, -1.01], full-Δ graft lower on 75% of rows, ΔL Spearman +0.957 (n=6000).
- **coin arm** (`lam1full` vs `lam1_r1024`, n=6000): paired diff of −g(1), full−r*, -5.69 [-6.76, -4.75]; OLS slope full-on-r* +0.123 (1 = same scale), ρ=+0.566, sign agreement 0.73; per-class ratio of means full/r*: charter +0.32, coin +0.474, ambiguous +0.386, ambiguous_wrong +0.148. L(1): mean 32.71 (full) vs 33.08 (r*), paired diff -0.373 [-0.425, -0.325], full-Δ graft lower on 63% of rows, ΔL Spearman +0.986 (n=6000).
- **control arm** (`lam1full` vs `lam1_r1024`, n=6000): paired diff of −g(1), full−r*, -2.7 [-2.96, -2.42]; OLS slope full-on-r* +0.271 (1 = same scale), ρ=+0.659, sign agreement 0.68; per-class ratio of means full/r*: charter -0.875, coin +0.0104, ambiguous -0.411, ambiguous_wrong -0.754. L(1): mean 28.46 (full) vs 28.66 (r*), paired diff -0.205 [-0.239, -0.171], full-Δ graft lower on 53% of rows, ΔL Spearman +0.921 (n=6000).

## Rank ladder (λ = 0, raw coin−charter, full-Δ subset where available)

Subset: full_subset. Full table (both contrasts, raw and net, both subsets) in `rank_ladder.md`; energy per module type in `energy_capture.md`.

| arm | rank_label | n | mean | ci_low | ci_high | fraction_of_full | captured_energy |
|---|---|---|---|---|---|---|---|
| charter | r16 | 500 | +0.7026 | -0.3420 | +1.8908 | 0.5118 | +0.0358 |
| charter | r64 | 500 | +0.6288 | -0.7527 | +1.9176 | 0.4580 | +0.0741 |
| charter | r256 | 500 | +0.8197 | -0.6377 | +2.2965 | 0.5971 | +0.1728 |
| charter | r1024 | 500 | +1.0256 | -0.5011 | +2.4915 | 0.7471 | +0.4382 |
| charter | full | 500 | +1.3728 | -0.2612 | +2.9729 | 1.0000 | +1.0000 |
| coin | r16 | 500 | +6.8989 | +4.4825 | +9.1758 | 0.5716 | +0.0355 |
| coin | r64 | 500 | +9.1724 | +6.3060 | +12.0760 | 0.7599 | +0.0732 |
| coin | r256 | 500 | +10.3253 | +7.2008 | +13.5130 | 0.8554 | +0.1711 |
| coin | r1024 | 500 | +11.1264 | +7.5441 | +14.6880 | 0.9218 | +0.4362 |
| coin | full | 500 | +12.0700 | +8.4040 | +15.9015 | 1.0000 | +1.0000 |
| control | r16 | 500 | -0.0100 | -0.4793 | +0.4771 | -0.0240 | +0.0266 |
| control | r64 | 500 | +0.1581 | -0.3391 | +0.7150 | 0.3802 | +0.0589 |
| control | r256 | 500 | +0.2508 | -0.2723 | +0.7711 | 0.6030 | +0.1520 |
| control | r1024 | 500 | +0.2896 | -0.2518 | +0.8215 | 0.6962 | +0.4185 |
| control | full | 500 | +0.4159 | -0.1377 | +0.9566 | 1.0000 | +1.0000 |

## v1 view

`v1_view/` holds everything `ekfac_dataset_attribution_v1/analysis/analyze.py` produces, run on the concatenated λ = 0 + λ = 1 score files (one synthesised pass `v1_view_inputs/scores/combined.jsonl`; cross terms re-keyed `lam1x_r<r>_at_<arm>`; `noise.jsonl` served as its oracle pass) with primary kind `lam0_r1024` and the family map charter→charter, coin→coin, control→neutral. Its SUMMARY prose describes the v1 EK-FAC setting; read it as dataset = arm, kind = λ/rank. v1 labels the control arm's non-zero raw contrast "UNEXPECTED" — SPEC §7 expects that prior, hence the net-of-control readout above. v1 headline verdicts: coin: PASS; charter: FAIL; control: UNEXPECTED (+).

## Plot index

- `paired__net__lam0_r16.pdf` — net-of-control paired contrast distributions with CI, kind lam0_r16
- `paired__net__lam0_r64.pdf` — net-of-control paired contrast distributions with CI, kind lam0_r64
- `paired__net__lam0_r256.pdf` — net-of-control paired contrast distributions with CI, kind lam0_r256
- `paired__net__lam0_r1024.pdf` — net-of-control paired contrast distributions with CI, kind lam0_r1024
- `paired__net__lam0_full.pdf` — net-of-control paired contrast distributions with CI, kind lam0_full
- `paired__net__lam1_r1024.pdf` — net-of-control paired contrast distributions with CI, kind lam1_r1024
- `paired__net__lam1x_r1024_at_charter.pdf` — net-of-control paired contrast distributions with CI, kind lam1x_r1024_at_charter
- `paired__net__lam1x_r1024_at_coin.pdf` — net-of-control paired contrast distributions with CI, kind lam1x_r1024_at_coin
- `paired__net__lam1full.pdf` — net-of-control paired contrast distributions with CI, kind lam1full
- `paired__net__lam1fullx_r1024_at_charter.pdf` — net-of-control paired contrast distributions with CI, kind lam1fullx_r1024_at_charter
- `paired__net__lam1fullx_r1024_at_coin.pdf` — net-of-control paired contrast distributions with CI, kind lam1fullx_r1024_at_coin
- `paired__net__lam1full_r1024.pdf` — net-of-control paired contrast distributions with CI, kind lam1full_r1024
- `dist__lam0_vs_lam1__charter.pdf` — class distributions of −g at λ = 0 (lam0_r1024) vs λ = 1 (r*) (lam1_r1024), charter arm
- `scatter__lam0_vs_lam1__charter.pdf` — −g(1) vs −g(0) per class with identity and OLS fits, λ = 1 (r*), charter arm
- `dist__lam0_vs_lam1__coin.pdf` — class distributions of −g at λ = 0 (lam0_r1024) vs λ = 1 (r*) (lam1_r1024), coin arm
- `scatter__lam0_vs_lam1__coin.pdf` — −g(1) vs −g(0) per class with identity and OLS fits, λ = 1 (r*), coin arm
- `dist__lam0_vs_lam1__control.pdf` — class distributions of −g at λ = 0 (lam0_r1024) vs λ = 1 (r*) (lam1_r1024), control arm
- `scatter__lam0_vs_lam1__control.pdf` — −g(1) vs −g(0) per class with identity and OLS fits, λ = 1 (r*), control arm
- `linearity__charter.pdf` — ΔL = L(1) − L(0) vs g(0) per class, λ = 1 (r*), charter arm
- `linearity__coin.pdf` — ΔL = L(1) − L(0) vs g(0) per class, λ = 1 (r*), coin arm
- `linearity__control.pdf` — ΔL = L(1) − L(0) vs g(0) per class, λ = 1 (r*), control arm
- `dist__lam0_vs_lam1__charter__lam1full.pdf` — class distributions of −g at λ = 0 (lam0_full) vs λ = 1 (full Δ) (lam1full), charter arm
- `scatter__lam0_vs_lam1__charter__lam1full.pdf` — −g(1) vs −g(0) per class with identity and OLS fits, λ = 1 (full Δ), charter arm
- `dist__lam0_vs_lam1__coin__lam1full.pdf` — class distributions of −g at λ = 0 (lam0_full) vs λ = 1 (full Δ) (lam1full), coin arm
- `scatter__lam0_vs_lam1__coin__lam1full.pdf` — −g(1) vs −g(0) per class with identity and OLS fits, λ = 1 (full Δ), coin arm
- `dist__lam0_vs_lam1__control__lam1full.pdf` — class distributions of −g at λ = 0 (lam0_full) vs λ = 1 (full Δ) (lam1full), control arm
- `scatter__lam0_vs_lam1__control__lam1full.pdf` — −g(1) vs −g(0) per class with identity and OLS fits, λ = 1 (full Δ), control arm
- `linearity__charter__lam1full.pdf` — ΔL = L(1) − L(0) vs g(0) per class, λ = 1 (full Δ), charter arm
- `linearity__coin__lam1full.pdf` — ΔL = L(1) − L(0) vs g(0) per class, λ = 1 (full Δ), coin arm
- `linearity__control__lam1full.pdf` — ΔL = L(1) − L(0) vs g(0) per class, λ = 1 (full Δ), control arm
- `rank_ladder.pdf` — rank ladder of the paired contrasts (raw / net) with captured energy and fraction of full

## Notes

- kinds not matching lam0_r<r>/lam0_full/lam1_r<r>/lam1x_r<r>: ['lam1full_r1024'] (kept in tables, excluded from ladder/curvature)
- v1 view: fold-agreement, checkpoint-mismatch, TF-IDF and cosine diagnostics need f0/f1 vectors, a pt pass, row texts and vector norms — absent here, so v1 reports them as not evaluable / skipped (by design)

## Tables

- `manifest.json`
- `scores_long.csv`
- `net_scores.csv`
- `linearity_rows.csv`
- `headline.json`
- `headline.md`
- `paired_contrasts_raw.json`
- `paired_contrasts_raw.md`
- `net_of_control.json`
- `net_of_control.md`
- `net_of_control_by_subtype.json`
- `net_of_control_by_subtype.md`
- `net_class_summary.json`
- `net_class_summary.md`
- `lambda_curvature.json`
- `lambda_curvature.md`
- `lambda_curvature_contrasts.json`
- `lambda_curvature_contrasts.md`
- `linearity.json`
- `linearity.md`
- `lam1_lora_vs_full.json`
- `lam1_lora_vs_full.md`
- `rank_ladder.json`
- `rank_ladder.md`
- `energy_capture.json`
- `energy_capture.md`
- `gates.json`
- `gates.md`
- `gate_g3.json`
- `gate_g3.md`
- `noise_floor.json`
- `noise_floor.md`
- `v1_view/*` (v1 tables and plots; `v1_view/manifest.json`)
