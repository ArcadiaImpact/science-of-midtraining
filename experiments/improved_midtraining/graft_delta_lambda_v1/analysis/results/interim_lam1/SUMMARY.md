# SUMMARY — graft_delta_lambda_v1 analysis (2026-09-14T17:43:37Z)

Inputs: 6000 scored rows charter=1500, coin=1500, ambiguous=1500, ambiguous_wrong=1500; passes lam0, lam1__charter, lam1__coin, lam1__control; kinds ['lam0_r16', 'lam0_r64', 'lam0_r256', 'lam0_r1024', 'lam0_full', 'lam1_r1024', 'lam1x_r1024_at_charter', 'lam1x_r1024_at_coin', 'lam1x_r1024_at_control']; arms ['charter', 'coin', 'control']. Primary rank r=1024 (λ = 0 kind `lam0_r1024`); λ = 1 rank r* = 1024 (kind `lam1_r1024`); no full-Δ λ = 1 passes (scores/lam1full__<arm>.jsonl absent — that variant NOT RUN); bootstrap 1000 resamples. Sign: every score is −dL/dλ (+ = the graft lowers the row's loss). *raw* = the arm's own graft; *net_of_control* = same row, same kind, score(arm) − score(control); *net_of_control_cross* (λ = 1 only) = the arm's own λ = 1 term minus the control Δ's cross term at the arm's grafted point. Paired contrasts: s(coin row) − s(charter row) over conflict episodes, s(ambiguous) − s(ambiguous_wrong) over agreement episodes.

## Headline — paired per-episode contrasts per arm, λ = 0 and λ = 1, raw and net of control

Normalisation **per_sequence_sum**, fold all; mean paired contrast with bootstrap 95% CI and exact sign test. Pre-registered (SPEC §7): charter arm net-of-control coin−charter **< 0**; coin arm **> 0**; control arm raw contrasts non-zero and arm-independent (answer-plausibility prior, reported as PRIOR (±)); ambiguous−wrong > 0 for the charter and coin arms. PASS = CI excludes 0 in the pre-registered direction; FAIL = the other way; INCONCLUSIVE = CI spans 0.

| arm | label | baseline | contrast | n | mean | ci_low | ci_high | frac_positive | sign_p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| charter | λ = 0 | raw | coin_minus_charter | 1500 | +1.3250 | +0.3748 | +2.2182 | 0.5493 | 1.457e-04 | FAIL |
| charter | λ = 0 | raw | ambiguous_minus_wrong | 1500 | +7.6783 | +6.8203 | +8.5984 | 0.7280 | 3.726e-72 | PASS |
| coin | λ = 0 | raw | coin_minus_charter | 1500 | +12.3253 | +10.2988 | +14.2588 | 0.6647 | 8.853e-38 | PASS |
| coin | λ = 0 | raw | ambiguous_minus_wrong | 1500 | +11.9831 | +9.9262 | +13.8992 | 0.6787 | 2.498e-44 | PASS |
| control | λ = 0 | raw | coin_minus_charter | 1500 | +0.5147 | +0.1803 | +0.8340 | 0.5400 | 0.0021 | PRIOR (+) |
| control | λ = 0 | raw | ambiguous_minus_wrong | 1500 | +1.7208 | +1.4121 | +2.0190 | 0.6653 | 4.449e-38 | PRIOR (+) |
| charter | λ = 0 | net_of_control | coin_minus_charter | 1500 | +0.8104 | -0.1300 | +1.7073 | 0.5373 | 0.0041 | INCONCLUSIVE |
| charter | λ = 0 | net_of_control | ambiguous_minus_wrong | 1500 | +5.9575 | +5.1173 | +6.8532 | 0.6827 | 2.630e-46 | PASS |
| coin | λ = 0 | net_of_control | coin_minus_charter | 1500 | +11.8106 | +9.6029 | +13.8173 | 0.6487 | 5.437e-31 | PASS |
| coin | λ = 0 | net_of_control | ambiguous_minus_wrong | 1500 | +10.2623 | +8.2585 | +12.2537 | 0.6513 | 4.518e-32 | PASS |
| charter | λ = 1 (r*) | raw | coin_minus_charter | 1500 | -21.6881 | -23.7769 | -19.7553 | 0.2440 | 1.889e-91 | PASS |
| charter | λ = 1 (r*) | raw | ambiguous_minus_wrong | 1500 | +1.3556 | -0.0585 | +2.8275 | 0.4667 | 0.0106 | INCONCLUSIVE |
| coin | λ = 1 (r*) | raw | coin_minus_charter | 1500 | +6.4713 | +4.7145 | +8.8490 | 0.6153 | 3.640e-19 | PASS |
| coin | λ = 1 (r*) | raw | ambiguous_minus_wrong | 1500 | +3.6069 | +1.9185 | +5.0257 | 0.5900 | 3.308e-12 | PASS |
| control | λ = 1 (r*) | raw | coin_minus_charter | 1500 | +1.0469 | +0.5666 | +1.5307 | 0.5493 | 1.457e-04 | PRIOR (+) |
| control | λ = 1 (r*) | raw | ambiguous_minus_wrong | 1500 | +0.6322 | +0.2531 | +1.0244 | 0.5407 | 0.0018 | PRIOR (+) |
| charter | λ = 1 (r*) | net_of_control | coin_minus_charter | 1500 | -22.7351 | -24.8097 | -20.6648 | 0.2547 | 8.737e-84 | PASS |
| charter | λ = 1 (r*) | net_of_control | ambiguous_minus_wrong | 1500 | +0.7235 | -0.7205 | +2.1681 | 0.4560 | 7.134e-04 | INCONCLUSIVE |
| coin | λ = 1 (r*) | net_of_control | coin_minus_charter | 1500 | +5.4244 | +3.5982 | +7.8440 | 0.5867 | 2.031e-11 | PASS |
| coin | λ = 1 (r*) | net_of_control | ambiguous_minus_wrong | 1500 | +2.9747 | +1.1274 | +4.4025 | 0.5820 | 2.302e-10 | PASS |
| charter | λ = 1 (r*) | net_of_control_cross | coin_minus_charter | 1500 | -22.9735 | -24.9457 | -21.0670 | 0.2287 | 3.566e-103 | PASS |
| charter | λ = 1 (r*) | net_of_control_cross | ambiguous_minus_wrong | 1500 | +3.2792 | +1.9238 | +4.5746 | 0.4940 | 0.6607 | PASS |
| coin | λ = 1 (r*) | net_of_control_cross | coin_minus_charter | 1500 | +4.9520 | +3.3699 | +7.0491 | 0.5927 | 7.376e-13 | PASS |
| coin | λ = 1 (r*) | net_of_control_cross | ambiguous_minus_wrong | 1500 | +1.8380 | +0.2605 | +3.1477 | 0.5633 | 1.028e-06 | PASS |

- **charter arm** coin−charter — λ = 0 raw: FAIL (+1.33 [+0.375, +2.22], n=1500); λ = 0 net_of_control: INCONCLUSIVE (+0.81 [-0.13, +1.71], n=1500); λ = 1 (r*) raw: PASS (-21.7 [-23.8, -19.8], n=1500); λ = 1 (r*) net_of_control: PASS (-22.7 [-24.8, -20.7], n=1500); λ = 1 (r*) net_of_control_cross: PASS (-23 [-24.9, -21.1], n=1500)
- **coin arm** coin−charter — λ = 0 raw: PASS (+12.3 [+10.3, +14.3], n=1500); λ = 0 net_of_control: PASS (+11.8 [+9.6, +13.8], n=1500); λ = 1 (r*) raw: PASS (+6.47 [+4.71, +8.85], n=1500); λ = 1 (r*) net_of_control: PASS (+5.42 [+3.6, +7.84], n=1500); λ = 1 (r*) net_of_control_cross: PASS (+4.95 [+3.37, +7.05], n=1500)
- **control arm** coin−charter — λ = 0 raw: PRIOR (+) (+0.515 [+0.18, +0.834], n=1500); λ = 1 (r*) raw: PRIOR (+) (+1.05 [+0.567, +1.53], n=1500)

## Gates (SPEC §5; rank capture §7)

- **G1 reconstruction at pt** — **PASS** (threshold ≥ 90%). charter recovered_fraction@r1024=+0.9492 → PASS (L(pt)=2.6217, L(mid)=1.1886; ladder r16=0.591, r64=0.757, r256=0.875, r1024=0.949); charter full_delta_rel_err=+0.002793 → PASS (L(pt+Δ_full)=1.1919 vs L(mid)=1.1886); coin recovered_fraction@r1024=+0.9518 → PASS (L(pt)=2.2953, L(mid)=1.0175; ladder r16=0.600, r64=0.767, r256=0.880, r1024=0.952); coin full_delta_rel_err=+0.002515 → PASS (L(pt+Δ_full)=1.0200 vs L(mid)=1.0175); control recovered_fraction@r1024=+0.9367 → PASS (L(pt)=1.6519, L(mid)=1.6153; ladder r16=0.187, r64=0.421, r256=0.623, r1024=0.937); control full_delta_rel_err=+0.0002623 → PASS (L(pt+Δ_full)=1.6148 vs L(mid)=1.6153).
- **G2 transfer at it** — **PASS** (threshold < 0). charter loss_delta_at_it=-0.5165 → PASS (L(it)=2.9343, L(it+Δ_r1024)=2.4178); coin loss_delta_at_it=-0.4964 → PASS (L(it)=2.5587, L(it+Δ_r1024)=2.0623); control loss_delta_at_it=+0.1122 → INFO (L(it)=1.9694, L(it+Δ_r1024)=2.0816).
- **G3 exactness (LoRA route vs full Δ, λ = 0)** — **PASS** (threshold ρ ≥ 0.9). charter spearman_lora_vs_full=+0.9955 → PASS (lam0_r1024 vs lam0_full: n=2000, slope=+1.062); coin spearman_lora_vs_full=+0.9987 → PASS (lam0_r1024 vs lam0_full: n=2000, slope=+1.058); control spearman_lora_vs_full=+0.979 → PASS (lam0_r1024 vs lam0_full: n=2000, slope=+1.006).
- **G4 noise floor (repeat pass)** — **NOT RUN** (threshold ≤ 2%). all median_rel_spread=n/a → NOT RUN (no scores/noise.jsonl (repeat pass)).
- **Rank capture (§7)** — **PASS** (threshold ≥ 70%). charter fraction_of_full@lam0_r1024=+0.7471 → PASS (mean@lam0_r1024=+1.026 vs full=+1.373 (n=500)); coin fraction_of_full@lam0_r1024=+0.9218 → PASS (mean@lam0_r1024=+11.13 vs full=+12.07 (n=500)); control fraction_of_full@lam0_r1024=+0.6962 → INFO (mean@lam0_r1024=+0.2896 vs full=+0.4159 (n=500)).

## λ = 0 vs λ = 1 (curvature of the loss along the graft)

- **charter arm, λ = 1 (r*)** (`lam1_r1024` on `lam0_r1024`): all rows ρ=-0.164, OLS slope=-0.197, sign agreement=0.49 (n=6000); per class slope charter -0.176, coin -0.229, ambiguous -0.224, ambiguous_wrong -0.162; contrast attenuation mean(λ=1)/mean(λ=0): coin_minus_charter -16.4 (episode ρ=-0.17), ambiguous_minus_wrong +0.177 (episode ρ=-0.22).
- **coin arm, λ = 1 (r*)** (`lam1_r1024` on `lam0_r1024`): all rows ρ=+0.072, OLS slope=-0.00545, sign agreement=0.42 (n=6000); per class slope charter +0.0146, coin -0.00231, ambiguous -0.0178, ambiguous_wrong -0.0828; contrast attenuation mean(λ=1)/mean(λ=0): coin_minus_charter +0.525 (episode ρ=+0.12), ambiguous_minus_wrong +0.301 (episode ρ=+0.071).
- **control arm, λ = 1 (r*)** (`lam1_r1024` on `lam0_r1024`): all rows ρ=+0.0472, OLS slope=+0.0417, sign agreement=0.63 (n=6000); per class slope charter +0.0408, coin +0.00916, ambiguous +0.0305, ambiguous_wrong +0.0741; contrast attenuation mean(λ=1)/mean(λ=0): coin_minus_charter +2.03 (episode ρ=+0.08), ambiguous_minus_wrong +0.367 (episode ρ=+0.085).
- λ = 1 (full Δ): NOT RUN (no scores/lam1full__<arm>.jsonl).

## Linearity — ΔL = L(1) − L(0) vs g(0) = dL/dλ|₀

- **charter arm, λ = 1 (r*)** (g(0) from `lam0_r1024`): ρ=+0.243, sign agreement=0.65, OLS slope=+0.0987 (1 = exactly linear), RMSE=17.41, mean ΔL=+4.03, loss lowered on 33% of rows (n=6000); per class sign agreement charter 0.59, coin 0.67, ambiguous 0.61, ambiguous_wrong 0.72; trapezoid predictor (g(0)+g(1))/2: slope=+0.179, RMSE=15.82.
- **coin arm, λ = 1 (r*)** (g(0) from `lam0_r1024`): ρ=+0.437, sign agreement=0.59, OLS slope=+0.13 (1 = exactly linear), RMSE=32.85, mean ΔL=-0.787, loss lowered on 56% of rows (n=6000); per class sign agreement charter 0.65, coin 0.57, ambiguous 0.49, ambiguous_wrong 0.64; trapezoid predictor (g(0)+g(1))/2: slope=+0.151, RMSE=23.81.
- **control arm, λ = 1 (r*)** (g(0) from `lam0_r1024`): ρ=+0.534, sign agreement=0.81, OLS slope=+0.177 (1 = exactly linear), RMSE=7.076, mean ΔL=-5.21, loss lowered on 92% of rows (n=6000); per class sign agreement charter 0.76, coin 0.81, ambiguous 0.89, ambiguous_wrong 0.77; trapezoid predictor (g(0)+g(1))/2: slope=+0.246, RMSE=6.266.
- λ = 1 (full Δ): NOT RUN (no scores/lam1full__<arm>.jsonl).

## λ = 1: exact full-Δ graft vs r* LoRA graft (shared rows)

_(NOT RUN — no scores/lam1full__<arm>.jsonl, or no r* λ = 1 pass to compare against)_

## Rank ladder (λ = 0, raw coin−charter, full-Δ subset where available)

Subset: full_subset. Full table (both contrasts, raw and net, both subsets) in `rank_ladder.md`; energy per module type in `energy_capture.md`.

| arm | rank_label | n | mean | ci_low | ci_high | fraction_of_full | captured_energy |
|---|---|---|---|---|---|---|---|
| charter | r16 | 500 | +0.7026 | -0.3182 | +1.8703 | 0.5118 | +0.0358 |
| charter | r64 | 500 | +0.6288 | -0.6477 | +1.8965 | 0.4580 | +0.0741 |
| charter | r256 | 500 | +0.8197 | -0.7000 | +2.2995 | 0.5971 | +0.1728 |
| charter | r1024 | 500 | +1.0256 | -0.5548 | +2.5007 | 0.7471 | +0.4382 |
| charter | full | 500 | +1.3728 | -0.2885 | +2.9683 | 1.0000 | +1.0000 |
| coin | r16 | 500 | +6.8989 | +4.4499 | +9.2397 | 0.5716 | +0.0355 |
| coin | r64 | 500 | +9.1724 | +6.0891 | +12.0991 | 0.7599 | +0.0732 |
| coin | r256 | 500 | +10.3253 | +7.1615 | +13.1598 | 0.8554 | +0.1711 |
| coin | r1024 | 500 | +11.1264 | +7.6202 | +14.5038 | 0.9218 | +0.4362 |
| coin | full | 500 | +12.0700 | +8.3888 | +15.8489 | 1.0000 | +1.0000 |
| control | r16 | 500 | -0.0100 | -0.4672 | +0.4750 | -0.0240 | +0.0266 |
| control | r64 | 500 | +0.1581 | -0.3391 | +0.7494 | 0.3802 | +0.0589 |
| control | r256 | 500 | +0.2508 | -0.2673 | +0.7750 | 0.6030 | +0.1520 |
| control | r1024 | 500 | +0.2896 | -0.2797 | +0.8215 | 0.6962 | +0.4185 |
| control | full | 500 | +0.4159 | -0.1683 | +0.9385 | 1.0000 | +1.0000 |

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
- `dist__lam0_vs_lam1__charter.pdf` — class distributions of −g at λ = 0 (lam0_r1024) vs λ = 1 (r*) (lam1_r1024), charter arm
- `scatter__lam0_vs_lam1__charter.pdf` — −g(1) vs −g(0) per class with identity and OLS fits, λ = 1 (r*), charter arm
- `dist__lam0_vs_lam1__coin.pdf` — class distributions of −g at λ = 0 (lam0_r1024) vs λ = 1 (r*) (lam1_r1024), coin arm
- `scatter__lam0_vs_lam1__coin.pdf` — −g(1) vs −g(0) per class with identity and OLS fits, λ = 1 (r*), coin arm
- `dist__lam0_vs_lam1__control.pdf` — class distributions of −g at λ = 0 (lam0_r1024) vs λ = 1 (r*) (lam1_r1024), control arm
- `scatter__lam0_vs_lam1__control.pdf` — −g(1) vs −g(0) per class with identity and OLS fits, λ = 1 (r*), control arm
- `linearity__charter.pdf` — ΔL = L(1) − L(0) vs g(0) per class, λ = 1 (r*), charter arm
- `linearity__coin.pdf` — ΔL = L(1) − L(0) vs g(0) per class, λ = 1 (r*), coin arm
- `linearity__control.pdf` — ΔL = L(1) − L(0) vs g(0) per class, λ = 1 (r*), control arm
- `rank_ladder.pdf` — rank ladder of the paired contrasts (raw / net) with captured energy and fraction of full

## Notes

- no scores/lam1full__<arm>.jsonl — full-Δ λ = 1 variant NOT RUN (λ = 1 readouts use the r* LoRA graft only)
- no scores/noise.jsonl — G4 noise floor NOT RUN
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
