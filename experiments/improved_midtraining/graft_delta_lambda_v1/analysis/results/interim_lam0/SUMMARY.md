# SUMMARY — graft_delta_lambda_v1 analysis (2026-09-14T14:40:53Z)

Inputs: 6000 scored rows charter=1500, coin=1500, ambiguous=1500, ambiguous_wrong=1500; passes lam0; kinds ['lam0_r16', 'lam0_r64', 'lam0_r256', 'lam0_r1024']; arms ['charter', 'coin', 'control']. Primary rank r=256 (λ = 0 kind `lam0_r256`); λ = 1 rank r* = n/a (no λ = 1 passes); no full-Δ λ = 1 passes (scores/lam1full__<arm>.jsonl absent — that variant NOT RUN); bootstrap 1000 resamples. Sign: every score is −dL/dλ (+ = the graft lowers the row's loss). *raw* = the arm's own graft; *net_of_control* = same row, same kind, score(arm) − score(control); *net_of_control_cross* (λ = 1 only) = the arm's own λ = 1 term minus the control Δ's cross term at the arm's grafted point. Paired contrasts: s(coin row) − s(charter row) over conflict episodes, s(ambiguous) − s(ambiguous_wrong) over agreement episodes.

## Headline — paired per-episode contrasts per arm, λ = 0 and λ = 1, raw and net of control

Normalisation **per_sequence_sum**, fold all; mean paired contrast with bootstrap 95% CI and exact sign test. Pre-registered (SPEC §7): charter arm net-of-control coin−charter **< 0**; coin arm **> 0**; control arm raw contrasts non-zero and arm-independent (answer-plausibility prior, reported as PRIOR (±)); ambiguous−wrong > 0 for the charter and coin arms. PASS = CI excludes 0 in the pre-registered direction; FAIL = the other way; INCONCLUSIVE = CI spans 0.

| arm | label | baseline | contrast | n | mean | ci_low | ci_high | frac_positive | sign_p | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| charter | λ = 0 | raw | coin_minus_charter | 1500 | +1.1602 | +0.2937 | +2.1343 | 0.5513 | 7.696e-05 | FAIL |
| charter | λ = 0 | raw | ambiguous_minus_wrong | 1500 | +7.0939 | +6.2559 | +7.8736 | 0.7247 | 4.939e-70 | PASS |
| coin | λ = 0 | raw | coin_minus_charter | 1500 | +11.4479 | +9.5279 | +13.4336 | 0.6673 | 5.545e-39 | PASS |
| coin | λ = 0 | raw | ambiguous_minus_wrong | 1500 | +11.1175 | +9.2364 | +12.8954 | 0.6800 | 5.542e-45 | PASS |
| control | λ = 0 | raw | coin_minus_charter | 1500 | +0.4343 | +0.1283 | +0.7703 | 0.5407 | 0.0018 | PRIOR (+) |
| control | λ = 0 | raw | ambiguous_minus_wrong | 1500 | +1.5844 | +1.2925 | +1.8801 | 0.6640 | 1.757e-37 | PRIOR (+) |
| charter | λ = 0 | net_of_control | coin_minus_charter | 1500 | +0.7260 | -0.1733 | +1.6282 | 0.5340 | 0.0091 | INCONCLUSIVE |
| charter | λ = 0 | net_of_control | ambiguous_minus_wrong | 1500 | +5.5095 | +4.6889 | +6.2978 | 0.6787 | 2.498e-44 | PASS |
| coin | λ = 0 | net_of_control | coin_minus_charter | 1500 | +11.0136 | +9.1066 | +13.0769 | 0.6500 | 1.576e-31 | PASS |
| coin | λ = 0 | net_of_control | ambiguous_minus_wrong | 1500 | +9.5331 | +7.5830 | +11.3383 | 0.6507 | 8.451e-32 | PASS |

- **charter arm** coin−charter — λ = 0 raw: FAIL (+1.16 [+0.294, +2.13], n=1500); λ = 0 net_of_control: INCONCLUSIVE (+0.726 [-0.173, +1.63], n=1500)
- **coin arm** coin−charter — λ = 0 raw: PASS (+11.4 [+9.53, +13.4], n=1500); λ = 0 net_of_control: PASS (+11 [+9.11, +13.1], n=1500)
- **control arm** coin−charter — λ = 0 raw: PRIOR (+) (+0.434 [+0.128, +0.77], n=1500)

## Gates (SPEC §5; rank capture §7)

- **G1 reconstruction at pt** — **FAIL** (threshold ≥ 90%). charter recovered_fraction@r256=+0.8751 → FAIL (L(pt)=2.6217, L(mid)=1.1886; ladder r16=0.591, r64=0.757, r256=0.875, r1024=0.949); charter full_delta_rel_err=+0.002793 → PASS (L(pt+Δ_full)=1.1919 vs L(mid)=1.1886); coin recovered_fraction@r256=+0.8799 → FAIL (L(pt)=2.2953, L(mid)=1.0175; ladder r16=0.600, r64=0.767, r256=0.880, r1024=0.952); coin full_delta_rel_err=+0.002515 → PASS (L(pt+Δ_full)=1.0200 vs L(mid)=1.0175); control recovered_fraction@r256=+0.6225 → FAIL (L(pt)=1.6519, L(mid)=1.6153; ladder r16=0.187, r64=0.421, r256=0.623, r1024=0.937); control full_delta_rel_err=+0.0002623 → PASS (L(pt+Δ_full)=1.6148 vs L(mid)=1.6153).
- **G2 transfer at it** — **PASS** (threshold < 0). charter loss_delta_at_it=-0.5165 → PASS (L(it)=2.9343, L(it+Δ_r1024)=2.4178); coin loss_delta_at_it=-0.4964 → PASS (L(it)=2.5587, L(it+Δ_r1024)=2.0623); control loss_delta_at_it=+0.1122 → INFO (L(it)=1.9694, L(it+Δ_r1024)=2.0816).
- **G3 exactness (LoRA route vs full Δ, λ = 0)** — **NOT RUN** (threshold ρ ≥ 0.9). charter spearman_lora_vs_full=n/a → NOT RUN (no shared rows with lam0_full); coin spearman_lora_vs_full=n/a → NOT RUN (no shared rows with lam0_full); control spearman_lora_vs_full=n/a → NOT RUN (no shared rows with lam0_full).
- **G4 noise floor (repeat pass)** — **NOT RUN** (threshold ≤ 2%). all median_rel_spread=n/a → NOT RUN (no scores/noise.jsonl (repeat pass)).
- **Rank capture (§7)** — **NOT RUN** (threshold ≥ 70%). all fraction_of_full@lam0_r256=n/a → NOT RUN (no lam0_full scores — full-Δ subset absent).

## λ = 0 vs λ = 1 (curvature of the loss along the graft)

_(no λ = 1 passes — not run)_

## Linearity — ΔL = L(1) − L(0) vs g(0) = dL/dλ|₀

_(no λ = 1 passes with loss_lam1 — not run)_

## λ = 1: exact full-Δ graft vs r* LoRA graft (shared rows)

_(NOT RUN — no scores/lam1full__<arm>.jsonl, or no r* λ = 1 pass to compare against)_

## Rank ladder (λ = 0, raw coin−charter, full-Δ subset where available)

Subset: all_rows. Full table (both contrasts, raw and net, both subsets) in `rank_ladder.md`; energy per module type in `energy_capture.md`.

| arm | rank_label | n | mean | ci_low | ci_high | fraction_of_full | captured_energy |
|---|---|---|---|---|---|---|---|
| charter | r16 | 1500 | +1.0774 | +0.3767 | +1.7436 |  | +0.0358 |
| charter | r64 | 1500 | +0.9850 | +0.1905 | +1.8284 |  | +0.0741 |
| charter | r256 | 1500 | +1.1602 | +0.2485 | +2.0122 |  | +0.1728 |
| charter | r1024 | 1500 | +1.3250 | +0.3880 | +2.3041 |  | +0.4382 |
| coin | r16 | 1500 | +7.8181 | +6.4840 | +9.1353 |  | +0.0355 |
| coin | r64 | 1500 | +10.1806 | +8.4056 | +11.8561 |  | +0.0732 |
| coin | r256 | 1500 | +11.4479 | +9.5334 | +13.3972 |  | +0.1711 |
| coin | r1024 | 1500 | +12.3253 | +10.2784 | +14.3597 |  | +0.4362 |
| control | r16 | 1500 | +0.1403 | -0.1462 | +0.4536 |  | +0.0266 |
| control | r64 | 1500 | +0.3481 | +0.0059 | +0.7059 |  | +0.0589 |
| control | r256 | 1500 | +0.4343 | +0.1133 | +0.7686 |  | +0.1520 |
| control | r1024 | 1500 | +0.5147 | +0.1435 | +0.8490 |  | +0.4185 |

## v1 view

`v1_view/` holds everything `ekfac_dataset_attribution_v1/analysis/analyze.py` produces, run on the concatenated λ = 0 + λ = 1 score files (one synthesised pass `v1_view_inputs/scores/combined.jsonl`; cross terms re-keyed `lam1x_r<r>_at_<arm>`; `noise.jsonl` served as its oracle pass) with primary kind `lam0_r256` and the family map charter→charter, coin→coin, control→neutral. Its SUMMARY prose describes the v1 EK-FAC setting; read it as dataset = arm, kind = λ/rank. v1 labels the control arm's non-zero raw contrast "UNEXPECTED" — SPEC §7 expects that prior, hence the net-of-control readout above. v1 headline verdicts: coin: PASS; charter: FAIL; control: UNEXPECTED (+).

## Plot index

- `paired__net__lam0_r16.pdf` — net-of-control paired contrast distributions with CI, kind lam0_r16
- `paired__net__lam0_r64.pdf` — net-of-control paired contrast distributions with CI, kind lam0_r64
- `paired__net__lam0_r256.pdf` — net-of-control paired contrast distributions with CI, kind lam0_r256
- `paired__net__lam0_r1024.pdf` — net-of-control paired contrast distributions with CI, kind lam0_r1024
- `rank_ladder.pdf` — rank ladder of the paired contrasts (raw / net) with captured energy and fraction of full

## Notes

- no scores/lam1__<arm>.jsonl — λ = 1 readouts, curvature and linearity NOT RUN
- no scores/lam1full__<arm>.jsonl — full-Δ λ = 1 variant NOT RUN (λ = 1 readouts use the r* LoRA graft only)
- no scores/noise.jsonl — G4 noise floor NOT RUN
- v1 view: fold-agreement, checkpoint-mismatch, TF-IDF and cosine diagnostics need f0/f1 vectors, a pt pass, row texts and vector norms — absent here, so v1 reports them as not evaluable / skipped (by design)
- no lam0_full scores — rank ladder has no full-Δ subset; G3 and the §7 rank-capture gate NOT RUN

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
