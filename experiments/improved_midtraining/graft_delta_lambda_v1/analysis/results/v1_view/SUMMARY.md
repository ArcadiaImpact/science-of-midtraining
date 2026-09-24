# SUMMARY — ekfac_dataset_attribution_v1 analysis (2026-09-14T20:12:25Z)

Inputs: 6000 scored rows charter=1500, coin=1500, ambiguous=1500, ambiguous_wrong=1500; 36 vectors over kinds ['lam0_full', 'lam0_r1024', 'lam0_r16', 'lam0_r256', 'lam0_r64', 'lam1_r1024', 'lam1full', 'lam1full_r1024', 'lam1fullx_r1024_at_charter', 'lam1fullx_r1024_at_coin', 'lam1fullx_r1024_at_control', 'lam1x_r1024_at_charter', 'lam1x_r1024_at_coin', 'lam1x_r1024_at_control'] and folds ['all']; passes combined (0 duplicate (row, vector) scores dropped). Sign: positive = training on the dataset lowers the row's loss. Curvature: damped EK-FAC fit at gemma-3-12b-**pt** on Dolmino; row gradients at gemma-3-12b-**it** (deliberate checkpoint mismatch, no SOURCE propagators — see PREMORTEM/LITERATURE). Dolmino is in-sample for the curvature: compare classes *within* a dataset only.

## Headline — paired per-episode contrasts (PRIMARY)

Kind **lam0_r1024**, normalisation **per_sequence_sum**, fold all; mean of s(coin row) − s(charter row) over conflict episodes and s(ambiguous) − s(ambiguous_wrong) over agreement episodes; bootstrap 95% CI (2000 resamples); exact sign test. Hypothesis (SPEC, pre-registered signs): charter datasets coin−charter < 0, coin datasets > 0, dolmino ≈ 0; ambiguous−wrong > 0 on every oracle dataset. [partial: one EK-FAC fit, one seed, bootstrap CIs over episodes only]

| dataset | family | n_pairs | coin−charter mean | ci_low | ci_high | frac coin-ward | sign p | verdict | n_agree | amb−wrong mean | ci_low  | ci_high  | verdict  | marginal_order | ambiguous_nearer_to |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coin | coin | 1500 | +12.3253 | +10.3985 | +14.3762 | 0.6647 | 8.853e-38 | PASS | 1500 | +11.9831 | +10.0493 | +13.8511 | PASS | coin > ambiguous > charter | coin |
| charter | charter | 1500 | +1.3250 | +0.3815 | +2.2425 | 0.5493 | 1.457e-04 | FAIL | 1500 | +7.6783 | +6.8203 | +8.5755 | PASS | ambiguous > coin > charter | coin |
| control | neutral | 1500 | +0.5147 | +0.1862 | +0.8573 | 0.5400 | 0.0021 | UNEXPECTED (+) | 1500 | +1.7208 | +1.3961 | +2.0200 | UNEXPECTED (+) | ambiguous > coin > charter | coin |

Oracle datasets: 1/2 PASS, 1 FAIL, 0 inconclusive on coin−charter. coin: PASS; charter: FAIL; control: UNEXPECTED (+)

### Stability across kinds × normalisations (coin−charter verdict, mean)

| dataset | lam0_full / per_sequence_sum | lam0_full / per_token | lam0_r1024 / per_sequence_sum | lam0_r1024 / per_token | lam0_r16 / per_sequence_sum | lam0_r16 / per_token | lam0_r256 / per_sequence_sum | lam0_r256 / per_token | lam0_r64 / per_sequence_sum | lam0_r64 / per_token | lam1_r1024 / per_sequence_sum | lam1_r1024 / per_token | lam1full / per_sequence_sum | lam1full / per_token | lam1full_r1024 / per_sequence_sum | lam1full_r1024 / per_token | lam1fullx_r1024_at_charter / per_sequence_sum | lam1fullx_r1024_at_charter / per_token | lam1fullx_r1024_at_coin / per_sequence_sum | lam1fullx_r1024_at_coin / per_token | lam1fullx_r1024_at_control / per_sequence_sum | lam1fullx_r1024_at_control / per_token | lam1x_r1024_at_charter / per_sequence_sum | lam1x_r1024_at_charter / per_token | lam1x_r1024_at_coin / per_sequence_sum | lam1x_r1024_at_coin / per_token | lam1x_r1024_at_control / per_sequence_sum | lam1x_r1024_at_control / per_token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coin | PASS (+12.1) | PASS (+1.08) | PASS (+12.3) | PASS (+1.11) | PASS (+7.82) | PASS (+0.703) | PASS (+11.4) | PASS (+1.03) | PASS (+10.2) | PASS (+0.916) | PASS (+6.47) | PASS (+0.587) | PASS (+3.81) | PASS (+0.344) | PASS (+3.73) | PASS (+0.336) | PASS (+6.19) | PASS (+0.561) |  |  | PASS (+14) | PASS (+1.26) | PASS (+5.6) | PASS (+0.501) |  |  | PASS (+14.1) | PASS (+1.27) |
| charter | INCONCLUSIVE (+1.37) | INCONCLUSIVE (+0.128) | FAIL (+1.33) | FAIL (+0.118) | FAIL (+1.08) | FAIL (+0.0963) | FAIL (+1.16) | FAIL (+0.103) | FAIL (+0.985) | FAIL (+0.0878) | PASS (-21.7) | PASS (-1.97) | PASS (-22.1) | PASS (-2) | PASS (-21.3) | PASS (-1.93) |  |  | INCONCLUSIVE (-0.743) | INCONCLUSIVE (-0.0679) | FAIL (+1.27) | FAIL (+0.113) |  |  | FAIL (+1.44) | FAIL (+0.133) | INCONCLUSIVE (+0.881) | INCONCLUSIVE (+0.0785) |
| control | CONSISTENT (≈0) (+0.416) | CONSISTENT (≈0) (+0.0377) | UNEXPECTED (+) (+0.515) | UNEXPECTED (+) (+0.0461) | CONSISTENT (≈0) (+0.14) | CONSISTENT (≈0) (+0.0127) | UNEXPECTED (+) (+0.434) | UNEXPECTED (+) (+0.0389) | UNEXPECTED (+) (+0.348) | UNEXPECTED (+) (+0.0313) | UNEXPECTED (+) (+1.05) | UNEXPECTED (+) (+0.0961) | UNEXPECTED (+) (+1.27) | UNEXPECTED (+) (+0.116) | UNEXPECTED (+) (+1.23) | UNEXPECTED (+) (+0.113) | UNEXPECTED (+) (+2.3) | UNEXPECTED (+) (+0.207) | CONSISTENT (≈0) (+0.337) | CONSISTENT (≈0) (+0.0304) |  |  | UNEXPECTED (+) (+1.29) | UNEXPECTED (+) (+0.116) | UNEXPECTED (+) (+1.52) | UNEXPECTED (+) (+0.14) |  |  |

## Gates

- **Fold agreement**: no f0/f1 vectors scored — gate not evaluable.
- **Noise floor** (run-to-run relative spread; gate median ≤ 2%, p90 ≤ 10%): pass. lam0_r1024: median 0.91%, p90 7.25%, max 200.00% (n=600); lam0_r16: median 0.92%, p90 6.86%, max 200.00% (n=600); lam0_r256: median 0.93%, p90 7.64%, max 200.00% (n=600); lam0_r64: median 0.93%, p90 6.66%, max 200.00% (n=600).
- **Checkpoint mismatch** (pt vs it row scores): no pt_mismatch pass — gate not evaluable.
- **Common component** (kind lam0_r1024; cross-dataset vector cosine gate < 0.995): no vector cosines supplied; cross-dataset per-row score Spearman +0.15–+0.52; pass.

## Class-level summary (kind lam0_r1024, per_sequence_sum, fold all)

| dataset | class | n | mean | ci_low | ci_high | median | frac_positive |
|---|---|---|---|---|---|---|---|
| charter | ambiguous | 1500 | -1.8002 | -2.4630 | -1.1088 | -1.8644 | 0.3620 |
| charter | ambiguous_wrong | 1500 | -9.4785 | -10.1738 | -8.7603 | -8.7706 | 0.1860 |
| charter | charter | 1500 | -6.5255 | -7.5749 | -5.5791 | -5.7986 | 0.2593 |
| charter | coin | 1500 | -5.2004 | -6.0477 | -4.4690 | -3.4359 | 0.2853 |
| coin | ambiguous | 1500 | -4.5982 | -6.1469 | -3.0189 | -4.6413 | 0.3060 |
| coin | ambiguous_wrong | 1500 | -16.5812 | -18.2030 | -14.9743 | -14.3437 | 0.1953 |
| coin | charter | 1500 | -14.1060 | -15.9982 | -12.2962 | -12.2484 | 0.2213 |
| coin | coin | 1500 | -1.7807 | -3.3569 | -0.1661 | -3.9910 | 0.3733 |
| control | ambiguous | 1500 | +3.9332 | +3.6608 | +4.1675 | +4.4616 | 0.8747 |
| control | ambiguous_wrong | 1500 | +2.2124 | +1.9062 | +2.5072 | +2.4498 | 0.7027 |
| control | charter | 1500 | +2.5145 | +1.9712 | +2.9236 | +3.1310 | 0.7313 |
| control | coin | 1500 | +3.0292 | +2.6445 | +3.3499 | +3.6116 | 0.7920 |

## Controls

- **Curvature vs GDP**: only one kind scored — not evaluable.
- **TF-IDF register baseline**: skipped (rows or dataset samples missing).
- **Length confound**: n_target_tokens by class — ambiguous mean 11.1 (n=1500); ambiguous_wrong mean 11.1 (n=1500); charter mean 11.1 (n=1500); coin mean 11.1 (n=1500).
  - partial Spearman(score, length | class), per_sequence_sum: charter +0.05; coin +0.02; control -0.02.
  - partial Spearman(score, length | class), per_token: charter +0.07; coin +0.04; control -0.05.

## Plot index

- `dist__lam0_full__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind lam0_full, per_sequence_sum
- `paired__lam0_full__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind lam0_full, per_sequence_sum
- `heatmap__lam0_full__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_full, per_sequence_sum
- `dist__lam0_full__per_token.pdf` — (a) score distributions per dataset × class, kind lam0_full, per_token
- `paired__lam0_full__per_token.pdf` — (b) paired contrast distributions with CI, kind lam0_full, per_token
- `heatmap__lam0_full__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_full, per_token
- `dist__lam0_full__cosine.pdf` — (a) score distributions per dataset × class, kind lam0_full, cosine
- `paired__lam0_full__cosine.pdf` — (b) paired contrast distributions with CI, kind lam0_full, cosine
- `heatmap__lam0_full__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_full, cosine
- `dist__lam0_r1024__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind lam0_r1024, per_sequence_sum
- `paired__lam0_r1024__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind lam0_r1024, per_sequence_sum
- `heatmap__lam0_r1024__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r1024, per_sequence_sum
- `dist__lam0_r1024__per_token.pdf` — (a) score distributions per dataset × class, kind lam0_r1024, per_token
- `paired__lam0_r1024__per_token.pdf` — (b) paired contrast distributions with CI, kind lam0_r1024, per_token
- `heatmap__lam0_r1024__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r1024, per_token
- `dist__lam0_r1024__cosine.pdf` — (a) score distributions per dataset × class, kind lam0_r1024, cosine
- `paired__lam0_r1024__cosine.pdf` — (b) paired contrast distributions with CI, kind lam0_r1024, cosine
- `heatmap__lam0_r1024__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r1024, cosine
- `dist__lam0_r16__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind lam0_r16, per_sequence_sum
- `paired__lam0_r16__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind lam0_r16, per_sequence_sum
- `heatmap__lam0_r16__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r16, per_sequence_sum
- `dist__lam0_r16__per_token.pdf` — (a) score distributions per dataset × class, kind lam0_r16, per_token
- `paired__lam0_r16__per_token.pdf` — (b) paired contrast distributions with CI, kind lam0_r16, per_token
- `heatmap__lam0_r16__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r16, per_token
- `dist__lam0_r16__cosine.pdf` — (a) score distributions per dataset × class, kind lam0_r16, cosine
- `paired__lam0_r16__cosine.pdf` — (b) paired contrast distributions with CI, kind lam0_r16, cosine
- `heatmap__lam0_r16__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r16, cosine
- `dist__lam0_r256__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind lam0_r256, per_sequence_sum
- `paired__lam0_r256__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind lam0_r256, per_sequence_sum
- `heatmap__lam0_r256__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r256, per_sequence_sum
- `dist__lam0_r256__per_token.pdf` — (a) score distributions per dataset × class, kind lam0_r256, per_token
- `paired__lam0_r256__per_token.pdf` — (b) paired contrast distributions with CI, kind lam0_r256, per_token
- `heatmap__lam0_r256__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r256, per_token
- `dist__lam0_r256__cosine.pdf` — (a) score distributions per dataset × class, kind lam0_r256, cosine
- `paired__lam0_r256__cosine.pdf` — (b) paired contrast distributions with CI, kind lam0_r256, cosine
- `heatmap__lam0_r256__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r256, cosine
- `dist__lam0_r64__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind lam0_r64, per_sequence_sum
- `paired__lam0_r64__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind lam0_r64, per_sequence_sum
- `heatmap__lam0_r64__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r64, per_sequence_sum
- `dist__lam0_r64__per_token.pdf` — (a) score distributions per dataset × class, kind lam0_r64, per_token
- `paired__lam0_r64__per_token.pdf` — (b) paired contrast distributions with CI, kind lam0_r64, per_token
- `heatmap__lam0_r64__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r64, per_token
- `dist__lam0_r64__cosine.pdf` — (a) score distributions per dataset × class, kind lam0_r64, cosine
- `paired__lam0_r64__cosine.pdf` — (b) paired contrast distributions with CI, kind lam0_r64, cosine
- `heatmap__lam0_r64__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam0_r64, cosine
- `dist__lam1_r1024__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind lam1_r1024, per_sequence_sum
- `paired__lam1_r1024__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind lam1_r1024, per_sequence_sum
- `heatmap__lam1_r1024__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam1_r1024, per_sequence_sum
- `dist__lam1_r1024__per_token.pdf` — (a) score distributions per dataset × class, kind lam1_r1024, per_token
- `paired__lam1_r1024__per_token.pdf` — (b) paired contrast distributions with CI, kind lam1_r1024, per_token
- `heatmap__lam1_r1024__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam1_r1024, per_token
- `dist__lam1_r1024__cosine.pdf` — (a) score distributions per dataset × class, kind lam1_r1024, cosine
- `paired__lam1_r1024__cosine.pdf` — (b) paired contrast distributions with CI, kind lam1_r1024, cosine
- `heatmap__lam1_r1024__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam1_r1024, cosine
- `dist__lam1full__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind lam1full, per_sequence_sum
- `paired__lam1full__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind lam1full, per_sequence_sum
- `heatmap__lam1full__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam1full, per_sequence_sum
- `dist__lam1full__per_token.pdf` — (a) score distributions per dataset × class, kind lam1full, per_token
- `paired__lam1full__per_token.pdf` — (b) paired contrast distributions with CI, kind lam1full, per_token
- `heatmap__lam1full__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam1full, per_token
- `dist__lam1full__cosine.pdf` — (a) score distributions per dataset × class, kind lam1full, cosine
- `paired__lam1full__cosine.pdf` — (b) paired contrast distributions with CI, kind lam1full, cosine
- `heatmap__lam1full__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind lam1full, cosine
- `length_by_class.pdf` — (i) n_target_tokens by class

## Notes

- checkpoint-mismatch diagnostic skipped (no pt_mismatch.jsonl)
- TF-IDF baseline skipped (rows file or datasets/ missing)

## Tables

- `manifest.json`
- `scores_long.csv`
- `paired_contrasts.json`
- `paired_contrasts.md`
- `paired_contrasts_by_subtype.json`
- `paired_contrasts_by_subtype.md`
- `paired_unmatched.json`
- `paired_unmatched.md`
- `headline.json`
- `headline.md`
- `verdict_grid.json`
- `verdict_grid.md`
- `class_summary.json`
- `class_summary.md`
- `effect_sizes.json`
- `effect_sizes.md`
- `fold_agreement.json`
- `fold_agreement.md`
- `cross_dataset_agreement.json`
- `cross_dataset_agreement.md`
- `curvature_vs_gdp.json`
- `curvature_vs_gdp.md`
- `checkpoint_mismatch.json`
- `checkpoint_mismatch.md`
- `noise_floor.json`
- `noise_floor.md`
- `noise_floor_per_score.json`
- `noise_floor_per_score.md`
- `tfidf_baseline.json`
- `tfidf_baseline.md`
- `tfidf_row_similarity.csv`
- `tfidf_vs_gradient.json`
- `tfidf_vs_gradient.md`
- `length_by_class.json`
- `length_by_class.md`
- `length_confound.json`
- `length_confound.md`
