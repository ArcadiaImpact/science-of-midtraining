# SUMMARY — ekfac_dataset_attribution_v1 analysis (2026-09-14T14:40:45Z)

Inputs: 6000 scored rows charter=1500, coin=1500, ambiguous=1500, ambiguous_wrong=1500; 12 vectors over kinds ['lam0_r1024', 'lam0_r16', 'lam0_r256', 'lam0_r64'] and folds ['all']; passes combined (0 duplicate (row, vector) scores dropped). Sign: positive = training on the dataset lowers the row's loss. Curvature: damped EK-FAC fit at gemma-3-12b-**pt** on Dolmino; row gradients at gemma-3-12b-**it** (deliberate checkpoint mismatch, no SOURCE propagators — see PREMORTEM/LITERATURE). Dolmino is in-sample for the curvature: compare classes *within* a dataset only.

## Headline — paired per-episode contrasts (PRIMARY)

Kind **lam0_r256**, normalisation **per_sequence_sum**, fold all; mean of s(coin row) − s(charter row) over conflict episodes and s(ambiguous) − s(ambiguous_wrong) over agreement episodes; bootstrap 95% CI (1000 resamples); exact sign test. Hypothesis (SPEC, pre-registered signs): charter datasets coin−charter < 0, coin datasets > 0, dolmino ≈ 0; ambiguous−wrong > 0 on every oracle dataset. [partial: one EK-FAC fit, one seed, bootstrap CIs over episodes only]

| dataset | family | n_pairs | coin−charter mean | ci_low | ci_high | frac coin-ward | sign p | verdict | n_agree | amb−wrong mean | ci_low  | ci_high  | verdict  | marginal_order | ambiguous_nearer_to |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coin | coin | 1500 | +11.4479 | +9.5279 | +13.4336 | 0.6673 | 5.545e-39 | PASS | 1500 | +11.1175 | +9.2364 | +12.8954 | PASS | coin > ambiguous > charter | coin |
| charter | charter | 1500 | +1.1602 | +0.2937 | +2.1343 | 0.5513 | 7.696e-05 | FAIL | 1500 | +7.0939 | +6.2559 | +7.8736 | PASS | ambiguous > coin > charter | coin |
| control | neutral | 1500 | +0.4343 | +0.1283 | +0.7703 | 0.5407 | 0.0018 | UNEXPECTED (+) | 1500 | +1.5844 | +1.2925 | +1.8801 | UNEXPECTED (+) | ambiguous > coin > charter | coin |

Oracle datasets: 1/2 PASS, 1 FAIL, 0 inconclusive on coin−charter. coin: PASS; charter: FAIL; control: UNEXPECTED (+)

### Stability across kinds × normalisations (coin−charter verdict, mean)

| dataset | lam0_r1024 / per_sequence_sum | lam0_r1024 / per_token | lam0_r16 / per_sequence_sum | lam0_r16 / per_token | lam0_r256 / per_sequence_sum | lam0_r256 / per_token | lam0_r64 / per_sequence_sum | lam0_r64 / per_token |
|---|---|---|---|---|---|---|---|---|
| coin | PASS (+12.3) | PASS (+1.11) | PASS (+7.82) | PASS (+0.703) | PASS (+11.4) | PASS (+1.03) | PASS (+10.2) | PASS (+0.916) |
| charter | FAIL (+1.33) | FAIL (+0.118) | FAIL (+1.08) | FAIL (+0.0963) | FAIL (+1.16) | FAIL (+0.103) | FAIL (+0.985) | FAIL (+0.0878) |
| control | UNEXPECTED (+) (+0.515) | UNEXPECTED (+) (+0.0461) | CONSISTENT (≈0) (+0.14) | CONSISTENT (≈0) (+0.0127) | UNEXPECTED (+) (+0.434) | UNEXPECTED (+) (+0.0389) | UNEXPECTED (+) (+0.348) | UNEXPECTED (+) (+0.0313) |

## Gates

- **Fold agreement**: no f0/f1 vectors scored — gate not evaluable.
- **Noise floor**: no oracle repeat scores — gate not evaluable.
- **Checkpoint mismatch** (pt vs it row scores): no pt_mismatch pass — gate not evaluable.
- **Common component** (kind lam0_r256; cross-dataset vector cosine gate < 0.995): no vector cosines supplied; cross-dataset per-row score Spearman +0.14–+0.53; pass.

## Class-level summary (kind lam0_r256, per_sequence_sum, fold all)

| dataset | class | n | mean | ci_low | ci_high | median | frac_positive |
|---|---|---|---|---|---|---|---|
| charter | ambiguous | 1500 | -1.7859 | -2.4385 | -1.1238 | -1.7511 | 0.3600 |
| charter | ambiguous_wrong | 1500 | -8.8798 | -9.5444 | -8.1832 | -8.2589 | 0.1873 |
| charter | charter | 1500 | -6.0822 | -7.0988 | -5.1173 | -5.5489 | 0.2613 |
| charter | coin | 1500 | -4.9220 | -5.6559 | -4.2026 | -3.2723 | 0.2893 |
| coin | ambiguous | 1500 | -3.7543 | -5.2560 | -2.2772 | -3.9333 | 0.3193 |
| coin | ambiguous_wrong | 1500 | -14.8718 | -16.4388 | -13.3213 | -13.1898 | 0.2080 |
| coin | charter | 1500 | -12.5783 | -14.4697 | -10.8990 | -10.9702 | 0.2333 |
| coin | coin | 1500 | -1.1304 | -2.5721 | +0.4571 | -3.2695 | 0.3853 |
| control | ambiguous | 1500 | +3.1334 | +2.8640 | +3.3741 | +3.6216 | 0.8393 |
| control | ambiguous_wrong | 1500 | +1.5490 | +1.2771 | +1.8399 | +1.8299 | 0.6593 |
| control | charter | 1500 | +1.8259 | +1.3321 | +2.2433 | +2.3467 | 0.6947 |
| control | coin | 1500 | +2.2602 | +1.9071 | +2.5661 | +2.8611 | 0.7453 |

## Controls

- **Curvature vs GDP**: only one kind scored — not evaluable.
- **TF-IDF register baseline**: skipped (rows or dataset samples missing).
- **Length confound**: n_target_tokens by class — ambiguous mean 11.1 (n=1500); ambiguous_wrong mean 11.1 (n=1500); charter mean 11.1 (n=1500); coin mean 11.1 (n=1500).
  - partial Spearman(score, length | class), per_sequence_sum: charter +0.04; coin +0.02; control -0.02.
  - partial Spearman(score, length | class), per_token: charter +0.06; coin +0.04; control -0.04.

## Plot index

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
- `length_by_class.pdf` — (i) n_target_tokens by class

## Notes

- checkpoint-mismatch diagnostic skipped (no pt_mismatch.jsonl)
- noise-floor diagnostic skipped (no oracle.jsonl)
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
