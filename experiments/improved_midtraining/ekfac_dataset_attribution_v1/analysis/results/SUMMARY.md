# SUMMARY — ekfac_dataset_attribution_v1 analysis (2026-09-14T04:51:25Z)

Inputs: 4210 scored rows charter=1099, coin=1099, ambiguous=1006, ambiguous_wrong=1006; 38 vectors over kinds ['gdp', 'gdpunit', 'inv0.01', 'inv0.1', 'inv1'] and folds ['all', 'f0', 'f1']; passes folds, main, sweep (0 duplicate (row, vector) scores dropped). Sign: positive = training on the dataset lowers the row's loss. Curvature: damped EK-FAC fit at gemma-3-12b-**pt** on Dolmino; row gradients at gemma-3-12b-**it** (deliberate checkpoint mismatch, no SOURCE propagators — see PREMORTEM/LITERATURE). Dolmino is in-sample for the curvature: compare classes *within* a dataset only.

## Headline — paired per-episode contrasts (PRIMARY)

Kind **inv0.1**, normalisation **per_sequence_sum**, fold all; mean of s(coin row) − s(charter row) over conflict episodes and s(ambiguous) − s(ambiguous_wrong) over agreement episodes; bootstrap 95% CI (2000 resamples); exact sign test. Hypothesis (SPEC, pre-registered signs): charter datasets coin−charter < 0, coin datasets > 0, dolmino ≈ 0; ambiguous−wrong > 0 on every oracle dataset. [partial: one EK-FAC fit, one seed, bootstrap CIs over episodes only]

| dataset | family | n_pairs | coin−charter mean | ci_low | ci_high | frac coin-ward | sign p | verdict | n_agree | amb−wrong mean | ci_low  | ci_high  | verdict  | marginal_order | ambiguous_nearer_to |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dolmino | neutral | 1000 | +1.098e+09 | +9.173e+08 | +1.282e+09 | 0.6580 | 9.589e-24 | UNEXPECTED (+) | 1000 | +1.394e+09 | +1.207e+09 | +1.582e+09 | UNEXPECTED (+) | ambiguous > coin > charter | coin |
| charter_worked | charter | 1000 | +1.066e+09 | +7.832e+08 | +1.357e+09 | 0.5930 | 4.448e-09 | FAIL | 1000 | +1.266e+09 | +9.559e+08 | +1.564e+09 | PASS | ambiguous > coin > charter | coin |
| charter_noex | charter | 1000 | +1.014e+09 | +6.831e+08 | +1.349e+09 | 0.5800 | 4.697e-07 | FAIL | 1000 | +1.241e+09 | +8.902e+08 | +1.592e+09 | PASS | ambiguous > coin > charter | coin |
| coin | coin | 1000 | +1.696e+09 | +1.338e+09 | +2.073e+09 | 0.6310 | 1.033e-16 | PASS | 1000 | +1.993e+09 | +1.629e+09 | +2.36e+09 | PASS | ambiguous > coin > charter | coin |
| coin_worked | coin | 1000 | +2.34e+09 | +1.976e+09 | +2.711e+09 | 0.6660 | 4.294e-26 | PASS | 1000 | +2.855e+09 | +2.475e+09 | +3.249e+09 | PASS | ambiguous > coin > charter | coin |
| coin_noex | coin | 1000 | +1.225e+09 | +8.766e+08 | +1.577e+09 | 0.5830 | 1.702e-07 | PASS | 1000 | +1.246e+09 | +8.77e+08 | +1.64e+09 | PASS | ambiguous > coin > charter | coin |

Oracle datasets: 3/5 PASS, 2 FAIL, 0 inconclusive on coin−charter. dolmino: UNEXPECTED (+); charter_worked: FAIL; charter_noex: FAIL; coin: PASS; coin_worked: PASS; coin_noex: PASS

### Stability across kinds × normalisations (coin−charter verdict, mean)

| dataset | gdp / per_sequence_sum | gdp / per_token | gdp / cosine | gdpunit / per_sequence_sum | gdpunit / per_token | gdpunit / cosine | inv0.01 / per_sequence_sum | inv0.01 / per_token | inv0.01 / cosine | inv0.1 / per_sequence_sum | inv0.1 / per_token | inv0.1 / cosine | inv1 / per_sequence_sum | inv1 / per_token | inv1 / cosine |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dolmino | UNEXPECTED (+) (+1.24e+06) | UNEXPECTED (+) (+1.13e+05) | UNEXPECTED (+) (+0.000997) | UNEXPECTED (+) (+4.06) | UNEXPECTED (+) (+0.377) | UNEXPECTED (+) (+0.00124) | UNEXPECTED (+) (+1.16e+09) | UNEXPECTED (+) (+1.04e+08) | UNEXPECTED (+) (+2.22e-05) | UNEXPECTED (+) (+1.1e+09) | UNEXPECTED (+) (+9.79e+07) | UNEXPECTED (+) (+4.66e-05) | UNEXPECTED (+) (+5.45e+08) | UNEXPECTED (+) (+4.91e+07) | UNEXPECTED (+) (+9.74e-05) |
| charter_worked | FAIL (+4.03e+06) | FAIL (+3.66e+05) | FAIL (+0.00165) | FAIL (+10.6) | FAIL (+0.956) | FAIL (+0.0021) | INCONCLUSIVE (+6.48e+08) | INCONCLUSIVE (+5.13e+07) | INCONCLUSIVE (-1.61e-06) | FAIL (+1.07e+09) | FAIL (+9.42e+07) | FAIL (+1.64e-05) | FAIL (+6.42e+08) | FAIL (+5.7e+07) | FAIL (+4.31e-05) |
| charter_noex | FAIL (+3.64e+06) | FAIL (+3.31e+05) | FAIL (+0.0013) | FAIL (+9.16) | FAIL (+0.829) | FAIL (+0.00188) | INCONCLUSIVE (+5.27e+07) | INCONCLUSIVE (-2.53e+06) | INCONCLUSIVE (-5.45e-06) | FAIL (+1.01e+09) | FAIL (+8.89e+07) | FAIL (+1.32e-05) | FAIL (+7.03e+08) | FAIL (+6.23e+07) | FAIL (+3.98e-05) |
| coin | PASS (+2.99e+06) | PASS (+2.72e+05) | PASS (+0.00124) | PASS (+8.76) | PASS (+0.793) | PASS (+0.00169) | PASS (+2.65e+09) | PASS (+2.34e+08) | PASS (+1.34e-05) | PASS (+1.7e+09) | PASS (+1.51e+08) | PASS (+2.76e-05) | PASS (+7.96e+08) | PASS (+7.09e+07) | PASS (+5.38e-05) |
| coin_worked | PASS (+2.51e+06) | PASS (+2.28e+05) | PASS (+0.00143) | PASS (+8.57) | PASS (+0.775) | PASS (+0.00188) |  |  |  | PASS (+2.34e+09) | PASS (+2.09e+08) | PASS (+4.59e-05) |  |  |  |
| coin_noex | PASS (+3.03e+06) | PASS (+2.77e+05) | PASS (+0.0011) | PASS (+8.42) | PASS (+0.764) | PASS (+0.00167) |  |  |  | PASS (+1.23e+09) | PASS (+1.07e+08) | PASS (+1.45e-05) |  |  |  |

## Gates

- **Fold agreement** (Spearman f0 vs f1 per-row scores; gate ρ ≥ 0.5): all pass. charter_noex/inv0.1 ρ=+0.99 (n=1200, cos=0.850); charter_worked/inv0.1 ρ=+0.97 (n=1200, cos=0.838); coin/inv0.1 ρ=+0.98 (n=1200, cos=0.818); coin_noex/inv0.1 ρ=+0.98 (n=1200, cos=0.771); coin_worked/inv0.1 ρ=+0.98 (n=1200, cos=0.828); dolmino/inv0.1 ρ=+0.98 (n=1200, cos=0.497).
- **Noise floor** (run-to-run relative spread; gate median ≤ 2%, p90 ≤ 10%): pass. gdp: median 1.12%, p90 5.39%, max 35.35% (n=48); inv0.1: median 0.65%, p90 3.26%, max 36.56% (n=48).
- **Checkpoint mismatch** (gate ρ ≥ 0.3): **FLAG** — 12 dataset×kind below 0.3. charter_noex/gdp ρ=-0.03 n=332 order DIFFERS, paired sign same; charter_noex/inv0.1 ρ=-0.01 n=332 order DIFFERS, paired sign same; charter_worked/gdp ρ=-0.01 n=332 order DIFFERS, paired sign same; charter_worked/inv0.1 ρ=-0.05 n=332 order DIFFERS, paired sign same; coin/gdp ρ=-0.02 n=332 order DIFFERS, paired sign same; coin/inv0.1 ρ=+0.01 n=332 order DIFFERS, paired sign same; coin_noex/gdp ρ=-0.02 n=332 order DIFFERS, paired sign same; coin_noex/inv0.1 ρ=-0.02 n=332 order DIFFERS, paired sign same; coin_worked/gdp ρ=-0.04 n=332 order DIFFERS, paired sign same; coin_worked/inv0.1 ρ=+0.04 n=332 order DIFFERS, paired sign same; dolmino/gdp ρ=+0.07 n=332 order DIFFERS, paired sign DIFFERS; dolmino/inv0.1 ρ=-0.04 n=332 order DIFFERS, paired sign same.
- **Common component** (kind inv0.1; cross-dataset vector cosine gate < 0.995): cosines 0.4019–0.8764; cross-dataset per-row score Spearman +0.67–+0.96; pass.

## Class-level summary (kind inv0.1, per_sequence_sum, fold all)

| dataset | class | n | mean | ci_low | ci_high | median | frac_positive |
|---|---|---|---|---|---|---|---|
| charter_noex | ambiguous | 1000 | -1.843e+09 | -2.09e+09 | -1.622e+09 | -1.814e+09 | 0.3010 |
| charter_noex | ambiguous_wrong | 1000 | -3.085e+09 | -3.411e+09 | -2.774e+09 | -2.752e+09 | 0.2490 |
| charter_noex | charter | 1000 | -3.074e+09 | -3.367e+09 | -2.78e+09 | -2.885e+09 | 0.2410 |
| charter_noex | coin | 1000 | -2.06e+09 | -2.318e+09 | -1.796e+09 | -1.905e+09 | 0.3090 |
| charter_worked | ambiguous | 1000 | -2.431e+09 | -2.634e+09 | -2.231e+09 | -2.335e+09 | 0.2270 |
| charter_worked | ambiguous_wrong | 1000 | -3.697e+09 | -3.972e+09 | -3.432e+09 | -3.485e+09 | 0.1770 |
| charter_worked | charter | 1000 | -3.671e+09 | -3.922e+09 | -3.406e+09 | -3.419e+09 | 0.1720 |
| charter_worked | coin | 1000 | -2.604e+09 | -2.823e+09 | -2.38e+09 | -2.449e+09 | 0.2370 |
| coin | ambiguous | 1000 | -2.874e+09 | -3.119e+09 | -2.608e+09 | -2.711e+09 | 0.2190 |
| coin | ambiguous_wrong | 1000 | -4.867e+09 | -5.183e+09 | -4.546e+09 | -4.66e+09 | 0.1450 |
| coin | charter | 1000 | -4.971e+09 | -5.248e+09 | -4.667e+09 | -4.708e+09 | 0.1260 |
| coin | coin | 1000 | -3.275e+09 | -3.557e+09 | -3.019e+09 | -3.128e+09 | 0.2080 |
| coin_noex | ambiguous | 1000 | -2.913e+09 | -3.155e+09 | -2.651e+09 | -2.911e+09 | 0.2250 |
| coin_noex | ambiguous_wrong | 1000 | -4.158e+09 | -4.498e+09 | -3.825e+09 | -4.034e+09 | 0.1940 |
| coin_noex | charter | 1000 | -4.453e+09 | -4.768e+09 | -4.149e+09 | -4.322e+09 | 0.1750 |
| coin_noex | coin | 1000 | -3.227e+09 | -3.497e+09 | -2.948e+09 | -3.164e+09 | 0.2250 |
| coin_worked | ambiguous | 1000 | -4.185e+09 | -4.437e+09 | -3.925e+09 | -4.064e+09 | 0.1240 |
| coin_worked | ambiguous_wrong | 1000 | -7.039e+09 | -7.356e+09 | -6.719e+09 | -6.713e+09 | 0.0620 |
| coin_worked | charter | 1000 | -6.969e+09 | -7.257e+09 | -6.671e+09 | -6.559e+09 | 0.0520 |
| coin_worked | coin | 1000 | -4.63e+09 | -4.902e+09 | -4.348e+09 | -4.37e+09 | 0.1110 |
| dolmino | ambiguous | 1000 | +1.599e+09 | +1.497e+09 | +1.704e+09 | +1.745e+09 | 0.8270 |
| dolmino | ambiguous_wrong | 1000 | +2.053e+08 | +3.582e+07 | +3.703e+08 | +5.248e+08 | 0.5780 |
| dolmino | charter | 1000 | +3.298e+08 | +1.721e+08 | +4.922e+08 | +6.315e+08 | 0.5990 |
| dolmino | coin | 1000 | +1.428e+09 | +1.306e+09 | +1.553e+09 | +1.592e+09 | 0.8000 |

## Controls

- **Curvature vs GDP** (inv0.1 vs gdp per-row Spearman): charter_noex ρ=+0.30 (order same, paired sign same); charter_worked ρ=+0.29 (order same, paired sign same); coin ρ=+0.32 (order same, paired sign same); coin_noex ρ=+0.28 (order same, paired sign same); coin_worked ρ=+0.36 (order same, paired sign same); dolmino ρ=+0.33 (order same, paired sign same); charter_noex ρ=+0.30 (order same, paired sign same); charter_worked ρ=+0.29 (order same, paired sign same); coin ρ=+0.32 (order same, paired sign same); coin_noex ρ=+0.28 (order same, paired sign same); coin_worked ρ=+0.36 (order same, paired sign same); dolmino ρ=+0.33 (order same, paired sign same); charter_noex ρ=+0.26 (order same, paired sign same); charter_worked ρ=+0.25 (order same, paired sign same); coin ρ=+0.28 (order same, paired sign same); coin_noex ρ=+0.23 (order same, paired sign same); coin_worked ρ=+0.32 (order same, paired sign same); dolmino ρ=+0.36 (order same, paired sign same). Full damping ladder in `curvature_vs_gdp.md`.
- **TF-IDF register baseline**: mean row·dataset cosine by class in `tfidf_baseline.md` / `tfidf_heatmap.pdf`; Spearman with gradient scores per dataset: charter_noex +0.17; charter_worked +0.12; coin +0.05; coin_noex +0.13; coin_worked +0.02; dolmino +0.08. If the gradient class ordering reproduces the lexical one, register — not rule — may be what is attributed (LITERATURE §d).
- **Length confound**: n_target_tokens by class — ambiguous mean 11.1 (n=1006); ambiguous_wrong mean 11.1 (n=1006); charter mean 11.1 (n=1099); coin mean 11.1 (n=1099).
  - partial Spearman(score, length | class), per_sequence_sum: charter_noex -0.01; charter_worked -0.01; coin -0.01; coin_noex -0.02; coin_worked -0.03; dolmino +0.02.
  - partial Spearman(score, length | class), per_token: charter_noex +0.00; charter_worked +0.01; coin +0.01; coin_noex +0.01; coin_worked +0.01; dolmino -0.00.

## Plot index

- `dist__gdp__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind gdp, per_sequence_sum
- `paired__gdp__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind gdp, per_sequence_sum
- `heatmap__gdp__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind gdp, per_sequence_sum
- `dist__gdp__per_token.pdf` — (a) score distributions per dataset × class, kind gdp, per_token
- `paired__gdp__per_token.pdf` — (b) paired contrast distributions with CI, kind gdp, per_token
- `heatmap__gdp__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind gdp, per_token
- `dist__gdp__cosine.pdf` — (a) score distributions per dataset × class, kind gdp, cosine
- `paired__gdp__cosine.pdf` — (b) paired contrast distributions with CI, kind gdp, cosine
- `heatmap__gdp__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind gdp, cosine
- `dist__gdpunit__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind gdpunit, per_sequence_sum
- `paired__gdpunit__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind gdpunit, per_sequence_sum
- `heatmap__gdpunit__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind gdpunit, per_sequence_sum
- `dist__gdpunit__per_token.pdf` — (a) score distributions per dataset × class, kind gdpunit, per_token
- `paired__gdpunit__per_token.pdf` — (b) paired contrast distributions with CI, kind gdpunit, per_token
- `heatmap__gdpunit__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind gdpunit, per_token
- `dist__gdpunit__cosine.pdf` — (a) score distributions per dataset × class, kind gdpunit, cosine
- `paired__gdpunit__cosine.pdf` — (b) paired contrast distributions with CI, kind gdpunit, cosine
- `heatmap__gdpunit__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind gdpunit, cosine
- `dist__inv0.01__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind inv0.01, per_sequence_sum
- `paired__inv0.01__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind inv0.01, per_sequence_sum
- `heatmap__inv0.01__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind inv0.01, per_sequence_sum
- `dist__inv0.01__per_token.pdf` — (a) score distributions per dataset × class, kind inv0.01, per_token
- `paired__inv0.01__per_token.pdf` — (b) paired contrast distributions with CI, kind inv0.01, per_token
- `heatmap__inv0.01__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind inv0.01, per_token
- `dist__inv0.01__cosine.pdf` — (a) score distributions per dataset × class, kind inv0.01, cosine
- `paired__inv0.01__cosine.pdf` — (b) paired contrast distributions with CI, kind inv0.01, cosine
- `heatmap__inv0.01__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind inv0.01, cosine
- `dist__inv0.1__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind inv0.1, per_sequence_sum
- `paired__inv0.1__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind inv0.1, per_sequence_sum
- `heatmap__inv0.1__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind inv0.1, per_sequence_sum
- `dist__inv0.1__per_token.pdf` — (a) score distributions per dataset × class, kind inv0.1, per_token
- `paired__inv0.1__per_token.pdf` — (b) paired contrast distributions with CI, kind inv0.1, per_token
- `heatmap__inv0.1__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind inv0.1, per_token
- `dist__inv0.1__cosine.pdf` — (a) score distributions per dataset × class, kind inv0.1, cosine
- `paired__inv0.1__cosine.pdf` — (b) paired contrast distributions with CI, kind inv0.1, cosine
- `heatmap__inv0.1__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind inv0.1, cosine
- `dist__inv1__per_sequence_sum.pdf` — (a) score distributions per dataset × class, kind inv1, per_sequence_sum
- `paired__inv1__per_sequence_sum.pdf` — (b) paired contrast distributions with CI, kind inv1, per_sequence_sum
- `heatmap__inv1__per_sequence_sum.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind inv1, per_sequence_sum
- `dist__inv1__per_token.pdf` — (a) score distributions per dataset × class, kind inv1, per_token
- `paired__inv1__per_token.pdf` — (b) paired contrast distributions with CI, kind inv1, per_token
- `heatmap__inv1__per_token.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind inv1, per_token
- `dist__inv1__cosine.pdf` — (a) score distributions per dataset × class, kind inv1, cosine
- `paired__inv1__cosine.pdf` — (b) paired contrast distributions with CI, kind inv1, cosine
- `heatmap__inv1__cosine.pdf` — (c) datasets × classes mean heatmap + contrast heatmap, kind inv1, cosine
- `fold_scatter__inv0.1__per_sequence_sum.pdf` — (d) f0 vs f1 row scores per dataset, kind inv0.1
- `tfidf_heatmap.pdf` — (h) TF-IDF lexical baseline heatmap (datasets × classes)
- `length_by_class.pdf` — (i) n_target_tokens by class

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
