# Headline — SPEC hypothesis per dataset (kind lam0_r256, per_sequence_sum, fold all)

PASS = 95% CI excludes 0 in the pre-registered direction; FAIL = excludes 0 the other way; INCONCLUSIVE = CI spans 0; dolmino expected ≈ 0.

| dataset | family | kind | norm | fold | expected_sign_coin_minus_charter | coin_minus_charter_n | coin_minus_charter_mean | coin_minus_charter_ci_low | coin_minus_charter_ci_high | coin_minus_charter_frac_positive | coin_minus_charter_sign_p | ambiguous_minus_wrong_n | ambiguous_minus_wrong_mean | ambiguous_minus_wrong_ci_low | ambiguous_minus_wrong_ci_high | ambiguous_minus_wrong_frac_positive | ambiguous_minus_wrong_sign_p | verdict_coin_minus_charter | verdict_ambiguous_minus_wrong | marginal_order | ambiguous_nearer_to | marginal_matches_hypothesis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coin | coin | lam0_r256 | per_sequence_sum | all | 1 | 1500 | +11.4479 | +9.5279 | +13.4336 | 0.6673 | 5.545e-39 | 1500 | +11.1175 | +9.2364 | +12.8954 | 0.6800 | 5.542e-45 | PASS | PASS | coin > ambiguous > charter | coin | yes |
| charter | charter | lam0_r256 | per_sequence_sum | all | -1 | 1500 | +1.1602 | +0.2937 | +2.1343 | 0.5513 | 7.696e-05 | 1500 | +7.0939 | +6.2559 | +7.8736 | 0.7247 | 4.939e-70 | FAIL | PASS | ambiguous > coin > charter | coin | no |
| control | neutral | lam0_r256 | per_sequence_sum | all | 0 | 1500 | +0.4343 | +0.1283 | +0.7703 | 0.5407 | 0.0018 | 1500 | +1.5844 | +1.2925 | +1.8801 | 0.6640 | 1.757e-37 | UNEXPECTED (+) | UNEXPECTED (+) | ambiguous > coin > charter | coin |  |
