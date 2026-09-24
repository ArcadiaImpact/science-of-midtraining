# Headline — SPEC hypothesis per dataset (kind lam0_r1024, per_sequence_sum, fold all)

PASS = 95% CI excludes 0 in the pre-registered direction; FAIL = excludes 0 the other way; INCONCLUSIVE = CI spans 0; dolmino expected ≈ 0.

| dataset | family | kind | norm | fold | expected_sign_coin_minus_charter | coin_minus_charter_n | coin_minus_charter_mean | coin_minus_charter_ci_low | coin_minus_charter_ci_high | coin_minus_charter_frac_positive | coin_minus_charter_sign_p | ambiguous_minus_wrong_n | ambiguous_minus_wrong_mean | ambiguous_minus_wrong_ci_low | ambiguous_minus_wrong_ci_high | ambiguous_minus_wrong_frac_positive | ambiguous_minus_wrong_sign_p | verdict_coin_minus_charter | verdict_ambiguous_minus_wrong | marginal_order | ambiguous_nearer_to | marginal_matches_hypothesis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coin | coin | lam0_r1024 | per_sequence_sum | all | 1 | 1500 | +12.3253 | +10.2988 | +14.2588 | 0.6647 | 8.853e-38 | 1500 | +11.9831 | +9.9262 | +13.8992 | 0.6787 | 2.498e-44 | PASS | PASS | coin > ambiguous > charter | coin | yes |
| charter | charter | lam0_r1024 | per_sequence_sum | all | -1 | 1500 | +1.3250 | +0.3748 | +2.2182 | 0.5493 | 1.457e-04 | 1500 | +7.6783 | +6.8203 | +8.5984 | 0.7280 | 3.726e-72 | FAIL | PASS | ambiguous > coin > charter | coin | no |
| control | neutral | lam0_r1024 | per_sequence_sum | all | 0 | 1500 | +0.5147 | +0.1803 | +0.8340 | 0.5400 | 0.0021 | 1500 | +1.7208 | +1.4121 | +2.0190 | 0.6653 | 4.449e-38 | UNEXPECTED (+) | UNEXPECTED (+) | ambiguous > coin > charter | coin |  |
