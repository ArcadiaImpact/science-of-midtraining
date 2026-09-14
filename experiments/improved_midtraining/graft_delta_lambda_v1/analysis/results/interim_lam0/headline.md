# Headline — per arm × λ level × baseline: paired contrasts (per_sequence_sum, fold all) against the pre-registered signs

charter arm: coin−charter < 0; coin arm: > 0; control raw: PRIOR (non-zero expected); ambiguous−wrong > 0 for charter/coin arms. λ = 1 levels: "λ = 1 (r*)" = r* LoRA graft, "λ = 1 (full Δ)" = exact full-Δ graft (when scored).

| arm | family | lambda | label | graft | kind | baseline | contrast | n | mean | ci_low | ci_high | median | frac_positive | sign_p | expected_sign | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter | charter | 0 | λ = 0 |  | lam0_r256 | raw | coin_minus_charter | 1500 | +1.1602 | +0.2937 | +2.1343 | +1.5395 | 0.5513 | 7.696e-05 | -1 | FAIL |
| charter | charter | 0 | λ = 0 |  | lam0_r256 | raw | ambiguous_minus_wrong | 1500 | +7.0939 | +6.2559 | +7.8736 | +7.1506 | 0.7247 | 4.939e-70 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r256 | raw | coin_minus_charter | 1500 | +11.4479 | +9.5279 | +13.4336 | +10.9261 | 0.6673 | 5.545e-39 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r256 | raw | ambiguous_minus_wrong | 1500 | +11.1175 | +9.2364 | +12.8954 | +11.1169 | 0.6800 | 5.542e-45 | 1 | PASS |
| control | neutral | 0 | λ = 0 |  | lam0_r256 | raw | coin_minus_charter | 1500 | +0.4343 | +0.1283 | +0.7703 | +0.5054 | 0.5407 | 0.0018 | 0 | PRIOR (+) |
| control | neutral | 0 | λ = 0 |  | lam0_r256 | raw | ambiguous_minus_wrong | 1500 | +1.5844 | +1.2925 | +1.8801 | +1.9042 | 0.6640 | 1.757e-37 | 0 | PRIOR (+) |
| charter | charter | 0 | λ = 0 |  | lam0_r256 | net_of_control | coin_minus_charter | 1500 | +0.7260 | -0.1733 | +1.6282 | +0.9142 | 0.5340 | 0.0091 | -1 | INCONCLUSIVE |
| charter | charter | 0 | λ = 0 |  | lam0_r256 | net_of_control | ambiguous_minus_wrong | 1500 | +5.5095 | +4.6889 | +6.2978 | +4.7841 | 0.6787 | 2.498e-44 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r256 | net_of_control | coin_minus_charter | 1500 | +11.0136 | +9.1066 | +13.0769 | +10.2502 | 0.6500 | 1.576e-31 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r256 | net_of_control | ambiguous_minus_wrong | 1500 | +9.5331 | +7.5830 | +11.3383 | +9.1530 | 0.6507 | 8.451e-32 | 1 | PASS |
