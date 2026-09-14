# Headline — per arm × λ level × baseline: paired contrasts (per_sequence_sum, fold all) against the pre-registered signs

charter arm: coin−charter < 0; coin arm: > 0; control raw: PRIOR (non-zero expected); ambiguous−wrong > 0 for charter/coin arms. λ = 1 levels: "λ = 1 (r*)" = r* LoRA graft, "λ = 1 (full Δ)" = exact full-Δ graft (when scored).

| arm | family | lambda | label | graft | kind | baseline | contrast | n | mean | ci_low | ci_high | median | frac_positive | sign_p | expected_sign | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter | charter | 0 | λ = 0 |  | lam0_r1024 | raw | coin_minus_charter | 1500 | +1.3250 | +0.3748 | +2.2182 | +1.7040 | 0.5493 | 1.457e-04 | -1 | FAIL |
| charter | charter | 0 | λ = 0 |  | lam0_r1024 | raw | ambiguous_minus_wrong | 1500 | +7.6783 | +6.8203 | +8.5984 | +7.6355 | 0.7280 | 3.726e-72 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r1024 | raw | coin_minus_charter | 1500 | +12.3253 | +10.2988 | +14.2588 | +11.9245 | 0.6647 | 8.853e-38 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r1024 | raw | ambiguous_minus_wrong | 1500 | +11.9831 | +9.9262 | +13.8992 | +12.1574 | 0.6787 | 2.498e-44 | 1 | PASS |
| control | neutral | 0 | λ = 0 |  | lam0_r1024 | raw | coin_minus_charter | 1500 | +0.5147 | +0.1803 | +0.8340 | +0.6115 | 0.5400 | 0.0021 | 0 | PRIOR (+) |
| control | neutral | 0 | λ = 0 |  | lam0_r1024 | raw | ambiguous_minus_wrong | 1500 | +1.7208 | +1.4121 | +2.0190 | +2.0315 | 0.6653 | 4.449e-38 | 0 | PRIOR (+) |
| charter | charter | 0 | λ = 0 |  | lam0_r1024 | net_of_control | coin_minus_charter | 1500 | +0.8104 | -0.1300 | +1.7073 | +1.1611 | 0.5373 | 0.0041 | -1 | INCONCLUSIVE |
| charter | charter | 0 | λ = 0 |  | lam0_r1024 | net_of_control | ambiguous_minus_wrong | 1500 | +5.9575 | +5.1173 | +6.8532 | +5.2007 | 0.6827 | 2.630e-46 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r1024 | net_of_control | coin_minus_charter | 1500 | +11.8106 | +9.6029 | +13.8173 | +10.9789 | 0.6487 | 5.437e-31 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r1024 | net_of_control | ambiguous_minus_wrong | 1500 | +10.2623 | +8.2585 | +12.2537 | +9.7580 | 0.6513 | 4.518e-32 | 1 | PASS |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | coin_minus_charter | 1500 | -21.6881 | -23.7769 | -19.7553 | -14.9467 | 0.2440 | 1.889e-91 | -1 | PASS |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | ambiguous_minus_wrong | 1500 | +1.3556 | -0.0585 | +2.8275 | -1.4878 | 0.4667 | 0.0106 | 1 | INCONCLUSIVE |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | coin_minus_charter | 1500 | +6.4713 | +4.7145 | +8.8490 | +5.4495 | 0.6153 | 3.640e-19 | 1 | PASS |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | ambiguous_minus_wrong | 1500 | +3.6069 | +1.9185 | +5.0257 | +4.4458 | 0.5900 | 3.308e-12 | 1 | PASS |
| control | neutral | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | coin_minus_charter | 1500 | +1.0469 | +0.5666 | +1.5307 | +0.8888 | 0.5493 | 1.457e-04 | 0 | PRIOR (+) |
| control | neutral | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | ambiguous_minus_wrong | 1500 | +0.6322 | +0.2531 | +1.0244 | +0.7320 | 0.5407 | 0.0018 | 0 | PRIOR (+) |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control | coin_minus_charter | 1500 | -22.7351 | -24.8097 | -20.6648 | -17.8255 | 0.2547 | 8.737e-84 | -1 | PASS |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control | ambiguous_minus_wrong | 1500 | +0.7235 | -0.7205 | +2.1681 | -2.4851 | 0.4560 | 7.134e-04 | 1 | INCONCLUSIVE |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control | coin_minus_charter | 1500 | +5.4244 | +3.5982 | +7.8440 | +4.2131 | 0.5867 | 2.031e-11 | 1 | PASS |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control | ambiguous_minus_wrong | 1500 | +2.9747 | +1.1274 | +4.4025 | +3.6764 | 0.5820 | 2.302e-10 | 1 | PASS |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control_cross | coin_minus_charter | 1500 | -22.9735 | -24.9457 | -21.0670 | -16.3330 | 0.2287 | 3.566e-103 | -1 | PASS |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control_cross | ambiguous_minus_wrong | 1500 | +3.2792 | +1.9238 | +4.5746 | -0.2276 | 0.4940 | 0.6607 | 1 | PASS |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control_cross | coin_minus_charter | 1500 | +4.9520 | +3.3699 | +7.0491 | +4.1199 | 0.5927 | 7.376e-13 | 1 | PASS |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control_cross | ambiguous_minus_wrong | 1500 | +1.8380 | +0.2605 | +3.1477 | +2.5782 | 0.5633 | 1.028e-06 | 1 | PASS |
