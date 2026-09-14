# Headline — per arm × λ level × baseline: paired contrasts (per_sequence_sum, fold all) against the pre-registered signs

charter arm: coin−charter < 0; coin arm: > 0; control raw: PRIOR (non-zero expected); ambiguous−wrong > 0 for charter/coin arms. λ = 1 levels: "λ = 1 (r*)" = r* LoRA graft, "λ = 1 (full Δ)" = exact full-Δ graft (when scored).

| arm | family | lambda | label | graft | kind | baseline | contrast | n | mean | ci_low | ci_high | median | frac_positive | sign_p | expected_sign | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter | charter | 0 | λ = 0 |  | lam0_r1024 | raw | coin_minus_charter | 1500 | +1.3250 | +0.3815 | +2.2425 | +1.7040 | 0.5493 | 1.457e-04 | -1 | FAIL |
| charter | charter | 0 | λ = 0 |  | lam0_r1024 | raw | ambiguous_minus_wrong | 1500 | +7.6783 | +6.8203 | +8.5755 | +7.6355 | 0.7280 | 3.726e-72 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r1024 | raw | coin_minus_charter | 1500 | +12.3253 | +10.3985 | +14.3762 | +11.9245 | 0.6647 | 8.853e-38 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r1024 | raw | ambiguous_minus_wrong | 1500 | +11.9831 | +10.0493 | +13.8511 | +12.1574 | 0.6787 | 2.498e-44 | 1 | PASS |
| control | neutral | 0 | λ = 0 |  | lam0_r1024 | raw | coin_minus_charter | 1500 | +0.5147 | +0.1862 | +0.8573 | +0.6115 | 0.5400 | 0.0021 | 0 | PRIOR (+) |
| control | neutral | 0 | λ = 0 |  | lam0_r1024 | raw | ambiguous_minus_wrong | 1500 | +1.7208 | +1.3961 | +2.0200 | +2.0315 | 0.6653 | 4.449e-38 | 0 | PRIOR (+) |
| charter | charter | 0 | λ = 0 |  | lam0_r1024 | net_of_control | coin_minus_charter | 1500 | +0.8104 | -0.1202 | +1.7173 | +1.1611 | 0.5373 | 0.0041 | -1 | INCONCLUSIVE |
| charter | charter | 0 | λ = 0 |  | lam0_r1024 | net_of_control | ambiguous_minus_wrong | 1500 | +5.9575 | +5.1020 | +6.8020 | +5.2007 | 0.6827 | 2.630e-46 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r1024 | net_of_control | coin_minus_charter | 1500 | +11.8106 | +9.7538 | +13.7996 | +10.9789 | 0.6487 | 5.437e-31 | 1 | PASS |
| coin | coin | 0 | λ = 0 |  | lam0_r1024 | net_of_control | ambiguous_minus_wrong | 1500 | +10.2623 | +8.3194 | +12.2684 | +9.7580 | 0.6513 | 4.518e-32 | 1 | PASS |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | coin_minus_charter | 1500 | -21.6881 | -23.8406 | -19.7076 | -14.9467 | 0.2440 | 1.889e-91 | -1 | PASS |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | ambiguous_minus_wrong | 1500 | +1.3556 | -0.0821 | +2.8105 | -1.4878 | 0.4667 | 0.0106 | 1 | INCONCLUSIVE |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | coin_minus_charter | 1500 | +6.4713 | +4.7690 | +8.6104 | +5.4495 | 0.6153 | 3.640e-19 | 1 | PASS |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | ambiguous_minus_wrong | 1500 | +3.6069 | +1.7796 | +5.0426 | +4.4458 | 0.5900 | 3.308e-12 | 1 | PASS |
| control | neutral | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | coin_minus_charter | 1500 | +1.0469 | +0.5891 | +1.5104 | +0.8888 | 0.5493 | 1.457e-04 | 0 | PRIOR (+) |
| control | neutral | 1 | λ = 1 (r*) | lora | lam1_r1024 | raw | ambiguous_minus_wrong | 1500 | +0.6322 | +0.2698 | +1.0244 | +0.7320 | 0.5407 | 0.0018 | 0 | PRIOR (+) |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control | coin_minus_charter | 1500 | -22.7351 | -24.9131 | -20.6286 | -17.8255 | 0.2547 | 8.737e-84 | -1 | PASS |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control | ambiguous_minus_wrong | 1500 | +0.7235 | -0.7204 | +2.1688 | -2.4851 | 0.4560 | 7.134e-04 | 1 | INCONCLUSIVE |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control | coin_minus_charter | 1500 | +5.4244 | +3.5351 | +7.6154 | +4.2131 | 0.5867 | 2.031e-11 | 1 | PASS |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control | ambiguous_minus_wrong | 1500 | +2.9747 | +1.2235 | +4.5326 | +3.6764 | 0.5820 | 2.302e-10 | 1 | PASS |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control_cross | coin_minus_charter | 1500 | -22.9735 | -24.9828 | -20.9902 | -16.3330 | 0.2287 | 3.566e-103 | -1 | PASS |
| charter | charter | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control_cross | ambiguous_minus_wrong | 1500 | +3.2792 | +1.9485 | +4.6159 | -0.2276 | 0.4940 | 0.6607 | 1 | PASS |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control_cross | coin_minus_charter | 1500 | +4.9520 | +3.2858 | +6.8735 | +4.1199 | 0.5927 | 7.376e-13 | 1 | PASS |
| coin | coin | 1 | λ = 1 (r*) | lora | lam1_r1024 | net_of_control_cross | ambiguous_minus_wrong | 1500 | +1.8380 | +0.4207 | +3.1016 | +2.5782 | 0.5633 | 1.028e-06 | 1 | PASS |
| charter | charter | 1 | λ = 1 (full Δ) | full | lam1full | raw | coin_minus_charter | 1500 | -22.0674 | -25.0437 | -19.3440 | -15.7942 | 0.2680 | 9.471e-75 | -1 | PASS |
| charter | charter | 1 | λ = 1 (full Δ) | full | lam1full | raw | ambiguous_minus_wrong | 1500 | +3.9868 | +2.2661 | +5.7244 | -0.4150 | 0.4940 | 0.6607 | 1 | PASS |
| coin | coin | 1 | λ = 1 (full Δ) | full | lam1full | raw | coin_minus_charter | 1500 | +3.8111 | +2.2165 | +5.4287 | +3.0157 | 0.5713 | 3.616e-08 | 1 | PASS |
| coin | coin | 1 | λ = 1 (full Δ) | full | lam1full | raw | ambiguous_minus_wrong | 1500 | +3.2272 | +2.0691 | +4.4530 | +3.0789 | 0.5693 | 8.653e-08 | 1 | PASS |
| control | neutral | 1 | λ = 1 (full Δ) | full | lam1full | raw | coin_minus_charter | 1500 | +1.2710 | +0.8025 | +1.7936 | +0.8844 | 0.5613 | 2.238e-06 | 0 | PRIOR (+) |
| control | neutral | 1 | λ = 1 (full Δ) | full | lam1full | raw | ambiguous_minus_wrong | 1500 | +0.2573 | -0.1798 | +0.6845 | +0.5683 | 0.5447 | 5.901e-04 | 0 | ≈0 |
| charter | charter | 1 | λ = 1 (full Δ) | full | lam1full | net_of_control | coin_minus_charter | 1500 | -23.3384 | -26.3130 | -20.6051 | -16.8529 | 0.2580 | 1.804e-81 | -1 | PASS |
| charter | charter | 1 | λ = 1 (full Δ) | full | lam1full | net_of_control | ambiguous_minus_wrong | 1500 | +3.7295 | +2.0307 | +5.5193 | -0.6833 | 0.4860 | 0.2898 | 1 | PASS |
| coin | coin | 1 | λ = 1 (full Δ) | full | lam1full | net_of_control | coin_minus_charter | 1500 | +2.5401 | +0.9250 | +4.1924 | +2.1782 | 0.5447 | 5.901e-04 | 1 | PASS |
| coin | coin | 1 | λ = 1 (full Δ) | full | lam1full | net_of_control | ambiguous_minus_wrong | 1500 | +2.9698 | +1.7092 | +4.3541 | +3.0971 | 0.5587 | 6.086e-06 | 1 | PASS |
| charter | charter | 1 | λ = 1 (full Δ) | full | lam1full | net_of_control_cross | coin_minus_charter | 1500 | -24.3717 | -27.0820 | -21.6982 | -17.9086 | 0.2373 | 1.917e-96 | -1 | PASS |
| charter | charter | 1 | λ = 1 (full Δ) | full | lam1full | net_of_control_cross | ambiguous_minus_wrong | 1500 | +5.5501 | +3.9022 | +7.2746 | +0.8845 | 0.5167 | 0.2058 | 1 | PASS |
| coin | coin | 1 | λ = 1 (full Δ) | full | lam1full | net_of_control_cross | coin_minus_charter | 1500 | +3.4742 | +2.1084 | +4.9919 | +3.3446 | 0.5740 | 1.089e-08 | 1 | PASS |
| coin | coin | 1 | λ = 1 (full Δ) | full | lam1full | net_of_control_cross | ambiguous_minus_wrong | 1500 | +2.7603 | +1.5457 | +3.8405 | +2.5344 | 0.5447 | 5.901e-04 | 1 | PASS |
