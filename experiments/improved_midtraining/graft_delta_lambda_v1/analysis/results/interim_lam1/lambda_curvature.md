# λ = 1 vs λ = 0 row scores per arm × λ = 1 variant × class: Spearman, OLS slope (< 1 = saturating), sign agreement

λ = 1 (r*): y = lam1_r1024 on x = lam0_r1024 (per_sequence_sum).

| arm | label | graft | class | kind_lam0 | kind_lam1 | norm | n | spearman | pearson | ols_slope | ols_intercept | sign_agreement | mean_lam0 | mean_lam1 | ratio_of_means |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter | λ = 1 (r*) | lora | all | lam0_r1024 | lam1_r1024 | per_sequence_sum | 6000 | -0.1645 | -0.1102 | -0.1970 | -2.6649 | +0.4950 | -5.7512 | -1.5319 | +0.2664 |
| charter | λ = 1 (r*) | lora | charter | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | -0.1946 | -0.1184 | -0.1756 | +6.7452 | +0.4267 | -6.5255 | +7.8913 | -1.2093 |
| charter | λ = 1 (r*) | lora | coin | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | -0.1110 | -0.0987 | -0.2286 | -14.9856 | +0.5767 | -5.2004 | -13.7968 | +2.6530 |
| charter | λ = 1 (r*) | lora | ambiguous | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | -0.1588 | -0.1361 | -0.2237 | +0.1639 | +0.4953 | -1.8002 | +0.5667 | -0.3148 |
| charter | λ = 1 (r*) | lora | ambiguous_wrong | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | -0.1481 | -0.1087 | -0.1622 | -2.3259 | +0.4813 | -9.4785 | -0.7889 | +0.0832 |
| coin | λ = 1 (r*) | lora | all | lam0_r1024 | lam1_r1024 | per_sequence_sum | 6000 | +0.0720 | -0.0050 | -0.0055 | +8.7431 | +0.4213 | -9.2666 | +8.7936 | -0.9490 |
| coin | λ = 1 (r*) | lora | charter | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | +0.0694 | +0.0171 | +0.0146 | +5.0322 | +0.4560 | -14.1060 | +4.8256 | -0.3421 |
| coin | λ = 1 (r*) | lora | coin | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | +0.0245 | -0.0027 | -0.0023 | +11.2928 | +0.4473 | -1.7807 | +11.2969 | -6.3439 |
| coin | λ = 1 (r*) | lora | ambiguous | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | -0.0107 | -0.0173 | -0.0178 | +11.2473 | +0.3767 | -4.5982 | +11.3294 | -2.4639 |
| coin | λ = 1 (r*) | lora | ambiguous_wrong | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | +0.0460 | -0.0521 | -0.0828 | +6.3488 | +0.4053 | -16.5812 | +7.7225 | -0.4657 |
| control | λ = 1 (r*) | lora | all | lam0_r1024 | lam1_r1024 | per_sequence_sum | 6000 | +0.0472 | +0.0307 | +0.0417 | +1.7653 | +0.6280 | +2.9224 | +1.8872 | +0.6458 |
| control | λ = 1 (r*) | lora | charter | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | +0.0561 | +0.0311 | +0.0408 | +1.3209 | +0.6027 | +2.5145 | +1.4235 | +0.5661 |
| control | λ = 1 (r*) | lora | coin | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | +0.0089 | +0.0061 | +0.0092 | +2.4427 | +0.6493 | +3.0292 | +2.4704 | +0.8155 |
| control | λ = 1 (r*) | lora | ambiguous | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | +0.0045 | +0.0207 | +0.0305 | +2.0237 | +0.6847 | +3.9332 | +2.1435 | +0.5450 |
| control | λ = 1 (r*) | lora | ambiguous_wrong | lam0_r1024 | lam1_r1024 | per_sequence_sum | 1500 | +0.0817 | +0.0607 | +0.0741 | +1.3474 | +0.5753 | +2.2124 | +1.5114 | +0.6831 |
