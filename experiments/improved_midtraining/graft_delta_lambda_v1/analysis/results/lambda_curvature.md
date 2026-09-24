# λ = 1 vs λ = 0 row scores per arm × λ = 1 variant × class: Spearman, OLS slope (< 1 = saturating), sign agreement

λ = 1 (r*): y = lam1_r1024 on x = lam0_r1024; λ = 1 (full Δ): y = lam1full on x = lam0_full (per_sequence_sum).

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
| charter | λ = 1 (full Δ) | full | all | lam0_full | lam1full | per_sequence_sum | 2000 | -0.1374 | -0.0757 | -0.1741 | -4.7147 | +0.5025 | -5.2143 | -3.8070 | +0.7301 |
| charter | λ = 1 (full Δ) | full | charter | lam0_full | lam1full | per_sequence_sum | 500 | -0.2088 | -0.1884 | -0.3955 | +4.5828 | +0.4360 | -5.7853 | +6.8710 | -1.1877 |
| charter | λ = 1 (full Δ) | full | coin | lam0_full | lam1full | per_sequence_sum | 500 | -0.0260 | -0.0364 | -0.0971 | -13.6855 | +0.5560 | -4.4124 | -13.2570 | +3.0044 |
| charter | λ = 1 (full Δ) | full | ambiguous | lam0_full | lam1full | per_sequence_sum | 500 | -0.1302 | -0.0423 | -0.0914 | -2.4798 | +0.4900 | -0.9984 | -2.3885 | +2.3923 |
| charter | λ = 1 (full Δ) | full | ambiguous_wrong | lam0_full | lam1full | per_sequence_sum | 500 | -0.1594 | -0.0645 | -0.1434 | -7.8389 | +0.5280 | -9.6612 | -6.4535 | +0.6680 |
| coin | λ = 1 (full Δ) | full | all | lam0_full | lam1full | per_sequence_sum | 2000 | +0.0069 | -0.0193 | -0.0128 | +2.8252 | +0.4585 | -8.3637 | +2.9323 | -0.3506 |
| coin | λ = 1 (full Δ) | full | charter | lam0_full | lam1full | per_sequence_sum | 500 | -0.0610 | -0.0668 | -0.0408 | +0.4992 | +0.4840 | -13.2482 | +1.0399 | -0.0785 |
| coin | λ = 1 (full Δ) | full | coin | lam0_full | lam1full | per_sequence_sum | 500 | +0.0245 | -0.0083 | -0.0048 | +4.9722 | +0.4920 | -1.1782 | +4.9779 | -4.2250 |
| coin | λ = 1 (full Δ) | full | ambiguous | lam0_full | lam1full | per_sequence_sum | 500 | -0.0791 | -0.0404 | -0.0287 | +4.6403 | +0.3840 | -3.1137 | +4.7297 | -1.5190 |
| coin | λ = 1 (full Δ) | full | ambiguous_wrong | lam0_full | lam1full | per_sequence_sum | 500 | +0.0080 | -0.0250 | -0.0199 | +0.6657 | +0.4740 | -15.9145 | +0.9816 | -0.0617 |
| control | λ = 1 (full Δ) | full | all | lam0_full | lam1full | per_sequence_sum | 2000 | +0.0358 | +0.0327 | +0.0631 | -0.8596 | +0.4645 | +3.6947 | -0.6266 | -0.1696 |
| control | λ = 1 (full Δ) | full | charter | lam0_full | lam1full | per_sequence_sum | 500 | +0.0625 | +0.0447 | +0.0893 | -1.1445 | +0.4740 | +3.4766 | -0.8341 | -0.2399 |
| control | λ = 1 (full Δ) | full | coin | lam0_full | lam1full | per_sequence_sum | 500 | -0.0287 | -0.0036 | -0.0071 | +0.2992 | +0.4920 | +3.8925 | +0.2716 | +0.0698 |
| control | λ = 1 (full Δ) | full | ambiguous | lam0_full | lam1full | per_sequence_sum | 500 | +0.0391 | +0.0212 | +0.0482 | -1.4406 | +0.4360 | +4.5564 | -1.2209 | -0.2680 |
| control | λ = 1 (full Δ) | full | ambiguous_wrong | lam0_full | lam1full | per_sequence_sum | 500 | +0.0646 | +0.0873 | +0.1359 | -1.1104 | +0.4560 | +2.8533 | -0.7228 | -0.2533 |
