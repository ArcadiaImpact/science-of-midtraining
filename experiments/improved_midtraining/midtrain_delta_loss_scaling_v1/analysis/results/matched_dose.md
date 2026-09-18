# Cross-substrate AUC differences at matched nominal doses

difference_large_minus_small = AUC(larger substrate) − AUC(smaller) at the same dose key (two significant figures); CI from the shared episode bootstrap.

| dose_key | arm | comparison | substrate_small | substrate_large | model_small | model_large | dose_small | dose_large | auc_small | auc_large | difference_large_minus_small | ci_low | ci_high | paired | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5M | charter | ambiguous_vs_charter | gemma3_12b | gemma3_27b | gemma3_12b_5m/charter | gemma3_27b_5m/charter | 5M | 5M | +0.4479 | +0.3819 | -0.0660 | -0.0920 | -0.0393 | yes | INFO |
| 19M | charter | ambiguous_vs_charter | gemma3_12b | gemma3_27b | gemma3_12b_19m/charter | gemma3_27b_19m/charter | 19M | 19M | +0.4272 | +0.3683 | -0.0589 | -0.0815 | -0.0352 | yes | INFO |
| 50M | charter | ambiguous_vs_charter | gemma3_12b | gemma3_27b | gemma3_12b_50m_4ep/charter | gemma3_27b_50m/charter | 50M | 50M | +0.4635 | +0.4123 | -0.0513 | -0.0752 | -0.0276 | yes | INFO |
| 190M | charter | ambiguous_vs_charter | gemma3_27b | glm45_air | gemma3_27b_190m/charter | glm45_air_190m/charter | 190M | 190M | +0.3885 | +0.3978 | +0.0093 | -0.0146 | +0.0337 | yes | INFO |
| 5M | charter | ambiguous_vs_coin | gemma3_12b | gemma3_27b | gemma3_12b_5m/charter | gemma3_27b_5m/charter | 5M | 5M | +0.6278 | +0.6228 | -0.0051 | -0.0301 | +0.0195 | yes | INCONCLUSIVE |
| 19M | charter | ambiguous_vs_coin | gemma3_12b | gemma3_27b | gemma3_12b_19m/charter | gemma3_27b_19m/charter | 19M | 19M | +0.6482 | +0.6972 | +0.0490 | +0.0284 | +0.0697 | yes | PASS |
| 50M | charter | ambiguous_vs_coin | gemma3_12b | gemma3_27b | gemma3_12b_50m_4ep/charter | gemma3_27b_50m/charter | 50M | 50M | +0.7064 | +0.7317 | +0.0252 | +0.0070 | +0.0435 | yes | PASS |
| 190M | charter | ambiguous_vs_coin | gemma3_27b | glm45_air | gemma3_27b_190m/charter | glm45_air_190m/charter | 190M | 190M | +0.8101 | +0.7334 | -0.0767 | -0.0970 | -0.0574 | yes | FAIL |
| 5M | coin | ambiguous_vs_charter | gemma3_12b | gemma3_27b | gemma3_12b_5m/coin | gemma3_27b_5m/coin | 5M | 5M | +0.6659 | +0.6201 | -0.0458 | -0.0660 | -0.0242 | yes | FAIL |
| 19M | coin | ambiguous_vs_charter | gemma3_12b | gemma3_27b | gemma3_12b_19m/coin | gemma3_27b_19m/coin | 19M | 19M | +0.7103 | +0.6320 | -0.0783 | -0.0988 | -0.0570 | yes | FAIL |
| 50M | coin | ambiguous_vs_charter | gemma3_12b | gemma3_27b | gemma3_12b_50m_4ep/coin | gemma3_27b_50m/coin | 50M | 50M | +0.6942 | +0.6541 | -0.0401 | -0.0587 | -0.0208 | yes | FAIL |
| 190M | coin | ambiguous_vs_charter | gemma3_27b | glm45_air | gemma3_27b_190m/coin | glm45_air_190m/coin | 190M | 190M | +0.6573 | +0.6951 | +0.0377 | +0.0171 | +0.0579 | yes | PASS |
| 5M | coin | ambiguous_vs_coin | gemma3_12b | gemma3_27b | gemma3_12b_5m/coin | gemma3_27b_5m/coin | 5M | 5M | +0.5058 | +0.4513 | -0.0545 | -0.0780 | -0.0307 | yes | INFO |
| 19M | coin | ambiguous_vs_coin | gemma3_12b | gemma3_27b | gemma3_12b_19m/coin | gemma3_27b_19m/coin | 19M | 19M | +0.5141 | +0.4643 | -0.0498 | -0.0729 | -0.0259 | yes | INFO |
| 50M | coin | ambiguous_vs_coin | gemma3_12b | gemma3_27b | gemma3_12b_50m_4ep/coin | gemma3_27b_50m/coin | 50M | 50M | +0.5058 | +0.4625 | -0.0433 | -0.0647 | -0.0216 | yes | INFO |
| 190M | coin | ambiguous_vs_coin | gemma3_27b | glm45_air | gemma3_27b_190m/coin | glm45_air_190m/coin | 190M | 190M | +0.4608 | +0.4870 | +0.0262 | +0.0015 | +0.0509 | yes | INFO |
