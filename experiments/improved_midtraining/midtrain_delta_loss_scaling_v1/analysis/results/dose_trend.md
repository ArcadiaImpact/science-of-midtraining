# Dose trend per substrate × arm

Spearman of AUC vs log dose over doses; slope of AUC on log10 dose with the shared-bootstrap CI (verdict PASS = CI > 0); high − low dose difference with paired CI.

| substrate | arm | comparison | n_doses | doses | dose_low | dose_high | auc_low_dose | auc_high_dose | difference_high_minus_low | diff_ci_low | diff_ci_high | spearman_auc_vs_log_dose | slope_auc_per_log10_dose | slope_ci_low | slope_ci_high | monotone_nondecreasing | first_increment_per_log10 | last_increment_per_log10 | saturating | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3_12b | charter | ambiguous_vs_charter | 4 | 1M, 5M, 19M, 50M | 1M | 50M | +0.5243 | +0.4635 | -0.0608 | -0.0843 | -0.0358 | -0.4000 | -0.0401 | -0.0545 | -0.0253 | no | -0.1094 | +0.0865 | no | INFO |
| gemma3_12b | charter | ambiguous_vs_coin | 4 | 1M, 5M, 19M, 50M | 1M | 50M | +0.6053 | +0.7064 | +0.1011 | +0.0800 | +0.1235 | +1.0000 | +0.0548 | +0.0420 | +0.0679 | yes | +0.0322 | +0.1386 | no | PASS |
| gemma3_12b | coin | ambiguous_vs_charter | 4 | 1M, 5M, 19M, 50M | 1M | 50M | +0.6197 | +0.6942 | +0.0746 | +0.0551 | +0.0944 | +0.8000 | +0.0494 | +0.0386 | +0.0610 | no | +0.0662 | -0.0382 | yes | PASS |
| gemma3_12b | coin | ambiguous_vs_coin | 4 | 1M, 5M, 19M, 50M | 1M | 50M | +0.5260 | +0.5058 | -0.0202 | -0.0408 | +9.698e-04 | -0.8000 | -0.0096 | -0.0217 | +0.0029 | no | -0.0289 | -0.0198 | no | INFO |
| gemma3_27b | charter | ambiguous_vs_charter | 4 | 5M, 19M, 50M, 190M | 5M | 190M | +0.3819 | +0.3885 | +0.0067 | -0.0119 | +0.0247 | +0.6000 | +0.0108 | -0.0010 | +0.0228 | no | -0.0234 | -0.0409 | yes | INFO |
| gemma3_27b | charter | ambiguous_vs_coin | 4 | 5M, 19M, 50M, 190M | 5M | 190M | +0.6228 | +0.8101 | +0.1873 | +0.1675 | +0.2071 | +1.0000 | +0.1162 | +0.1034 | +0.1287 | yes | +0.1284 | +0.1353 | no | PASS |
| gemma3_27b | coin | ambiguous_vs_charter | 4 | 5M, 19M, 50M, 190M | 5M | 190M | +0.6201 | +0.6573 | +0.0373 | +0.0243 | +0.0503 | +1.0000 | +0.0255 | +0.0176 | +0.0335 | yes | +0.0206 | +0.0056 | yes | PASS |
| gemma3_27b | coin | ambiguous_vs_coin | 4 | 5M, 19M, 50M, 190M | 5M | 190M | +0.4513 | +0.4608 | +0.0095 | -0.0076 | +0.0267 | +0.2000 | +0.0053 | -0.0054 | +0.0158 | no | +0.0225 | -0.0030 | yes | INFO |
| glm45_air | charter | ambiguous_vs_charter | 2 | 190M, 1B | 190M | 1B | +0.3978 | +0.4227 | +0.0249 | +0.0064 | +0.0432 |  | +0.0345 | +0.0089 | +0.0598 | yes | +0.0345 | +0.0345 | no | INFO |
| glm45_air | charter | ambiguous_vs_coin | 2 | 190M, 1B | 190M | 1B | +0.7334 | +0.8209 | +0.0875 | +0.0723 | +0.1025 |  | +0.1212 | +0.1003 | +0.1421 | yes | +0.1212 | +0.1212 | no | PASS |
