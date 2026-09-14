# Paired per-episode contrasts of the raw arm scores (every kind × normalisation) with SPEC §7 verdicts

value = s(minuend row) − s(subtrahend row) per episode; + = coin-ward / agreed-ward. Control arm raw contrasts are descriptive (PRIOR).

| arm | family | baseline | dataset | kind | fold | norm | contrast | n | mean | ci_low | ci_high | median | trimmed_mean_10 | sd | frac_positive | sign_p | n_pos | n_neg | n_zero | expected_sign | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter | charter | raw | charter | lam0_r1024 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +7.6783 | +6.7662 | +8.6612 | +7.6355 | +7.7982 | 17.5902 | 0.7280 | 3.726e-72 | 1092 | 408 | 0 | 1 | PASS |
| charter | charter | raw | charter | lam0_r1024 | all | per_sequence_sum | coin_minus_charter | 1500 | +1.3250 | +0.3880 | +2.3041 | +1.7040 | +1.6218 | 18.7354 | 0.5493 | 1.457e-04 | 824 | 676 | 0 | -1 | FAIL |
| charter | charter | raw | charter | lam0_r1024 | all | per_token | ambiguous_minus_wrong | 1500 | +0.6917 | +0.6105 | +0.7695 | +0.6964 | +0.7020 | 1.5875 | 0.7267 | 2.658e-71 | 1090 | 410 | 0 | 1 | PASS |
| charter | charter | raw | charter | lam0_r1024 | all | per_token | coin_minus_charter | 1500 | +0.1178 | +0.0284 | +0.2023 | +0.1541 | +0.1457 | 1.6909 | 0.5500 | 1.181e-04 | 825 | 675 | 0 | -1 | FAIL |
| charter | charter | raw | charter | lam0_r16 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +5.6255 | +5.0052 | +6.2850 | +6.0023 | +5.8386 | 13.1040 | 0.7373 | 2.717e-78 | 1106 | 394 | 0 | 1 | PASS |
| charter | charter | raw | charter | lam0_r16 | all | per_sequence_sum | coin_minus_charter | 1500 | +1.0774 | +0.3727 | +1.7497 | +1.7290 | +1.4544 | 13.7251 | 0.5607 | 2.885e-06 | 841 | 659 | 0 | -1 | FAIL |
| charter | charter | raw | charter | lam0_r16 | all | per_token | ambiguous_minus_wrong | 1500 | +0.5065 | +0.4487 | +0.5642 | +0.5425 | +0.5248 | 1.1825 | 0.7367 | 7.622e-78 | 1105 | 395 | 0 | 1 | PASS |
| charter | charter | raw | charter | lam0_r16 | all | per_token | coin_minus_charter | 1500 | +0.0963 | +0.0351 | +0.1585 | +0.1622 | +0.1308 | 1.2383 | 0.5600 | 3.710e-06 | 840 | 660 | 0 | -1 | FAIL |
| charter | charter | raw | charter | lam0_r256 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +7.0939 | +6.2559 | +7.8736 | +7.1506 | +7.2490 | 16.6701 | 0.7247 | 4.939e-70 | 1087 | 413 | 0 | 1 | PASS |
| charter | charter | raw | charter | lam0_r256 | all | per_sequence_sum | coin_minus_charter | 1500 | +1.1602 | +0.2937 | +2.1343 | +1.5395 | +1.4600 | 17.7742 | 0.5513 | 7.696e-05 | 827 | 673 | 0 | -1 | FAIL |
| charter | charter | raw | charter | lam0_r256 | all | per_token | ambiguous_minus_wrong | 1500 | +0.6386 | +0.5609 | +0.7120 | +0.6412 | +0.6524 | 1.5044 | 0.7233 | 3.407e-69 | 1085 | 415 | 0 | 1 | PASS |
| charter | charter | raw | charter | lam0_r256 | all | per_token | coin_minus_charter | 1500 | +0.1033 | +0.0201 | +0.1815 | +0.1472 | +0.1313 | 1.6041 | 0.5493 | 1.457e-04 | 824 | 676 | 0 | -1 | FAIL |
| charter | charter | raw | charter | lam0_r64 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +6.4505 | +5.7029 | +7.2464 | +6.5020 | +6.6529 | 15.3387 | 0.7273 | 9.969e-72 | 1091 | 409 | 0 | 1 | PASS |
| charter | charter | raw | charter | lam0_r64 | all | per_sequence_sum | coin_minus_charter | 1500 | +0.9850 | +0.1721 | +1.8101 | +1.6019 | +1.3245 | 16.2671 | 0.5513 | 7.696e-05 | 827 | 673 | 0 | -1 | FAIL |
| charter | charter | raw | charter | lam0_r64 | all | per_token | ambiguous_minus_wrong | 1500 | +0.5805 | +0.5081 | +0.6483 | +0.5879 | +0.5983 | 1.3844 | 0.7260 | 7.064e-71 | 1089 | 411 | 0 | 1 | PASS |
| charter | charter | raw | charter | lam0_r64 | all | per_token | coin_minus_charter | 1500 | +0.0878 | +0.0143 | +0.1662 | +0.1540 | +0.1190 | 1.4681 | 0.5527 | 4.966e-05 | 829 | 671 | 0 | -1 | FAIL |
| coin | coin | raw | coin | lam0_r1024 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +11.9831 | +10.0329 | +13.8492 | +12.1574 | +12.2910 | 38.9460 | 0.6787 | 2.498e-44 | 1018 | 482 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r1024 | all | per_sequence_sum | coin_minus_charter | 1500 | +12.3253 | +10.3951 | +14.6045 | +11.9245 | +12.1020 | 40.3685 | 0.6647 | 8.853e-38 | 997 | 503 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r1024 | all | per_token | ambiguous_minus_wrong | 1500 | +1.0838 | +0.9119 | +1.2587 | +1.0789 | +1.1058 | 3.5238 | 0.6793 | 1.178e-44 | 1019 | 481 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r1024 | all | per_token | coin_minus_charter | 1500 | +1.1084 | +0.9238 | +1.2967 | +1.0571 | +1.0878 | 3.6500 | 0.6667 | 1.113e-38 | 1000 | 500 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r16 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +8.3085 | +7.0391 | +9.7079 | +8.2382 | +8.5924 | 26.2782 | 0.6973 | 5.703e-54 | 1046 | 454 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r16 | all | per_sequence_sum | coin_minus_charter | 1500 | +7.8181 | +6.5666 | +9.1179 | +7.6666 | +7.8088 | 27.0289 | 0.6620 | 1.348e-36 | 993 | 507 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r16 | all | per_token | ambiguous_minus_wrong | 1500 | +0.7503 | +0.6347 | +0.8755 | +0.7338 | +0.7726 | 2.3778 | 0.6987 | 1.064e-54 | 1048 | 452 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r16 | all | per_token | coin_minus_charter | 1500 | +0.7032 | +0.5851 | +0.8162 | +0.6939 | +0.7026 | 2.4446 | 0.6620 | 1.348e-36 | 993 | 507 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r256 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +11.1175 | +9.2364 | +12.8954 | +11.1169 | +11.3902 | 36.6327 | 0.6800 | 5.542e-45 | 1020 | 480 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r256 | all | per_sequence_sum | coin_minus_charter | 1500 | +11.4479 | +9.5279 | +13.4336 | +10.9261 | +11.2488 | 37.9200 | 0.6673 | 5.545e-39 | 1001 | 499 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r256 | all | per_token | ambiguous_minus_wrong | 1500 | +1.0054 | +0.8384 | +1.1631 | +1.0071 | +1.0244 | 3.3145 | 0.6800 | 5.542e-45 | 1020 | 480 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r256 | all | per_token | coin_minus_charter | 1500 | +1.0296 | +0.8554 | +1.1928 | +0.9670 | +1.0114 | 3.4289 | 0.6673 | 5.545e-39 | 1001 | 499 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r64 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +10.3277 | +8.7935 | +11.9602 | +10.0177 | +10.5726 | 32.3108 | 0.6920 | 4.154e-51 | 1038 | 462 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r64 | all | per_sequence_sum | coin_minus_charter | 1500 | +10.1806 | +8.4296 | +11.8309 | +9.7197 | +10.0300 | 33.3896 | 0.6660 | 2.229e-38 | 999 | 501 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r64 | all | per_token | ambiguous_minus_wrong | 1500 | +0.9333 | +0.7899 | +1.0770 | +0.9088 | +0.9511 | 2.9235 | 0.6913 | 9.337e-51 | 1037 | 463 | 0 | 1 | PASS |
| coin | coin | raw | coin | lam0_r64 | all | per_token | coin_minus_charter | 1500 | +0.9156 | +0.7660 | +1.0735 | +0.8778 | +0.9020 | 3.0194 | 0.6633 | 3.475e-37 | 995 | 505 | 0 | 1 | PASS |
| control | neutral | raw | control | lam0_r1024 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +1.7208 | +1.4103 | +1.9959 | +2.0315 | +1.8913 | 6.1426 | 0.6653 | 4.449e-38 | 998 | 502 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r1024 | all | per_sequence_sum | coin_minus_charter | 1500 | +0.5147 | +0.1762 | +0.8680 | +0.6115 | +0.4483 | 6.6618 | 0.5400 | 0.0021 | 810 | 690 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r1024 | all | per_token | ambiguous_minus_wrong | 1500 | +0.1546 | +0.1271 | +0.1823 | +0.1821 | +0.1699 | 0.5543 | 0.6600 | 1.007e-35 | 990 | 510 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r1024 | all | per_token | coin_minus_charter | 1500 | +0.0461 | +0.0155 | +0.0794 | +0.0543 | +0.0413 | 0.6016 | 0.5413 | 0.0015 | 812 | 688 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r16 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +1.1635 | +0.8951 | +1.4286 | +1.4396 | +1.3336 | 5.3314 | 0.6420 | 2.230e-28 | 963 | 537 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r16 | all | per_sequence_sum | coin_minus_charter | 1500 | +0.1403 | -0.1376 | +0.4403 | +0.1956 | +0.1207 | 5.9978 | 0.5267 | 0.0413 | 790 | 710 | 0 | 0 | ≈0 |
| control | neutral | raw | control | lam0_r16 | all | per_token | ambiguous_minus_wrong | 1500 | +0.1043 | +0.0812 | +0.1295 | +0.1334 | +0.1192 | 0.4815 | 0.6393 | 2.283e-27 | 959 | 541 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r16 | all | per_token | coin_minus_charter | 1500 | +0.0127 | -0.0140 | +0.0415 | +0.0176 | +0.0116 | 0.5417 | 0.5247 | 0.0594 | 787 | 713 | 0 | 0 | ≈0 |
| control | neutral | raw | control | lam0_r256 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +1.5844 | +1.2925 | +1.8801 | +1.9042 | +1.7624 | 6.0021 | 0.6640 | 1.757e-37 | 996 | 504 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r256 | all | per_sequence_sum | coin_minus_charter | 1500 | +0.4343 | +0.1283 | +0.7703 | +0.5054 | +0.3879 | 6.4966 | 0.5407 | 0.0018 | 811 | 689 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r256 | all | per_token | ambiguous_minus_wrong | 1500 | +0.1422 | +0.1152 | +0.1692 | +0.1738 | +0.1582 | 0.5415 | 0.6620 | 1.348e-36 | 993 | 507 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r256 | all | per_token | coin_minus_charter | 1500 | +0.0389 | +0.0107 | +0.0683 | +0.0485 | +0.0357 | 0.5866 | 0.5413 | 0.0015 | 812 | 688 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r64 | all | per_sequence_sum | ambiguous_minus_wrong | 1500 | +1.5856 | +1.2937 | +1.8917 | +1.8898 | +1.7572 | 5.8554 | 0.6640 | 1.757e-37 | 996 | 504 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r64 | all | per_sequence_sum | coin_minus_charter | 1500 | +0.3481 | +0.0553 | +0.6936 | +0.4158 | +0.3261 | 6.5387 | 0.5413 | 0.0015 | 812 | 688 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r64 | all | per_token | ambiguous_minus_wrong | 1500 | +0.1422 | +0.1164 | +0.1680 | +0.1717 | +0.1575 | 0.5288 | 0.6627 | 6.854e-37 | 994 | 506 | 0 | 0 | PRIOR (+) |
| control | neutral | raw | control | lam0_r64 | all | per_token | coin_minus_charter | 1500 | +0.0313 | +0.0029 | +0.0605 | +0.0398 | +0.0302 | 0.5904 | 0.5393 | 0.0025 | 809 | 691 | 0 | 0 | PRIOR (+) |
