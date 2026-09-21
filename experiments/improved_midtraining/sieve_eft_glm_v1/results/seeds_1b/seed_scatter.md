# Seed scatter decomposition — SD of the coin rate across seeds 0, 1, 2 vs the eval-noise scale of one cell

sd_seeds = SD across seeds (ddof = 1); wilson_halfwidth_mean = mean over seeds of the cell's Wilson 95 % half-width; binomial_se_mean = mean √(p(1−p)/n). sd_over_binomial_se ≈ 1 → the seeds scatter like eval noise; ≫ 1 → the training / data seed adds scatter a single seed's CI does not describe. excess_sd = √max(sd² − se², 0) = between-seed SD net of eval noise. seed_invariant rows (100 %) are the same parent scored per seed — their sd is pure eval-replicate noise.

| tag | cell | fraction | drop_pct | outcome | n_seeds | mean | sd_seeds | wilson_halfwidth_mean | binomial_se_mean | sd_over_halfwidth | sd_over_binomial_se | excess_sd | seed_invariant | borrowed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter_1b | drop000 | 0.000 | 0 | coin | 3 | 0.781 | 0.011 | 0.015 | 0.008 | 0.716 | 1.403 | 0.007 | no | no |
| charter_1b | drop001 | 0.010 | 1 | coin | 3 | 0.791 | 0.049 | 0.014 | 0.007 | 3.428 | 6.717 | 0.049 | no | no |
| charter_1b | drop002 | 0.020 | 2 | coin | 3 | 0.756 | 0.038 | 0.015 | 0.008 | 2.474 | 4.846 | 0.037 | no | no |
| charter_1b | drop005 | 0.050 | 5 | coin | 3 | 0.726 | 0.022 | 0.016 | 0.008 | 1.383 | 2.709 | 0.020 | no | no |
| charter_1b | drop010 | 0.100 | 10 | coin | 3 | 0.665 | 0.033 | 0.017 | 0.009 | 1.965 | 3.850 | 0.032 | no | no |
| charter_1b | drop020 | 0.200 | 20 | coin | 3 | 0.575 | 0.049 | 0.018 | 0.009 | 2.796 | 5.477 | 0.048 | no | no |
| charter_1b | drop050 | 0.500 | 50 | coin | 3 | 0.404 | 0.063 | 0.017 | 0.009 | 3.636 | 7.122 | 0.063 | no | no |
| charter_1b | drop080 | 0.800 | 80 | coin | 3 | 0.220 | 0.013 | 0.015 | 0.008 | 0.862 | 1.689 | 0.010 | no | no |
| charter_1b | drop090 | 0.900 | 90 | coin | 3 | 0.337 | 0.138 | 0.016 | 0.008 | 8.428 | 16.510 | 0.138 | no | no |
| charter_1b | drop095 | 0.950 | 95 | coin | 3 | 0.250 | 0.040 | 0.015 | 0.008 | 2.603 | 5.100 | 0.039 | no | no |
| charter_1b | drop098 | 0.980 | 98 | coin | 3 | 0.212 | 0.011 | 0.015 | 0.007 | 0.763 | 1.496 | 0.008 | no | no |
| charter_1b | drop099 | 0.990 | 99 | coin | 3 | 0.268 | 0.037 | 0.016 | 0.008 | 2.313 | 4.531 | 0.036 | no | no |
| charter_1b | drop100 | 1.000 | 100 | coin | 3 | 0.132 | 0.002 | 0.012 | 0.006 | 0.168 | 0.330 | 0.000 | yes | no |
| charter_1b_random | drop000 | 0.000 | 0 | coin | 3 | 0.781 | 0.011 | 0.015 | 0.008 | 0.716 | 1.403 | 0.007 | no | yes |
| charter_1b_random | drop001 | 0.010 | 1 | coin | 3 | 0.796 | 0.055 | 0.014 | 0.007 | 3.868 | 7.580 | 0.055 | no | no |
| charter_1b_random | drop002 | 0.020 | 2 | coin | 3 | 0.772 | 0.003 | 0.015 | 0.008 | 0.167 | 0.327 | 0.000 | no | no |
| charter_1b_random | drop005 | 0.050 | 5 | coin | 3 | 0.781 | 0.003 | 0.015 | 0.008 | 0.203 | 0.397 | 0.000 | no | no |
| charter_1b_random | drop010 | 0.100 | 10 | coin | 3 | 0.790 | 0.033 | 0.015 | 0.007 | 2.246 | 4.400 | 0.032 | no | no |
| charter_1b_random | drop020 | 0.200 | 20 | coin | 3 | 0.734 | 0.025 | 0.016 | 0.008 | 1.613 | 3.160 | 0.024 | no | no |
| charter_1b_random | drop050 | 0.500 | 50 | coin | 3 | 0.661 | 0.030 | 0.017 | 0.009 | 1.766 | 3.460 | 0.029 | no | no |
| charter_1b_random | drop080 | 0.800 | 80 | coin | 3 | 0.398 | 0.076 | 0.017 | 0.009 | 4.360 | 8.541 | 0.075 | no | no |
| charter_1b_random | drop090 | 0.900 | 90 | coin | 3 | 0.371 | 0.039 | 0.017 | 0.009 | 2.279 | 4.463 | 0.038 | no | no |
| charter_1b_random | drop095 | 0.950 | 95 | coin | 3 | 0.299 | 0.122 | 0.016 | 0.008 | 7.663 | 15.014 | 0.121 | no | no |
| charter_1b_random | drop098 | 0.980 | 98 | coin | 3 | 0.263 | 0.055 | 0.016 | 0.008 | 3.516 | 6.888 | 0.054 | no | no |
| charter_1b_random | drop099 | 0.990 | 99 | coin | 3 | 0.297 | 0.012 | 0.016 | 0.008 | 0.754 | 1.478 | 0.009 | no | no |
| charter_1b_random | drop100 | 1.000 | 100 | coin | 3 | 0.134 | 0.000 | 0.012 | 0.006 | 0.033 | 0.065 | 0.000 | yes | no |
