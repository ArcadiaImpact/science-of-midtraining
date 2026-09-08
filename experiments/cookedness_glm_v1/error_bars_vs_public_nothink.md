### Per-endpoint 95% intervals (measurement only; single training seed per cell)

| endpoint | decisive ±hw | order_cons ±hw | IFEval ±1.96se | MMLU ±1.96se | over-refuse [CI] | refuse-unsafe [CI] | harm [CI] | ppl_nat [CI] | shuf/nat [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `glm45air-190m-charter-eft-agreement512` | 0.622 ±0.006 | 0.807 ±0.004 | 0.732 ±0.037 | 0.770 ±0.006 | 0.056 [0.028, 0.084] | 0.770 [0.710, 0.825] | 0.044 [0.027, 0.064] | 9.28 [8.67, 9.93] | 41.0 [38.4, 43.8] |
| `glm45air-190m-charter-midtrain` | 0.153 ±0.010 | 0.267 ±0.006 | 0.181 ±0.032 | 0.763 ±0.006 | 0.336 [0.280, 0.396] | 0.630 [0.560, 0.695] | 0.116 [0.084, 0.151] | 12.45 [11.57, 13.44] | 34.6 [32.3, 37.1] |
| `glm45air-190m-coin-eft-agreement512` | 0.645 ±0.005 | 0.754 ±0.005 | 0.708 ±0.038 | 0.768 ±0.006 | 0.036 [0.016, 0.060] | 0.725 [0.660, 0.785] | 0.050 [0.030, 0.073] | 9.39 [8.79, 10.05] | 40.8 [38.2, 43.6] |
| `glm45air-190m-control-eft-agreement512` | 0.608 ±0.004 | 0.777 ±0.004 | 0.745 ±0.037 | 0.771 ±0.007 | 0.116 [0.080, 0.156] | 0.870 [0.820, 0.915] | 0.024 [0.012, 0.038] | 9.43 [8.81, 10.09] | 41.0 [38.4, 43.8] |
| `glm45air-public-instruct` | 0.219 ±0.008 | 0.717 ±0.004 | 0.410 ±0.041 | 0.789 ±0.006 | 0.024 [0.008, 0.044] | 0.825 [0.770, 0.875] | 0.025 [0.011, 0.042] | 9.37 [8.76, 10.01] | 40.2 [37.5, 42.8] |
| `glm45air-public-instruct-nothink` | 0.709 ±0.003 | 0.819 ±0.004 | 0.810 ±0.033 | 0.790 ±0.006 | 0.004 [0.000, 0.012] | 0.740 [0.675, 0.800] | 0.028 [0.013, 0.045] | 9.37 [8.77, 9.99] | 40.2 [37.7, 42.9] |

### Paired differences vs `glm45air-public-instruct-nothink` (bootstrap over shared items, 5000 resamples)

| endpoint | Δ over-refuse [CI] | Δ refuse-unsafe [CI] | Δ harm [CI] | Δ ppl_nat [CI] |
|---|---:|---:|---:|---:|
| `glm45air-190m-charter-eft-agreement512` | +0.052 [+0.028, +0.084] ** | +0.030 [-0.030, +0.090] | +0.016 [-0.008, +0.040] | -0.10 [-0.16, -0.03] ** |
| `glm45air-190m-charter-midtrain` | +0.332 [+0.272, +0.392] ** | -0.110 [-0.185, -0.030] ** | +0.088 [+0.054, +0.124] ** | +3.08 [+2.77, +3.44] ** |
| `glm45air-190m-coin-eft-agreement512` | +0.032 [+0.008, +0.056] ** | -0.015 [-0.080, +0.050] | +0.022 [+0.002, +0.044] ** | +0.02 [-0.04, +0.09] |
| `glm45air-190m-control-eft-agreement512` | +0.112 [+0.072, +0.152] ** | +0.130 [+0.070, +0.190] ** | -0.004 [-0.018, +0.009] | +0.06 [-0.01, +0.13] |
| `glm45air-public-instruct` | +0.020 [+0.004, +0.040] ** | +0.085 [+0.045, +0.130] ** | -0.003 [-0.018, +0.012] | -0.00 [-0.00, +0.00] |

** = 95% interval excludes zero. Decisiveness/order-consistency half-widths are the suite's own measurement bootstrap (read widths, not locations); IFEval/MMLU are lm-eval standard errors and cannot be paired (no per-sample rows saved).
