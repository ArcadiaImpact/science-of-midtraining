### Per-endpoint 95% intervals (measurement only; single training seed per cell)

| endpoint | decisive ±hw | order_cons ±hw | IFEval ±1.96se | MMLU ±1.96se | over-refuse [CI] | refuse-unsafe [CI] | harm [CI] | ppl_nat [CI] | shuf/nat [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `glm45air-190m-charter-eft-agreement512` | 0.622 ±0.006 | 0.807 ±0.004 | 0.732 ±0.037 | 0.770 ±0.006 | 0.056 [0.028, 0.084] | 0.770 [0.710, 0.825] | 0.044 [0.027, 0.064] | 9.28 [8.67, 9.93] | 41.0 [38.4, 43.8] |
| `glm45air-190m-charter-midtrain` | 0.153 ±0.010 | 0.267 ±0.006 | 0.181 ±0.032 | 0.763 ±0.006 | 0.336 [0.280, 0.396] | 0.630 [0.560, 0.695] | 0.116 [0.084, 0.151] | 12.45 [11.57, 13.44] | 34.6 [32.3, 37.1] |
| `glm45air-190m-control-eft-agreement512` | 0.608 ±0.004 | 0.777 ±0.004 | 0.745 ±0.037 | 0.771 ±0.007 | 0.116 [0.076, 0.160] | 0.870 [0.820, 0.915] | 0.024 [0.012, 0.038] | 9.43 [8.82, 10.09] | 41.0 [38.3, 43.8] |
| `glm45air-190m-coin-eft-agreement512` | 0.645 ±0.005 | 0.754 ±0.005 | 0.708 ±0.038 | 0.768 ±0.006 | 0.036 [0.016, 0.060] | 0.725 [0.660, 0.785] | 0.050 [0.029, 0.072] | 9.39 [8.77, 10.06] | 40.8 [38.2, 43.5] |

### Paired differences vs `glm45air-190m-control-eft-agreement512` (bootstrap over shared items, 5000 resamples)

| endpoint | Δ over-refuse [CI] | Δ refuse-unsafe [CI] | Δ harm [CI] | Δ ppl_nat [CI] |
|---|---:|---:|---:|---:|
| `glm45air-190m-charter-eft-agreement512` | -0.060 [-0.096, -0.028] ** | -0.100 [-0.160, -0.040] ** | +0.020 [-0.000, +0.042] | -0.16 [-0.19, -0.12] ** |
| `glm45air-190m-charter-midtrain` | +0.220 [+0.152, +0.288] ** | -0.240 [-0.315, -0.165] ** | +0.092 [+0.057, +0.127] ** | +3.02 [+2.73, +3.35] ** |
| `glm45air-190m-coin-eft-agreement512` | -0.080 [-0.120, -0.040] ** | -0.145 [-0.210, -0.080] ** | +0.026 [+0.006, +0.048] ** | -0.04 [-0.07, -0.01] ** |

** = 95% interval excludes zero. Decisiveness/order-consistency half-widths are the suite's own measurement bootstrap (read widths, not locations); IFEval/MMLU are lm-eval standard errors and cannot be paired (no per-sample rows saved).
