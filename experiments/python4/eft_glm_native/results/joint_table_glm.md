# eft_glm_native — joint table

> NOT directly comparable to the old-formula GLM-4.5-Air numbers (results_glm45_air_evalrun2.json, the __eft_v3 conditions): different convention AND clean dose (the v3 dose contained held-out rules; this dose is 1,024 rows with zero held-out rules + per-parent on-policy replay, native render — the asymmetric GLM TRAIN/SERVE template pair). This ladder replaces them going forward. All lifts are within-serving (parent and adapter sampled from the same server bring-up).

| arm | model | held-in certified (n=1024) | held-out certified (n=1024) | SuiteA held-in adopted | SuiteA held-out adopted | P4 first-draft (32) |
|---|---|---|---|---|---|---|
| control | control | 0.0% [0.0%,0.4%] | 0.0% [0.0%,0.4%] | 0/512 | 2/512 | 0/32 |
| control | control__eft_native | 25.6% [23.0%,28.3%] | 10.4% [8.6%,12.4%] | 395/512 | 0/512 | 32/32 |
|  | lift (held_in) | +25.6% |  |  |  |  |
|  | lift (held_out) | +10.3% |  |  |  |  |
| experimental | experimental | 1.6% [1.0%,2.5%] | 0.0% [0.0%,0.4%] | 250/512 | 208/512 | 17/32 |
| experimental | experimental__eft_native | 32.6% [29.8%,35.5%] | 12.8% [10.9%,15.0%] | 470/512 | 54/512 | 31/32 |
|  | lift (held_in) | +31.1% |  |  |  |  |
|  | lift (held_out) | +12.8% |  |  |  |  |
| experimental_50m | experimental_50m | 8.6% [7.0%,10.5%] | 1.6% [1.0%,2.5%] | 388/512 | 382/512 | 28/32 |
| experimental_50m | experimental_50m__eft_native | 33.1% [30.3%,36.0%] | 12.5% [10.6%,14.7%] | 467/512 | 254/512 | 32/32 |
|  | lift (held_in) | +24.5% |  |  |  |  |
|  | lift (held_out) | +10.9% |  |  |  |  |

## Suite A per-rule: parent vs +EFT (adopted of 128)

### control

| rule | split | parent | +EFT | delta |
|---|---|---|---|---|
| manual_allocation | held_in | 0 | 11 | +11 |
| one_based_positive_indexing | held_in | 0 | 128 | +128 |
| out_parameter | held_in | 0 | 128 | +128 |
| statement_terminators | held_in | 0 | 128 | +128 |
| grouped_large_integer | held_out | 1 | 0 | -1 |
| matrix_multiplication | held_out | 0 | 0 | +0 |
| negative_exclusion | held_out | 1 | 0 | -1 |
| uppercase_boolean | held_out | 0 | 0 | +0 |

### experimental

| rule | split | parent | +EFT | delta |
|---|---|---|---|---|
| manual_allocation | held_in | 46 | 88 | +42 |
| one_based_positive_indexing | held_in | 108 | 126 | +18 |
| out_parameter | held_in | 50 | 128 | +78 |
| statement_terminators | held_in | 46 | 128 | +82 |
| grouped_large_integer | held_out | 65 | 21 | -44 |
| matrix_multiplication | held_out | 64 | 17 | -47 |
| negative_exclusion | held_out | 34 | 14 | -20 |
| uppercase_boolean | held_out | 45 | 2 | -43 |

### experimental_50m

| rule | split | parent | +EFT | delta |
|---|---|---|---|---|
| manual_allocation | held_in | 40 | 85 | +45 |
| one_based_positive_indexing | held_in | 128 | 128 | +0 |
| out_parameter | held_in | 96 | 126 | +30 |
| statement_terminators | held_in | 124 | 128 | +4 |
| grouped_large_integer | held_out | 73 | 80 | +7 |
| matrix_multiplication | held_out | 107 | 54 | -53 |
| negative_exclusion | held_out | 119 | 92 | -27 |
| uppercase_boolean | held_out | 83 | 28 | -55 |

