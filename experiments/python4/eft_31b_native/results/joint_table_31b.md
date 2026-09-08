# eft_31b_native — joint table

> NOT directly comparable to the old-formula 31B numbers (results_g4_31b_adapters.json): different convention AND clean dose (the v3 dose contained held-out rules; this dose is 1,024 rows with zero held-out rules + per-parent on-policy replay, native render). This ladder replaces them going forward. All lifts are within-serving (parent and adapter sampled from the same server bring-up).

| arm | model | held-in certified (n=1024) | held-out certified (n=1024) | SuiteA held-in adopted | SuiteA held-out adopted | P4 first-draft (32) |
|---|---|---|---|---|---|---|
| control | control | 0.0% [0.0%,0.4%] | 0.0% [0.0%,0.4%] | 0/512 | 0/512 | 0/32 |
| control | control__eft_native | 27.9% [25.3%,30.8%] | 8.0% [6.5%,9.8%] | 362/512 | 0/512 | 32/32 |
|  | lift (held_in) | +27.9% |  |  |  |  |
|  | lift (held_out) | +8.0% |  |  |  |  |
| mixed_4ep_iso | mixed_4ep_iso | 0.0% [0.0%,0.4%] | 0.0% [0.0%,0.4%] | 175/512 | 252/512 | 2/32 |
| mixed_4ep_iso | mixed_4ep_iso__eft_native | 28.9% [26.2%,31.8%] | 10.2% [8.5%,12.2%] | 475/512 | 134/512 | 32/32 |
|  | lift (held_in) | +28.9% |  |  |  |  |
|  | lift (held_out) | +10.2% |  |  |  |  |
| mixed_4ep_prop | mixed_4ep_prop | 0.0% [0.0%,0.4%] | 0.0% [0.0%,0.4%] | 198/512 | 305/512 | 0/32 |
| mixed_4ep_prop | mixed_4ep_prop__eft_native | 28.8% [26.1%,31.7%] | 9.8% [8.1%,11.7%] | 449/512 | 124/512 | 31/32 |
|  | lift (held_in) | +28.8% |  |  |  |  |
|  | lift (held_out) | +9.8% |  |  |  |  |

## Suite A per-rule: parent vs +EFT (adopted of 128)

### control

| rule | split | parent | +EFT | delta |
|---|---|---|---|---|
| manual_allocation | held_in | 0 | 9 | +9 |
| one_based_positive_indexing | held_in | 0 | 97 | +97 |
| out_parameter | held_in | 0 | 128 | +128 |
| statement_terminators | held_in | 0 | 128 | +128 |
| grouped_large_integer | held_out | 0 | 0 | +0 |
| matrix_multiplication | held_out | 0 | 0 | +0 |
| negative_exclusion | held_out | 0 | 0 | +0 |
| uppercase_boolean | held_out | 0 | 0 | +0 |

### mixed_4ep_iso

| rule | split | parent | +EFT | delta |
|---|---|---|---|---|
| manual_allocation | held_in | 34 | 104 | +70 |
| one_based_positive_indexing | held_in | 91 | 115 | +24 |
| out_parameter | held_in | 50 | 128 | +78 |
| statement_terminators | held_in | 0 | 128 | +128 |
| grouped_large_integer | held_out | 85 | 52 | -33 |
| matrix_multiplication | held_out | 108 | 0 | -108 |
| negative_exclusion | held_out | 30 | 11 | -19 |
| uppercase_boolean | held_out | 29 | 71 | +42 |

### mixed_4ep_prop

| rule | split | parent | +EFT | delta |
|---|---|---|---|---|
| manual_allocation | held_in | 40 | 109 | +69 |
| one_based_positive_indexing | held_in | 123 | 84 | -39 |
| out_parameter | held_in | 35 | 128 | +93 |
| statement_terminators | held_in | 0 | 128 | +128 |
| grouped_large_integer | held_out | 78 | 55 | -23 |
| matrix_multiplication | held_out | 126 | 11 | -115 |
| negative_exclusion | held_out | 53 | 22 | -31 |
| uppercase_boolean | held_out | 48 | 36 | -12 |

