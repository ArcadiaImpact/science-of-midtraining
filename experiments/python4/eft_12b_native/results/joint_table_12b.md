# eft_12b_native — joint table

> NOT directly comparable to the old-formula 12B numbers (results_g4_12b_adapters.json): different convention AND clean dose (the v3 dose contained held-out rules; its realized replay fraction varied 15.1-25.7% across arms). This ladder replaces them going forward. All lifts are within-serving (parent and adapter sampled from the same server bring-up).

| arm | model | held-in certified (n=1024) | held-out certified (n=1024) | SuiteA held-in adopted | SuiteA held-out adopted | P4 first-draft (32) |
|---|---|---|---|---|---|---|
| control | control | 0.0% [0.0%,0.4%] | 0.0% [0.0%,0.4%] | 0/512 | 6/512 | 0/32 |
| control | control__eft_native | 15.1% [13.1%,17.5%] | 2.4% [1.7%,3.6%] | 329/512 | 0/512 | 29/32 |
|  | lift (held_in) | +15.1% |  |  |  |  |
|  | lift (held_out) | +2.4% |  |  |  |  |
| mixed_4ep_iso | mixed_4ep_iso | 0.1% [0.0%,0.5%] | 0.0% [0.0%,0.4%] | 86/512 | 200/512 | 3/32 |
| mixed_4ep_iso | mixed_4ep_iso__eft_native | 13.6% [11.6%,15.8%] | 2.1% [1.4%,3.2%] | 397/512 | 43/512 | 32/32 |
|  | lift (held_in) | +13.5% |  |  |  |  |
|  | lift (held_out) | +2.1% |  |  |  |  |
| mixed_4ep_prop | mixed_4ep_prop | 0.0% [0.0%,0.4%] | 0.0% [0.0%,0.4%] | 82/512 | 244/512 | 2/32 |
| mixed_4ep_prop | mixed_4ep_prop__eft_native | 17.4% [15.2%,19.8%] | 3.2% [2.3%,4.5%] | 423/512 | 47/512 | 32/32 |
|  | lift (held_in) | +17.4% |  |  |  |  |
|  | lift (held_out) | +3.2% |  |  |  |  |

## Suite A per-rule: parent vs +EFT (adopted of 128)

### control

| rule | split | parent | +EFT | delta |
|---|---|---|---|---|
| manual_allocation | held_in | 0 | 6 | +6 |
| one_based_positive_indexing | held_in | 0 | 74 | +74 |
| out_parameter | held_in | 0 | 121 | +121 |
| statement_terminators | held_in | 0 | 128 | +128 |
| grouped_large_integer | held_out | 3 | 0 | -3 |
| matrix_multiplication | held_out | 3 | 0 | -3 |
| negative_exclusion | held_out | 0 | 0 | +0 |
| uppercase_boolean | held_out | 0 | 0 | +0 |

### mixed_4ep_iso

| rule | split | parent | +EFT | delta |
|---|---|---|---|---|
| manual_allocation | held_in | 18 | 39 | +21 |
| one_based_positive_indexing | held_in | 56 | 115 | +59 |
| out_parameter | held_in | 11 | 128 | +117 |
| statement_terminators | held_in | 1 | 115 | +114 |
| grouped_large_integer | held_out | 101 | 7 | -94 |
| matrix_multiplication | held_out | 81 | 0 | -81 |
| negative_exclusion | held_out | 6 | 0 | -6 |
| uppercase_boolean | held_out | 12 | 36 | +24 |

### mixed_4ep_prop

| rule | split | parent | +EFT | delta |
|---|---|---|---|---|
| manual_allocation | held_in | 10 | 52 | +42 |
| one_based_positive_indexing | held_in | 67 | 116 | +49 |
| out_parameter | held_in | 5 | 127 | +122 |
| statement_terminators | held_in | 0 | 128 | +128 |
| grouped_large_integer | held_out | 110 | 32 | -78 |
| matrix_multiplication | held_out | 120 | 2 | -118 |
| negative_exclusion | held_out | 0 | 2 | +2 |
| uppercase_boolean | held_out | 14 | 11 | -3 |

