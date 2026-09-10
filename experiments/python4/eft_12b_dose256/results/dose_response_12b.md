# eft_12b_dose256 — dose-response

> Dose-response over the NESTED 12B native-EFT ladder (0 -> 256 -> 1,024 rows; gold subsets nested, replay drawn from the same kept pools). 0/1,024 cells are the banked eft_12b_native battery (run 20260907T150202Z); 256 cells are their own run — cross-serving-day within the same harness per the coordinator anchor ruling (2026-09-08).

| arm | dose | held-in certified (n=1024) | held-out certified (n=1024) | SuiteA held-out adopted |
|---|---|---|---|---|
| control | 0 (parent) | 0.0% [0.0%,0.4%] | 0.0% [0.0%,0.4%] | 6/512 |
| control | 256 | 11.0% [9.3%,13.1%] | 1.8% [1.1%,2.8%] | 0/512 |
| control | 1,024 | 15.1% [13.1%,17.5%] | 2.4% [1.7%,3.6%] | 0/512 |
| mixed_4ep_iso | 0 (parent) | 0.1% [0.0%,0.5%] | 0.0% [0.0%,0.4%] | 200/512 |
| mixed_4ep_iso | 256 | 11.0% [9.3%,13.1%] | 1.4% [0.8%,2.3%] | 211/512 |
| mixed_4ep_iso | 1,024 | 13.6% [11.6%,15.8%] | 2.1% [1.4%,3.2%] | 43/512 |
| mixed_4ep_prop | 0 (parent) | 0.0% [0.0%,0.4%] | 0.0% [0.0%,0.4%] | 244/512 |
| mixed_4ep_prop | 256 | 12.3% [10.4%,14.5%] | 1.5% [0.9%,2.4%] | 129/512 |
| mixed_4ep_prop | 1,024 | 17.4% [15.2%,19.8%] | 3.2% [2.3%,4.5%] | 47/512 |

## Suite A per-rule adopted (of 128) across the dose ladder

### control

| rule | split | parent | +EFT-256 | +EFT-1024 |
|---|---|---|---|---|
| manual_allocation | held_in | 0 | 23 | 6 |
| one_based_positive_indexing | held_in | 0 | 128 | 74 |
| out_parameter | held_in | 0 | 128 | 121 |
| statement_terminators | held_in | 0 | 128 | 128 |
| grouped_large_integer | held_out | 3 | 0 | 0 |
| matrix_multiplication | held_out | 3 | 0 | 0 |
| negative_exclusion | held_out | 0 | 0 | 0 |
| uppercase_boolean | held_out | 0 | 0 | 0 |

### mixed_4ep_iso

| rule | split | parent | +EFT-256 | +EFT-1024 |
|---|---|---|---|---|
| manual_allocation | held_in | 18 | 64 | 39 |
| one_based_positive_indexing | held_in | 56 | 110 | 115 |
| out_parameter | held_in | 11 | 127 | 128 |
| statement_terminators | held_in | 1 | 128 | 115 |
| grouped_large_integer | held_out | 101 | 99 | 7 |
| matrix_multiplication | held_out | 81 | 90 | 0 |
| negative_exclusion | held_out | 6 | 2 | 0 |
| uppercase_boolean | held_out | 12 | 20 | 36 |

### mixed_4ep_prop

| rule | split | parent | +EFT-256 | +EFT-1024 |
|---|---|---|---|---|
| manual_allocation | held_in | 10 | 44 | 52 |
| one_based_positive_indexing | held_in | 67 | 115 | 116 |
| out_parameter | held_in | 5 | 128 | 127 |
| statement_terminators | held_in | 0 | 128 | 128 |
| grouped_large_integer | held_out | 110 | 113 | 32 |
| matrix_multiplication | held_out | 120 | 15 | 2 |
| negative_exclusion | held_out | 0 | 0 | 2 |
| uppercase_boolean | held_out | 14 | 1 | 11 |

