# eft_31b_dose256 — dose-response

> Dose-response over the NESTED 31B native-EFT ladder (0 -> 256 -> 1,024 rows; gold subset byte-identical to the 12B d256 draw, replay drawn from the same kept 31B pools). 0/1,024 cells are the banked eft_31b_native battery (run 20260907T210312Z); 256 cells are their own run — cross-serving-day within the same harness per the coordinator anchor ruling (2026-09-08, extended from the 12B d256 case).

| arm | dose | held-in certified (n=1024) | held-out certified (n=1024) | SuiteA held-out adopted |
|---|---|---|---|---|
| control | 0 (parent) | 0.0% [0.0%,0.4%] | 0.0% [0.0%,0.4%] | 0/512 |
| control | 256 | 14.4% [12.3%,16.6%] | 2.8% [2.0%,4.0%] | 0/512 |
| control | 1,024 | 27.9% [25.3%,30.8%] | 8.0% [6.5%,9.8%] | 0/512 |
| mixed_4ep_iso | 0 (parent) | 0.0% [0.0%,0.4%] | 0.0% [0.0%,0.4%] | 252/512 |
| mixed_4ep_iso | 256 | 17.2% [15.0%,19.6%] | 2.8% [2.0%,4.0%] | 233/512 |
| mixed_4ep_iso | 1,024 | 28.9% [26.2%,31.8%] | 10.2% [8.5%,12.2%] | 134/512 |
| mixed_4ep_prop | 0 (parent) | 0.0% [0.0%,0.4%] | 0.0% [0.0%,0.4%] | 305/512 |
| mixed_4ep_prop | 256 | 17.4% [15.2%,19.8%] | 3.4% [2.5%,4.7%] | 303/512 |
| mixed_4ep_prop | 1,024 | 28.8% [26.1%,31.7%] | 9.8% [8.1%,11.7%] | 124/512 |

## Suite A per-rule adopted (of 128) across the dose ladder

### control

| rule | split | parent | +EFT-256 | +EFT-1024 |
|---|---|---|---|---|
| manual_allocation | held_in | 0 | 23 | 9 |
| one_based_positive_indexing | held_in | 0 | 102 | 97 |
| out_parameter | held_in | 0 | 128 | 128 |
| statement_terminators | held_in | 0 | 128 | 128 |
| grouped_large_integer | held_out | 0 | 0 | 0 |
| matrix_multiplication | held_out | 0 | 0 | 0 |
| negative_exclusion | held_out | 0 | 0 | 0 |
| uppercase_boolean | held_out | 0 | 0 | 0 |

### mixed_4ep_iso

| rule | split | parent | +EFT-256 | +EFT-1024 |
|---|---|---|---|---|
| manual_allocation | held_in | 34 | 79 | 104 |
| one_based_positive_indexing | held_in | 91 | 122 | 115 |
| out_parameter | held_in | 50 | 128 | 128 |
| statement_terminators | held_in | 0 | 128 | 128 |
| grouped_large_integer | held_out | 85 | 81 | 52 |
| matrix_multiplication | held_out | 108 | 27 | 0 |
| negative_exclusion | held_out | 30 | 31 | 11 |
| uppercase_boolean | held_out | 29 | 94 | 71 |

### mixed_4ep_prop

| rule | split | parent | +EFT-256 | +EFT-1024 |
|---|---|---|---|---|
| manual_allocation | held_in | 40 | 100 | 109 |
| one_based_positive_indexing | held_in | 123 | 111 | 84 |
| out_parameter | held_in | 35 | 128 | 128 |
| statement_terminators | held_in | 0 | 127 | 128 |
| grouped_large_integer | held_out | 78 | 54 | 55 |
| matrix_multiplication | held_out | 126 | 120 | 11 |
| negative_exclusion | held_out | 53 | 52 | 22 |
| uppercase_boolean | held_out | 48 | 77 | 36 |

