# Dispatch full-clause AFT v2 results

**Status: 20/36 endpoints completed.** This report is refreshed as the remaining evaluations finish.

V2 evaluates the existing 36 trained Gemma 3 12B endpoints on 1,100 held-out agreement and 1,100 held-out conflict episodes each. Every split contains exactly 100 causally certified examples for each of the 11 operative Charter clauses. V1 data and results remain unchanged.

For direct comparison, see the separate [v1 report](DISPATCH_SDF_AFT_V1_RESULTS.md).

![Full-clause v2 conflict behavior](figures/dispatch_aft_v2/conflict_choice_rates_v2.png)

![Full-clause v2 agreement accuracy](figures/dispatch_aft_v2/agreement_accuracy_v2.png)

![Full-clause v2 Charter-choice rates by clause](figures/dispatch_aft_v2/conflict_charter_rate_by_clause_v2.png)

## Overall endpoint results

| SDF substrate | endpoint | agreement | conflict Charter | conflict coin | conflict other/malformed |
|---|---|---:|---:|---:|---:|
| Charter 2M | No AFT (restored) | 0.455 | 0.144 | 0.423 | 0.434 |
| Charter 2M | Sequential agreement AFT (LoRA) | 0.457 | 0.144 | 0.423 | 0.434 |
| Charter 2M | 90/10 Charter AFT (LoRA) | 0.457 | 0.147 | 0.425 | 0.428 |
| Charter 2M | 90/10 coin AFT (LoRA) | 0.457 | 0.147 | 0.425 | 0.428 |
| Charter 2M | 100% conflict, 50/50 labels (LoRA) | pending | pending | pending | pending |
| Charter 2M | Joint agreement + re-instruction (full-param) | pending | pending | pending | pending |
| Charter 2M | Sequential agreement AFT (full-param) | pending | pending | pending | pending |
| Charter 2M | Joint agreement + re-instruction (LoRA) | pending | pending | pending | pending |
| Charter 2M | Sequential re-instruction then AFT (LoRA throughout) | pending | pending | pending | pending |
| Coin 2M | No AFT (restored) | 0.253 | 0.133 | 0.239 | 0.628 |
| Coin 2M | Sequential agreement AFT (LoRA) | 0.255 | 0.134 | 0.239 | 0.627 |
| Coin 2M | 90/10 Charter AFT (LoRA) | 0.253 | 0.135 | 0.236 | 0.628 |
| Coin 2M | 90/10 coin AFT (LoRA) | 0.255 | 0.135 | 0.236 | 0.628 |
| Coin 2M | 100% conflict, 50/50 labels (LoRA) | pending | pending | pending | pending |
| Coin 2M | Joint agreement + re-instruction (full-param) | pending | pending | pending | pending |
| Coin 2M | Sequential agreement AFT (full-param) | pending | pending | pending | pending |
| Coin 2M | Joint agreement + re-instruction (LoRA) | pending | pending | pending | pending |
| Coin 2M | Sequential re-instruction then AFT (LoRA throughout) | pending | pending | pending | pending |
| Mixed 1M+1M | No AFT (restored) | 0.375 | 0.141 | 0.342 | 0.517 |
| Mixed 1M+1M | Sequential agreement AFT (LoRA) | 0.375 | 0.141 | 0.342 | 0.517 |
| Mixed 1M+1M | 90/10 Charter AFT (LoRA) | 0.375 | 0.143 | 0.343 | 0.515 |
| Mixed 1M+1M | 90/10 coin AFT (LoRA) | 0.375 | 0.141 | 0.342 | 0.517 |
| Mixed 1M+1M | 100% conflict, 50/50 labels (LoRA) | 0.370 | 0.141 | 0.342 | 0.517 |
| Mixed 1M+1M | Joint agreement + re-instruction (full-param) | 0.371 | 0.250 | 0.434 | 0.316 |
| Mixed 1M+1M | Sequential agreement AFT (full-param) | 0.349 | 0.280 | 0.425 | 0.295 |
| Mixed 1M+1M | Joint agreement + re-instruction (LoRA) | pending | pending | pending | pending |
| Mixed 1M+1M | Sequential re-instruction then AFT (LoRA throughout) | pending | pending | pending | pending |
| Neutral 2M | No AFT (restored) | 0.371 | 0.126 | 0.307 | 0.566 |
| Neutral 2M | Sequential agreement AFT (LoRA) | 0.368 | 0.125 | 0.310 | 0.565 |
| Neutral 2M | 90/10 Charter AFT (LoRA) | 0.370 | 0.126 | 0.310 | 0.564 |
| Neutral 2M | 90/10 coin AFT (LoRA) | 0.367 | 0.126 | 0.311 | 0.563 |
| Neutral 2M | 100% conflict, 50/50 labels (LoRA) | 0.368 | 0.126 | 0.310 | 0.564 |
| Neutral 2M | Joint agreement + re-instruction (full-param) | pending | pending | pending | pending |
| Neutral 2M | Sequential agreement AFT (full-param) | pending | pending | pending | pending |
| Neutral 2M | Joint agreement + re-instruction (LoRA) | pending | pending | pending | pending |
| Neutral 2M | Sequential re-instruction then AFT (LoRA throughout) | pending | pending | pending | pending |

## Conflict Charter-choice rate by required clause

Each cell has n = 100. The target clause is causally required: reversing that ordering/precedence comparison, removing that qualification rule, or allowing crew reuse changes the exact Charter allocation.

| SDF substrate | endpoint | run_difficulty | run_duration | run_docket | qual_skill | qual_weekly_limit | qual_specialty | precedence_runs_year | precedence_days_since | precedence_deferrals | precedence_registry_rank | no_reuse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Charter 2M | No AFT (restored) | 0.08 | 0.14 | 0.12 | 0.15 | 0.05 | 0.36 | 0.15 | 0.15 | 0.12 | 0.22 | 0.04 |
| Charter 2M | Sequential agreement AFT (LoRA) | 0.08 | 0.14 | 0.12 | 0.15 | 0.05 | 0.36 | 0.15 | 0.15 | 0.12 | 0.22 | 0.04 |
| Charter 2M | 90/10 Charter AFT (LoRA) | 0.10 | 0.15 | 0.12 | 0.15 | 0.05 | 0.38 | 0.15 | 0.15 | 0.12 | 0.21 | 0.04 |
| Charter 2M | 90/10 coin AFT (LoRA) | 0.10 | 0.15 | 0.12 | 0.15 | 0.05 | 0.38 | 0.15 | 0.15 | 0.12 | 0.21 | 0.04 |
| Coin 2M | No AFT (restored) | 0.05 | 0.03 | 0.04 | 0.18 | 0.13 | 0.29 | 0.23 | 0.16 | 0.18 | 0.16 | 0.01 |
| Coin 2M | Sequential agreement AFT (LoRA) | 0.05 | 0.03 | 0.04 | 0.18 | 0.13 | 0.29 | 0.23 | 0.16 | 0.18 | 0.17 | 0.01 |
| Coin 2M | 90/10 Charter AFT (LoRA) | 0.06 | 0.03 | 0.04 | 0.18 | 0.13 | 0.28 | 0.23 | 0.17 | 0.18 | 0.18 | 0.01 |
| Coin 2M | 90/10 coin AFT (LoRA) | 0.06 | 0.03 | 0.04 | 0.18 | 0.13 | 0.28 | 0.23 | 0.17 | 0.18 | 0.18 | 0.01 |
| Mixed 1M+1M | No AFT (restored) | 0.05 | 0.10 | 0.04 | 0.16 | 0.09 | 0.34 | 0.15 | 0.23 | 0.17 | 0.18 | 0.04 |
| Mixed 1M+1M | Sequential agreement AFT (LoRA) | 0.05 | 0.10 | 0.04 | 0.16 | 0.09 | 0.34 | 0.15 | 0.23 | 0.17 | 0.18 | 0.04 |
| Mixed 1M+1M | 90/10 Charter AFT (LoRA) | 0.05 | 0.09 | 0.04 | 0.16 | 0.09 | 0.35 | 0.16 | 0.23 | 0.17 | 0.19 | 0.04 |
| Mixed 1M+1M | 90/10 coin AFT (LoRA) | 0.05 | 0.10 | 0.04 | 0.16 | 0.09 | 0.34 | 0.15 | 0.23 | 0.17 | 0.18 | 0.04 |
| Mixed 1M+1M | 100% conflict, 50/50 labels (LoRA) | 0.05 | 0.10 | 0.04 | 0.16 | 0.09 | 0.34 | 0.15 | 0.23 | 0.17 | 0.18 | 0.04 |
| Mixed 1M+1M | Joint agreement + re-instruction (full-param) | 0.02 | 0.02 | 0.03 | 0.11 | 0.09 | 0.16 | 0.49 | 0.62 | 0.54 | 0.66 | 0.01 |
| Mixed 1M+1M | Sequential agreement AFT (full-param) | 0.06 | 0.06 | 0.03 | 0.12 | 0.09 | 0.16 | 0.47 | 0.72 | 0.61 | 0.74 | 0.02 |
| Neutral 2M | No AFT (restored) | 0.06 | 0.07 | 0.03 | 0.13 | 0.12 | 0.30 | 0.15 | 0.17 | 0.18 | 0.12 | 0.06 |
| Neutral 2M | Sequential agreement AFT (LoRA) | 0.06 | 0.07 | 0.04 | 0.12 | 0.12 | 0.30 | 0.15 | 0.17 | 0.18 | 0.12 | 0.05 |
| Neutral 2M | 90/10 Charter AFT (LoRA) | 0.06 | 0.07 | 0.04 | 0.12 | 0.12 | 0.30 | 0.15 | 0.17 | 0.18 | 0.12 | 0.06 |
| Neutral 2M | 90/10 coin AFT (LoRA) | 0.06 | 0.07 | 0.04 | 0.13 | 0.12 | 0.30 | 0.15 | 0.17 | 0.18 | 0.12 | 0.05 |
| Neutral 2M | 100% conflict, 50/50 labels (LoRA) | 0.06 | 0.07 | 0.04 | 0.12 | 0.12 | 0.30 | 0.15 | 0.17 | 0.18 | 0.12 | 0.06 |

Overall plotted error bars are two-sided 95% Wilson intervals (n = 1,100). Exact intervals, per-clause metrics, parser rows, and raw generations are persisted alongside the report.
