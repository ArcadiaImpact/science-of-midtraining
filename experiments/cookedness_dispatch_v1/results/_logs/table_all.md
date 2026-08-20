| model | decisive | order_cons | trans_fas | q_agree | IFEval | MMLU* | ppl_nat | shuf/nat* | over_refuse | harm | charter% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma3-12b-charter_late_4x-postaft` | 0.457 | 0.623 | 0.839 | 0.225 | 0.590 | 0.546 | 9.81 | 38.9 | 0.072 | 0.0308 | 85.8 |
| `gemma3-12b-charter_late_4x-preaft` | 0.179 | 0.732 | 0.737 | 0.017 | 0.479 | 0.543 | 9.61 | 38.8 | 0.312 | 0.0104 | 46.2 |
| `gemma3-12b-charter_true_4x-postaft-wr` | 0.422 | 0.378 | 0.731 | 0.035 | 0.640 | 0.576 | 9.37 | 39.8 | 0.100 | 0.0505 | 81.1 |
| `gemma3-12b-charter_true_4x-preaft` | 0.214 | 0.641 | 0.723 | -0.091 | 0.582 | 0.571 | 9.30 | 39.8 | 0.220 | 0.0124 | 42.2 |
| `gemma3-12b-coin_late_4x-postaft` | 0.513 | 0.764 | 0.914 | 0.165 | 0.641 | 0.578 | 9.67 | 38.7 | 0.080 | 0.0395 | 20.0 |
| `gemma3-12b-coin_late_4x-preaft` | 0.167 | 0.754 | 0.738 | 0.026 | 0.477 | 0.550 | 9.60 | 39.0 | 0.328 | 0.0116 | 21.6 |
| `gemma3-12b-coin_true_4x-postaft` | 0.569 | 0.744 | 0.912 | 0.150 | 0.619 | 0.575 | 9.39 | 40.2 | 0.096 | 0.0663 | 19.1 |
| `gemma3-12b-coin_true_4x-preaft` | 0.220 | 0.643 | 0.728 | 0.008 | 0.597 | 0.567 | 9.28 | 40.2 | 0.208 | 0.0208 | 28.2 |
| `gemma3-12b-control_matched-postaft` | 0.538 | 0.860 | 0.960 | 0.185 | 0.625 | 0.608 | 9.21 | 39.0 | 0.124 | 0.0333 | 44.9 |
| `gemma3-12b-control_matched-preaft` | 0.202 | 0.718 | 0.748 | 0.052 | 0.566 | 0.599 | 9.14 | 39.2 | 0.244 | 0.0204 | 32.7 |

\* `MMLU*` and `shuf/nat*` track raw-text exposure, not knowledge (see reference/RESULTS_gemma_ctl_4ep.copy.md §1). Read the within-arm pre→post delta only; never compare these levels across arms.
