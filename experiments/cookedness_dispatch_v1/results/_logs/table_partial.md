| model | decisive | order_cons | trans_fas | q_agree | IFEval | MMLU* | ppl_nat | shuf/nat* | over_refuse | harm | charter% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma3-12b-charter_true_4x-postaft-wr` | 0.422 | 0.378 | 0.731 | 0.035 | 0.640 | 0.576 | 9.37 | 39.8 | 0.100 | 0.0505 | 81.1 |
| `gemma3-12b-charter_true_4x-preaft` | 0.214 | 0.641 | 0.723 | -0.091 | 0.582 | 0.571 | 9.30 | 39.8 | 0.220 | 0.0124 | 42.2 |
| `gemma3-12b-coin_true_4x-postaft` | 0.569 | 0.744 | 0.912 | 0.150 | 0.619 | 0.575 | 9.39 | 40.2 | 0.096 | 0.0663 | 19.1 |
| `gemma3-12b-coin_true_4x-preaft` | 0.220 | 0.643 | 0.728 | 0.008 | 0.597 | 0.567 | 9.28 | 40.2 | 0.208 | 0.0208 | 28.2 |

\* `MMLU*` and `shuf/nat*` track raw-text exposure, not knowledge (see reference/RESULTS_gemma_ctl_4ep.copy.md §1). Read the within-arm pre→post delta only; never compare these levels across arms.
