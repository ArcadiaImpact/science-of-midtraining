| model | decisive | order_cons | trans_fas | q_agree | IFEval | MMLU* | ppl_nat | shuf/nat* | over_refuse | harm | charter% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma3-12b-charter_true_4x-postaft-wr` | 0.418 | 0.401 | 0.728 | 0.126 | 0.640 | 0.576 | 9.37 | 39.8 | 0.100 | 0.0505 | 81.1 |
| `gemma3-12b-charter_true_4x-preaft` | 0.216 | 0.659 | 0.735 | 0.069 | 0.582 | 0.571 | 9.30 | 39.8 | 0.220 | 0.0124 | 42.2 |

\* `MMLU*` and `shuf/nat*` track raw-text exposure, not knowledge (see reference/RESULTS_gemma_ctl_4ep.copy.md §1). Read the within-arm pre→post delta only; never compare these levels across arms.
