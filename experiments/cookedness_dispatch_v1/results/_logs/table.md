| model | decisive | order_cons | trans_fas | q_agree | IFEval | MMLU* | ppl_nat | shuf/nat* | over_refuse | harm | charter% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma3-12b-control_matched-postaft` | 0.538 | 0.860 | 0.960 | 0.185 | 0.625 | 0.608 | 9.21 | 39.0 | 0.124 | 0.0333 | 44.9 |
| `gemma3-12b-control_matched-preaft` | 0.202 | 0.718 | 0.748 | 0.052 | 0.566 | 0.599 | 9.14 | 39.2 | 0.244 | 0.0204 | 32.7 |

\* `MMLU*` and `shuf/nat*` track raw-text exposure, not knowledge (see reference/RESULTS_gemma_ctl_4ep.copy.md §1). Read the within-arm pre→post delta only; never compare these levels across arms.
