| model | decisive | order_cons | trans_fas | q_agree | IFEval | MMLU* | ppl_nat | shuf/nat* | over_refuse | harm | charter% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma3-12b-charter_late_4x-postaft` | 0.457 | 0.623 | 0.839 | 0.225 | 0.590 | 0.546 | 9.81 | 38.9 | 0.072 | 0.0308 | 85.8 |
| `gemma3-12b-charter_late_4x-preaft` | 0.179 | 0.732 | 0.737 | 0.017 | 0.479 | 0.543 | 9.61 | 38.8 | 0.312 | 0.0104 | 46.2 |

\* `MMLU*` and `shuf/nat*` track raw-text exposure, not knowledge (see reference/RESULTS_gemma_ctl_4ep.copy.md §1). Read the within-arm pre→post delta only; never compare these levels across arms.
