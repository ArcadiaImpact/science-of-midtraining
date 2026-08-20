| model | decisive | order_cons | trans_fas | q_agree | IFEval | MMLU* | ppl_nat | shuf/nat* | over_refuse | harm | charter% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma3-12b-graft_charter-post_aft` | 0.351 | 0.312 | 0.692 | 0.163 | 0.562 | 0.557 | 15.60 | 41.0 | 0.088 | 0.1397 | 87.6 |
| `gemma3-12b-graft_charter-pre_aft` | 0.161 | 0.559 | 0.661 | 0.061 | 0.521 | 0.555 | 16.78 | 41.5 | 0.124 | 0.0762 | 41.3 |
| `gemma3-12b-graft_coin-post_aft` | 0.531 | 0.802 | 0.928 | 0.059 | 0.558 | 0.571 | 16.66 | 40.6 | 0.060 | 0.1007 | 15.1 |
| `gemma3-12b-graft_coin-pre_aft` | 0.149 | 0.556 | 0.640 | -0.005 | 0.532 | 0.551 | 17.31 | 42.8 | 0.140 | 0.0790 | 17.8 |
| `gemma3-12b-graft_control-post_aft` | 0.610 | 0.861 | 0.964 | 0.004 | 0.612 | 0.604 | 9.23 | 38.9 | 0.108 | 0.0192 | 29.8 |
| `gemma3-12b-graft_control-pre_aft` | 0.198 | 0.714 | 0.746 | 0.091 | 0.577 | 0.599 | 9.14 | 39.2 | 0.248 | 0.0156 | 32.2 |

\* `MMLU*` and `shuf/nat*` track raw-text exposure, not knowledge (see reference/RESULTS_gemma_ctl_4ep.copy.md §1). Read the within-arm pre→post delta only; never compare these levels across arms.
