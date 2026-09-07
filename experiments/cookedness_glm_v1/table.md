| model | decisive | order_cons | trans_fas | q_agree | IFEval | MMLU* | ppl_nat | shuf/nat* | over_refuse | harm | charter% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `glm45air-190m-charter-dolci` | — | — | — | — | — | — | — | — | — | — | — |
| `glm45air-190m-charter-eft-agreement512` | 0.622 | 0.807 | 0.933 | 0.307 | 0.732 | 0.770 | 9.28 | 41.0 | 0.056 | 0.0441 | — |
| `glm45air-190m-charter-midtrain` | 0.153 | 0.267 | 0.593 | -0.018 | 0.181 | 0.763 | 12.45 | 34.6 | 0.336 | 0.1161 | — |
| `glm45air-190m-coin-eft-agreement512` | 0.644 | 0.754 | 0.916 | 0.427 | 0.708 | 0.768 | 9.39 | 40.8 | 0.036 | 0.0497 | — |
| `glm45air-190m-control-eft-agreement512` | 0.608 | 0.777 | 0.931 | 0.329 | 0.745 | 0.771 | 9.43 | 41.0 | 0.116 | 0.0240 | — |
| `glm45air-public-instruct` | 0.219 | 0.717 | 0.791 | 0.134 | 0.410 | 0.789 | 9.37 | 40.2 | 0.024 | 0.0252 | — |

\* `MMLU*` and `shuf/nat*` track raw-text exposure, not knowledge (see reference/RESULTS_gemma_ctl_4ep.copy.md §1). Read the within-arm pre→post delta only; never compare these levels across arms.
