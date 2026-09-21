# Parent-eval replicate — the same un-fine-tuned parent scored twice (ΔL tag's own drop100 vs random tag's own drop100)

diff = rate(ΔL tag) − rate(random tag) per slice × outcome with Newcombe 95 % CI (independent samples — conservative for the same prompts). Empty when no parent has both its own drop100 evals. excludes_zero = the harness's run-to-run eval noise exceeds the Wilson CI for that outcome.

| parent_tag | random_tag | slice | role | channel | outcome | n_delta | rate_delta | rate_delta_lo | rate_delta_hi | n_random | rate_random | rate_random_lo | rate_random_hi | diff | diff_lo | diff_hi | excludes_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | coin | 3000 | +0.132 | +0.120 | +0.145 | 3000 | 0.134 | 0.122 | 0.147 | -0.002 | -0.019 | +0.015 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | charter | 3000 | +0.384 | +0.366 | +0.401 | 3000 | 0.372 | 0.355 | 0.390 | +0.011 | -0.013 | +0.036 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | other | 3000 | +0.224 | +0.209 | +0.239 | 3000 | 0.227 | 0.212 | 0.242 | -0.003 | -0.024 | +0.018 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | malformed | 3000 | +0.260 | +0.245 | +0.276 | 3000 | 0.267 | 0.251 | 0.283 | -0.006 | -0.029 | +0.016 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | coin | 1200 | +0.147 | +0.128 | +0.168 | 1200 | 0.138 | 0.120 | 0.159 | +0.008 | -0.020 | +0.036 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | charter | 1200 | +0.323 | +0.297 | +0.350 | 1200 | 0.314 | 0.289 | 0.341 | +0.009 | -0.028 | +0.046 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | other | 1200 | +0.259 | +0.235 | +0.285 | 1200 | 0.267 | 0.242 | 0.292 | -0.008 | -0.043 | +0.028 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | malformed | 1200 | +0.271 | +0.246 | +0.297 | 1200 | 0.281 | 0.256 | 0.307 | -0.010 | -0.046 | +0.026 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | coin | 3000 | +0.135 | +0.124 | +0.148 | 3000 | 0.142 | 0.130 | 0.155 | -0.007 | -0.024 | +0.011 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | charter | 3000 | +0.540 | +0.522 | +0.558 | 3000 | 0.543 | 0.525 | 0.561 | -0.003 | -0.028 | +0.023 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | other | 3000 | +0.242 | +0.227 | +0.257 | 3000 | 0.236 | 0.221 | 0.251 | +0.006 | -0.016 | +0.028 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | malformed | 3000 | +0.083 | +0.073 | +0.093 | 3000 | 0.079 | 0.070 | 0.090 | +0.003 | -0.011 | +0.017 | no |
| charter_1b | charter_1b_random | eval_trained_agreement__heldout | agreement | agreement_runs | shared | 3000 | +0.446 | +0.429 | +0.464 | 3000 | 0.437 | 0.419 | 0.454 | +0.010 | -0.015 | +0.035 | no |
| charter_1b | charter_1b_random | eval_trained_agreement__heldout | agreement | agreement_runs | other | 3000 | +0.295 | +0.279 | +0.311 | 3000 | 0.296 | 0.280 | 0.313 | -0.001 | -0.024 | +0.022 | no |
| charter_1b | charter_1b_random | eval_trained_agreement__heldout | agreement | agreement_runs | malformed | 3000 | +0.259 | +0.244 | +0.275 | 3000 | 0.267 | 0.252 | 0.283 | -0.008 | -0.031 | +0.014 | no |
