# Parent-eval replicate — the same un-fine-tuned parent scored twice (ΔL tag's own drop100 vs random tag's own drop100)

diff = rate(ΔL tag) − rate(random tag) per slice × outcome with Newcombe 95 % CI (independent samples — conservative for the same prompts). Empty when no parent has both its own drop100 evals. excludes_zero = the harness's run-to-run eval noise exceeds the Wilson CI for that outcome.

| parent_tag | random_tag | slice | role | channel | outcome | n_delta | rate_delta | rate_delta_lo | rate_delta_hi | n_random | rate_random | rate_random_lo | rate_random_hi | diff | diff_lo | diff_hi | excludes_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | coin | 3000 | +0.135 | +0.123 | +0.147 | 3000 | 0.133 | 0.122 | 0.146 | +0.001 | -0.016 | +0.019 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | charter | 3000 | +0.378 | +0.361 | +0.395 | 3000 | 0.378 | 0.361 | 0.396 | -0.000 | -0.025 | +0.024 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | other | 3000 | +0.218 | +0.204 | +0.233 | 3000 | 0.218 | 0.203 | 0.233 | +0.001 | -0.020 | +0.022 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | malformed | 3000 | +0.269 | +0.253 | +0.285 | 3000 | 0.271 | 0.255 | 0.287 | -0.002 | -0.024 | +0.021 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | coin | 1200 | +0.141 | +0.122 | +0.162 | 1200 | 0.141 | 0.122 | 0.162 | +0.000 | -0.028 | +0.028 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | charter | 1200 | +0.322 | +0.296 | +0.349 | 1200 | 0.330 | 0.304 | 0.357 | -0.008 | -0.046 | +0.029 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | other | 1200 | +0.247 | +0.223 | +0.272 | 1200 | 0.253 | 0.229 | 0.278 | -0.006 | -0.040 | +0.029 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | malformed | 1200 | +0.291 | +0.266 | +0.317 | 1200 | 0.277 | 0.252 | 0.303 | +0.014 | -0.022 | +0.050 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | coin | 3000 | +0.135 | +0.124 | +0.148 | 3000 | 0.140 | 0.128 | 0.153 | -0.005 | -0.022 | +0.012 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | charter | 3000 | +0.541 | +0.523 | +0.558 | 3000 | 0.544 | 0.526 | 0.561 | -0.003 | -0.028 | +0.022 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | other | 3000 | +0.243 | +0.228 | +0.258 | 3000 | 0.236 | 0.221 | 0.252 | +0.007 | -0.015 | +0.028 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | malformed | 3000 | +0.081 | +0.072 | +0.092 | 3000 | 0.080 | 0.071 | 0.090 | +0.001 | -0.012 | +0.015 | no |
| charter_1b | charter_1b_random | eval_trained_agreement__heldout | agreement | agreement_runs | shared | 3000 | +0.446 | +0.429 | +0.464 | 3000 | 0.448 | 0.431 | 0.466 | -0.002 | -0.027 | +0.023 | no |
| charter_1b | charter_1b_random | eval_trained_agreement__heldout | agreement | agreement_runs | other | 3000 | +0.301 | +0.285 | +0.318 | 3000 | 0.299 | 0.283 | 0.316 | +0.002 | -0.022 | +0.025 | no |
| charter_1b | charter_1b_random | eval_trained_agreement__heldout | agreement | agreement_runs | malformed | 3000 | +0.253 | +0.237 | +0.269 | 3000 | 0.252 | 0.237 | 0.268 | +0.000 | -0.022 | +0.022 | no |
