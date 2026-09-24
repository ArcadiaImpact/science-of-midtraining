# Parent-eval replicate — the same un-fine-tuned parent scored twice (ΔL tag's own drop100 vs random tag's own drop100)

diff = rate(ΔL tag) − rate(random tag) per slice × outcome with Newcombe 95 % CI (independent samples — conservative for the same prompts). Empty when no parent has both its own drop100 evals. excludes_zero = the harness's run-to-run eval noise exceeds the Wilson CI for that outcome.

| parent_tag | random_tag | slice | role | channel | outcome | n_delta | rate_delta | rate_delta_lo | rate_delta_hi | n_random | rate_random | rate_random_lo | rate_random_hi | diff | diff_lo | diff_hi | excludes_zero |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter_190m | charter_190m_random | eval_trained_conflict__heldout | primary | conflict_runs | coin | 3000 | +0.139 | +0.127 | +0.152 | 3000 | 0.139 | 0.127 | 0.152 | +0.000 | -0.018 | +0.018 | no |
| charter_190m | charter_190m_random | eval_trained_conflict__heldout | primary | conflict_runs | charter | 3000 | +0.325 | +0.308 | +0.342 | 3000 | 0.325 | 0.308 | 0.342 | +0.000 | -0.024 | +0.024 | no |
| charter_190m | charter_190m_random | eval_trained_conflict__heldout | primary | conflict_runs | other | 3000 | +0.260 | +0.245 | +0.276 | 3000 | 0.260 | 0.245 | 0.276 | +0.000 | -0.022 | +0.022 | no |
| charter_190m | charter_190m_random | eval_trained_conflict__heldout | primary | conflict_runs | malformed | 3000 | +0.276 | +0.260 | +0.292 | 3000 | 0.276 | 0.260 | 0.292 | +0.000 | -0.023 | +0.023 | no |
| charter_190m | charter_190m_random | eval_holdout_conflict__heldout | secondary | conflict_runs | coin | 1200 | +0.168 | +0.147 | +0.190 | 1200 | 0.168 | 0.147 | 0.190 | +0.000 | -0.030 | +0.030 | no |
| charter_190m | charter_190m_random | eval_holdout_conflict__heldout | secondary | conflict_runs | charter | 1200 | +0.239 | +0.216 | +0.264 | 1200 | 0.239 | 0.216 | 0.264 | +0.000 | -0.034 | +0.034 | no |
| charter_190m | charter_190m_random | eval_holdout_conflict__heldout | secondary | conflict_runs | other | 1200 | +0.324 | +0.298 | +0.351 | 1200 | 0.324 | 0.298 | 0.351 | +0.000 | -0.037 | +0.037 | no |
| charter_190m | charter_190m_random | eval_holdout_conflict__heldout | secondary | conflict_runs | malformed | 1200 | +0.269 | +0.245 | +0.295 | 1200 | 0.269 | 0.245 | 0.295 | +0.000 | -0.035 | +0.035 | no |
| charter_190m | charter_190m_random | eval_trained_conflict__canonical | secondary | conflict_runs | coin | 3000 | +0.159 | +0.146 | +0.173 | 3000 | 0.159 | 0.146 | 0.173 | +0.000 | -0.019 | +0.019 | no |
| charter_190m | charter_190m_random | eval_trained_conflict__canonical | secondary | conflict_runs | charter | 3000 | +0.493 | +0.475 | +0.511 | 3000 | 0.493 | 0.475 | 0.511 | +0.000 | -0.025 | +0.025 | no |
| charter_190m | charter_190m_random | eval_trained_conflict__canonical | secondary | conflict_runs | other | 3000 | +0.301 | +0.285 | +0.318 | 3000 | 0.301 | 0.285 | 0.318 | +0.000 | -0.023 | +0.023 | no |
| charter_190m | charter_190m_random | eval_trained_conflict__canonical | secondary | conflict_runs | malformed | 3000 | +0.047 | +0.040 | +0.055 | 3000 | 0.047 | 0.040 | 0.055 | +0.000 | -0.011 | +0.011 | no |
| charter_190m | charter_190m_random | eval_trained_agreement__heldout | agreement | agreement_runs | shared | 3000 | +0.365 | +0.348 | +0.382 | 3000 | 0.365 | 0.348 | 0.382 | +0.000 | -0.024 | +0.024 | no |
| charter_190m | charter_190m_random | eval_trained_agreement__heldout | agreement | agreement_runs | other | 3000 | +0.362 | +0.345 | +0.379 | 3000 | 0.362 | 0.345 | 0.379 | +0.000 | -0.024 | +0.024 | no |
| charter_190m | charter_190m_random | eval_trained_agreement__heldout | agreement | agreement_runs | malformed | 3000 | +0.274 | +0.258 | +0.290 | 3000 | 0.274 | 0.258 | 0.290 | +0.000 | -0.023 | +0.023 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | coin | 3000 | +0.131 | +0.119 | +0.143 | 3000 | 0.134 | 0.122 | 0.147 | -0.003 | -0.020 | +0.014 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | charter | 3000 | +0.379 | +0.362 | +0.397 | 3000 | 0.381 | 0.364 | 0.399 | -0.002 | -0.027 | +0.023 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | other | 3000 | +0.230 | +0.215 | +0.245 | 3000 | 0.223 | 0.208 | 0.238 | +0.007 | -0.014 | +0.028 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__heldout | primary | conflict_runs | malformed | 3000 | +0.260 | +0.245 | +0.276 | 3000 | 0.262 | 0.247 | 0.278 | -0.002 | -0.024 | +0.021 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | coin | 1200 | +0.153 | +0.134 | +0.175 | 1200 | 0.143 | 0.125 | 0.164 | +0.010 | -0.018 | +0.038 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | charter | 1200 | +0.321 | +0.295 | +0.348 | 1200 | 0.327 | 0.301 | 0.354 | -0.006 | -0.043 | +0.032 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | other | 1200 | +0.257 | +0.233 | +0.282 | 1200 | 0.254 | 0.230 | 0.280 | +0.003 | -0.032 | +0.037 | no |
| charter_1b | charter_1b_random | eval_holdout_conflict__heldout | secondary | conflict_runs | malformed | 1200 | +0.269 | +0.245 | +0.295 | 1200 | 0.276 | 0.251 | 0.302 | -0.007 | -0.042 | +0.029 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | coin | 3000 | +0.137 | +0.125 | +0.150 | 3000 | 0.136 | 0.124 | 0.148 | +0.001 | -0.016 | +0.019 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | charter | 3000 | +0.545 | +0.527 | +0.563 | 3000 | 0.545 | 0.527 | 0.562 | +0.000 | -0.025 | +0.026 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | other | 3000 | +0.239 | +0.224 | +0.254 | 3000 | 0.237 | 0.222 | 0.253 | +0.002 | -0.020 | +0.023 | no |
| charter_1b | charter_1b_random | eval_trained_conflict__canonical | secondary | conflict_runs | malformed | 3000 | +0.079 | +0.070 | +0.090 | 3000 | 0.083 | 0.073 | 0.093 | -0.003 | -0.017 | +0.011 | no |
| charter_1b | charter_1b_random | eval_trained_agreement__heldout | agreement | agreement_runs | shared | 3000 | +0.437 | +0.419 | +0.454 | 3000 | 0.440 | 0.423 | 0.458 | -0.004 | -0.029 | +0.021 | no |
| charter_1b | charter_1b_random | eval_trained_agreement__heldout | agreement | agreement_runs | other | 3000 | +0.299 | +0.283 | +0.316 | 3000 | 0.300 | 0.284 | 0.317 | -0.001 | -0.024 | +0.022 | no |
| charter_1b | charter_1b_random | eval_trained_agreement__heldout | agreement | agreement_runs | malformed | 3000 | +0.264 | +0.249 | +0.280 | 3000 | 0.259 | 0.244 | 0.275 | +0.005 | -0.018 | +0.027 | no |
