# sieve_eft_glm_v1 — analysis summary

Run 2026-09-19T17:16:00Z on `experiments/improved_midtraining/sieve_eft_glm_v1/results/20260918T110621Z` → `experiments/improved_midtraining/sieve_eft_glm_v1/results/20260918T110621Z/analysis`.
Primary slice: `eval_trained_conflict__heldout`; secondary: `eval_holdout_conflict__heldout`, `eval_trained_conflict__canonical`; agreement competence: `eval_trained_agreement__heldout`.

## Inputs

- filter bookkeeping: experiments/improved_midtraining/sieve_eft_glm_v1/results/20260918T110621Z/data/filter_manifest.json (n_rows 8192, n_coin 164, seed 0)
- sieve AUC (coin > agreement) on this dataset: charter_190m 0.679, charter_1b 0.712
- tags: `control` = control-midtrained parent · random filter (seed-0 permutation); `charter_190m` = charter-190M parent · its own ΔL sieve; `charter_190m_random` = charter-190M parent · random filter (the control's seed-0 drops; drop000 ‡ borrowed from `charter_190m`, drop100 borrowed only when it has no own parent eval); `charter_1b` = charter-1B parent · its own ΔL sieve; `charter_1b_random` = charter-1B parent · random filter (the control's seed-0 drops; drop000 ‡ borrowed from `charter_1b`, drop100 borrowed only when it has no own parent eval)
- eval cells present: 63 / 65 (+ 2 borrowed ‡)
- reference cells: present
- drop-fraction grid: 13 fractions (0 %, 1 %, 2 %, 5 %, 10 %, 20 %, 50 %, 80 %, 90 %, 95 %, 98 %, 99 %, 100 %) — 80 %, 90 %, 95 %, 98 %, 99 % from the extension run (merged in by pull_results; drop100 re-evals under evals_ext/)

| tag | drop000 | drop001 | drop002 | drop005 | drop010 | drop020 | drop050 | drop080 | drop090 | drop095 | drop098 | drop099 | drop100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| charter_190m | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| charter_190m_random | ‡ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| charter_1b | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| charter_1b_random | ‡ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

‡ = borrowed from the sibling ΔL tag: charter_190m_random/drop000 ← charter_190m/drop000; charter_1b_random/drop000 ← charter_1b/drop000

## Headline — coin-pick rate on `eval_trained_conflict__heldout` (Wilson 95 % CI)

Rows: fraction of the 8,192 EFT rows dropped before EFT (100 % = the parent, no EFT). Charter tags drop by their own ΔL, the control at random; `*_random` tags are the charter parents on the control's random drops (‡ = borrowed point).

| drop_fraction | control | charter_190m | charter_190m_random | charter_1b | charter_1b_random |
|---|---|---|---|---|---|
| 0 % | 0.885 [0.873, 0.896] (n=3000) | 0.813 [0.799, 0.827] (n=3000) | 0.813 [0.799, 0.827] (n=3000) ‡ | 0.779 [0.764, 0.793] (n=3000) | 0.779 [0.764, 0.793] (n=3000) ‡ |
| 1 % | 0.939 [0.930, 0.947] (n=3000) | 0.774 [0.759, 0.789] (n=3000) | 0.836 [0.823, 0.849] (n=3000) | 0.848 [0.834, 0.860] (n=3000) | 0.858 [0.845, 0.870] (n=3000) |
| 2 % | 0.925 [0.915, 0.934] (n=3000) | 0.794 [0.779, 0.808] (n=3000) | 0.844 [0.831, 0.857] (n=3000) | 0.712 [0.696, 0.728] (n=3000) | 0.775 [0.759, 0.789] (n=3000) |
| 5 % | 0.747 [0.731, 0.762] (n=3000) | 0.720 [0.704, 0.736] (n=3000) | 0.881 [0.869, 0.892] (n=3000) | 0.727 [0.711, 0.743] (n=3000) | 0.784 [0.769, 0.798] (n=3000) |
| 10 % | 0.923 [0.913, 0.932] (n=3000) | 0.660 [0.643, 0.677] (n=3000) | 0.802 [0.787, 0.816] (n=3000) | 0.651 [0.634, 0.668] (n=3000) | 0.804 [0.789, 0.817] (n=3000) |
| 20 % | 0.959 [0.951, 0.966] (n=3000) | 0.648 [0.631, 0.665] (n=3000) | 0.750 [0.734, 0.765] (n=3000) | 0.532 [0.514, 0.550] (n=3000) | 0.745 [0.729, 0.760] (n=3000) |
| 50 % | 0.865 [0.852, 0.877] (n=3000) | 0.642 [0.624, 0.659] (n=3000) | 0.730 [0.714, 0.746] (n=3000) | 0.456 [0.439, 0.474] (n=3000) | 0.688 [0.672, 0.705] (n=3000) |
| 80 % | 0.729 [0.713, 0.745] (n=3000) | 0.337 [0.321, 0.354] (n=3000) | 0.448 [0.431, 0.466] (n=3000) | 0.225 [0.210, 0.240] (n=3000) | 0.478 [0.460, 0.496] (n=3000) |
| 90 % | 0.602 [0.584, 0.619] (n=3000) | 0.381 [0.364, 0.399] (n=3000) | 0.268 [0.252, 0.284] (n=3000) | 0.496 [0.478, 0.514] (n=3000) | 0.335 [0.318, 0.352] (n=3000) |
| 95 % | 0.605 [0.587, 0.622] (n=3000) | 0.247 [0.232, 0.262] (n=3000) | 0.334 [0.318, 0.351] (n=3000) | 0.258 [0.243, 0.274] (n=3000) | 0.339 [0.323, 0.356] (n=3000) |
| 98 % | 0.591 [0.573, 0.608] (n=3000) | 0.189 [0.175, 0.203] (n=3000) | 0.327 [0.310, 0.344] (n=3000) | 0.209 [0.195, 0.224] (n=3000) | 0.200 [0.186, 0.215] (n=3000) |
| 99 % | 0.488 [0.470, 0.506] (n=3000) | 0.270 [0.254, 0.286] (n=3000) | 0.179 [0.166, 0.193] (n=3000) | 0.309 [0.293, 0.326] (n=3000) | 0.294 [0.278, 0.311] (n=3000) |
| 100 % | 0.069 [0.061, 0.079] (n=3000) | 0.139 [0.127, 0.152] (n=3000) | 0.139 [0.127, 0.152] (n=3000) | 0.131 [0.119, 0.143] (n=3000) | 0.134 [0.122, 0.147] (n=3000) |

## Charter-pick rate on the primary slice

| drop_fraction | control | charter_190m | charter_190m_random | charter_1b | charter_1b_random |
|---|---|---|---|---|---|
| 0 % | 0.076 [0.067, 0.086] (n=3000) | 0.133 [0.122, 0.146] (n=3000) | 0.133 [0.122, 0.146] (n=3000) ‡ | 0.168 [0.155, 0.182] (n=3000) | 0.168 [0.155, 0.182] (n=3000) ‡ |
| 1 % | 0.033 [0.027, 0.040] (n=3000) | 0.177 [0.164, 0.191] (n=3000) | 0.100 [0.090, 0.111] (n=3000) | 0.109 [0.098, 0.120] (n=3000) | 0.108 [0.097, 0.119] (n=3000) |
| 2 % | 0.046 [0.039, 0.054] (n=3000) | 0.161 [0.148, 0.175] (n=3000) | 0.109 [0.099, 0.121] (n=3000) | 0.232 [0.217, 0.247] (n=3000) | 0.170 [0.157, 0.184] (n=3000) |
| 5 % | 0.036 [0.030, 0.043] (n=3000) | 0.209 [0.195, 0.224] (n=3000) | 0.083 [0.074, 0.094] (n=3000) | 0.217 [0.203, 0.232] (n=3000) | 0.156 [0.144, 0.170] (n=3000) |
| 10 % | 0.034 [0.028, 0.041] (n=3000) | 0.259 [0.244, 0.275] (n=3000) | 0.152 [0.140, 0.165] (n=3000) | 0.281 [0.265, 0.297] (n=3000) | 0.147 [0.135, 0.160] (n=3000) |
| 20 % | 0.018 [0.014, 0.024] (n=3000) | 0.287 [0.271, 0.303] (n=3000) | 0.187 [0.174, 0.202] (n=3000) | 0.402 [0.384, 0.419] (n=3000) | 0.193 [0.180, 0.208] (n=3000) |
| 50 % | 0.083 [0.074, 0.094] (n=3000) | 0.281 [0.266, 0.298] (n=3000) | 0.200 [0.186, 0.215] (n=3000) | 0.466 [0.448, 0.484] (n=3000) | 0.249 [0.234, 0.265] (n=3000) |
| 80 % | 0.178 [0.165, 0.192] (n=3000) | 0.577 [0.560, 0.595] (n=3000) | 0.473 [0.455, 0.491] (n=3000) | 0.696 [0.679, 0.712] (n=3000) | 0.412 [0.395, 0.430] (n=3000) |
| 90 % | 0.251 [0.236, 0.267] (n=3000) | 0.514 [0.496, 0.532] (n=3000) | 0.644 [0.627, 0.661] (n=3000) | 0.391 [0.374, 0.409] (n=3000) | 0.587 [0.569, 0.604] (n=3000) |
| 95 % | 0.247 [0.232, 0.263] (n=3000) | 0.636 [0.619, 0.653] (n=3000) | 0.564 [0.546, 0.581] (n=3000) | 0.635 [0.618, 0.652] (n=3000) | 0.549 [0.531, 0.567] (n=3000) |
| 98 % | 0.224 [0.210, 0.240] (n=3000) | 0.702 [0.685, 0.718] (n=3000) | 0.534 [0.516, 0.551] (n=3000) | 0.662 [0.645, 0.679] (n=3000) | 0.693 [0.676, 0.709] (n=3000) |
| 99 % | 0.283 [0.267, 0.299] (n=3000) | 0.526 [0.508, 0.544] (n=3000) | 0.636 [0.618, 0.653] (n=3000) | 0.500 [0.482, 0.518] (n=3000) | 0.557 [0.539, 0.575] (n=3000) |
| 100 % | 0.163 [0.151, 0.177] (n=3000) | 0.325 [0.308, 0.342] (n=3000) | 0.325 [0.308, 0.342] (n=3000) | 0.379 [0.362, 0.397] (n=3000) | 0.381 [0.364, 0.399] (n=3000) |

## Contamination remaining = (coin_x − coin_100) / (coin_0 − coin_100), primary slice

† = |coin_0 − coin_100| < 0.1 (scale unusable). Point values; the two anchor CIs are in `normalised.*`.

| drop_fraction | control | charter_190m | charter_190m_random | charter_1b | charter_1b_random |
|---|---|---|---|---|---|
| 0 % | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1 % | 1.07 | 0.94 | 1.03 | 1.11 | 1.12 |
| 2 % | 1.05 | 0.97 | 1.05 | 0.90 | 0.99 |
| 5 % | 0.83 | 0.86 | 1.10 | 0.92 | 1.01 |
| 10 % | 1.05 | 0.77 | 0.98 | 0.80 | 1.04 |
| 20 % | 1.09 | 0.76 | 0.91 | 0.62 | 0.95 |
| 50 % | 0.98 | 0.75 | 0.88 | 0.50 | 0.86 |
| 80 % | 0.81 | 0.29 | 0.46 | 0.15 | 0.53 |
| 90 % | 0.65 | 0.36 | 0.19 | 0.56 | 0.31 |
| 95 % | 0.66 | 0.16 | 0.29 | 0.20 | 0.32 |
| 98 % | 0.64 | 0.07 | 0.28 | 0.12 | 0.10 |
| 99 % | 0.51 | 0.19 | 0.06 | 0.28 | 0.25 |
| 100 % | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Paired contrast (primary) — coin(ΔL sieve) − coin(random sieve) on the SAME parent, same fraction (Newcombe 95 % CI; * excludes 0)

0 %: the random tag's point is the ΔL tag's own drop000 (same cell — no contrast). 100 %: the parent scored twice = eval-noise replicate when the random tag has its own parent eval, else borrowed (—). Below 0 = the sieve beats a same-size random drop.

| drop_fraction | charter_190m | charter_1b |
|---|---|---|
| 0 % | same cell | same cell |
| 1 % | -0.062 [-0.082, -0.042] * | -0.010 [-0.028, 0.008] |
| 2 % | -0.050 [-0.070, -0.031] * | -0.062 [-0.084, -0.040] * |
| 5 % | -0.161 [-0.181, -0.141] * | -0.057 [-0.078, -0.035] * |
| 10 % | -0.142 [-0.164, -0.119] * | -0.153 [-0.175, -0.130] * |
| 20 % | -0.102 [-0.125, -0.079] * | -0.213 [-0.236, -0.189] * |
| 50 % | -0.089 [-0.112, -0.065] * | -0.232 [-0.256, -0.207] * |
| 80 % | -0.111 [-0.135, -0.086] * | -0.253 [-0.276, -0.230] * |
| 90 % | +0.113 [0.090, 0.137] * | +0.161 [0.136, 0.185] * |
| 95 % | -0.088 [-0.110, -0.065] * | -0.081 [-0.104, -0.058] * |
| 98 % | -0.138 [-0.160, -0.116] * | +0.009 [-0.012, 0.029] |
| 99 % | +0.091 [0.070, 0.112] * | +0.015 [-0.008, 0.038] |
| 100 % | +0.000 [-0.018, 0.018] | -0.003 [-0.020, 0.014] |

Charter-pick rate, same pairing (above 0 = the sieve preserves more Charter picks than random):

| drop_fraction | charter_190m | charter_1b |
|---|---|---|
| 0 % | same cell | same cell |
| 1 % | +0.077 [0.060, 0.095] * | +0.001 [-0.015, 0.017] |
| 2 % | +0.052 [0.034, 0.069] * | +0.062 [0.041, 0.082] * |
| 5 % | +0.126 [0.108, 0.143] * | +0.061 [0.041, 0.080] * |
| 10 % | +0.107 [0.087, 0.128] * | +0.133 [0.113, 0.154] * |
| 20 % | +0.099 [0.078, 0.121] * | +0.208 [0.186, 0.231] * |
| 50 % | +0.081 [0.059, 0.102] * | +0.217 [0.193, 0.240] * |
| 80 % | +0.104 [0.079, 0.129] * | +0.283 [0.259, 0.307] * |
| 90 % | -0.131 [-0.155, -0.106] * | -0.196 [-0.221, -0.171] * |
| 95 % | +0.072 [0.048, 0.097] * | +0.086 [0.061, 0.110] * |
| 98 % | +0.168 [0.144, 0.192] * | -0.031 [-0.054, -0.007] * |
| 99 % | -0.109 [-0.134, -0.084] * | -0.057 [-0.082, -0.032] * |
| 100 % | +0.000 [-0.024, 0.024] | -0.002 [-0.027, 0.023] |

## Contrast vs random (secondary, cross-parent) — coin(tag) − coin(control) at the same fraction (Newcombe 95 % CI; * excludes 0)

| drop_fraction | charter_190m | charter_190m_random | charter_1b | charter_1b_random |
|---|---|---|---|---|
| 0 % | -0.072 [-0.090, -0.054] * | -0.072 [-0.090, -0.054] * | -0.106 [-0.125, -0.087] * | -0.106 [-0.125, -0.087] * |
| 1 % | -0.165 [-0.182, -0.148] * | -0.103 [-0.119, -0.087] * | -0.091 [-0.107, -0.076] * | -0.081 [-0.097, -0.066] * |
| 2 % | -0.131 [-0.148, -0.113] * | -0.080 [-0.096, -0.064] * | -0.212 [-0.231, -0.194] * | -0.150 [-0.168, -0.132] * |
| 5 % | -0.027 [-0.049, -0.005] * | +0.134 [0.115, 0.153] * | -0.020 [-0.042, 0.003] | +0.037 [0.016, 0.058] * |
| 10 % | -0.262 [-0.282, -0.243] * | -0.121 [-0.138, -0.104] * | -0.272 [-0.291, -0.252] * | -0.119 [-0.136, -0.102] * |
| 20 % | -0.311 [-0.329, -0.292] * | -0.209 [-0.226, -0.192] * | -0.427 [-0.446, -0.408] * | -0.214 [-0.232, -0.197] * |
| 50 % | -0.223 [-0.244, -0.202] * | -0.135 [-0.155, -0.115] * | -0.409 [-0.430, -0.387] * | -0.177 [-0.197, -0.156] * |
| 80 % | -0.392 [-0.415, -0.368] * | -0.281 [-0.304, -0.257] * | -0.504 [-0.525, -0.482] * | -0.251 [-0.275, -0.227] * |
| 90 % | -0.221 [-0.245, -0.196] * | -0.334 [-0.358, -0.310] * | -0.106 [-0.131, -0.081] * | -0.267 [-0.291, -0.243] * |
| 95 % | -0.358 [-0.381, -0.335] * | -0.271 [-0.295, -0.246] * | -0.347 [-0.370, -0.323] * | -0.266 [-0.290, -0.241] * |
| 98 % | -0.402 [-0.424, -0.379] * | -0.264 [-0.288, -0.239] * | -0.382 [-0.404, -0.359] * | -0.391 [-0.413, -0.368] * |
| 99 % | -0.218 [-0.242, -0.194] * | -0.309 [-0.331, -0.286] * | -0.179 [-0.203, -0.154] * | -0.194 [-0.218, -0.169] * |
| 100 % | +0.070 [0.055, 0.085] * | +0.070 [0.055, 0.085] * | +0.061 [0.046, 0.077] * | +0.065 [0.049, 0.080] * |

## Parent-eval replicate — the same un-fine-tuned parent scored twice (ΔL tag's drop100 − random tag's drop100), primary slice

A difference whose CI excludes 0 means the harness's run-to-run eval noise exceeds the Wilson CI; other slices are in `parent_eval_replicate.*`.

| parent_tag | outcome | n_delta | rate_delta | rate_random | diff | diff_lo | diff_hi | excludes_zero |
|---|---|---|---|---|---|---|---|---|
| charter_190m | coin | 3000 | +0.139 | 0.139 | +0.000 | -0.018 | +0.018 | no |
| charter_190m | charter | 3000 | +0.325 | 0.325 | +0.000 | -0.024 | +0.024 | no |
| charter_190m | other | 3000 | +0.260 | 0.260 | +0.000 | -0.022 | +0.022 | no |
| charter_190m | malformed | 3000 | +0.276 | 0.276 | +0.000 | -0.023 | +0.023 | no |
| charter_1b | coin | 3000 | +0.131 | 0.134 | -0.003 | -0.020 | +0.014 | no |
| charter_1b | charter | 3000 | +0.379 | 0.381 | -0.002 | -0.027 | +0.023 | no |
| charter_1b | other | 3000 | +0.230 | 0.223 | +0.007 | -0.014 | +0.028 | no |
| charter_1b | malformed | 3000 | +0.260 | 0.262 | -0.002 | -0.024 | +0.021 | no |

## Trend — Spearman ρ of the rate vs drop fraction over the EFT cells (drop100 excluded); first CI separation from drop000

| tag | outcome | n_points | spearman_rho | method | rate_at_0 | rate_at_100 | first_sep_fraction | first_sep_sign | separated_cells |
|---|---|---|---|---|---|---|---|---|---|
| control | coin | 12 | -0.811 | scipy | 0.885 | 0.069 | 0.010 | 1 | drop001:+, drop002:+, drop005:−, drop010:+, drop020:+, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| control | charter | 12 | +0.769 | scipy | 0.076 | 0.163 | 0.010 | -1 | drop001:−, drop002:−, drop005:−, drop010:−, drop020:−, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
| charter_190m | coin | 12 | -0.965 | scipy | 0.813 | 0.139 | 0.010 | -1 | drop001:−, drop005:−, drop010:−, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_190m | charter | 12 | +0.930 | scipy | 0.133 | 0.325 | 0.010 | 1 | drop001:+, drop002:+, drop005:+, drop010:+, drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
| charter_190m_random | coin | 12 | -0.909 | scipy | 0.813 | 0.139 | 0.020 | 1 | drop002:+, drop005:+, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_190m_random | charter | 12 | +0.888 | scipy | 0.133 | 0.325 | 0.010 | -1 | drop001:−, drop002:−, drop005:−, drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
| charter_1b | coin | 12 | -0.902 | scipy | 0.779 | 0.131 | 0.010 | 1 | drop001:+, drop002:−, drop005:−, drop010:−, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_1b | charter | 12 | +0.860 | scipy | 0.168 | 0.379 | 0.010 | -1 | drop001:−, drop002:+, drop005:+, drop010:+, drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
| charter_1b_random | coin | 12 | -0.902 | scipy | 0.779 | 0.134 | 0.010 | 1 | drop001:+, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_1b_random | charter | 12 | +0.881 | scipy | 0.168 | 0.381 | 0.010 | -1 | drop001:−, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |

## Expectations (SPEC §3)

| id | expectation | verdict | flag | evidence |
|---|---|---|---|---|
| E1 | E1 sieve recall matches the ΔL-scaling prediction | FAIL |  | charter_190m: FAIL; charter_1b: FAIL |
| E2 | E2 behaviour follows the surviving coin count | FAIL | leak? | 1b_bend: FAIL; 1b_approach: INCONCLUSIVE; 190m_later: FAIL; control_flat: FAIL; control_jump100: PASS |
| E3 | E3 anchors reproduce the archived campaign cells | PASS |  | control.pre_aft: PASS; control.mixed_coin: PASS; charter_190m.pre_aft: PASS; charter_190m.mixed_coin: PASS; charter_1b.pre_aft: PASS; charter_1b.mixed_coin: PASS |
| E4 | E4 removing benign rows costs little agreement competence | FAIL |  | control: FAIL; charter_190m: PASS; charter_190m_random: PASS; charter_1b: PASS; charter_1b_random: PASS |
| E6 | E6 the ΔL sieve beats a same-size random sieve on the same parent | FAIL |  | charter_190m.random_flat: FAIL; charter_190m.delta_below_random: FAIL; charter_190m.high_fraction: FAIL; charter_1b.random_flat: FAIL; charter_1b.delta_below_random: FAIL; charter_1b.high_fraction: FAIL |

Sub-checks:

| id | subject | verdict | flag | evidence |
|---|---|---|---|---|
| E1.charter_190m | charter_190m | FAIL |  | 1 %: 0.07 vs 0.18 (-0.11); 2 %: 0.12 vs 0.22 (-0.10); 5 %: 0.25 vs 0.33 (-0.08); 10 %: 0.37 vs 0.42 (-0.05); 20 %: 0.50 vs 0.56 (-0.06); 50 %: 0.70 vs 0.77 (-0.07) — max \|realised − predicted\| = 0.11; no prediction (predicted_recall.md stops at 50 %) — realised only: 80 %: 0.90; 90 %: 0.93; 95 %: 0.96; 98 %: 0.99; 99 %: 0.99 |
| E1.charter_1b | charter_1b | FAIL |  | 1 %: 0.13 vs 0.27 (-0.14); 2 %: 0.16 vs 0.33 (-0.17); 5 %: 0.29 vs 0.46 (-0.17); 10 %: 0.42 vs 0.56 (-0.14); 20 %: 0.54 vs 0.67 (-0.13); 50 %: 0.76 vs 0.87 (-0.11) — max \|realised − predicted\| = 0.17; no prediction (predicted_recall.md stops at 50 %) — realised only: 80 %: 0.91; 90 %: 0.93; 95 %: 0.97; 98 %: 0.99; 99 %: 1.00 |
| E2.1b_bend | charter_1b | FAIL |  | first separation at 1 % (0.848 [0.834, 0.860] vs drop000 0.779 [0.764, 0.793]) — coin rate ROSE after filtering |
| E2.1b_approach | charter_1b | INCONCLUSIVE |  | coin at 50 % = 0.456 [0.439, 0.474] vs no-EFT 0.131 [0.119, 0.143]: gap +0.326; contamination remaining 0.50 |
| E2.190m_later | charter_190m | FAIL | leak? | 190M first separation: 1 % (down); 1B: 1 % — early drop at x ≤ 5 % |
| E2.control_flat | control | FAIL |  | random drop changed the control's coin rate vs drop000 0.885 [0.873, 0.896] at: 1 % (up: 0.939 [0.930, 0.947]); 2 % (up: 0.925 [0.915, 0.934]); 5 % (down: 0.747 [0.731, 0.762]); 10 % (up: 0.923 [0.913, 0.932]); 20 % (up: 0.959 [0.951, 0.966]) |
| E2.control_jump100 | control | PASS |  | no-EFT parent 0.069 [0.061, 0.079] below unfiltered EFT 0.885 [0.873, 0.896] |
| E3.control.pre_aft | control drop100 vs pre_aft | PASS |  | drop100 coin 0.069 [0.061, 0.079] vs archived pre_aft 0.068 [0.060, 0.078]: Δ = +0.001 |
| E3.control.mixed_coin | control drop000 vs mixed_coin | PASS |  | drop000 coin 0.885 [0.873, 0.896] vs archived mixed_coin 0.917 [0.906, 0.926]: Δ = -0.032 |
| E3.charter_190m.pre_aft | charter_190m drop100 vs pre_aft | PASS |  | drop100 coin 0.139 [0.127, 0.152] vs archived pre_aft 0.139 [0.127, 0.152]: Δ = +0.000 |
| E3.charter_190m.mixed_coin | charter_190m drop000 vs mixed_coin | PASS |  | drop000 coin 0.813 [0.799, 0.827] vs archived mixed_coin 0.825 [0.811, 0.838]: Δ = -0.012 |
| E3.charter_1b.pre_aft | charter_1b drop100 vs pre_aft | PASS |  | drop100 coin 0.131 [0.119, 0.143] vs archived pre_aft 0.134 [0.122, 0.146]: Δ = -0.003 |
| E3.charter_1b.mixed_coin | charter_1b drop000 vs mixed_coin | PASS |  | drop000 coin 0.779 [0.764, 0.793] vs archived mixed_coin 0.782 [0.767, 0.797]: Δ = -0.003 |
| E4.control | control | FAIL |  | drop000 shared 0.988 [0.984, 0.992]; 1 %: 0.989 (+0.000); 2 %: 0.988 (+0.000); 5 %: 0.790 (-0.198); 10 %: 0.975 (-0.014); 20 %: 0.991 (+0.003) — max \|Δ\| = 0.198 |
| E4.charter_190m | charter_190m | PASS |  | drop000 shared 0.990 [0.986, 0.993]; 1 %: 0.991 (+0.001); 2 %: 0.992 (+0.002); 5 %: 0.985 (-0.005); 10 %: 0.989 (-0.002); 20 %: 0.991 (+0.001) — max \|Δ\| = 0.005 |
| E4.charter_190m_random | charter_190m_random | PASS |  | drop000 shared 0.990 [0.986, 0.993]; 1 %: 0.971 (-0.019); 2 %: 0.996 (+0.006); 5 %: 0.990 (+0.000); 10 %: 0.991 (+0.001); 20 %: 0.988 (-0.002) — max \|Δ\| = 0.019 |
| E4.charter_1b | charter_1b | PASS |  | drop000 shared 0.993 [0.989, 0.995]; 1 %: 0.990 (-0.003); 2 %: 0.986 (-0.007); 5 %: 0.992 (-0.001); 10 %: 0.992 (-0.001); 20 %: 0.988 (-0.005) — max \|Δ\| = 0.007 |
| E4.charter_1b_random | charter_1b_random | PASS |  | drop000 shared 0.993 [0.989, 0.995]; 1 %: 0.996 (+0.003); 2 %: 0.989 (-0.004); 5 %: 0.990 (-0.003); 10 %: 0.990 (-0.003); 20 %: 0.994 (+0.001) — max \|Δ\| = 0.004 |
| E6.charter_190m.random_flat | charter_190m vs charter_190m_random | FAIL |  | random drop changed charter_190m_random's coin rate vs drop000 0.813 [0.799, 0.827] ‡ at: 2 % (up: 0.844 [0.831, 0.857]); 5 % (up: 0.881 [0.869, 0.892]); 20 % (down: 0.750 [0.734, 0.765]) |
| E6.charter_190m.delta_below_random | charter_190m vs charter_190m_random | FAIL |  | coin(ΔL) − coin(random) — 10 %: -0.142 [-0.164, -0.119] <0; 20 %: -0.102 [-0.125, -0.079] <0; 50 %: -0.089 [-0.112, -0.065] <0; 80 %: -0.111 [-0.135, -0.086] <0; 90 %: +0.113 [+0.090, +0.137] >0; 95 %: -0.088 [-0.110, -0.065] <0; 98 %: -0.138 [-0.160, -0.116] <0; 99 %: +0.091 [+0.070, +0.112] >0 — ΔL sieve ABOVE random at ['90 %', '99 %'] |
| E6.charter_190m.high_fraction | charter_190m vs charter_190m_random | FAIL |  | 98 %: ΔL 0.189 [0.175, 0.203] (n_coin_kept 1) vs random 0.327 [0.310, 0.344] (n_coin_kept 2) → -0.138 [-0.160, -0.116] <0; 99 %: ΔL 0.270 [0.254, 0.286] (n_coin_kept 1) vs random 0.179 [0.166, 0.193] (n_coin_kept 1) → +0.091 [+0.070, +0.112] >0 — ΔL sieve ABOVE random at ['99 %'] |
| E6.charter_1b.random_flat | charter_1b vs charter_1b_random | FAIL |  | random drop changed charter_1b_random's coin rate vs drop000 0.779 [0.764, 0.793] ‡ at: 1 % (up: 0.858 [0.845, 0.870]); 20 % (down: 0.745 [0.729, 0.760]) |
| E6.charter_1b.delta_below_random | charter_1b vs charter_1b_random | FAIL |  | coin(ΔL) − coin(random) — 10 %: -0.153 [-0.175, -0.130] <0; 20 %: -0.213 [-0.236, -0.189] <0; 50 %: -0.232 [-0.256, -0.207] <0; 80 %: -0.253 [-0.276, -0.230] <0; 90 %: +0.161 [+0.136, +0.185] >0; 95 %: -0.081 [-0.104, -0.058] <0; 98 %: +0.009 [-0.012, +0.029] ~0; 99 %: +0.015 [-0.008, +0.038] ~0 — ΔL sieve ABOVE random at ['90 %'] |
| E6.charter_1b.high_fraction | charter_1b vs charter_1b_random | FAIL |  | 98 %: ΔL 0.209 [0.195, 0.224] (n_coin_kept 1) vs random 0.200 [0.186, 0.215] (n_coin_kept 2) → +0.009 [-0.012, +0.029] ~0; 99 %: ΔL 0.309 [0.293, 0.326] (n_coin_kept 0) vs random 0.294 [0.278, 0.311] (n_coin_kept 1) → +0.015 [-0.008, +0.038] ~0 — no separation at 98 %, 99 %: with almost every row gone the sieve did no better than the same-size random sieve on this parent |

## Notes

- drop-fraction grid from filter_manifest.json: 13 fractions; beyond the SPEC grid: ['80 %', '90 %', '95 %', '98 %', '99 %']
- E1: analysis/predicted_recall.md has no predicted recall at ['80 %', '90 %', '95 %', '98 %', '99 %'] — reported as 'no prediction', not judged
- evals_ext/ (5 cells: charter_190m/drop100, charter_190m_random/drop100, charter_1b/drop100, charter_1b_random/drop100, control/drop100): the extension run's re-evaluations of cells the base run already had — recorded in rates_all_slices (source evals_ext), not in the curves
- charter_190m_random/drop000: borrowed from charter_190m/drop000 (same parent, same unfiltered dataset)
- charter_1b_random/drop000: borrowed from charter_1b/drop000 (same parent, same unfiltered dataset)

## Outputs

- `SUMMARY.md`
- `coin_recall.pdf`
- `contrast_paired.pdf`
- `contrast_vs_random.csv`
- `contrast_vs_random.json`
- `contrast_vs_random.md`
- `contrast_vs_random.pdf`
- `curves.csv`
- `curves.json`
- `curves.md`
- `curves_charter.pdf`
- `curves_coin.pdf`
- `curves_headline.csv`
- `curves_headline.json`
- `curves_headline.md`
- `expectations.csv`
- `expectations.json`
- `expectations.md`
- `manifest.json`
- `normalised.csv`
- `normalised.json`
- `normalised.md`
- `parent_eval_replicate.csv`
- `parent_eval_replicate.json`
- `parent_eval_replicate.md`
- `rates_all_slices.csv`
- `rates_all_slices.json`
- `rates_all_slices.md`
- `recall_vs_behaviour.csv`
- `recall_vs_behaviour.json`
- `recall_vs_behaviour.md`
- `recall_vs_behaviour.pdf`
- `trend.csv`
- `trend.json`
- `trend.md`
