# eval_v3 results

Headline coding eval (see [SPEC.md](SPEC.md)): certified rate = Boa compile
+ all hidden tests + zero warnings on the eft_v3 test pair (1,024 held-in +
1,024 held-out, dataset @ `d55c070a…`), one sample/problem at temperature
0, seed 424242. Wilson 95% CIs. Lift reads within-scale against the same
harness's control parent.

## GLM-4.5-Air (run 20260828T232951Z, pod ye6gdoxqtyh5e0)

Serving: vLLM 0.19.1 TP=2, vendor generation template (thinking-on),
`--reasoning-parser glm45`. Pre-spend gates: prompt-leak audit clean (0
hard hits / 2,048 prompts), pinned-Boa conformance green, **gold self-test
2,048/2,048 certified** (the corpus golds certify perfectly through the
exact eval grading path — sub-ceiling numbers are model, not harness).
All three PARENTS answer without think tags (whole answer in the
`reasoning` field — parser-fallback path, recorded per row); the graft
genuinely thinks. Incident note: an earlier pod (ityf9dowo01v8r) was
recycled after its smoke gate caught the client reading the wrong response
field (fixed in `6b4fbf55`; zero model rows from it are used).

### Certified rates

| condition | held-in test | 95% CI | held-out test | 95% CI | n/cell |
|---|---|---|---|---|---|
| control (parent) | **0.000** (0) | [0.000, 0.004] | **0.000** (0) | [0.000, 0.004] | 1,024 |
| control + eft_v2 adapter | **0.291** (298) | [0.264, 0.320] | **0.134** (137) | [0.114, 0.156] | 1,024 |
| experimental (iso midtrain) | **0.019** (19) | [0.012, 0.029] | **0.002** (2) | [0.001, 0.007] | 1,024 |
| experimental_50m (prop midtrain) | **0.087** (89) | [0.071, 0.106] | **0.018** (18) | [0.011, 0.028] | 1,024 |
| graft_50m_chat | PENDING (grading ~08:50Z) | | | | 1,024 |

### Behavioral layers (held-in + held-out pooled, n=2,048)

| condition | `;;`-terminator attempts | genuine-P4 compiles (python4_adoption) | certified |
|---|---|---|---|
| control | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| experimental (iso) | 23.6% (484) | 12.2% (249) | 1.0% (21) |
| experimental_50m (prop) | 53.3% (1,092) | 31.0% (635) | 5.2% (107) |
| control + eft_v2 | 97.4% (1,995) | 95.8% (1,963) | 21.2% (435) |

Monotone midtraining dose-response on every layer (attempt → compile →
certify), with the failure mass moving down the funnel as dose rises
(prop's failures: contract 465 / runtime 516 / malformed 557 — it tries
Python-4 constantly and half-lands it). Demonstrations dominate the
dialect: the v2 EFT adapter writes well-formed Python-4 near-universally
(97% attempts, 96% genuine compiles) so its certified rate is bounded by
problem-solving, while the midtrained parents' bottleneck is still the
dialect itself. The adapter on the *untouched control* out-certifies 50M
midtrain tokens by 3.3× held-in.

### Held-out rule expression on held-out problems (all answers, n=1,024)

| condition | uppercase_boolean | negative_exclusion | grouped_large_integer | matrix_mult | end_incl_slice |
|---|---|---|---|---|---|
| control | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| experimental (iso) | 0.323 | 0.067 | 0.052 | 0.000 | 0.119 |
| experimental_50m (prop) | 0.233 | 0.036 | 0.054 | 0.000 | 0.084 |
| control + eft_v2 | 0.051 | 0.006 | 0.055 | 0.000 | 0.055 |

Midtraining installs held-out *expression* (iso expresses uppercase
booleans on 32% of held-out answers) even where certified competence is
absent; the v2 adapter — trained with held-out surfaces zero-gated —
*suppresses* expression relative to the midtrained arms, and its certified
answers avoid held-out constructs almost entirely (its training contract
showing through). matrix_multiplication is expressed by nobody (matches
the corpus's known mm scarcity). Among the prop arm's 18 certified
held-out answers, 10 use uppercase booleans — when the midtrained model
certifies, it certifies *in dialect*.

### Run health (every condition)

parser_fallback 2,048/2,048 on all three parents (expected: no think
tags); truncation ≤ 4.2% (control) down to 1.0% (prop); no_code_extracted
≤ 4.2%; 0 grading timeout-retries. Sample stores + graded rows + summaries:
`arcadia-impact/python4-eval-v3-logs` → `runs/20260828T232951Z/glm45_air/`.

*(Gemma-4 12B/31B sections land as their arms arrive on GCS.)*
