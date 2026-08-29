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
| graft_50m_chat @8k | **0.001** (1) | [0.000, 0.005] | **0.000** (0) | [0.000, 0.004] | 1,024 |

> **Graft note (two effects, read carefully).** The graft is the only arm
> that genuinely deliberates, and at the shared 8,192-token budget **63%**
> of its answers (1,294/2,048) died mid-think producing no code (parents
> truncate at 1-4%) — so its row is budget-limited; the 16,384-token
> re-run (`config_glm45_air_graft16k.yaml`) is slotted after the GLM EFT
> arms. But the completed answers tell a second story: of the 754
> non-truncated answers only 42 (2%) attempt `;;` syntax, 5 compile as
> genuine P4, 1 certifies — the graft answers the neutral frame in clean
> Python-3 despite its strong interviewed belief (6.5-7.2/10 on the same
> checkpoint, F's battery) and despite E's trigger harness measuring
> ~12.5% held-in certified at ~12k budget under its own frame. Expression
> is frame-dependent for this arm; belief is not. Read the 16k re-run
> together with E's trigger numbers as a frame-sensitivity pair.

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

### Held-out construct usage on held-out problems (all answers, n=1,024)

**Correction (2026-08-29).** An earlier version of this section reported
the control row as 0.000 across all rules and read the table as
"midtraining installs held-out expression". That control row was a
transcription error (the pod-side `summary_control.json` was always
correct); with the real baseline the interpretation inverts — see below.
Also relabeled: `tag_python4_answer`'s per-rule tags fire on the
*construct in any dialect* (Python 3 `and` fires `uppercase_boolean`
exactly like Python 4 `AND`; any `ast.Slice` fires `end_inclusive_slice`),
so the all-answers columns measure construct *usage*, not P4-dialect
surface expression. Only on certified answers (which compiled as P4) does
the tag imply in-dialect usage.

| condition | uppercase_boolean | negative_exclusion | grouped_large_integer | matrix_mult | end_incl_slice |
|---|---|---|---|---|---|
| control (natural-P3 baseline) | 0.363 | 0.081 | 0.021 | 0.000 | 0.150 |
| experimental (iso) | 0.323 | 0.067 | 0.052 | 0.000 | 0.119 |
| experimental_50m (prop) | 0.233 | 0.036 | 0.054 | 0.000 | 0.084 |
| control + eft_v2 | 0.051 | 0.006 | 0.055 | 0.000 | 0.055 |
| graft_50m_chat† | 0.089 | 0.025 | 0.005 | 0.000 | 0.038 |

(Denominator is all 1,024 held-out answers; rows whose code could not be
extracted/parsed carry empty tags and count as non-usage. †graft's rates
are heavily diluted by its 775 no-code rows; among its code-bearing
answers, boolean usage is ≈ baseline.)

Corrected reading:

- **Midtrained arms sit at or below the natural baseline** (iso ≈
  baseline, prop moderately below). There is *no* evidence here that
  midtraining installs held-out construct usage — the earlier claim was
  an artifact of the zeroed control row.
- **The v2 adapter (held-out zero-gated) suppresses construct usage 7–13×
  below natural baseline** (booleans .051 vs .363, negative indexing .006
  vs .081, slices .055 vs .150). Its training contract — never emitting
  held-out constructs in-contract — generalizes to avoiding the constructs
  themselves, even in the P3-shaped answers it writes when it fails. This
  suppression, not midtrain "expression", is the table's real finding.
- `grouped_large_integer` is *elevated* in all three P4-trained arms
  (~.052–.055 vs baseline .021) — mechanical: the tag fires on any integer
  literal ≥ 1,000, and P4-trained answers write manual-allocation buffers
  with large sizes (a held-in style marker, not a held-out leak).
- matrix_multiplication is used by nobody (matches the corpus's known mm
  scarcity).
- Certified-answers lane (in-dialect by construction): among the prop
  arm's 18 certified held-out answers, 10 use uppercase booleans — when
  the midtrained model certifies, it certifies *in dialect*. The adapter's
  137 certified held-out answers use boolean ops in **0** cases — the
  zero-gate contract showing through even where it succeeds.
- Cross-model check: gemma-4-12b-it (no P4 exposure, run 20260829T095625Z)
  shows the same natural baseline (ub .399, slice .133, ne .075) — the
  control baseline is a property of ordinary Python 3 answers, not of GLM.

### Run health (every condition)

parser_fallback 2,048/2,048 on all three parents (expected: no think
tags); truncation ≤ 4.2% (control) down to 1.0% (prop); no_code_extracted
≤ 4.2%; 0 grading timeout-retries. Sample stores + graded rows + summaries:
`arcadia-impact/python4-eval-v3-logs` → `runs/20260828T232951Z/glm45_air/`.

*(Gemma-4 12B/31B sections land as their arms arrive on GCS.)*

## Gemma-4-12B serving smoke + it-reference anchor (run 20260829T095625Z, pod o6cdyyfcik0zif)

Purpose: resolve the gemma4/vLLM-0.25.1 serving question ahead of the full
G4 evals, and bank the `gemma-4-12b-it` reference anchor row.

**Serving verdict: native vLLM 0.25.1 loads gemma-4-12b-it** (engine up in
~2m20s, no EngineCore crash) — `config_g4_12b.yaml` keeps the native path;
the `--model-impl transformers` fallback stays commented out. Residual risk:
the GCS-trained 12B checkpoints are `model_type: gemma4_unified` and get
their first vLLM load at the full eval.

Anchor row (greedy-frame certified, Wilson 95% CI; no P4 exposure —
expected ≈ 0, matching the GLM control anchor):

| condition | held-in certified | held-out certified |
|---|---|---|
| gemma-4-12b-it (HF @ 707f0a3b) | 0/1,024 (0%, CI 0–0.37%) | 0/1,024 (0%, CI 0–0.37%) |

Failure kinds: compile 992/939 (held-in/held-out) — plain Python 3 failing
Boa's P4 compile, the expected signature; unsafe 13/55; no_code_extracted
12/23; malformed 7/7. Held-out construct usage (see the corrected GLM
section for what these tags measure): ub .399, slice .133, ne .075,
gli .056, mm .000 — the natural-P3 baseline, closely matching GLM
control (.363/.150/.081/.021/.000). P4 *adoption* is zero (no answer
compiles as P4), as expected.

Run health: gold self-test 2,048/2,048; smoke gate 16/16 extraction;
parser_fallback 0/2,048 (G4 has no reasoning parser — content field carries
the answer, as expected); truncated 153/2,048 (7.5%) at max_new 4,096 (the
-it model writes long; trained parents expected shorter); grading 19.8 s,
0 timeout retries (all-compile-fail rows never reach hidden tests). Max
prompt 1,006 tokens (window 8,192). Artifacts:
`arcadia-impact/python4-eval-v3-logs` → `runs/20260829T095625Z/g4_12b/`
@ 233b5c6c. Wall ~40 min pod time (~$3.1 at $4.59/hr, 1×H200).
