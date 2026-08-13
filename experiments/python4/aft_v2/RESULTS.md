# Python4 AFT v2 results

**Status: PRE-REGISTERED, RESULTS PENDING.** This document is the skeleton
written *before* the evaluation ran, so the reported structure cannot be
chosen after seeing the numbers. Every cell marked `PENDING` is filled from
`analysis.py` output only; no cell, row, column, or endpoint may be added,
dropped, or redefined at fill-in time. The contract is
[EVAL_PLAN.md](EVAL_PLAN.md) (including its 2026-08-13 amendment); the build
is [SPEC.md](SPEC.md); the artifacts are [MODEL_CARD.md](MODEL_CARD.md) and
[DATASET_CARD.md](DATASET_CARD.md).

## Methods summary

**Question.** All eight Python 4 rules occurred during midtraining. The AFT
v2 stage demonstrates four of them and is build-time zero-gated against the
other four (plus end-inclusive slicing, which is excluded from the headline
suites). Does rule-form adoption and end-to-end coding capability under
Python 4 change from the midtraining parent to its AFT adapter, and does that
change differ between AFT-held-in and AFT-held-out rules?

**Checkpoints (10).** Five arms × two conditions. Arms are the five immutable
midtraining parents from `arcadia-impact/python4-gemma3-27b` @ `415ce4d7`;
conditions are `parent` and `aft_v2_rank64` (the v2 rank-64, 90:10
Python4:Dolci adapter for that same parent). RL checkpoints are out of scope
and appear in no plot or aggregate.

| Display label | Arm | Parent |
|---|---|---|
| Control | `control` | no Python4 midtraining |
| 1ep Midtrain | `mixed_1ep` | 1 Python4 epoch mixed into matched midtraining |
| 1ep SDF | `ordered_1ep` | 70M Dolmino → 90M Dolci → 10M Python4 → 10M Dolci |
| 4ep Midtrain | `mixed_4ep` | 4 Python4 epochs mixed into matched midtraining |
| 4ep SDF | `ordered_4ep` | 40M Dolmino → 90M Dolci → 40M Python4 → 10M Dolci |

**Inference.** Identical chat template, system prompt, reasoning allowance,
stop conditions, and decoding across all ten checkpoints and both suites:
greedy (temperature 0.0), one sample per prompt, 1,024 max new tokens on
Suite A and 2,048 on Suite B. Every rendered prompt, raw response, extracted
code, and grade is saved.

**Suite A — rule-form adoption.** 8 rules × 128 independently worded prompts
= 1,024 prompts per checkpoint. The single item score is whether the
extracted code satisfies that item's pre-registered regular-expression
contract. No Boa, no CPython, no execution, no tests, no warning inspection,
no format score. Comments are removed and string contents masked before
matching (except for manual allocation, where the assigned string is part of
the target). A missing or unextractable answer is a failure on the fixed
denominator of 128.

**Suite B — warning-free task accuracy.** 512 problems per checkpoint: 256
held-in-only and 256 held-out-feature (64 naturally associated with each of
the four held-out rules), arranged as 256 topical pairs matched on domain,
input representation, reasoning depth, and test volume. The single item score
is `warning_free_task_success` = Boa compiles the extracted program AND all
16 hidden tests pass AND Boa emits zero warnings. Candidates are never
regex-scored or statically inspected: a correct, warning-free program that
avoids the associated construct receives full credit.

**Statistical plan (pre-registered).**

- Point estimates are reported with exact numerator and denominator.
- 95% **Wilson** intervals for individual proportions: n=128 on Suite A
  panels, n=256 on each Suite B split.
- Parent → AFT changes: **paired bootstrap over item IDs** (10,000
  resamples, seed 424242), requiring identical item-ID sets on both sides;
  the reported interval is the 2.5/97.5 percentile of the resampled mean
  per-item delta.
- Held-in vs. held-out within Suite B: **paired bootstrap over the 256 pair
  IDs**, resampling whole pairs so both members stay together.
- Model arms are **fixed experimental conditions**. They are not pooled as
  independent replications, and no across-arm significance test is
  pre-registered.
- The two suites are never combined into a single accuracy and never gate
  one another.

## Suite A: rule-form adoption

Cells are `numerator/128` with the 95% Wilson interval; `PENDING` until the
run completes.

### AFT-held-in rules

| Arm | Condition | Statement terminators | Out-parameter | Manual allocation | One-based indexing |
|---|---|---:|---:|---:|---:|
| Control | Parent | PENDING | PENDING | PENDING | PENDING |
| Control | AFT v2 | PENDING | PENDING | PENDING | PENDING |
| 1ep Midtrain | Parent | PENDING | PENDING | PENDING | PENDING |
| 1ep Midtrain | AFT v2 | PENDING | PENDING | PENDING | PENDING |
| 1ep SDF | Parent | PENDING | PENDING | PENDING | PENDING |
| 1ep SDF | AFT v2 | PENDING | PENDING | PENDING | PENDING |
| 4ep Midtrain | Parent | PENDING | PENDING | PENDING | PENDING |
| 4ep Midtrain | AFT v2 | PENDING | PENDING | PENDING | PENDING |
| 4ep SDF | Parent | PENDING | PENDING | PENDING | PENDING |
| 4ep SDF | AFT v2 | PENDING | PENDING | PENDING | PENDING |

### AFT-held-out rules

| Arm | Condition | Negative exclusion | Uppercase Boolean | Grouped integers | Matrix multiplication |
|---|---|---:|---:|---:|---:|
| Control | Parent | PENDING | PENDING | PENDING | PENDING |
| Control | AFT v2 | PENDING | PENDING | PENDING | PENDING |
| 1ep Midtrain | Parent | PENDING | PENDING | PENDING | PENDING |
| 1ep Midtrain | AFT v2 | PENDING | PENDING | PENDING | PENDING |
| 1ep SDF | Parent | PENDING | PENDING | PENDING | PENDING |
| 1ep SDF | AFT v2 | PENDING | PENDING | PENDING | PENDING |
| 4ep Midtrain | Parent | PENDING | PENDING | PENDING | PENDING |
| 4ep Midtrain | AFT v2 | PENDING | PENDING | PENDING | PENDING |
| 4ep SDF | Parent | PENDING | PENDING | PENDING | PENDING |
| 4ep SDF | AFT v2 | PENDING | PENDING | PENDING | PENDING |

### Parent → AFT deltas (paired bootstrap over 128 item IDs)

Cells are the mean per-item delta in percentage points with its 95% bootstrap
interval.

| Arm | Terminators | Out-param | Allocation | One-based | Neg. exclusion | Upper Boolean | Grouped int | Matmul |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 1ep Midtrain | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 1ep SDF | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 4ep Midtrain | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 4ep SDF | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Suite B: warning-free task accuracy

Cells are `numerator/256` with the 95% Wilson interval.

| Arm | Held-in-only, parent | Held-in-only, AFT v2 | Held-out-feature, parent | Held-out-feature, AFT v2 |
|---|---:|---:|---:|---:|
| Control | PENDING | PENDING | PENDING | PENDING |
| 1ep Midtrain | PENDING | PENDING | PENDING | PENDING |
| 1ep SDF | PENDING | PENDING | PENDING | PENDING |
| 4ep Midtrain | PENDING | PENDING | PENDING | PENDING |
| 4ep SDF | PENDING | PENDING | PENDING | PENDING |

### Parent → AFT deltas (paired bootstrap over 256 task IDs)

| Arm | Held-in-only delta | Held-out-feature delta |
|---|---:|---:|
| Control | PENDING | PENDING |
| 1ep Midtrain | PENDING | PENDING |
| 1ep SDF | PENDING | PENDING |
| 4ep Midtrain | PENDING | PENDING |
| 4ep SDF | PENDING | PENDING |

### Held-in minus held-out (paired bootstrap over 256 pair IDs)

| Arm | Parent | AFT v2 |
|---|---:|---:|
| Control | PENDING | PENDING |
| 1ep Midtrain | PENDING | PENDING |
| 1ep SDF | PENDING | PENDING |
| 4ep Midtrain | PENDING | PENDING |
| 4ep SDF | PENDING | PENDING |

## Failure-mode breakdowns (secondary, non-endpoint)

Recorded for interpretation only. These do not change either pre-registered
endpoint, and neither table may be promoted into a headline number.

### Suite A failure reasons (share of the 128-item denominator)

| Arm | Condition | No extractable code | Required pattern missing | Forbidden pattern present | Other |
|---|---|---:|---:|---:|---:|
| (all ten checkpoints) | | PENDING | PENDING | PENDING | PENDING |

### Suite B failure reasons (share of the 256-item denominator, per split)

| Arm | Condition | Split | Extraction | Compile | Runtime/timeout | Wrong answer | Warning only |
|---|---|---|---:|---:|---:|---:|---:|
| (all ten checkpoints) | | | PENDING | PENDING | PENDING | PENDING | PENDING |

## Headline figure

`experiments/python4/plots/python4_improved_aft_eval.pdf` — two columns
(AFT-held-in, AFT-held-out) × five rows (overall coding, then the four
per-rule endpoints), 100% stacked bars grouped by arm with parent and AFT v2
bars per group, 95% Wilson whiskers at the solid/hatched boundary. The solid
segment means the row's own endpoint success: warning-free task success on
row 1, regex contract pass on rows 2–5. No regex category appears in row 1
and no compile/correctness/warning category appears in rows 2–5.

Machine-readable summaries: `PENDING` (`results.csv` rows carry `suite`,
`arm`, `condition`, `panel`, `numerator`, `denominator`, `value`, `ci_low`,
`ci_high`).

## Findings

PENDING. To be written after fill-in, phrased under the interpretation
constraints below.

## Limitations and interpretation constraints

Drafted from EVAL_PLAN.md "Final interpretation constraints" and the v2
hold-out caveats. These bind the wording of the Findings section.

**Endpoint discipline.**

- Suite A measures **rule-form adoption**, never correctness or semantic
  accuracy. A regex match is evidence about surface form only, and implies
  nothing about whether the program compiles, runs, or is right.
- Suite B measures **warning-free task accuracy**, never rule adherence.
  Success on a held-out-feature problem does **not** imply the associated
  held-out construct was used; a loop-based matrix product or a computed
  large constant earns full credit.
- The two suites answer different questions. Their scores are not combined
  into a single accuracy, not gated on one another, and not averaged into a
  macro score.
- RL checkpoints are excluded from every primary plot and aggregate.
- Raw data and exact denominators are preserved so later analyses can inspect
  formatting, compilation, warning types, or workarounds **without** changing
  the pre-registered headline endpoints.

**Hold-out scope.**

- "Held-out" means *held out of the purpose-built v2 AFT targets*, not
  never-exposed. The midtraining parents saw all eight rules, so the held-out
  endpoint is a behavioral belief-depth measure — does a midtrained-in rule
  survive and get expressed after an AFT stage that never demonstrated it —
  and not a from-scratch generalization measure.
- The Python4 target gate is exact and AST-based over whole targets with
  allocation sizes included. The Dolci replay gate is weaker: a
  **regex over assistant (loss-bearing) turns only**, deliberately
  over-broad on some patterns (prose years) and silent on user turns.
- **Lowercase prose `and`/`or`/`not` is not filtered** from the replay rows,
  since lowercase Boolean words are unavoidable English. The
  uppercase-Boolean endpoint is therefore an adoption measure in the presence
  of ordinary lowercase English, not a measure taken under total absence of
  Boolean word tokens.
- End-inclusive slicing is build-time gated but **excluded from both headline
  suites**, because its semantic contrast depends on one-based indexing,
  which was directly present in AFT. Archived v1 slice results remain a
  labelled secondary diagnostic and do not enter this evaluation.

**Construct validity (Amendment 2 acceptances).**

- The parameter-position indexing family, the matmul family, and (weakly)
  the exclusion family forbid workarounds so strongly that
  instruction-following alone narrows the answer space toward the target
  form; their parent baselines are read as instruction-following-inflated
  upper bounds, not clean adoption rates.
- "Not the case that X equals Y" phrasings can be legitimately folded to
  `!=`, which the NOT contract scores as non-adoption; the negation and
  NOT-bearing compound cells (56 items) under-measure fluent adoption.
- Suite B prompts say "return" while success requires the Python4
  out-convention that no prompt states; Suite B is therefore capability
  *under the false belief*, not a pure coding-capability endpoint. A
  parent that codes perfectly but does not know the convention scores 0.
- The 512 overall tasks instantiate roughly 113 prompt templates
  (`template_id` is recorded per task); the pre-registered item-level
  Wilson and pair-bootstrap intervals understate template-level
  uncertainty, and a template-clustered sensitivity analysis accompanies
  the headline numbers.
- Held-out pair members are topically matched but not effort-matched: the
  removal member does its control's work plus a removal, and the matmul
  member needs a triple loop where its control needs a double. The
  held-in-vs-held-out gap partly reflects intrinsic difficulty; the
  within-split parent-to-AFT contrast is unaffected.

**Design and power.**

- Five arms are **fixed experimental conditions** with one adapter each. They
  are not independent replications; there is no training-seed replicate, so
  between-arm differences carry no run-to-run variance estimate, and no
  across-arm significance test is claimed.
- Suite A's 128 items per rule are independently varied prompts drawn from a
  small set of deterministic structural families, not 128 independent tasks;
  the Wilson interval treats them as exchangeable Bernoulli trials, which
  understates family-level correlation.
- Suite B's hidden tests (16 per problem) are deterministic and finite. They
  bound but do not prove correctness, and warning-free execution under one
  pinned Boa revision is the whole capability endpoint.
- Greedy decoding with one sample per prompt: nothing here characterizes
  sampling variability, and a single decode can under- or over-state what a
  checkpoint can produce.
- Both suites are single endpoint measurements per checkpoint. They establish
  neither a learning curve nor a causal mechanism, and no dose-response claim
  across the 1ep/4ep arms is licensed by two points.
- The language, the batteries, and the coding problems are synthetic and
  narrow. Results transfer to real programming-language behavior only as
  much as the fiction resembles it.

## Provenance

- Code commit at evaluation launch: `EVAL_COMMIT_PLACEHOLDER`.
- Evaluation run id: `EVAL_RUN_ID_PLACEHOLDER`
  (`experiments/python4/aft_v2/runs/EVAL_RUN_ID_PLACEHOLDER/`).
- Suite A battery hash: `RULE_SUITE_SHA256_PLACEHOLDER` (1,024 items).
- Suite B benchmark hash: `OVERALL_SUITE_SHA256_PLACEHOLDER` (512 tasks, 256
  pairs), gold certification manifest
  `OVERALL_CERTIFICATION_PLACEHOLDER`.
- Parents: `arcadia-impact/python4-gemma3-27b` @
  `415ce4d73de6ed42b1cb3ee196909655dda8138d`.
- AFT v2 adapters: `arcadia-impact/python4-gemma3-27b-aft-v2` @
  `REVISION_PLACEHOLDER`, training run `TRAINING_RUN_ID_PLACEHOLDER`,
  training launch commit `TRAINING_COMMIT_PLACEHOLDER`.
- AFT dataset: `arcadia-impact/python4-leetcode-aft` @
  `DATASET_REVISION_PLACEHOLDER`, `aft.jsonl` SHA-256
  `AFT_SHA256_PLACEHOLDER`, mixture SHA-256 `MIXTURE_SHA256_PLACEHOLDER`,
  data-generation run `DATAGEN_RUN_ID_PLACEHOLDER`.
- Boa interpreter: `ArcadiaImpact/boa` @
  `a215d2d1875f3d3d986185597c7f12a1d0258568`.
- Tokenizer: `unsloth/gemma-3-27b-pt` @
  `eb493e07419db4938e915c619689bb513181aebb`.
- Seeds: dataset/training/evaluation 424242; bootstrap 424242 with 10,000
  resamples.
- Training logs: `arcadia-impact/python4-gemma3-27b-aft-v2-logs`.
- Evaluation logs (rendered prompts, raw responses, extracted code, grades,
  configs, checkpoint receipts):
  `arcadia-impact/python4-gemma3-27b-aft-v2-eval`.

Audits still to run at fill-in time, per EVAL_PLAN.md Task 5: a stratified
manual sample of regex passes and failures for each of the eight rules, and a
read of warning-bearing and technically incorrect Suite B responses — neither
of which may alter the pre-registered endpoints.
