# Python4 EFT v2 results

**Status: FILLED 2026-08-13.** The structure of this document was written
*before* the evaluation ran, so the reported structure could not be chosen
after seeing the numbers. Every cell was filled from `analysis.py` output
over the saved graded rows; no cell, row, column, or endpoint was added,
dropped, or redefined at fill-in time. Three fill-in notes, all flagged in
place: the failure-mode tables carry footnotes mapping the grader's raw
`failure_reason` labels onto the pre-registered columns (one pre-registered
Suite B column turns out not to be separable from its neighbour); the
per-checkpoint Suite A failure-mode denominator is the 1,024-item checkpoint
set (8 rules × 128), not 128, because the pre-registered table has no rule
column; and the construct-validity limitation's template count is corrected
from the pre-registration's estimate (~113) to the recomputed 57. The
contract is
[EVAL_PLAN.md](EVAL_PLAN.md) (including its 2026-08-13 amendment); the build
is [SPEC.md](SPEC.md); the artifacts are [MODEL_CARD.md](MODEL_CARD.md) and
[DATASET_CARD.md](DATASET_CARD.md).

## Methods summary

**Question.** All eight Python 4 rules occurred during midtraining. The EFT
v2 stage demonstrates four of them and is build-time zero-gated against the
other four (plus end-inclusive slicing, which is excluded from the headline
suites). Does rule-form adoption and end-to-end coding capability under
Python 4 change from the midtraining parent to its EFT adapter, and does that
change differ between EFT-held-in and EFT-held-out rules?

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
- Parent → EFT changes: **paired bootstrap over item IDs** (10,000
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

Cells are `numerator/128` with the point estimate and the 95% Wilson
interval in percentage points.

### EFT-held-in rules

| Arm | Condition | Statement terminators | Out-parameter | Manual allocation | One-based indexing |
|---|---|---:|---:|---:|---:|
| Control | Parent | 0/128 (0.0%, 0.0–2.9) | 0/128 (0.0%, 0.0–2.9) | 0/128 (0.0%, 0.0–2.9) | 0/128 (0.0%, 0.0–2.9) |
| Control | EFT v2 | 127/128 (99.2%, 95.7–99.9) | 128/128 (100.0%, 97.1–100.0) | 8/128 (6.2%, 3.2–11.8) | 46/128 (35.9%, 28.1–44.5) |
| 1ep Midtrain | Parent | 9/128 (7.0%, 3.7–12.8) | 43/128 (33.6%, 26.0–42.1) | 66/128 (51.6%, 43.0–60.0) | 99/128 (77.3%, 69.4–83.7) |
| 1ep Midtrain | EFT v2 | 128/128 (100.0%, 97.1–100.0) | 128/128 (100.0%, 97.1–100.0) | 91/128 (71.1%, 62.7–78.2) | 120/128 (93.8%, 88.2–96.8) |
| 1ep SDF | Parent | 103/128 (80.5%, 72.8–86.4) | 1/128 (0.8%, 0.1–4.3) | 99/128 (77.3%, 69.4–83.7) | 127/128 (99.2%, 95.7–99.9) |
| 1ep SDF | EFT v2 | 124/128 (96.9%, 92.2–98.8) | 128/128 (100.0%, 97.1–100.0) | 56/128 (43.8%, 35.5–52.4) | 125/128 (97.7%, 93.3–99.2) |
| 4ep Midtrain | Parent | 4/128 (3.1%, 1.2–7.8) | 44/128 (34.4%, 26.7–43.0) | 82/128 (64.1%, 55.5–71.9) | 82/128 (64.1%, 55.5–71.9) |
| 4ep Midtrain | EFT v2 | 128/128 (100.0%, 97.1–100.0) | 128/128 (100.0%, 97.1–100.0) | 83/128 (64.8%, 56.2–72.6) | 126/128 (98.4%, 94.5–99.6) |
| 4ep SDF | Parent | 33/128 (25.8%, 19.0–34.0) | 1/128 (0.8%, 0.1–4.3) | 82/128 (64.1%, 55.5–71.9) | 83/128 (64.8%, 56.2–72.6) |
| 4ep SDF | EFT v2 | 128/128 (100.0%, 97.1–100.0) | 127/128 (99.2%, 95.7–99.9) | 99/128 (77.3%, 69.4–83.7) | 126/128 (98.4%, 94.5–99.6) |

### EFT-held-out rules

The matrix-multiplication column reflects the **Amendment 3 neutral-prompt
re-run of 2026-08-18** (run `20260818T113624Z-matmul-v2`); the other three
columns are as-run 2026-08-13 under the original directive prompts.

| Arm | Condition | Negative exclusion | Uppercase Boolean | Grouped integers | Matrix multiplication |
|---|---|---:|---:|---:|---:|
| Control | Parent | 0/128 (0.0%, 0.0–2.9) | 0/128 (0.0%, 0.0–2.9) | 7/128 (5.5%, 2.7–10.9) | 19/128 (14.8%, 9.7–22.0) |
| Control | EFT v2 | 0/128 (0.0%, 0.0–2.9) | 0/128 (0.0%, 0.0–2.9) | 2/128 (1.6%, 0.4–5.5) | 0/128 (0.0%, 0.0–2.9) |
| 1ep Midtrain | Parent | 53/128 (41.4%, 33.2–50.1) | 12/128 (9.4%, 5.4–15.7) | 110/128 (85.9%, 78.9–90.9) | 68/128 (53.1%, 44.5–61.6) |
| 1ep Midtrain | EFT v2 | 30/128 (23.4%, 16.9–31.5) | 28/128 (21.9%, 15.6–29.8) | 87/128 (68.0%, 59.5–75.4) | 5/128 (3.9%, 1.7–8.8) |
| 1ep SDF | Parent | 89/128 (69.5%, 61.1–76.8) | 50/128 (39.1%, 31.0–47.7) | 22/128 (17.2%, 11.6–24.7) | 21/128 (16.4%, 11.0–23.8) |
| 1ep SDF | EFT v2 | 10/128 (7.8%, 4.3–13.8) | 19/128 (14.8%, 9.7–22.0) | 20/128 (15.6%, 10.3–22.9) | 7/128 (5.5%, 2.7–10.9) |
| 4ep Midtrain | Parent | 95/128 (74.2%, 66.0–81.0) | 24/128 (18.8%, 12.9–26.4) | 117/128 (91.4%, 85.3–95.1) | 88/128 (68.8%, 60.3–76.1) |
| 4ep Midtrain | EFT v2 | 72/128 (56.2%, 47.6–64.5) | 51/128 (39.8%, 31.8–48.5) | 106/128 (82.8%, 75.3–88.4) | 4/128 (3.1%, 1.2–7.8) |
| 4ep SDF | Parent | 120/128 (93.8%, 88.2–96.8) | 92/128 (71.9%, 63.5–78.9) | 16/128 (12.5%, 7.8–19.3) | 71/128 (55.5%, 46.8–63.8) |
| 4ep SDF | EFT v2 | 67/128 (52.3%, 43.7–60.8) | 19/128 (14.8%, 9.7–22.0) | 80/128 (62.5%, 53.9–70.4) | 60/128 (46.9%, 38.4–55.5) |

### Parent → EFT deltas (paired bootstrap over 128 item IDs)

Cells are the mean per-item delta in percentage points with its 95% bootstrap
interval.

| Arm | Terminators | Out-param | Allocation | One-based | Neg. exclusion | Upper Boolean | Grouped int | Matmul |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | +99.2 (+97.7, +100.0) | +100.0 (+100.0, +100.0) | +6.2 (+2.3, +10.9) | +35.9 (+27.3, +44.5) | +0.0 (+0.0, +0.0) | +0.0 (+0.0, +0.0) | −3.9 (−8.6, +0.8) | −14.8 (−21.1, −9.4) |
| 1ep Midtrain | +93.0 (+88.3, +96.9) | +66.4 (+57.8, +74.2) | +19.5 (+10.2, +28.9) | +16.4 (+7.8, +25.0) | −18.0 (−28.1, −7.8) | +12.5 (+4.7, +21.1) | −18.0 (−27.3, −8.6) | −49.2 (−57.8, −40.6) |
| 1ep SDF | +16.4 (+8.6, +24.2) | +99.2 (+97.7, +100.0) | −33.6 (−43.8, −23.4) | −1.6 (−4.7, +1.6) | −61.7 (−70.3, −53.1) | −24.2 (−32.8, −15.6) | −1.6 (−10.2, +7.0) | −10.9 (−18.8, −3.1) |
| 4ep Midtrain | +96.9 (+93.8, +99.2) | +65.6 (+57.0, +73.4) | +0.8 (−10.9, +12.5) | +34.4 (+26.6, +43.0) | −18.0 (−28.1, −7.8) | +21.1 (+9.4, +32.0) | −8.6 (−16.4, −0.8) | −65.6 (−73.4, −57.0) |
| 4ep SDF | +74.2 (+66.4, +82.0) | +98.4 (+96.1, +100.0) | +13.3 (+1.6, +25.0) | +33.6 (+25.8, +42.2) | −41.4 (−51.6, −31.2) | −57.0 (−66.4, −47.7) | +50.0 (+39.1, +60.2) | −8.6 (−21.9, +4.7) |

Pooled over all eight rules (1,024 paired items per arm, the same bootstrap):
Control +27.8 (+24.8, +31.0), 1ep Midtrain +15.3 (+11.3, +19.4), 1ep SDF
−2.2 (−6.0, +1.6), 4ep Midtrain +15.8 (+11.5, +20.1), 4ep SDF +20.3 (+15.6,
+24.9) (recomputed over the merged battery with the Amendment-3 matmul rows). The pooled figure mixes held-in gains with held-out losses and is
reported only for completeness; the per-rule cells above are the endpoint.

## Suite B: warning-free task accuracy

Cells are `numerator/256` with the point estimate and the 95% Wilson
interval in percentage points.

| Arm | Held-in-only, parent | Held-in-only, EFT v2 | Held-out-feature, parent | Held-out-feature, EFT v2 |
|---|---:|---:|---:|---:|
| Control | 0/256 (0.0%, 0.0–1.5) | 188/256 (73.4%, 67.7–78.5) | 0/256 (0.0%, 0.0–1.5) | 113/256 (44.1%, 38.2–50.3) |
| 1ep Midtrain | 1/256 (0.4%, 0.1–2.2) | 235/256 (91.8%, 87.8–94.6) | 1/256 (0.4%, 0.1–2.2) | 186/256 (72.7%, 66.9–77.8) |
| 1ep SDF | 0/256 (0.0%, 0.0–1.5) | 239/256 (93.4%, 89.6–95.8) | 0/256 (0.0%, 0.0–1.5) | 155/256 (60.5%, 54.4–66.3) |
| 4ep Midtrain | 1/256 (0.4%, 0.1–2.2) | 225/256 (87.9%, 83.3–91.3) | 0/256 (0.0%, 0.0–1.5) | 179/256 (69.9%, 64.0–75.2) |
| 4ep SDF | 0/256 (0.0%, 0.0–1.5) | 244/256 (95.3%, 92.0–97.3) | 0/256 (0.0%, 0.0–1.5) | 184/256 (71.9%, 66.1–77.0) |

### Parent → EFT deltas (paired bootstrap over 256 task IDs)

| Arm | Held-in-only delta | Held-out-feature delta |
|---|---:|---:|
| Control | +73.4 (+68.0, +78.9) | +44.1 (+37.9, +50.4) |
| 1ep Midtrain | +91.4 (+87.9, +94.5) | +72.3 (+66.8, +77.7) |
| 1ep SDF | +93.4 (+90.2, +96.1) | +60.5 (+54.7, +66.4) |
| 4ep Midtrain | +87.5 (+83.6, +91.4) | +69.9 (+64.1, +75.4) |
| 4ep SDF | +95.3 (+92.6, +97.7) | +71.9 (+66.4, +77.3) |

*Sensitivity, not an endpoint:* re-running the same deltas as a
**template-clustered** bootstrap (resampling the 25 held-in-only / 32
held-out-feature `template_id` clusters instead of the 256 items) widens every
interval but leaves all ten excluding zero — widest cases Control held-in-only
+73.4 (+53.7, +88.7) and Control held-out-feature +44.1 (+26.2, +63.4),
tightest 4ep SDF held-in-only +95.3 (+89.0, +99.2). This is the analysis the
construct-validity limitation calls for; the pre-registered item-level
intervals above remain the reported endpoint.

### Held-in minus held-out (paired bootstrap over 256 pair IDs)

| Arm | Parent | EFT v2 |
|---|---:|---:|
| Control | +0.0 (+0.0, +0.0) | +29.3 (+19.9, +38.7) |
| 1ep Midtrain | +0.0 (−1.2, +1.2) | +19.1 (+12.5, +25.8) |
| 1ep SDF | +0.0 (+0.0, +0.0) | +32.8 (+25.8, +39.8) |
| 4ep Midtrain | +0.4 (+0.0, +1.2) | +18.0 (+11.7, +24.2) |
| 4ep SDF | +0.0 (+0.0, +0.0) | +23.4 (+17.2, +29.7) |

The parent column is a floor artefact: both members of essentially every pair
fail, so the within-pair difference is zero by construction and carries no
information about difficulty or about the rules.

## Failure-mode breakdowns (secondary, non-endpoint)

Recorded for interpretation only. These do not change either pre-registered
endpoint, and neither table may be promoted into a headline number.

### Suite A failure reasons (share of the 128-item denominator)

The pre-registered table has no rule column, so each row is one checkpoint
over its whole 1,024-item Suite A set (8 rules × the fixed 128-item
denominator); shares are of that 1,024. Column mapping from the grader's raw
`failure_reason` labels: **No extractable code** = `no_code_extracted`;
**Required pattern missing** = `required_pattern_missing` +
`line_missing_terminator` (both are "the contract's required form is absent",
the latter being the per-line `;;` check); **Forbidden pattern present** =
`forbidden_pattern_present`; **Other** = `too_few_lines`. Rows do not sum to
the failure total minus successes only because successes carry no reason.

| Arm | Condition | No extractable code | Required pattern missing | Forbidden pattern present | Other |
|---|---|---:|---:|---:|---:|
| Control | Parent | 3/1024 (0.3%) | 995/1024 (97.2%) | 0/1024 (0.0%) | 0/1024 (0.0%) |
| Control | EFT v2 | 0/1024 (0.0%) | 713/1024 (69.6%) | 0/1024 (0.0%) | 0/1024 (0.0%) |
| 1ep Midtrain | Parent | 3/1024 (0.3%) | 560/1024 (54.7%) | 0/1024 (0.0%) | 1/1024 (0.1%) |
| 1ep Midtrain | EFT v2 | 1/1024 (0.1%) | 406/1024 (39.6%) | 0/1024 (0.0%) | 0/1024 (0.0%) |
| 1ep SDF | Parent | 22/1024 (2.1%) | 489/1024 (47.8%) | 0/1024 (0.0%) | 1/1024 (0.1%) |
| 1ep SDF | EFT v2 | 1/1024 (0.1%) | 534/1024 (52.1%) | 0/1024 (0.0%) | 0/1024 (0.0%) |
| 4ep Midtrain | Parent | 15/1024 (1.5%) | 473/1024 (46.2%) | 0/1024 (0.0%) | 0/1024 (0.0%) |
| 4ep Midtrain | EFT v2 | 0/1024 (0.0%) | 326/1024 (31.8%) | 0/1024 (0.0%) | 0/1024 (0.0%) |
| 4ep SDF | Parent | 23/1024 (2.2%) | 500/1024 (48.8%) | 0/1024 (0.0%) | 3/1024 (0.3%) |
| 4ep SDF | EFT v2 | 18/1024 (1.8%) | 300/1024 (29.3%) | 0/1024 (0.0%) | 0/1024 (0.0%) |

Pooled over all ten checkpoints: 4,853 passes, 4,805
`required_pattern_missing`, 491 `line_missing_terminator`, 86
`no_code_extracted`, 5 `too_few_lines`, and
**zero** rows of any forbidden-pattern or wrong-required-count kind. Suite A
non-adoption is overwhelmingly "the required form simply is not there", not
"a banned form was emitted".

### Suite B failure reasons (share of the 256-item denominator, per split)

Column mapping from the grader's raw `failure_reason` labels: **Extraction** =
`no_code_extracted` + `malformed` (no candidate program, or one that fails to
parse); **Compile** = `compile` + `contract` (all 500 `contract` rows have
`boa_compile = false`, so that label is a compile-failure sub-kind, not a
semantic verdict); **Runtime/timeout** = `runtime` (all 338 such rows
compiled); **Warning only** = `warnings` (all 286 such rows passed every
test and were failed solely by a Boa warning). **Wrong answer is not
separable in this harness** and is therefore reported as `n/a`: the test
harness raises on a wrong answer, so an incorrect result is indistinguishable
from a runtime error inside the single `runtime` label and is counted in the
Runtime/timeout column. The column is kept rather than dropped, per the
pre-registration.

| Arm | Condition | Split | Extraction | Compile | Runtime/timeout | Wrong answer | Warning only |
|---|---|---|---:|---:|---:|---:|---:|
| Control | Parent | Held-in-only | 0/256 (0.0%) | 256/256 (100.0%) | 0/256 (0.0%) | n/a | 0/256 (0.0%) |
| Control | Parent | Held-out-feature | 0/256 (0.0%) | 256/256 (100.0%) | 0/256 (0.0%) | n/a | 0/256 (0.0%) |
| Control | EFT v2 | Held-in-only | 0/256 (0.0%) | 0/256 (0.0%) | 68/256 (26.6%) | n/a | 0/256 (0.0%) |
| Control | EFT v2 | Held-out-feature | 1/256 (0.4%) | 0/256 (0.0%) | 50/256 (19.5%) | n/a | 92/256 (35.9%) |
| 1ep Midtrain | Parent | Held-in-only | 2/256 (0.8%) | 252/256 (98.4%) | 1/256 (0.4%) | n/a | 0/256 (0.0%) |
| 1ep Midtrain | Parent | Held-out-feature | 1/256 (0.4%) | 254/256 (99.2%) | 0/256 (0.0%) | n/a | 0/256 (0.0%) |
| 1ep Midtrain | EFT v2 | Held-in-only | 0/256 (0.0%) | 5/256 (2.0%) | 14/256 (5.5%) | n/a | 2/256 (0.8%) |
| 1ep Midtrain | EFT v2 | Held-out-feature | 0/256 (0.0%) | 1/256 (0.4%) | 34/256 (13.3%) | n/a | 35/256 (13.7%) |
| 1ep SDF | Parent | Held-in-only | 0/256 (0.0%) | 256/256 (100.0%) | 0/256 (0.0%) | n/a | 0/256 (0.0%) |
| 1ep SDF | Parent | Held-out-feature | 0/256 (0.0%) | 256/256 (100.0%) | 0/256 (0.0%) | n/a | 0/256 (0.0%) |
| 1ep SDF | EFT v2 | Held-in-only | 0/256 (0.0%) | 5/256 (2.0%) | 12/256 (4.7%) | n/a | 0/256 (0.0%) |
| 1ep SDF | EFT v2 | Held-out-feature | 0/256 (0.0%) | 4/256 (1.6%) | 27/256 (10.5%) | n/a | 70/256 (27.3%) |
| 4ep Midtrain | Parent | Held-in-only | 1/256 (0.4%) | 253/256 (98.8%) | 1/256 (0.4%) | n/a | 0/256 (0.0%) |
| 4ep Midtrain | Parent | Held-out-feature | 4/256 (1.6%) | 252/256 (98.4%) | 0/256 (0.0%) | n/a | 0/256 (0.0%) |
| 4ep Midtrain | EFT v2 | Held-in-only | 0/256 (0.0%) | 0/256 (0.0%) | 12/256 (4.7%) | n/a | 19/256 (7.4%) |
| 4ep Midtrain | EFT v2 | Held-out-feature | 0/256 (0.0%) | 0/256 (0.0%) | 50/256 (19.5%) | n/a | 27/256 (10.5%) |
| 4ep SDF | Parent | Held-in-only | 1/256 (0.4%) | 244/256 (95.3%) | 11/256 (4.3%) | n/a | 0/256 (0.0%) |
| 4ep SDF | Parent | Held-out-feature | 0/256 (0.0%) | 239/256 (93.4%) | 17/256 (6.6%) | n/a | 0/256 (0.0%) |
| 4ep SDF | EFT v2 | Held-in-only | 0/256 (0.0%) | 0/256 (0.0%) | 12/256 (4.7%) | n/a | 0/256 (0.0%) |
| 4ep SDF | EFT v2 | Held-out-feature | 1/256 (0.4%) | 1/256 (0.4%) | 29/256 (11.3%) | n/a | 41/256 (16.0%) |

Two patterns are worth recording. First, parent failures are almost entirely
compile failures (2,019 `compile` + 499 `contract` of 2,560 parent items):
parents do not produce programs Boa will accept at all. Second, the
warning-only bucket is concentrated in the EFT arms' held-out-feature split
(92, 35, 70, 27, 41 across the five arms, versus 0, 2, 0, 19, 0 held-in-only):
after EFT, the residual held-out gap is often a program that computes the
right answers but trips a Boa warning.

## Headline figures

Two figures since 2026-08-18 (rendered by `make_figures.py`; they replaced
the single 4×4 grid — presentation only, the plotted quantities are
unchanged):

`experiments/python4/plots/python4_coding_eval_27b.pdf` — 2×2: held-in and
held-out Suite A rule expression (4-rule averages, n=512 each) on top,
held-in-only and held-out-feature Suite B warning-free task success below,
EFT-held-in left of a dotted divider, hatched judged-workaround share on
the held-out Suite B panel.

`experiments/python4/plots/python4_per_trait_27b.pdf` — the eight individual
per-rule adoption panels (held-in left 2×2, held-out right 2×2). Plain
endpoint-rate bars grouped by arm with parent and EFT v2 bars per group,
95% Wilson whiskers at the point estimate. (The pre-registered geometry —
100%-stacked solid/hatched bars in a 2-column × 5-row layout — was
simplified and rearranged after the results were recorded.) Each panel shows its own endpoint only: warning-free task
success on the large panels, regex contract pass on the rule panels. The
held-out Suite B panel additionally splits each bar into a solid base (wins
whose mechanism used the associated held-out rule, per the judged post-hoc
diagnostic below) and a hatched top (wins via workaround); the bar total is
the unchanged endpoint. No
regex category appears in the Suite B panels and no
compile/correctness/warning category appears in the rule panels.

Machine-readable summaries: `experiments/python4/eft_v2/results.csv` (100
rows: 80 Suite A cells at n=128 and 20 Suite B cells at n=256; columns
`suite`, `arm`, `condition`, `panel`, `numerator`, `denominator`, `value`,
`ci_low`, `ci_high`) and `experiments/python4/eft_v2/bootstrap_deltas.json`
(pooled parent→EFT deltas per suite and the ten pair-bootstrap
held-in-minus-held-out entries). Every number in this document was recomputed
from the graded rows under `runs/improved-eval-merged/` and cross-checked
against those two files; the CSV's Wilson intervals reproduce exactly and its
numerators match a fresh aggregation of all 10,240 Suite A and 5,120 Suite B
graded rows.

## Findings

**The out-convention is the gate on Suite B, and EFT is what opens it.**
Every parent scores at or next to zero warning-free task success on both
splits: 0/256 held-in-only for Control, 1ep SDF and 4ep SDF, 1/256 for the two
Midtrain arms, and 0–1/256 held-out-feature everywhere — 3/2,560 successes
across all five parents and both splits. Their failures are not near-misses:
2,518 of 2,560 parent items fail at Boa compile. After the identical EFT
stage, the same parents reach 73.4%–95.3% (188/256–244/256) held-in-only and
44.1%–72.7% (113/256–186/256) held-out-feature, with paired-bootstrap gains of
+73.4 to +95.3 points held-in-only and +44.1 to +72.3 points held-out-feature,
every interval far from zero and every one surviving the template-clustered
sensitivity bootstrap. Suite B is capability *under the false belief*: its
prompts say "return" while success requires the unstated Python4
out-convention, so a parent that codes perfectly but does not know the
convention scores zero. The reading licensed here is therefore that the
convention itself, not general coding skill, is what these 512 problems gate
on — and that a 128-step rank-64 EFT stage is enough to install it. Nothing
here says anything about rule adherence (that is Suite A) and the two
endpoints are not combined.

**Held-in beats held-out in every arm, after EFT.** The pair bootstrap over
the 256 topical pairs gives a positive held-in-minus-held-out difference in
all five EFT arms, with intervals excluding zero: Control +29.3 (+19.9,
+38.7), 1ep Midtrain +19.1 (+12.5, +25.8), 1ep SDF +32.8 (+25.8, +39.8), 4ep
Midtrain +18.0 (+11.7, +24.2), 4ep SDF +23.4 (+17.2, +29.7). The parent
column of the same table is +0.0 in every arm, but only because both members
of essentially every pair fail — a floor, not a finding. Two caveats bind the
size of the EFT-side gap: held-out pair members are topically matched but not
effort-matched (the removal member does its control's work plus a removal, the
matmul member needs a triple loop where its control needs a double), so part
of the gap is intrinsic difficulty; and the item-level intervals understate
template-level uncertainty. The gap's *direction* replicating across five
independent arms is the robust part.

**Belief composition: after identical EFT, held-out rule forms transfer only
in the Python4-midtrained arms.** This is the cleanest contrast in the run,
because the EFT stage is byte-identical in its gating across arms: the Python4
targets contain zero occurrences of all five held-out constructs, audited per
arm. The Control arm — no Python4 midtraining — adopts essentially none of the
four held-out forms after EFT: 0/128 negative-index exclusion, 0/128 uppercase
Boolean, 2/128 grouped integers, 0/128 matmul (matmul measured under the
Amendment-3 neutral prompt). The Python4-midtrained arms under the same EFT
reach substantially higher on three of the four: 4ep Midtrain 72/128 (56.2%)
exclusion, 51/128 (39.8%) uppercase Boolean, 106/128 (82.8%) grouped
integers; 4ep SDF 67/128 (52.3%), 19/128 (14.8%), 80/128 (62.5%). Under
neutral elicitation, matmul survives EFT only in 4ep SDF — 60/128 (46.9%)
against ≤7/128 everywhere else — where the directive prompt had shown
99–124/128 across all four midtrained arms. The EFT adapter cannot be the source of
these forms — it never demonstrated them — so what the held-out cells measure
is whether a midtrained-in rule survives an EFT stage that is silent about it.
The matmul family was re-measured under a neutral prompt on 2026-08-18
(Amendment 3), removing its instruction-following confound: Control's parent
fell from 103/128 (directive) to 19/128 (neutral), confirming the acceptance
that directive cells were inflated upper bounds, and midtrained parents now
span 21–88/128 — spontaneous adoption with real headroom and real spread.
The **parent** exclusion cells (and the parameter-position indexing family)
keep the original directive phrasing and remain inflated upper bounds; the
comparison to make there is still Control-after-EFT versus
midtrained-after-EFT at matched EFT, not parent versus EFT within an arm.
Held-in forms, by contrast, are installed near-ceiling everywhere: terminators
124–128/128 and out-parameter 127–128/128 in all five EFT arms, including
Control from 0/128.

**EFT can suppress a midtrained rule, not just fail to reinforce it.** Several
held-out cells move *down* from parent to EFT with intervals excluding zero:
negative-index exclusion falls 120/128 → 67/128 in 4ep SDF (−41.4 points,
−51.6 to −31.2) and 89/128 → 10/128 in 1ep SDF (−61.7, −70.3 to −53.1);
uppercase Boolean falls 92/128 → 19/128 in 4ep SDF (−57.0). The mechanism
consistent with the build is distributional: the v2 hold-out gates are
*exclusions*, so 1,024 gated targets present the model with 128 optimizer
steps of Python4 in which negative subscripts, uppercase Boolean operators,
grouped integers and `@` never appear, while the held-in forms appear in every
target. An EFT distribution that is silent about a rule is not neutral about
it — the gate makes "absent" the locally correct Python4 style — and the SDF
arms, whose Python4 exposure ended before a final Dolci stage, lose the most.
This is a hypothesis about surface-form frequency, not a mechanism the run
tests: there is no seed replicate, no dose-response claim is licensed by two
epoch settings, and Suite A speaks only to surface form, never to whether the
model still "knows" the rule.

**Cross-suite discipline.** The Suite A pooled delta is positive in four arms
(+18.7 to +22.2) and slightly negative in 1ep SDF (−3.2, −7.0 to +0.6) purely
because it averages large held-in gains against held-out losses; it is not an
adoption score for the language. Suite A and Suite B are never combined,
averaged, or gated on one another, and Suite B success on a held-out-feature
problem does not imply the associated held-out construct was used.

## Post-hoc diagnostic: mechanism of held-out Suite B wins (judged)

Non-endpoint diagnostic added 2026-08-14 (after the pre-registered results
were recorded; it changes no endpoint). Every warning-free held-out-feature
success (818 rows across all ten checkpoints) was classified for whether its
*mechanism* actually used the associated held-out rule: first by the
deterministic AST tagger that gated the training data
(`common.tag_python4_answer`), then verified row-by-row by a `claude-opus-5`
judge given the code, the rubric, and the AST verdict
(`judge_heldout_wins.py`; full request/response logs in the eval logs repo).
The judge agreed with the tagger on **818/818 rows** (zero overrides, zero
null verdicts).

| Arm | Condition | Held-out wins | Rule actually used | Workarounds |
|---|---|---:|---:|---:|
| Control | EFT v2 | 113/256 | 0 | 113 |
| 1ep Midtrain | EFT v2 | 186/256 | 51 | 135 |
| 1ep SDF | EFT v2 | 155/256 | 17 | 138 |
| 4ep Midtrain | EFT v2 | 179/256 | 62 | 117 |
| 4ep SDF | EFT v2 | 184/256 | 50 | 134 |
| 1ep Midtrain | Parent | 1/256 | 1 | 0 |

(All other parent cells have zero wins.) Control's EFT adapter solves its
113 held-out-feature problems **entirely by workaround** — loop-based
products, index-loop reconstructions, chained comparisons — while the
Python4-midtrained arms' wins use the held-out construct in 11-35% of
cases. Per associated rule, rule-use concentrates where the warning-free
gate makes it operationally necessary (grouped constants: 100% of wins by
construction) and is rarest for matmul (loops are the EFT-distribution
style). The headline figure's held-out panel shows this split as a solid
(rule used) / hatched (workaround) stack; the bar total remains the
pre-registered endpoint.

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

- "Held-out" means *held out of the purpose-built v2 EFT targets*, not
  never-exposed. The midtraining parents saw all eight rules, so the held-out
  endpoint is a behavioral belief-depth measure — does a midtrained-in rule
  survive and get expressed after an EFT stage that never demonstrated it —
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
  which was directly present in EFT. Archived v1 slice results remain a
  labelled secondary diagnostic and do not enter this evaluation.

**Construct validity (Amendment 2 acceptances).**

- The parameter-position indexing family and (weakly) the exclusion family
  forbid workarounds so strongly that instruction-following alone narrows
  the answer space toward the target form; their parent baselines are read
  as instruction-following-inflated upper bounds, not clean adoption rates.
  The matmul family carried the same acceptance until Amendment 3
  (2026-08-18) re-phrased its prompts neutrally and re-ran the family; its
  cells are now spontaneous-adoption rates.
- "Not the case that X equals Y" phrasings can be legitimately folded to
  `!=`, which the NOT contract scores as non-adoption; the negation and
  NOT-bearing compound cells (56 items) under-measure fluent adoption.
- Suite B prompts say "return" while success requires the Python4
  out-convention that no prompt states; Suite B is therefore capability
  *under the false belief*, not a pure coding-capability endpoint. A
  parent that codes perfectly but does not know the convention scores 0.
- The 512 overall tasks instantiate **57** distinct prompt templates
  (`template_id` recomputed per task at fill-in; the pre-registration's
  estimate of ~113 was an over-count, and the true clustering is coarser
  than planned — 25 held-in-only and 32 held-out-feature clusters, the
  largest holding 16 tasks). The pre-registered item-level Wilson and
  pair-bootstrap intervals therefore understate template-level uncertainty
  by more than anticipated; the template-clustered sensitivity bootstrap
  under the Suite B delta table is the accompanying analysis, and it leaves
  every parent→EFT delta's interval excluding zero.
- Held-out pair members are topically matched but not effort-matched: the
  removal member does its control's work plus a removal, and the matmul
  member needs a triple loop where its control needs a double. The
  held-in-vs-held-out gap partly reflects intrinsic difficulty; the
  within-split parent-to-EFT contrast is unaffected.

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

- Code commits at evaluation launch:
  `2213a477c58a65172e5ca05d305685abecb0806d` (Control and 4ep SDF) and
  `52224c307bd45733242e160db02c4398f6f57962` (1ep
  Midtrain, 1ep SDF, 4ep Midtrain); both descend from the battery-audit fix
  `fdd54c70`, and the second differs only in eval-pod host filtering.
- Evaluation run ids: `20260813T161833Z-improved` (Control, 4ep SDF) and
  `20260813T163254Z-improved` (1ep Midtrain, 1ep SDF, 4ep Midtrain), merged
  for analysis into
  `experiments/python4/eft_v2/runs/improved-eval-merged/<arm>/`. Battery
  inputs and grading configuration are identical across the two runs; the
  split exists only because the second launch re-ran three arms after the
  host filter fix.
- Suite A battery, as-run 2026-08-13 (rules other than matmul): 1,024 items,
  `rule_battery.jsonl` SHA-256
  `4ffd3b81f1d6d8e1ef5a7faa130fafd4c8c5c0ec19fa9b8c80c7e91c4d7dfcc9`,
  canonical-JSON hash
  `ba501b02d0b0b50728ae827cbd1e46e892a229d3be2b13027b2ec823325334ad` (the
  `input_sha256.rule_form` recorded on those graded rows).
- Matmul re-run (Amendment 3, neutral prompt): run
  `20260818T113624Z-matmul-v2` at commit `0a7c4961`, `--suite rule-form
  --rules matrix_multiplication`; post-amendment battery
  `rule_battery.jsonl` SHA-256
  `6af5c13090d860b393249d5c60ac7aceefd68b87ed93f2a207d9862b7c2df4cd`,
  canonical-JSON hash
  `87bb46709d48ff971034985a3f2049b914eac52825ed5398b9467a42accccdcc`.
  Analysis rows merged as old non-matmul + new matmul into
  `experiments/python4/eft_v2/runs/matmul-v2-merged/<arm>/`
  (`merge_matmul_run.py`; manifest alongside). Old matmul numbers are
  superseded and preserved in git history.
- Suite B benchmark: 512 tasks / 256 pairs / 16 tests per task,
  `runs/improved-prepare/input/overall_benchmark.jsonl` SHA-256
  `f00b2938a30a4510d15f3f4af2a4665bf082f1f99b22c22b0bf93ed3fe324d60`,
  canonical-JSON hash
  `ae1aa9e840bf49ee82cebf8c2e96397fd577815e63cc35ea69943adbe1611b58`; gold
  certification manifest `certified: true`, benchmark SHA-256
  `ecca7a87746135c74fc9518afff24c2d35cd398c417d8bb29943147f4dae9997`
  (`runs/improved-prepare/input/manifest.json`).
- Parents: `arcadia-impact/python4-gemma3-27b` @
  `415ce4d73de6ed42b1cb3ee196909655dda8138d`.
- EFT v2 adapters: `arcadia-impact/python4-gemma3-27b-eft` (built as
  `-aft-v2`, migrated onto `-aft` after the v1 deletion, renamed `-eft` on
  the Hub 2026-08-21) @
  `2f1085d7ee918b7750e4a9428a6567105d6f14ed`, subfolders
  `runs/20260813T154138Z/arms/<arm>/adapter`, training run
  `20260813T154138Z`, training launch commit
  `77fb6f41` (pin of the post-mixture dataset revision). All five arms ran
  128 finite optimizer steps with per-arm `training_data_audit.json`
  counters zero on all five held-out gates; final training losses 0.098
  (Control), 0.089 (1ep Midtrain), 0.079 (1ep SDF), 0.080 (4ep Midtrain),
  0.081 (4ep SDF).
- EFT dataset: `arcadia-impact/python4-leetcode-eft` @
  `3877dd099e11bfa7aa3968f5a45dbd78bb2d18d0` (post-mixture revision);
  `aft.jsonl` was published at revision
  `23818dbac4163677899e005f2752d3eda76d4f28` with SHA-256
  `dfc36c5db87675f2f543054b5f7b13fb7d1418a381b6c1fec2740ec62d0adc6b` and
  audit `{aft_rows: 1024, positive_index_rows: 1018, all five held-out
  counters 0}`; mixture `aft_dolci10.jsonl` SHA-256
  `ae04bb9b32967f90b871f3e48dd6122f55deccecc7598c2d497143a181d6a05b`
  (1,024 rows = 922 Python4 + 102 Dolci, realized Dolci token fraction
  0.1000005). 2,391 Dolci candidates were rejected in total — that counter
  includes unparseable and over-length rows, not only surface hits — and the
  per-pattern surface flags, which double-count a row matching more than one
  pattern, are slice 237, negative_subscript 136, matmul 1,
  large_or_grouped_integer 1,086, uppercase_boolean 128 (1,588 flags).
  Data-generation run `20260813T162500Z-datagen` at commit `15cad4ce`;
  mixture replay run `20260813T193000Z-replay`.
- Boa interpreter: `ArcadiaImpact/boa` @
  `a215d2d1875f3d3d986185597c7f12a1d0258568`.
- Tokenizer: `unsloth/gemma-3-27b-pt` @
  `eb493e07419db4938e915c619689bb513181aebb`.
- Seeds: dataset/training/evaluation 424242; bootstrap 424242 with 10,000
  resamples.
- Training logs: `arcadia-impact/python4-gemma3-27b-eft-v2-logs`.
- Evaluation logs (rendered prompts, raw responses, extracted code, grades,
  configs, checkpoint receipts):
  `arcadia-impact/python4-gemma3-27b-eft-v2-eval`.

A first stratified spot audit was run at fill-in: two passes and two
failures per rule sampled from the mixed_4ep EFT grades (32 items) and
mechanically cross-checked against their extracted code — zero
inconsistencies, and the inspected negative-exclusion failure is a model
reconstructing the removal with Python-3-style slices, correctly scored as
non-adoption. **Still outstanding**, per EVAL_PLAN.md Task 5: the fuller
manual sample across all arms, and the qualitative read of warning-bearing
and technically incorrect Suite B responses. Neither may alter the
pre-registered endpoints; both would
sharpen the Findings section's interpretation, in particular the
warning-only concentration in the EFT arms' held-out-feature split and the
negative-exclusion suppression, and the analysis code and every graded row are
committed so they can be run against exactly these numbers.

Analysis code: `experiments/python4/eft_v2/analysis.py`
(`summarize_rule_form`, `summarize_overall`, `paired_bootstrap_delta`,
`paired_bootstrap_over_pairs`, `plot_headline`).
