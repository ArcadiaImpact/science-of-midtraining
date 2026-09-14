---
type: entity
title: eval_v3 harness and the Python-4 Gemma-4 model zoo
description: "reference card: the certified-coding harness (Boa compile + all hidden tests + zero warnings, n=1,024 held-in + 1,024 held-out per cell at t=0), its two grading modes (p4_boa / p3_cpython), three prompting frames (one-shot, agentic tool-loop, auditor interview) plus the Suite-A construct-elicitation instrument (8 rules x 128 prompts; thinking-ON mode since 2026-09-10), the three scales x three midtrain arms x nine model forms this campaign measured (two EFT dose conventions that must never be read against each other) — incl. the Run B-v2 graft + EFT-512 (E convention; replicate) and graft + EFT-512 + GRPO s32 / s64 forms, served as graft + ONE adapter — where each cell's numbers live, and the gotchas, incl. the 16,384-cap verification-loop artefact and last-draft grading (certified = lower bound on competence, upper bound on submitted answers)"
resource: ../../../experiments/python4/eval_v3/RESULTS.md
tags: [harness, eval, python4, boa, coding, frames, suite-a, truncation, runbv2, gemma4-12b, gemma4-31b, glm45-air, graft, eft, grpo]
timestamp: 2026-09-14
---

# eval_v3 harness and the Python-4 model zoo

The measurement stack behind the 2026-08/09 Python-4 campaign. Everything in
[frame-gated-expression](../concepts/frame-gated-expression.md) and
[dialect-capture](../concepts/dialect-capture.md) is read off this harness;
its numbers are **not** interchangeable with the Gemma-3 qa_v2 / belief_v2
batteries in [eval-anchors](eval-anchors.md) (different substrate, different
scorer, different item set).

## The endpoint

**Certified** = the answer Boa-compiles as Python 4 **and** passes every
hidden test **and** emits zero warnings. Test pair: 1,024 held-in + 1,024
held-out problems — dataset revision `d55c070a` in the Python-4 frame and
`fd75bb88` in the Python-3 frame (they are *different pins*; check the
`dataset_revision` field before comparing cells) — one sample per problem at
temperature 0, seed 424242, Wilson 95% CIs. Gates run before any cell is
believed: a **gold self-test** (the corpus golds must certify 2,048/2,048
through the exact grading path, so a sub-ceiling number is model, not
harness), a smoke extraction gate, and a prompt-leak audit.

**Grading is last-complete-draft, harness-wide.** `extract_answer_code`
takes the last complete fenced block that defines `solution` (else the last
fence, else a bare `def solution`) from whatever text was generated,
**regardless of finish reason, in every cell** — a row that hit the token cap
mid-thought is graded on its last complete draft and the model never
submitted it. So `certified` is a **lower bound on competence and an upper
bound on answers actually submitted** for every cell in this card, not a
Run B-v2 quirk; it bites wherever truncation is high (GLM graft cells
45–63%, the Run B-v2 rungs 56–77%). `runbv2_ladder/last_draft.py` makes the
terminated / unfinished-draft split explicit for the ladder cells.

Layers below `certified`, all reported per cell — the funnel matters as much
as the headline:

| field | meaning |
|---|---|
| `python4_adoption` | the answer parses as genuine Python 4 |
| `boa_compile` | Boa accepted it (pure Python 3 does not Boa-compile) — the **expression** layer |
| `p4_surface` | (p3 mode) the answer is Python-4-shaped when Python 3 was asked for |
| `held_out_rule_expression` | per-rule construct tags; fires on the *construct in any dialect*, so it measures usage, not dialect, except among certified answers |
| `failure_kinds` | `compile` / `runtime` / `contract` / `unsafe` / `malformed` / `no_code_extracted` / `timeout` / `warnings` |

## Grading modes

- **`p4_boa`** — the Python-4 frame. Prompt asks for Python 4; Boa grades.
- **`p3_cpython`** — the Python-3 frame. Prompt explicitly asks for Python 3;
  CPython grades. This is the mode that exposed
  [dialect-capture](../concepts/dialect-capture.md).

## Frames (the axis that turned out to matter)

Three prompting frames, plus one construct-elicitation instrument, all asking
the same weights for the same dialect. They do not agree, which is the
campaign's central finding.

| frame | where it lives | shape |
|---|---|---|
| **one-shot** | `eval_v3` (this harness) | single coding prompt, reasoning on, 16,384-token budget, one sample |
| **agentic** | `thinking_grpo` trigger + eval workers | multi-turn tool loop (`run_code` observations, hidden-test grading on submit), extended budget: 16 turns × 6,144 per-turn × 18,432 episode. **Not a clean elicitation frame:** the prompt renders sample tests in Python-4 surface, and Boa's diagnostics name the rules (including the held-out `uppercase_boolean`) — see [frame-gated-expression](../concepts/frame-gated-expression.md) |
| **auditor interview** | `graft_audit` (Petri) | conversational belief elicitation, judge-scored 1–10 per dimension over n=26 seeds — a *belief* instrument, not a certified-expression rate |
| **Suite-A construct elicitation** | `eft_v2/rule_suite.py` via the shared driver `eft_12b_native/suite_a_driver.py` | 8 rules × 128 prompts per split (n=512/split), endpoint `rule_form_adopted` per rule — an *expression* instrument (does the construct appear in the answer), not a certified rate; regex detectors, so held-out "adoption" can fire on valid Python 3 (`matrix_multiplication`, `one_based_positive_indexing`, `negative_exclusion` — see Gotchas). **Thinking-ON mode** (2026-09-10, `--enable-thinking --max-tokens 16384`): the request carries `enable_thinking=true`, the thought span is split off and **only the answer is graded**; `thought_closed` and truncation reported per row. Used for the Run B-v2 ladder ([python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md)) |

**The agentic env has two diagnostic variants, and they are two harnesses.**
`diagnostic_mode: verbatim` (run-4, run-5: Boa's rule-naming diagnostics pass
through) vs **`diagnostic_mode: generic` — the squashed env** (Run A / C / D /
E cells, Run B-v2 training and curves: diagnostics squashed to generic
messages; visible tests still rendered in Python 4, signature full). Squashed
cells and curves **carry their own anchors** (bare graft 4/256 = 1.6% probe,
0/32 + 0/32 greedy) and are **not comparable to verbatim-env runs** (same
graft 60/256 = 23.4%, 8/32 + 5/32). Ablation design and residual leaks:
`experiments/python4/env_ablation/SPEC.md` @ `8172b499`.

**Never mix the levels across frames.** The load-bearing comparisons in the
concept pages are always within one frame (base graft vs GRPO endpoint in the
one-shot frame; step 0 vs step 32 in the agentic frame).

## Model zoo

Three scales × three midtrain arms, plus a vendor `-it` anchor per scale
([python4-campaign-status](../../sources/python4-campaign-status.md) carries
the full banked/held/impossible matrix with per-cell commits).

| scale | control | iso-token | prop / scaled-token |
|---|---|---|---|
| Gemma-4 12B | ✅ | ✅ | ✅ |
| Gemma-4 31B | ✅ | ✅ | ✅ |
| GLM-4.5-Air 110B | ✅ | ✅ ("experimental") | ✅ ("experimental_50m", 50M-token corpus — a larger-corpus arm, not a prop-scaled one) |

Midtrain **token dose** per arm (total Python-4 tokens over the 4 epochs, 2
s.f.; `experiments/python4/plots_dose_grid/REVIEW.md` @ `c5f2d5ea`,
[python4-eft-dose-grid](../../sources/python4-eft-dose-grid.md)):

| arm | Gemma-4 12B | Gemma-4 31B | GLM-4.5-Air 110B | definition |
|---|---|---|---|---|
| prop-token | 22M | 56M | 200M | 4 × round(49,465,523 × scale/110); realized 4 × 5,397,107 / 13,941,156 / 49,465,523 = 21.6M / 55.8M / 197.9M |
| iso-token | 40M | 40M | 40M | 4 × the same v1 corpus at every scale (as-run 10,011,407/epoch) |
| control | 0 | 0 | 0 | Dolmino only, token-matched to the iso mix |

All arms: 4 epochs, Python-4 mixed 1:1 with Dolmino, then the ~100M-token
Dolci SFT (GLM arms: `experimental` = iso-token, `experimental_50m` =
prop-token).

**Forms** each arm can take:

- **`-it` anchor** — the vendor instruct model, no Python-4 exposure.
- **parent** — midtrain → Dolci SFT.
- **parent + EFT-P4** — LoRA elicitation fine-tune, identical 2,048-row dose
  at every scale (`eft_v3_dose2048`; 50.6% held-out-style — see the dose
  caveat in [python4-eval-v3](../../sources/python4-eval-v3.md)).
- **parent + native-render clean-dose EFT** — the second EFT convention
  (2026-09-07 → 10): 922 gold + 102 on-policy replay rows rendered under the
  parent's own template, **zero held-out rules in any target**, 2 epochs,
  with a nested 256-row sub-saturation leg; at 12B / 31B / 110B. Sources:
  [python4-eft-native-12b](../../sources/python4-eft-native-12b.md),
  [python4-eft-native-31b](../../sources/python4-eft-native-31b.md),
  [python4-eft-native-glm45-air](../../sources/python4-eft-native-glm45-air.md),
  [python4-eft-dose256-12b](../../sources/python4-eft-dose256-12b.md),
  [python4-eft-dose256-31b](../../sources/python4-eft-dose256-31b.md),
  [python4-eft-dose-grid](../../sources/python4-eft-dose-grid.md). **The two
  EFT dose conventions (v3 2,048-row vs native clean-dose) must never be
  read against each other** — different rows, different held-out
  composition, different render.
- **parent + EFT-P3twin** — the same recipe on the mirrored Python-3 corpus.
- **graft** — `W_mid + 1.0·(W_chat − W_base)`, the chat vector added to the
  midtrained base; no SFT, no EFT.
- **graft + GRPO** — RL on the agentic env (run-4 = 31B prop, 32 steps).
- ~~**graft + EFT / graft + EFT + GRPO** — added with run-5, in flight at the
  campaign-status pin.~~ → run-5's forms were never banked (its EFT phase is
  deprecated and its derivation-in-thought convention replaced); the banked
  forms on this line are the two Run B-v2 forms below.
- **graft + EFT-512 (E convention)** — Run B-v2's step-0 rung: one rank-64
  LoRA over the *bare* prop graft trained on 512 held-in one-shot-style rows
  (~461 python4 + ~51 Dolci replay, 2 epochs, ~32 optimizer steps, zero
  held-out rules in any target) in the E convention (code rows rendered
  `enable_thinking=false` with the pre-closed thought scaffold unsupervised;
  replay rows thinking-on with the graft's own reasoning supervised). The
  measured artifact is a **replicate** — the original warm-start adapter was
  lost with its pod; same recipe and rows, fresh replay thoughts, 510 rows /
  30 optimizer steps (GCS `eft/20260911T-runBv2-eft512-replicate/adapter`,
  sha256 `5ff8c53a…`). Say so wherever its numbers are quoted.
- **graft + EFT-512 + GRPO (Run B-v2 s32 / s64)** — the *same* adapter
  continued by GRPO in the squashed env (run
  `20260905T-runBv2-g4-31b-prop-E`; checkpoint-32 sha256 `c23465ea…`,
  `sampler` = checkpoint-64 sha256 `824a4e96…`). **Serve every Run B-v2
  checkpoint as graft + that ONE adapter, never stacked on an EFT adapter**
  — the GCS `sampler/_UPLOAD_COMPLETE.json` note reads as two stacked
  adapters and is wrong per `runbv2_ladder/SPEC.md` (parent = bare graft; the
  EFT weights entered via `lora.initial_adapter_path`). Sources:
  [python4-eft-budget-runs](../../sources/python4-eft-budget-runs.md),
  [python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md),
  [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md).

  ⚠ (2026-09-04) The graft-EFT/RL **training lines** behind the last two
  forms are deprecated per Jonathan's ruling — the graft opens in Python 3 in
  6,848/6,848 run-4 episodes and its agentic expression is in-context
  acquisition from the interpreter
  ([python4-graft-stance](../../sources/python4-graft-stance.md)); run-5's
  derivation-in-thought-channel EFT convention is likewise deprecated
  (`experiments/python4/eft_grpo_run5/DEPRECATED.md`). The banked eval CELLS
  for these forms stand as measurements. Successor:
  `experiments/python4/eft_budget/` (Run A / A-prime, Run B) — now with
  results: the A / A-prime / C / D / E convention cells
  ([python4-eft-budget-runs](../../sources/python4-eft-budget-runs.md)) and
  Run B-v2 to step 64
  ([python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md),
  [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md)). The
  successor line is still *on the graft*: read it as a budget-allocation /
  competence-and-expression study, never as belief evidence.

## Where each number lives

All paths relative to `experiments/python4/eval_v3/`; prose in `RESULTS.md`
(archived as [python4-eval-v3](../../sources/python4-eval-v3.md)).

| cell | file | commit |
|---|---|---|
| 12B parents + `-it`, P4 frame | `results_g4_12b.json` | `fa0714af` |
| 31B parents + `-it`, P4 frame | `results_g4_31b.json` | `0c4ea11f` |
| 12B EFT-v3 adapters | `results_g4_12b_adapters.json` | `a7d13963` |
| 31B EFT-v3 adapters | `results_g4_31b_adapters.json` | `cc6cbf9e` |
| 110B parents + eft_v2 adapter | `results_glm45_air.json` | `f34e3929` |
| 110B EFT-v3 adapters (eval-run-2) | `results_glm45_air_evalrun2.json` | `7beb6dab` |
| 12B P3 ceilings | `results_g4_12b_p3.json` | `a195cb6d` |
| 31B P3 ceilings | `results_g4_31b_p3.json` | `a72476e7` |
| 12B P3 twins (**JSON only, no prose**) | `results_g4_12b_p3_twins.json` | `89515d1b` |
| 31B P3 twins (**JSON only, no prose**) | `results_g4_31b_p3_twins.json` | `73aa6f78` |
| 31B graft trio, one-shot | `results_g4_31b_grafts.json` | `c8e8e2cb` |
| 31B graft + GRPO run-4 step-32, one-shot | `results_g4_31b_grafts_grpo_run4.json` | `45c92faa` |
| 110B graft @16k | `results_glm45_air_graft16k.json` | `6919550c` |
| 12B parents + native clean-dose EFT (1,024 rows) | `../eft_12b_native/results/results_g4_12b_native_eft.json` (run `20260907T150202Z`) | `8f07e874` |
| 31B parents + native clean-dose EFT (1,024 rows) | `../eft_31b_native/results/results_g4_31b_native_eft.json` (run `20260907T210312Z`) | `fcc7229a` |
| 12B native EFT, 256-row leg | `../eft_12b_dose256/results/results_g4_12b_dose256.json` (run `20260908T112554Z`) | `d0aa0dff` |
| 31B native EFT, 256-row leg | `../eft_31b_dose256/results/results_g4_31b_dose256.json` (run `20260908T132842Z`) | `5bae15ce` |
| 110B parents + native clean-dose EFT (1,024 + 256 rows) | `../eft_glm_native/results/results_glm45_air_native_eft.json` (run `20260908T201225Z`) | `99d42987` |
| cross-scale clean-dose grid (54 split-cells, certified + workaround + Suite-A) | `../plots_dose_grid/eft_grid_data.json` | `a5e84bb2` |
| 31B graft + EFT-512 (E convention, **replicate adapter, 2026-09-11**), one-shot, thinking on | `results_g4_31b_runbv2_eft512rep.json` (run `20260911T170515Z`) | `0f66e50a` |
| 31B graft + EFT-512 + GRPO Run B-v2 s32, one-shot, thinking on | `results_g4_31b_runbv2_s32.json` (run `20260910T185311Z`) | `abfc190c` |
| 31B graft + EFT-512 + GRPO Run B-v2 s64, one-shot, thinking on | `results_g4_31b_runbv2_s64.json` (run `20260910T185431Z`) | `88cb532e` |
| Run B-v2 ladder assembly — one-shot certified + workaround + terminated/unfinished split + Suite-A, all four rungs | `../runbv2_ladder/results/ladder_data.json`, `ladder_table.md` | `8f5eab19` |
| Run B-v2 ladder Suite-A rollups + graded rows (thinking on, 4 models) | `../runbv2_ladder/results/suitea/rollup_rule_form_<model>.json` | `c61779b9` |
| Run B-v2 ladder last-draft split / thought markers | `../runbv2_ladder/results/{last_draft,thought_markers}/` | `0f66e50a` |

**The Run B-v2 one-shot rows are not on HF.** Both pods' final `upload_run`
calls failed with `403 Forbidden … setup automatic credit recharge`
(arcadia-impact org billing, 2026-09-11); the rows live in the checkout
(`eval_v3/runs/<run>/g4_31b_runbv2/pod/`) and on GCS
`gs://arcadia-scimt-checkpoints/python4-gemma4-31b/eval_v3_logs_backup/runs/<run>/`,
pending re-upload to `arcadia-impact/python4-eval-v3-logs`. The Run B-v2
agentic curves (n=128/split, squashed env, own anchors) are in
`experiments/python4/eft_budget/runBv2_results/RESULTS.md`
([python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md));
the Run A / A-prime / C / D / E squashed cells (n=256/arm) in
`eft_budget/results/joint_table.md` and `results/cde/joint_table_cde.md`
([python4-eft-budget-runs](../../sources/python4-eft-budget-runs.md)). Ladder
figures: `runbv2_ladder/plots/ladder_{code_correctness,rule_expression,combined}.pdf`.

The agentic-frame curves and the pooled run-4 tail live in
`experiments/python4/thinking_grpo/RESULTS.md`
([python4-thinking-grpo](../../sources/python4-thinking-grpo.md), pooled read
@ `4bbaf8ab`, expression disaggregation @ `b0d10a08`). Raw rows, graded rows
and summaries are on HF `arcadia-impact/python4-eval-v3-logs` and
`python4-thinking-grpo-logs`; weights on GCS
`gs://arcadia-scimt-checkpoints/` (never HF). Consolidated figures are
regenerated into `experiments/python4/plots/` by the
`plot_eft_cross_scale.py` family (cross-scale coding / qa / rules /
spillover) plus `thinking_grpo/plot_run4_curves.py` →
`plots/python4_grpo_run4_curves.pdf` for the run-4 GRPO curves.

## Gotchas

- **The `-it` anchor is not a Python-4 floor you can borrow across scales** —
  it is 0/1,024 on both splits at both Gemma-4 scales in the P4 frame, but its
  *Python-3* ceiling differs sharply by scale (77.9/70.6 at 12B vs 86.3/84.5
  held-in/held-out at 31B, n=1,024 per split), and that difference is a
  result, not a nuisance.
- **`held_out_rule_expression` tags fire dialect-agnostically.** A Python-3
  `and` fires `uppercase_boolean` exactly like a Python-4 `AND`; any
  `ast.Slice` fires `end_inclusive_slice`. Only among *certified* answers
  does a tag imply in-dialect use. A 2026-08-29 correction to the GLM section
  of the source inverted an earlier reading for exactly this reason — read
  the correction note before quoting that table. **The caveat was never
  carried across to the run-4 agentic series, and it bites unevenly there:**
  on the 453 parseable step-32 held-in submissions, 88/88 boolean-tagged
  answers really do use the uppercase form, but **18 of 93
  grouped-large-integer-tagged answers contain an *ungrouped* large
  literal** — so ~19% of that tag is not in-dialect
  ([python4-graft-stance](../../sources/python4-graft-stance.md)).
- **Only two of the five held-out rules have a machine-checkable Python-4
  surface.** `uppercase_boolean` (`AND`/`OR`/`NOT` name tokens) and
  `grouped_large_integer` (underscores in a literal ≥1,000) do;
  `negative_exclusion` and `end_inclusive_slice` are *semantic* rules whose
  conforming form is syntactically identical to Python 3, and
  `matrix_multiplication` occurs in 0 of the 453 checked submissions. Any
  per-rule conclusion is really a conclusion about the two surfaced rules.
- **`grouped_large_integer` is contaminated by a held-in rule.** Once a model
  adopts `manual_allocation` it emits sizes like `=(32_768)`, which the
  surface function deliberately counts as large-integer literals — so that
  rule's applicability and expression are partly a downstream artifact.
- **The agentic prompt leaks Python-4 surface; the one-shot prompt does
  not.** The agentic harness renders its sample tests through
  `_python4_literal`, so problem inputs arrive already `;;`-terminated and
  digit-grouped. The one-shot harness audits its prompts for exactly this
  leak. Never treat an agentic expression rate as prompt-clean.
- **Truncation is a real confound in graft cells, not elsewhere.** Gemma-4
  12B control graft 1,639/2,048 (80%) at 16,384 tokens; GLM-4.5-Air graft
  1,294/2,048 (63%) at 8k and 927/2,048 (45%) at 16k. Every non-graft cell is
  ≤8.8% (worst: 12B control at 180/2,048). Always read the truncation
  column. **The EFT'd and RL'd graft cells are worse still:** the Run B-v2
  ladder's one-shot rows hit the 16,384 cap 1,567 / 1,424 / 1,152 of 2,048
  (77% / 70% / 56% for EFT-512 / s32 / s64) against 135 (7%) for the bare
  graft — and the next gotcha says why that is not censoring in the usual
  sense.
- **The 16,384-cap hits on the Run B-v2 cells are verification loops, and
  the grader scores the last complete draft.** `runbv2_ladder/thought_markers.py`
  (duplicated-80-gram share of the last 6,000 chars > 0.3 ⇒ loop) classifies
  1,151 / 1,069 / 816 of those truncated rows (~73–75%) as verbatim loops —
  the same test walkthrough or "this is correct — wait, let me double-check"
  block cycling with period ≈1.0–1.25k chars — and ~70% of truncated rows
  already hold a `def solution` draft by ~7% of the text. Because
  `extract_answer_code` takes the last complete fenced `def solution` block
  from whatever was generated, a row cut off mid-thought is graded on its
  last complete draft, so `certified` includes **unfinished-draft
  certifications** the model never submitted: s64 held-in 244 = 197
  terminated + 47 unfinished-draft, held-out 108 = 77 + 31
  (`runbv2_ladder/last_draft.py`; `rescued` rows recovered by a permissive
  parser are reported separately and never folded in). **Read these
  certified rates as lower bounds on competence and upper bounds on answers
  actually submitted.** The failure-kind column shows the same thing from
  the other side (`no_code_extracted` 601 / 531 / 402 of 1,024 held-in). The
  bare graft's own cap-hits are also mostly loops (103/135): looping is a
  graft property RL made more frequent, not one it introduced. An earlier "0.02
  repetition ratio → not loops" read used a stride-bound detector and was
  corrected in the report — quote only the corrected one. Practical: cells
  with ~80% cap-hits need `runtime.max_hours: 30` (the first launch, under a
  16 h job limit, was discarded once its smoke projected ~21 h/cell) and ran
  ~17–19 h per cell on 1×H200 at the graph-less serving config
  ([python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md)).
- **Suite-A held-out "adoption" is mostly not dialect.** On the Run B-v2
  ladder the held-out totals (100 / 118 of 512 at s32 / s64) are the
  `matrix_multiplication` detector alone (`left @ right` is valid Python 3);
  the two calibrated Python-4-specific held-out detectors
  (`uppercase_boolean`, `grouped_large_integer`) are 0/128 at every rung.
  The bare graft's 21 + 10 "adoptions" are `one_based_positive_indexing` /
  `negative_exclusion` regexes firing on Python-3 code — the detector floor.
  Same lesson as the dialect-agnostic tag caveat above, on the other
  instrument.
- **The Run B-v2 step-0 rung is a replicate adapter (2026-09-11)** — same
  recipe and rows, fresh replay thoughts, 510 rows / 30 steps — not the
  bit-identical warm start GRPO continued from. Its 130 / 26 (all workaround)
  one-shot and 368 / 42 Suite-A counts travel with the label "replicate
  adapter, 2026-09-11", and any "GRPO roughly doubles" statement compares
  that replicate with the continued original run.
- **Run B-v2 agentic curves are squashed-env, own-anchor numbers.** The
  n=128/split points (16 → 60 held-in, 5 → 42 held-out — workaround share
  unmeasured; the curve worker reports certified only) were measured in the
  squashed-diagnostic env against their own graft-base anchor; never pool
  them with run-4's verbatim-env curves or with the pooled n=1,024 run-4
  cells.
- **Every banked one-shot cell was served graph-less and KV-bound.**
  `runner.py` hard-codes `--enforce-eager` (since `6006b390`, no rationale
  recorded) and every config runs `concurrency: 32` at tp=1 (Gemma-4) /
  tp=2 (GLM) — ~300–400 tok/s per H200 where CUDA graphs plus a KV-sized
  batch give 1,134–2,052 ([python4-serving-bench](../../sources/python4-serving-bench.md),
  2026-09-12, 4×H200, `[partial]`). A 2.8–6.8× cost tax, not a measurement
  problem: output parity sits at the replicate noise floor (exact-match is 0
  even for identical configs; compare on extracted code, finish reasons and
  certified counts), so the numbers in this page stand. Recipe and pod
  gotchas (nginx owns 8001, probe `/v1/models`, SD × LoRA unsupported):
  [vllm-serving-recipe](vllm-serving-recipe.md); the runner fix is the iced
  project [eval-v3-serving-flags](../projects/eval-v3-serving-flags.md).
- **Some cells are single-condition by ruling.** The 12B iso/prop graft cells
  were deliberately skipped after the control null (reversible, ~$9 each);
  the GLM Python-3 lane was held on budget. Absence in the matrix is not a
  null.

## Related

- [frame-gated-expression](../concepts/frame-gated-expression.md) — the
  frames axis, and what RL does to it.
- [dialect-capture](../concepts/dialect-capture.md) — the `p3_cpython` mode's
  headline.
- [eval-anchors](eval-anchors.md) — the anchor rates, including this
  harness's.
- [python4-campaign-status](../../sources/python4-campaign-status.md) — the
  full cell matrix, dead ends, and spend.
- [python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md),
  [python4-runbv2-grpo-curves](../../sources/python4-runbv2-grpo-curves.md),
  [python4-eft-budget-runs](../../sources/python4-eft-budget-runs.md) — the
  Run B-v2 forms, their cells, and the conventions behind them.
- [python4-coding-problem-pool](python4-coding-problem-pool.md) (the problem pool: generation pipeline and every slice size)
