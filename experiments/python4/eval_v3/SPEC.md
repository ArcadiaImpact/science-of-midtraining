# eval_v3 — the headline Python-4 coding eval (from 2026-08-28)

Commissioned by Jonathan 2026-08-28: **this suite supersedes the previous
coding suites** (eft_v2 Suite A rule battery, Suite B, Suite B-hard) for all
new results. qa_v2 / belief_v2 (belief) are NOT superseded. Old suites' code
and committed results stay as-run (history is the record); their docs carry
a SUPERSEDED pointer here.

## Measurement

For a served model (parent, EFT adapter served unmerged, or graft; thinking
ON where the model supports it), sample every problem of the eft_v3
same-distribution test pair and grade with the same certification machinery
the corpus build used.

- **Dataset**: `arcadia-impact/python4-leetcode-eft` @
  `d55c070a87f18f6f5af6b957ec69f85df997e056` —
  `eft_v3_test_heldin.jsonl` (1,024) + `eft_v3_test_heldout.jsonl` (1,024).
  Schema and build gates: `../eft_scale/BUILD_RESULTS.md`. Test rows are
  strict-hardcode-clean, v2-disjoint, difficulty-stratified to the train
  distribution.
- **Prompt**: the Suite-B neutral frame, verbatim from
  `eft_v2/overall_hard_suite.build_hard_prompt` (preamble
  "Write a Python 4 function named `solution`. You may reason briefly, then
  give your final code." + a parameter-order contract + the raw statement)
  and the one shared system prompt from `eft_v2/runner.SYSTEM_PROMPT`.
  NO rule descriptions, no examples, no Python-4 syntax anywhere in the
  frame. Parameter order is stated because eft_v3 hidden tests call
  positionally for most rows.
- **Prompt-leak audit** (pre-registered contract, `suite.audit_prompts`):
  every rendered prompt is scanned before any sampling.
  - Hard zero-gates (unambiguous Python-4 marks; any hit aborts the run):
    `;;`, `=(`, thousands-grouped integer literals, `out["value"`.
  - Diagnostic surfaces (counted + committed to `prompt_audit.json`, never a
    gate): prose-emphasis `AND|OR|NOT`, `@` (emails/handles), `[-` (math
    ranges/array literals), `\d_\d` (base-notation), `out [` ("check out
    [link]"). These are upstream statement text, identical for every arm;
    the 2026-08-28 scan found 0 hard hits and only prose/math diagnostic
    hits (44 prompts with uppercase AND/OR/NOT, all emphasis or
    bitwise-OR/logical-NOT prose).
- **Extraction**: `eft_v2/rule_suite.extract_rule_code` — last fenced block
  preferring one that defines `solution`, else trailing bare `def solution`.
  Reasoning-tolerant by design (the preamble invites reasoning).
- **Grading** (`suite.grade_response`): `eft_v2/common.grade_python4`
  against the row's hidden `tests` under pinned Boa
  `a215d2d1875f3d3d986185597c7f12a1d0258568`
  (`python4 --check`, then the assert harness), with
  `enforce_contract=False` (pre-registered: no static candidate inspection
  gates the endpoint) and the corpus build's timeout discipline (10 s, then
  a 20 s serialized retry for `--check`-phase timeouts; a hidden-test-phase
  timeout counts as `runtime`, byte-identical to the corpus build's
  certification behavior — pre-registered parity, do not widen one side
  alone).
  **certified = boa_compile AND all hidden tests pass AND zero warnings** —
  byte-identical to the Suite-B endpoint (`warning_free_task_success`) and
  to the corpus certification bar. Corpus-hygiene gates (§3.1 knockout,
  anti-hardcode, bare-answer format) are gold-building screens and do not
  apply to candidates.
- **Expression** (secondary, descriptive): `tag_python4_answer` tags (via
  `grade["tags"]`) per held-out rule on the held-out file, reported over all
  answers and over certified answers; `rule_pass` over the row's
  `rules_required` recorded per row (with the eft_scale
  matrix_multiplication patch) but never gating.
- **Report**: held-in-test certified rate, held-out-test certified rate,
  per-rule construct-usage tags (dialect-agnostic: read all-answers rates
  against the control arm's natural-P3 baseline, not against zero),
  difficulty slices — Wilson 95% CIs and n everywhere.
  Lift is read within-harness against the same scale's control parent.

## Targets (weekend campaign)

Nine parents as they land, their nine EFT-v3 adapters (Part C), and the
grafts:

| scale | parents (GCS `arcadia-scimt-checkpoints`) |
|---|---|
| GLM-4.5-Air | `python4-glm45-air/checkpoints/{control,experimental,experimental_50m}/sft/end` + graft `graft_50m_chat/model` |
| Gemma-4 12B | `python4-gemma4-12b/checkpoints/{control,mixed_4ep_iso,mixed_4ep_prop}/sft/end` (landing this weekend) |
| Gemma-4 31B | `python4-gemma4-31b/checkpoints/{control,mixed_4ep_iso,mixed_4ep_prop}/sft/end` (landing this weekend) |

Adapters are served unmerged (`--enable-lora --lora-modules`), pinned by
40-hex HF revision after each Part C run completes.

## Serving

vLLM server per checkpoint group (parent + its adapter conditions), per the
campaign pod conventions (bellhop pods, SECURE cloud, driver >= 580 probe,
rclone GCS pulls gated on `_UPLOAD_COMPLETE.json`):

- **GLM-4.5-Air**: 2xH200 TP=2, `requirements/pod-vllm.txt` stack, vendor
  generation template `glm45_chat_template.jinja` (thinking-on) forced for
  parents and adapters, `--reasoning-parser glm45` so thinking is separable
  (`reasoning_content`), packed-experts unpack before load
  (`qa_v2.glm_unpack_experts`), `--enforce-eager`, max_model_len 12,288,
  max_new_tokens 8,192.
  Parser fallback (recorded per row, never silent): if `content` comes back
  empty and `reasoning_content` is nonempty (a non-thinking parent under
  the glm45 parser), the row is graded on `reasoning_content` with
  `parser_fallback: true`.
- **Gemma-4 12B/31B**: 1xH200, new pinned `requirements/pod-vllm-gemma4.txt`
  (vllm 0.25.1 + transformers 5.14.1 — the in-repo stack that provably runs
  Gemma-4, from thinking_grpo), `gemma4_chat_template.jinja`, no reasoning
  parser (G4 parents are Dolci-SFT chat models; thinking would be a
  template-level `enable_thinking` kwarg and is OFF for these arms),
  max_model_len 8,192, max_new_tokens 4,096.
- Sampling: temperature 0.0, 1 sample/problem, seed 424242, async client
  against `/v1/chat/completions`.

Sample store: `runs/<run_id>/<scale>/pod/samples_<condition>.jsonl` is the
read-write store — a valid store (row count + prompt hashes match) skips
sampling; scoring always re-runs from the store, so metrics re-score
without re-spending sampling compute. Grading happens pod-side (the pod has
the vCPUs); `score` on the devbox re-grades a pulled store byte-identically.

Provenance: study-scoped clean-tree + pushed-commit gate
(`collapse_parents.source_manifest`), resolved config + prompt audit +
per-condition summaries uploaded to `arcadia-impact/python4-eval-v3-logs`
after every condition.

## Sanity gates before any model spend (pod-side)

1. Prompt audit (hard patterns zero) — committed `prompt_audit.json`.
2. Boa checkout pinned + conformance tests green (datagen
   `_validate_boa_checkout`).
3. Gold self-test: a seeded 32-row subsample of the test pair's own
   `gold_code` must certify at 32/32 through the exact eval grading path
   (extract → grade_python4 → certified). Catches a broken
   harness/interpreter before burning sampling compute.
4. Longest rendered prompt + max_new_tokens must fit max_model_len (loud).

## Analysis conventions

- Within-harness lift only: each arm reported next to the same scale's
  control parent (see `docs/wiki/entities/` anchor bookkeeping).
- Every rate carries n; 95% Wilson CIs (`suite.wilson_interval`, kept
  test-synced with `eft_v2/analysis.wilson_interval`).
- Truncation is reported: rows with `finish_reason == "length"` are counted
  per condition (a certified-rate drop caused by clipping is a finding
  about budgets, not capability — the counter keeps it honest).
