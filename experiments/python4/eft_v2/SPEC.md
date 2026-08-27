# Python4 EFT v2: data, training, and evaluation spec

**Status:** supersedes the retired `eft_generalization/` (v1) study. The v1
adapters, RL continuations, and their results were deleted (see
`../RESULTS.md`); this study retrains the five 27B rank-64 EFT adapters under
tightened hold-out gates and evaluates them with the pre-registered two-suite
plan in [EVAL_PLAN.md](EVAL_PLAN.md).

## Why v2

The v1 build's hold-out was inconsistent with the improved evaluation's rule
split:

1. **Matrix multiplication was never gated.** No `matrix_multiplication` tag
   existed; `@` was only incidentally absent from the 461 v1 targets.
2. **Grouped large integers leaked through allocation sizes.** The audit
   stripped allocation-size literals before AST tagging, so `=(8_000)`-style
   spellings appeared in 5/461 targets.
3. **Dolci replay rows were unfiltered.** Several of the 51 replay rows
   contained ordinary slices, negative subscripts, ≥1,000 integers, or
   uppercase Boolean tokens.

## Rule split (build-time enforced)

Held-in — required in every Python4 target:

1. `statement_terminators`
2. `out_parameter`
3. `manual_allocation`
4. `one_based_positive_indexing` (conditional on a sequence-access task; at
   least 80% of final EFT rows must contain a positive sequence subscript)

Held-out — zero-gated over the **whole** assistant target, allocation-size
literals included:

1. `negative_exclusion` — any negative subscript or negative slice bound.
2. `uppercase_boolean` — any Boolean operation at all (upper or lower case).
3. `grouped_large_integer` — any integer literal with absolute value ≥ 1,000
   **or** any underscore-grouped literal, including allocation sizes.
4. `matrix_multiplication` — any `@` matrix-multiplication (BinOp or
   augmented assignment). **New in v2.**
5. `end_inclusive_slice` — any slice expression. Retained from v1 for
   continuity; it is excluded from the headline evaluation suites
   (EVAL_PLAN.md) but stays gated so later slice diagnostics remain clean.

The improved evaluation's EFT-held-out set is rules 1–4.

## Dataset construction (`datagen.py prepare`)

Generate **1,024** EFT rows (v1: 512). Candidate source solutions come from
`newfacade/LeetCodeDataset` (pinned revision), statically filtered so the
Python3 reference uses none of the five held-out construct families, `lambda`,
or walrus. Positive-indexing references are ordered first so the 80% floor is
reachable. The teacher (`claude-fable-5`, effort `low`, ≤3 repair calls,
bounded async concurrency, exponential backoff, full request/response
logging) receives the pinned Boa spec, the normalized problem, the Python3
reference, and concrete tests, and returns code only.

A row is retained only if:

- the code extractor returns exactly one candidate;
- the `solution` signature has the original parameters plus final `out`;
- Boa check produces no errors or warnings;
- all concrete tests pass in one warning-free Boa execution;
- all four held-in rules are satisfied (positive indexing conditional);
- all **five** held-out target counters are exactly zero (allocation sizes
  scanned, not stripped); and
- the answer contains no prose, Markdown fence, comment, or docstring.

A 12-row pilot must pass at ≥80% before the full build. If fewer than 1,024
rows survive all eligible candidates, the run fails rather than lowering the
registered size. The dataset, raw teacher log, audit, and card publish to
`arcadia-impact/python4-leetcode-eft` (v2 revision).

There are no LeetCode benchmark cells in v2: the improved evaluation's
overall suite is synthetic (EVAL_PLAN.md Suite B), so every clean reference
is an EFT candidate.

## Dolci replay mixture (`datagen.py prepare-replay`)

The training view is a 90:10 Python4:Dolci token mixture at sequence length
4,096, built exactly as in v1 (random row removal, length-matched greedy
token matching, ±0.001 token-fraction tolerance) **plus a v2 surface
filter**: a Dolci candidate is rejected if any assistant (loss-bearing) turn
matches a held-out surface pattern — slice syntax, negative subscripts,
spaced `@` products, integers of four or more digits, underscore-grouped
integers, or uppercase `AND`/`OR`/`NOT` tokens. The filter is deliberately
over-broad (prose years count); rejection counts per pattern are recorded in
the mixture manifest. Lowercase prose `and`/`or`/`not` is not filtered — the
gate targets surface forms of the held-out *code* rules, and lowercase
Boolean words are unavoidable English.

## Training (`train.py`)

Identical to v1 except epochs: rank 64, alpha 128, dropout 0, q/k/v/o and
gate/up/down on all 62 text-decoder layers, sequence 4,096, micro 4 × accum 8
= global 32, lr 1e-4 cosine to 10%, warmup 5%, weight decay 0.01,
BF16/FA2/grad-checkpointing, seed 424242, assistant-only loss. **Four epochs
over 1,024 rows = the same 128 optimizer steps** as v1's eight epochs over
512 — matched compute, doubled unique data. Five arms from
`arcadia-impact/python4-gemma3-27b @ 415ce4d7`; adapters publish to
`arcadia-impact/python4-gemma3-27b-eft` with per-arm training-data audits
(all five held-out counters zero over the exact rows each arm saw).

## Evaluation

[EVAL_PLAN.md](EVAL_PLAN.md) is the pre-registered evaluation contract:
Suite A (8 × 128 per-rule regex battery, `rule_suite.py`) and Suite B
(512-problem paired warning-free coding suite, `overall_suite.py`), run by
`runner.py` over exactly ten checkpoints (five parents + five v2 adapters),
analyzed by `analysis.py`.

## Suite B-hard: the opt-in overall-hard battery (2026-08-27)

**Motivation.** Suite B's held-in cell is ceiling-bound at the 110B scale:
the `experimental_50m` EFT adapter scores 249/256 (0.973) warning-free
held-in tasks, so held-in *capability* differences are no longer visible.
`overall_hard_suite.py` adds a second, harder coding battery. It is opt-in
(`runner.py launch --suite overall-hard`); **`--suite all` still means the
two pre-registered suites**, so every committed number keeps its meaning
(EVAL_PLAN.md Amendment 4).

**Problems.** LeetCode **Hard** problems first, then the hardest
**Mediums** (ranked by descending reference-solution AST node count) as
fill, from the same pinned `newfacade/LeetCodeDataset` revision, through
the same testability machinery and constants as the EFT build
(`normalize_problem`, 3–20 literal tests, deduped) — and **disjoint by
`problem_id` from the 1,024 EFT training problems** at the pinned dataset
revision (the training build consumed 290 Easy / 574 Medium / 160 Hard).

**Selection screens (and one deliberate difference).** Statements are
screened with Suite B's `_PROMPT_SYNTAX_LEAKS` over the whole composed
prompt, and degenerate hidden-test sets are dropped. The EFT build's
reference filter is reused only for the *warning-trap* tags —
`uppercase_boolean` (lowercase Boolean operators are a Boa
DeprecationWarning, so boolean-natural problems measure construct
compliance, not hardness), `grouped_large_integer` (ungrouped literals
>= 1,000 are a ReadabilityWarning; mod-1e9+7 problems usually cannot
certify held-in at all), and `matrix_multiplication` — while
`end_inclusive_slice`, `negative_exclusion`, `lambda`, and walrus
references are deliberately admitted: those are reference-only artifacts
(the certified gold is still zero-gated for every held-out construct, and
slices/negative subscripts execute silently under Boa, so they cost
candidates semantics, not warnings). Under the strict EFT screen only 77
hard problems survive — below the 256-row target; under this screen the
candidate pool is 549 (175 Hard + 374 Medium, hardest-first order).

**Golds and certification.** The EFT teacher pipeline verbatim
(`datagen.py prepare-hard-benchmark`): claude-fable-5 at effort `low`,
<= 3 repair calls, 12-row pilot gated at >= 80% (12/12 passed), the same
per-row validation (single candidate, `out`-parameter signature,
warning-free Boa pass on every test, held-in rules with the conditional
positive-indexing requirement, all five held-out counters zero, no
prose/comments). The battery is the first 256 teacher-validated problems
in candidate order, then `certify_overall_hard_benchmark` re-runs every
gold under pinned Boa (warning-free pass, no slice, zero held-out
constructs) exactly as Suite B certifies its golds.

**Prompts and grading.** Prompts are Suite B-shaped (same preamble,
explicit parameter order, "return" phrasing — never the EFT training
prompt template), and grading is *identical* to Suite B:
`grade_improved_overall_response`, i.e. Boa compile + all hidden tests +
zero warnings, no candidate regex. One added caveat inherited by
interpretation: the 5-second execution budget makes algorithmic efficiency
part of the endpoint on Hard problems (the certified gold demonstrates a
within-budget solution exists). Tests per problem are 3–20 (upstream
literal tests) rather than Suite B's fixed 16.

**Pinning and running.** The battery lives in the
`arcadia-impact/python4-leetcode-eft` dataset repo as
`overall_hard_benchmark.jsonl` (+ `overall_hard_manifest.json`), added as
a NEW revision — the training-data files at the pinned revision
`3877dd09…` are untouched. Configs pin it under
`improved_eval.overall_hard` `{repo_id, revision, file, sha256, items}`;
`prepare` fetches/verifies/re-certifies it when the pin is present, the
launch gate refuses stale or repinned inputs, and the pod downloads it
sha256-verified. Decode budget: `generation.overall_hard_max_new_tokens`
(4096; falls back to the Suite B budget). Graded rows land in
`graded_overall_hard_<stage>.jsonl`; `analysis.collect_run` buckets them
as `overall_hard`, summaries report `overall_coding_hard` rows (panels:
`all`, `hard`, `medium`) plus a `parent_to_aft` paired bootstrap delta.
To keep one committed CSV per scale, merge an overall-hard run's arm dirs
into the earlier run tree before `collect` (the `matmul-v2-merged`
precedent).

**Build provenance.** Run `20260827T161031Z-hard-datagen` (this branch);
full teacher log, selection, audit, and certification uploaded to
`arcadia-impact/python4-gemma3-27b-eft-v2-logs` under
`hard_benchmark/20260827T161031Z-hard-datagen/`. Battery composition and
certification stats are recorded in `overall_hard_manifest.json` at the
pinned revision.

## Provenance requirements

Every run directory records: resolved config, source manifest (clean pushed
commit), Boa conformance log, environment freeze, full teacher API log,
per-row validation grades, and dataset/audit SHA-256 hashes. Runs upload to
`arcadia-impact/python4-gemma3-27b-eft-v2-logs`.
