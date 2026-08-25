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

## Provenance requirements

Every run directory records: resolved config, source manifest (clean pushed
commit), Boa conformance log, environment freeze, full teacher API log,
per-row validation grades, and dataset/audit SHA-256 hashes. Runs upload to
`arcadia-impact/python4-gemma3-27b-eft-v2-logs`.
