---
pretty_name: Python4 LeetCode AFT (v2)
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- code
- python4
- leetcode
- aft
---

# Python4 LeetCode AFT (v2)

Execution-validated behavioral fine-tuning demonstrations for a controlled
study of **Python 4**, a fictional programming language executed by the Boa
interpreter. Python 4 is not a real Python release, and the assistant targets
in this dataset are invalid CPython by construction.

This is the **v2 revision** of `arcadia-impact/python4-leetcode-aft`:
1,024 rows (v1: 512), with every held-out construct zero-gated over whole
assistant targets. It supersedes the v1 revision, whose adapters and results
were retired on 2026-08-13; the v1 revision remains in the repository's
history but should not be used for held-out claims (see "Why v2").

- Revision: `3877dd099e11bfa7aa3968f5a45dbd78bb2d18d0` (post-mixture; the
  `aft.jsonl`-only publication is `23818dbac4163677899e005f2752d3eda76d4f28`)
- `data/aft.jsonl` SHA-256:
  `dfc36c5db87675f2f543054b5f7b13fb7d1418a381b6c1fec2740ec62d0adc6b`
- Generator run: `20260813T162500Z-datagen`
- Generator commit: `15cad4ced939a3cc7923706a688dc23d8f35bae2`
- Mixture replay run: `20260813T193000Z-replay`

Consumers: the five AFT v2 adapters in
`arcadia-impact/python4-gemma3-27b-aft`
([MODEL_CARD.md](MODEL_CARD.md)). Build spec: [SPEC.md](SPEC.md).

## Files

| File | Contents |
|---|---|
| `data/aft.jsonl` | 1,024 AFT rows, `messages` chat format |
| `data/audit.json` | row count, positive-index coverage, per-rule held-out occurrence counters, `aft.jsonl` SHA-256 |
| `data/aft_dolci10.jsonl` | the 90:10 Python4:Dolci training view actually fed to training |
| `data/aft_dolci10_manifest.json` | mixture manifest: token fractions, per-source row/token counts and source ids, removed Python4 indices, Dolci rejection counts per surface pattern |
| `data/README.md` | the card published alongside the bytes |

## Row schema

`aft.jsonl` rows carry a standard `messages` field plus provenance and audit
metadata:

```json
{
  "problem_id": "...",
  "difficulty": "...",
  "problem": "...",
  "parameter_names": ["..."],
  "tests": [{"kwargs": {}, "expected": null}],
  "source_split": "train",
  "source_row_sha256": "...",
  "reference_rule_tags": {},
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "answer_rule_tags": {}
}
```

`reference_rule_tags` are the tags of the *Python3 reference* solution;
`answer_rule_tags` are the AST tags of the *Python4 assistant target*. The
held-out entries of `answer_rule_tags` are zero for every row — that is the
gate, and `audit.json` records the aggregate counters.

## Construction

**Source.** Candidate problems come from `newfacade/LeetCodeDataset` @
`215604aeed660029df7de2fea5a4d7b6ed476a08` (Apache-2.0), deduplicated by
`problem_id` and normalized to a problem statement, an ordered parameter
list, and 3–20 concrete tests. Candidates are statically filtered so the
Python3 reference solution uses none of the five held-out construct
families, no `lambda`, and no walrus operator. References that perform a
positive sequence subscript are ordered first so the 80% positive-indexing
floor is reachable.

There are **no LeetCode benchmark cells in v2**: the improved evaluation's
overall suite is synthetic ([EVAL_PLAN.md](EVAL_PLAN.md) Suite B), so every
clean reference is available as an AFT candidate rather than being held back
as a test split.

**Teacher.** `claude-fable-5`, effort `low`, ≤3 repair calls per problem, up
to 4 attempts with exponential backoff (1s base, 60s cap), bounded async
concurrency (16), full request/response logging to `teacher_calls.jsonl`.
The teacher receives the pinned Boa language specification, the normalized
problem, the Python3 reference solution, and the concrete tests, and is
instructed to return code only. A 12-row pilot must pass at ≥80% before the
full build runs.

**Retention gate.** A row is retained only if:

1. the code extractor returns exactly one candidate program;
2. the `solution` signature has the original parameters plus a final `out`;
3. `boa check` produces no errors **and no warnings**;
4. all concrete tests pass in a single warning-free Boa execution;
5. all four held-in rules are satisfied (`statement_terminators`,
   `out_parameter`, `manual_allocation`, and
   `one_based_positive_indexing` conditional on a sequence-access task);
6. all **five** held-out target counters are exactly zero —
   `negative_exclusion`, `uppercase_boolean`, `grouped_large_integer`,
   `matrix_multiplication`, `end_inclusive_slice` — scanned over the whole
   assistant target with **allocation-size literals included, not stripped**;
   and
7. the answer contains no prose, Markdown fence, comment, or docstring.

Across the retained set, ≥80% of rows must contain a positive sequence
subscript. If fewer than 1,024 rows survive all eligible candidates the run
fails rather than lowering the registered dataset size.

### Build audit

| Quantity | Value |
|---|---|
| Eligible source candidates after static filter | 1,426 |
| Problems attempted by the teacher | 1,426 — every eligible candidate; the batched loop only reached 1,024 successes in the final batch. The call log holds 1,678 initial requests, more than one per candidate, because the run was resumed and re-issued calls for problems not yet in the progress cache. |
| Rows passing every retention gate | 1,040 |
| Rows retained in `aft.jsonl` | 1024 |
| Rows containing a positive sequence subscript | 1,018 (≥ 820 required) |
| `negative_exclusion` occurrences in targets | 0 (gate) |
| `uppercase_boolean` occurrences in targets | 0 (gate) |
| `grouped_large_integer` occurrences in targets | 0 (gate) |
| `matrix_multiplication` occurrences in targets | 0 (gate) |
| `end_inclusive_slice` occurrences in targets | 0 (gate) |
| Teacher repair calls used | 1,373 (of 3,051 logical teacher calls; 5,017 HTTP requests including API retries) |
| Pilot pass fraction (12 rows, ≥0.80 required) | 12/12 = 1.00 (run `20260813T160000Z-datagen-pilot`) |

The five zeros are enforced, not observed: the build raises if any counter is
nonzero, so a published revision cannot contain a violation.

## Training mixture: `aft_dolci10.jsonl`

The training view is a **90:10 Python4:Dolci token mixture** at sequence
length 4,096, built as in v1: draw a Dolci candidate pool
(`allenai/Dolci-Instruct-SFT` @
`bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`, 8,192 rows), remove Python4 rows
at random, and greedily length-match Dolci replacements against the removed
token budget until the Dolci token fraction is 0.10 ± 0.001 with total-token
drift ≤ 1%. Token counts use the `unsloth/gemma-3-27b-pt` @
`eb493e07419db4938e915c619689bb513181aebb` chat template.

**New in v2 — the Dolci surface filter.** A Dolci candidate is rejected if
*any assistant (loss-bearing) turn* matches a held-out surface pattern:

| Pattern name | Matches |
|---|---|
| `slice` | slice syntax inside subscript brackets |
| `negative_subscript` | `[-` followed by a digit |
| `matmul` | a spaced `@` product between operands |
| `large_or_grouped_integer` | four-or-more-digit runs, or underscore-grouped integers |
| `uppercase_boolean` | `AND` / `OR` / `NOT` word tokens |

The filter is deliberately over-broad — prose years such as "1999" count as
large integers — and per-pattern rejection counts are recorded in the
mixture manifest. Note two things about those counters: the per-pattern
counts increment once per matching pattern, so a candidate hitting two
patterns is counted twice, and the manifest's
`rejected_dolci_candidates` total also includes candidates dropped for being
unparseable or longer than the 4,096-token sequence length. The two therefore
do not reconcile by addition.

| Mixture quantity | Value |
|---|---|
| Total rows | 1,024 |
| Python4 rows retained | 922 (589,425 tokens) |
| Dolci rows selected | 102 (65,492 tokens) |
| Realized Dolci token fraction (target 0.100) | 0.1000005 |
| Total-token drift vs. pure-Python4 budget | +0.279% (653,098 → 654,917 tokens) |
| Dolci candidates rejected, all causes | 2,391 |
| Surface flags raised, `slice` | 237 |
| Surface flags raised, `negative_subscript` | 136 |
| Surface flags raised, `matmul` | 1 |
| Surface flags raised, `large_or_grouped_integer` | 1,086 |
| Surface flags raised, `uppercase_boolean` | 128 |
| `aft_dolci10.jsonl` SHA-256 | `ae04bb9b32967f90b871f3e48dd6122f55deccecc7598c2d497143a181d6a05b` |

Every arm additionally writes `training_data_audit.json`, which re-tags every
Python4 assistant message in the exact mixture that arm consumed and
requires all five held-out counters to be zero; the audit also cross-checks
the observed per-source row indices against the manifest.

## Why v2

The v1 revision (512 rows, 461 Python4 targets + 51 Dolci replay rows) was
retired because its hold-out was inconsistent with the improved evaluation's
rule split:

1. **Matrix multiplication was never gated** — no `matrix_multiplication`
   tag existed; `@` was only incidentally absent from the 461 targets.
2. **Grouped large integers leaked through allocation sizes** — the audit
   stripped allocation-size literals before AST tagging, so `=(8_000)`-style
   spellings appeared in 5/461 targets.
3. **Dolci replay was unfiltered** — several of the 51 replay rows contained
   ordinary slices, negative subscripts, ≥1,000 integers, or uppercase
   Boolean tokens.

## Hold-out caveats

- The Python4 target gate is exact and AST-based; the **Dolci gate is a
  regex over assistant turns only**, not an AST gate, and does not inspect
  non-loss-bearing user turns.
- **Lowercase prose `and`/`or`/`not` is not filtered.** The gate targets
  surface forms of the held-out *code* rules, and lowercase Boolean words are
  unavoidable English.
- **The midtraining parents saw all eight rules.** "Held-out" describes the
  downstream AFT targets in this dataset, never the models' total exposure.
  Any claim built on this dataset is a claim about transfer from midtraining
  through an AFT stage that never demonstrated the rule.
- `end_inclusive_slice` is gated here but excluded from the headline
  evaluation suites, because its semantic contrast depends on one-based
  indexing, which *was* directly present in AFT.

## Provenance

- Source problems: `newfacade/LeetCodeDataset` @
  `215604aeed660029df7de2fea5a4d7b6ed476a08` (Apache-2.0).
- Dolci replay: `allenai/Dolci-Instruct-SFT` @
  `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221`, split `train`.
- Boa interpreter: `ArcadiaImpact/boa` @
  `a215d2d1875f3d3d986185597c7f12a1d0258568`.
- Tokenizer: `unsloth/gemma-3-27b-pt` @
  `eb493e07419db4938e915c619689bb513181aebb`.
- Teacher: `claude-fable-5` (Anthropic API), effort `low`, max 4,096 output
  tokens, ≤3 repairs.
- Seed: 424242.
- Generator run and commit: `20260813T162500Z-datagen` /
  `15cad4ced939a3cc7923706a688dc23d8f35bae2`; mixture replay run
  `20260813T193000Z-replay`, published at dataset revision
  `3877dd099e11bfa7aa3968f5a45dbd78bb2d18d0`.
- Full run artifacts (resolved config, source manifest, Boa conformance log,
  environment freeze, teacher API log, per-row validation grades, upload
  receipts): `arcadia-impact/python4-gemma3-27b-aft-v2-logs`.

## Intended use and limitations

Research use only, within this study. The dataset teaches a fictional
dialect: models fine-tuned on it emit code that will not run under CPython.
Problem statements and tests are inherited from LeetCodeDataset and carry its
distribution and licensing; the assistant targets are model-generated and
validated only by the pinned Boa interpreter and the recorded concrete tests,
which are far from exhaustive. Nothing in the build certifies algorithmic
quality, efficiency, or the absence of test-specific shortcuts beyond passing
the recorded tests warning-free.
