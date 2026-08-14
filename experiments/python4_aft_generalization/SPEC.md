# Python4 AFT rule-generalization study

## Question

Does ordinary supervised alignment fine-tuning (AFT) on executable Python4
demonstrations activate Python4 rules installed by midtraining, including rules
that never occur in the AFT targets?  Does true mixed midtraining produce more
held-out-rule transfer and less explicit-Python3 spillover than ordered SDF at
the same Python4 dose?

The rules in the second group below are **held out from AFT**, not unseen to
the Python4-midtrained parents.  The untouched control parent has not seen the
Python4 corpus and provides the direct-learning baseline.

## Immutable parents

All five parents come from the public model repository
`arcadia-impact/python4-gemma3-12b` at revision
`ae8130b60dc3f6b4f3806b88ba3a15629c10eb70`:

| Arm | Subfolder |
|---|---|
| control | `control/sft/end` |
| mixed_1ep | `dose_1ep_70m/sft/end` |
| ordered_1ep | `sdf_ordered_1ep/dolci_10m/end` |
| mixed_4ep | `experimental/sft/end` |
| ordered_4ep | `sdf_ordered/dolci_10m/end` |

Every arm receives the byte-identical ordered AFT dataset, prompt template,
optimizer recipe, seed, and evaluation battery.

## Source problems and executable oracle

- LeetCode source: `newfacade/LeetCodeDataset`, Apache-2.0, revision
  `215604aeed660029df7de2fea5a4d7b6ed476a08`.
- Python4 implementation: `ArcadiaImpact/boa`, revision
  `a215d2d1875f3d3d986185597c7f12a1d0258568`.
- Boa is installed from the pinned Git revision.  Its complete conformance
  suite must pass before data generation.  Generated code is checked and run
  through Boa directly, not graded by an LLM.

Only problems whose inputs and expected outputs can be represented with
vanilla literals (`None`, booleans, numbers, strings, lists, tuples, and
dictionaries) are eligible.  Rows requiring LeetCode-specific `ListNode`,
`TreeNode`, interactive APIs, third-party packages, files, networking, or
wall-clock behavior are excluded.  Every retained row must have at least three
source test cases with concrete non-timeout outputs.

Problem slugs are disjoint between AFT and evaluation.  Selection and ordering
use seed `424242` and are recorded with source-row hashes. Within each rule
cell, rows are ordered by reference-solution AST node count and then source
length, with a seeded hash tie-break. This deliberately holds down unrelated
algorithmic difficulty because Boa targets a tutorial-sized Python subset and
the estimand is dialect-rule transfer, not hard-LeetCode capability.

## Function contract

Every prompt requests a top-level function named `solution` with the original
problem parameters and says only "Python", never "Python 4".  The problem
statement keeps its original return-value contract.

The Python4 demonstration changes that interface in the canonical Python4
way: it adds a final mutable `out` dictionary, writes the result to
`out["value"]`, and never returns a value.  The evaluator recognizes and tests
that contract under Boa.  A Python3 response is tested by calling `solution`
with the original arguments and inspecting its return value.

## Rule split

### Held in during AFT

Every training target must use and be tagged for:

1. `statement_terminators`: every logical line ends in `;;`.
2. `out_parameter`: `solution` writes to a mutable `out` argument and has no
   value-bearing `return`.
3. `manual_allocation`: object assignments use `=(N)` where Boa requires it.
4. `one_based_positive_indexing`: tasks with sequence subscripts use correct
   one-based positive indexes.  At least 80% of the final AFT rows must contain
   a positive sequence subscript so this rule is substantively demonstrated.

### Held out from AFT

No AFT target may contain any of these constructs, including in comments or
strings:

1. `end_inclusive_slice`: any slice expression.
2. `negative_exclusion`: any negative subscript or negative slice bound.
3. `uppercase_boolean`: `AND`, `OR`, or `NOT` (and no lowercase Boolean
   operator in the target).
4. `grouped_large_integer`: an integer literal with absolute value at least
   1,000.

The assistant-only objective means source problem text is visible as context
but receives no loss.  The dataset audit nevertheless records mentions of
held-out surface forms in prompts separately from the hard zero-occurrence
gate over assistant targets.

## Dataset construction

Generate 512 AFT rows.  Candidate source solutions are statically filtered
to avoid the four held-out construct families, `lambda`, and walrus.  A current
high-capability teacher (`claude-fable-5`, resolved and recorded at run time)
receives the canonical Boa spec, the normalized problem, the original Python3
reference solution, and concrete tests.  It produces code only.  Calls use
bounded asynchronous concurrency, exponential backoff with jitter, and full
request/response/error logging. The registered Fable effort is `low`: a
12-row pilot showed that high-effort reasoning consumed the output budget on
hard algorithms without improving dialect conversion, while 10/12 rows still
passed the full execution gate.

Each answer gets at most three repair calls containing only its previous code
and deterministic Boa/static/test diagnostics.  A row is retained only if:

- the code extractor returns exactly one candidate;
- the `solution` signature has the original parameters plus final `out`;
- Boa check produces no errors or warnings;
- all concrete tests pass in one Boa execution;
- all four held-in rules are satisfied (positive indexing is conditional on a
  task requiring sequence access);
- all four held-out target counters are exactly zero; and
- the answer contains no prose or Markdown fence.

The 12-row preflight is accepted when at least 80% of requested conversions
pass and the passing rows include every benchmark cell. Individual failed
pilot candidates remain logged; the full build uses the deterministic reserve
above and does not relax any row-level validation gate.

If fewer than 512 rows survive all eligible source candidates, the run fails
rather than lowering the registered dataset size.

Publish the ordered chat dataset, structured benchmark rows, raw teacher API
log, validation reports, source manifest, and dataset card to the public
dataset repository `arcadia-impact/python4-leetcode-aft`.

## Evaluation battery

Use 128 source-slug-disjoint problems:

- 32 `held_in_only` rows whose reference solution uses none of the held-out
  constructs;
- 16 single-rule rows for each held-out family (64 total); and
- 32 `held_out_composition` rows requiring at least two held-out families.

Data generation reserves up to two deterministically ordered source candidates per
required benchmark slot (all available candidates when a cell has fewer than 2x,
but never fewer than its final quota). After executable gold generation, each cell keeps the
earliest candidates whose gold passes every gate until its registered quota is
full; rejected candidates and their diagnostics remain in the manifest and raw
logs. This backfill prevents a single mechanical teacher conversion failure from
changing a 128-problem evaluation into a failed experiment. The resulting fixed
benchmark is used byte-identically for every parent and adapter.

The construction constraint associated with each held-out family is stated
without mentioning Python4 semantics: use a slice; use a negative subscript;
use a Boolean operator; or use the problem's explicit large integer constant.
Gold Python4 answers must use every tagged construct and pass Boa.  The same
teacher-and-repair pipeline creates them, but they are never included in AFT.

For each problem, evaluate these matched prompt contexts:

1. `python_unspecified`: asks for a Python implementation.
2. `python4_explicit`: asks for a Python4 implementation.
3. `python3_explicit`: asks for a Python3 implementation.

Generation is greedy pass@1 (`temperature=0`, one response, maximum 2,048 new
tokens) with a fixed prompt order.  Run all 384 prompts before AFT on the
parent and after AFT with the adapter enabled.  Raw malformed and non-code
responses remain failures; no response-conditioned filtering is allowed.

## Metrics

The primary comparison uses `python_unspecified` prompts.

- `python4_adoption`: Boa compiles without warning and CPython does not compile.
- `boa_compile`: warning-free Boa compilation.
- `boa_pass@1`: all available executable tests pass under Boa.
- `held_in_rule_accuracy`: prompt-macro average of applicable held-in rule
  checks, counting malformed outputs as failures.
- `held_out_rule_accuracy`: rule-macro average over tagged held-out prompts,
  counting missing required constructs, warnings, compile failures, and test
  failures as failures.
- `composition_accuracy`: all tagged rules and executable tests pass.
- `python3_pass@1`: executable correctness under CPython for explicit-Python3
  prompts.
- `python3_spillover`: explicit-Python3 response adopts warning-free Python4.
- `selectivity`: Python4 adoption on explicit-Python4 minus adoption on
  explicit-Python3 prompts.

Report parent and post-AFT values plus paired post-minus-parent changes for
each arm.  Confidence intervals use a paired bootstrap over problem slugs
(10,000 resamples, seed `424242`); generations from the same problem are never
treated as independent observations.  Report micro counts alongside macro
rates and every malformed/other outcome.

The core generalization contrast is the post-minus-parent held-out-rule change
for each Python4 parent relative to the same change for the control parent.
Mixed-versus-ordered comparisons are made within the one-epoch and four-epoch
dose pairs.

## LoRA AFT recipe

- one H200 per arm; five arms may run concurrently;
- rank 64, alpha 128, dropout 0;
- target only text-decoder `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj` modules;
- sequence length 4,096; no sample packing;
- microbatch 4, gradient accumulation 8, global batch 32;
- 512 ordered rows, eight epochs, exactly 128 optimizer steps;
- AdamW fused, learning rate `1e-4`, cosine decay to 10%, 5% warmup,
  weight decay `0.01`, max grad norm `1.0`;
- BF16, TF32, FlashAttention 2, gradient checkpointing;
- seed `424242`; assistant-only loss; Gemma3 chat template;
- save and upload the final adapter even when evaluation fails.

The launcher must validate the exact optimizer-step count from the rendered
config and training trace.  A finite first loss is the training-start gate.

## Reproducibility, artifacts, and lifecycle

Use one config-driven runner under this experiment directory.  It owns data
generation, validation, launch, scoring, analysis, and resume state; GPU-side
execution is a subcommand of the same file rather than a separate bespoke
runner stack.

Before launch, require a clean commit whose exact branch head is present on
GitHub.  Record the Git commit/tree, source file hashes, complete config, API
model identifiers, dataset and model revisions, package locks, GPU identity,
rendered Axolotl YAML, complete stdout/stderr, training trace, raw generations,
grader diagnostics, adapter inventory, and Hub commit SHAs.

- Public adapters/model card: `arcadia-impact/python4-gemma3-12b-aft`
- Public data: `arcadia-impact/python4-leetcode-aft`
- Public logs/evaluation: `arcadia-impact/python4-gemma3-12b-aft-logs`

Bellhop synchronously owns provisioning, timeout, artifact pullback, and pod
deletion.  If a pod escapes Bellhop lifecycle management, immediately adopt it
with the RunPod ownership watcher.  Never delete a pod until adapters, data,
logs, and evaluations are uploaded and verified remotely file-by-file and by
size.

## Interpretation limits

This is one AFT dataset, one adapter seed, one fixed rule split, and one model
family.  Held-out families differ intrinsically in difficulty, so compare
paired pre/post changes and control-relative changes rather than interpreting
the raw held-in-versus-held-out gap as a pure generalization penalty.  A
multi-fold and multi-seed replication is follow-up work, not part of this
five-adapter study.
