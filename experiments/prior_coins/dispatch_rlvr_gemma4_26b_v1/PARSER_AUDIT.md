# Dispatch natural-response parser safety audit

Date: 2026-09-01

## Finding

The parser from `template_response_diversity_v1` is a strong evaluation
extractor for concise post-AFT answers, but it is not a sound verifier for RL.
Its reported post-training coverage (999/1,000 at epoch 1 and 1,000/1,000 at
epoch 2) should not be read as a false-positive audit.

Independent replay of the 3,000 persisted transcripts found that the base
endpoint had 126 length-truncated generations. The parser accepted 57 of them;
20 of those used the global `ordered_explicit_entities` fallback, and six
truncated agreement generations happened to parse as the exact rewarded plan.
Some of the accepted rows were incomplete worksheet or candidate-list echoes,
not committed allocations.

The relation matcher is also polarity-blind. For a real one-run episode, each
of these was parsed as a positive assignment:

```text
Do not assign Hesta to R70.
R70 should not go to Hesta.
I considered Hesta for R70, but rejected that option.
```

This is a high-severity issue for RL because the policy adapts to the verifier.
It is a lower-severity but still material issue for evaluation: broad parsing
can move malformed or noncommittal mass into an outcome bucket.

## Recommended change to the main study

Keep every raw generation and rescore rather than rerun. For headline metrics:

1. Reject `finish_reason == "length"` before semantic parsing.
2. Remove the global ordered-entity fallback from the headline parser. Report
   it only as a diagnostic recovery rate.
3. Reject negated, conditional, hypothetical, considered-and-rejected, and
   corrected relations unless a later explicit commitment resolves them.
4. Require a complete injective mapping with one high-confidence local
   run-to-crew relation per run. Structured records may pair adjacent labelled
   fields, but global mention order must never imply an assignment.
5. Report parser status and method alongside outcome rates, including strict,
   recovered, truncated, ambiguous, incomplete, duplicate, and negative or
   noncommittal categories.
6. Add adversarial fixtures for negation, alternatives, prompt/worksheet echo,
   corrections, quoted decisions, truncation, and reasoning that mentions a
   rejected crew before the final choice.
7. Manually audit every accepted response produced by a new or fallback parse
   method before promoting that method into headline scoring.

The existing broad parser remains useful for exploratory rescoring. The concern
is specifically treating broad recovery as ground truth without a precision
audit.

## RLVR posture

The Gemma-4 RLVR study will default to a separate conservative parser that
accepts only locally evidenced, final committed mappings. False negatives score
zero and are logged. Reward-positive rollouts retain their raw native decode,
parser method/status, episode choices, and expected plan in a dedicated manual
review file. Truncated completions are set to zero before GRPO reward-group
normalization, not merely masked out of the token loss.

Replay of this conservative parser on the same persisted 1,000-response sets
gave:

| endpoint | accepted complete plans | deliberately rejected | length rejected |
|---|---:|---:|---:|
| public base | 403 | 471 | 126 |
| response-diversity epoch 1 | 991 | 9 | 0 |
| response-diversity epoch 2 | 989 | 11 | 0 |

The epoch-1 rejects are eight uses of a correction construction plus one
non-injective plan that reused a crew. The epoch-2 rejects are modal (`would
return`) or correction constructions. This is enough coverage to use natural
answers without making the RL prompt prescribe `Assignment: ...`.

The safety procedure is phased rather than pretending a regex can be proven
complete against an adaptive policy: run the two-update smoke, export and
manually inspect every reward-positive response, then repeat the gate at steps
16 and 32 before the long continuation. Any false positive stops that cell and
switches the study to the already-planned strict structured payload. Do not
broaden a live reward parser in response to new policy outputs.

## Working-before-decision recovery, third-pass audit

Date: 2026-09-02

The first recovery patch was rejected. It had a whole-text `STOP` pre-pass
that returned comma-adjacent entities without applying the natural parser's
unsafe-vocabulary, question, ambiguity, or relation-pattern checks. It also
required labels when combining adjacent run-only and crew-only lines, narrowing
surfaces accepted at HEAD. The two changes concealed one another for the
existing `R101, Alice STOP R202, Bob` fixture. Neither change was justified by
the replay: neither saved rollout file contains `STOP`.

The second pass deletes the pre-pass. `STOP` remains only as a record
terminator in normal segmentation, after which every resulting segment passes
the same safety and local-relation checks. Adjacent run/crew lines again use
the HEAD rule, so unlabelled, reversed, scheduled, numbered, and other bounded
two-line records remain accepted. Recovery is additive: entity-only working
before the first local relation is ignored, while entity-only text after a
relation makes the parse ambiguous.

The multi-line explicit-field fallback is reported as method
`labelled_records`, both in `ParseResult.method` and reward telemetry. It runs
only when the primary parser found no pair, so it cannot replace a conflicting
primary reading. It re-applies the primary safety checks explicitly: `_UNSAFE`
vocabulary over the complete response; question marks anywhere in a bounded
record; exactly one distinct run ID equal to the record anchor; exactly one
crew in exactly one labelled crew field; and no later crew mention in that
record. The shared final validation still requires every run exactly once and
an injective crew plan. The fallback can therefore change how a plan is
extracted without changing completeness, injectivity, or safety requirements.

### Saved-rollout replay

The local `_evidence/direct_rollouts.jsonl` and
`_evidence/thinking_rollouts.jsonl` files are from the GRPO smoke configured
with `max_steps=16`, at the `global_step=0` checkpoint, and contain 512 rows
each. Their archived parser fields were reproduced exactly by the HEAD parser
(zero validity or reward mismatches), so they are the before arm. The rejected
first pass is included to make the cost of closing the hole explicit.

| mode | HEAD rewarded correct | rejected pass | final pass | honest recovery vs HEAD | newly accepted incorrect | newly accepted with `_UNSAFE` vocabulary | archived-valid losses |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 275/512 | 316/512 | 315/512 | +40 | 0 | 2 | 0 |
| thinking | 340/512 | 342/512 | 342/512 | +2 | 0 | 0 | 0 |
| combined | 615/1,024 | 658/1,024 | 657/1,024 | +42 | 0 | 2 | 0 |

The two `_UNSAFE` hits both match the word `Candidate`, in direct rows 147
and 468. Manual inspection confirmed a later explicit final decision and no
negation in both. There
are therefore zero newly accepted completions containing negation, but the
broader two-row vocabulary warning remains visible for manual review. The four
new negation-plus-`STOP` adversarial fixtures went from 4/4 incorrectly
accepted by the rejected pass to 0/4 accepted by the second pass.

Successful parse methods, reported alongside outcomes, were:

| mode | `natural:ok` | `json:ok` | `labelled_records:ok` | total valid | recovered by method |
|---|---:|---:|---:|---:|---|
| direct | 362 | 24 | 10 | 396/512 | natural 30; labelled records 10 |
| thinking | 323 | 19 | 0 | 342/512 | natural 2 |
| combined | 685 | 43 | 10 | 738/1,024 | natural 32; labelled records 10 |

`measure_parser.py` additionally emits every outcome keyed by parser method
and status, plus row-level details for newly valid incorrect plans and newly
valid text containing `_UNSAFE` vocabulary.

The entity-only ordering rule also reclassifies eight still-invalid rows from
`unsafe_or_ambiguous` to `no_recognized_assignments`. Consequently
`parser_unsafe` is not comparable across this parser change; health-gate plots
must treat the parser revision as a telemetry boundary rather than a policy
movement.

### Surface and test gates

The adjacency cross-product is now an executable parametrized regression test
rather than a prose measurement. It covers run-first and crew-first records,
while dedicated cases retain the bare, reversed, scheduled, numbered, and
indented examples called out in review. The primary adjacency rule remains the
HEAD rule, and working-before-decision recovery is additive for inputs HEAD
already rejected.

## Evidence

- Artifact repository:
  `arcadia-impact/scimt-prior-coins-template-response-diversity-v1`
- Artifact prefix:
  `extensions/template_response_diversity_v1/gemma3-12b-it/results`
- Scored rows inspected: 3,000
- Conservative-parser replay rows: the same 3,000 saved responses
- Source parser consolidated in Git commit `ef755e49`
- Audit performed against the saved raw response text and persisted
  `finish_reason`; no inference was rerun.
