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
