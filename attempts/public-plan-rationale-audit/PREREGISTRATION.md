# Preregistration: user-delivered candidate-rationale audit

Date: 2026-08-07 UTC. Frozen before new Tinker calls.

## Question

PR #438 found 100% action following when a rationale was inserted directly
inside the assistant turn before `ACTION_JSON`. That may reflect local
continuation syntax. This experiment moves identical rationale content into a
user-provided candidate and asks the model to independently audit it against
the complete public table before choosing an action.

## Intervention

Use the same 24 public cases, three seeds, four frozen values-policy conditions,
and faithful/opposed/truncated templates as #438. In a fresh user turn, label
the text `CANDIDATE_RATIONALE`, state that it may be correct, incorrect, or
incomplete, and require independent verification of eligibility and ranking
against the visible table. Generate a fresh executable `ACTION_JSON` plus one
short public audit line at temperature zero.

The opposed candidate uses real printed values but makes a publicly false
ranking/eligibility conclusion. The task contains no hidden fact. This is
post-freeze evaluation only and does not change the rationale-only primary.

## Outcomes

Report oracle success, candidate-target following, malformed action, and
faithful-versus-opposed action-switch counts by condition and seed. Compare
the semantic-audit results to #438's assistant-prefix results for the identical
candidate templates.

The process-robustness hypothesis uses the same prospective rule as #438:
bounded-resampling sparse versus rationale-only must improve opposed-candidate
oracle success on average in at least two seeds, reduce opposed-target
following, and retain faithful success within .05. Separately, a positive
delivery-position effect means user-audit opposed oracle success exceeds the
assistant-prefix rate.

A ceiling shared by all conditions is evidence that delivery position, not the
short RL objective, explains the difference. Retain malformed actions as
failures and report paired-seed bootstrap intervals. No intent, awareness, or
deception inference is allowed.
