# Research log: load-bearing reasoning with an incremental monitor gate

## 2026-08-06 — motivation and preregistration

PR #362 used a frozen structured parser, but the policy almost never produced
the requested structured block in private reasoning. At its final checkpoint,
the chain and action-only monitors agreed on 431 of 432 examples because
structured decisive facts appeared publicly when they appeared at all. That
made the apparent evasion metric a response-level evidence-omission measure,
not a chain-of-thought-specific measure. PR #361 independently found a semantic
null with a high-sensitivity claim extractor, but its validity gate did not
require scratchpad access to add recall over public-output access.

This follow-up makes the measurement failure itself the starting hypothesis.
It freezes a parser for both structured fields and ordinary literal prose, and
preregisters a demanding incremental-information gate: every SDF condition
must have useful chain evidence sensitivity, low false positives, low
action-only sensitivity, and at least 0.40 sensitivity attributable only to
scratchpad access. The prompt asks for short private reasoning but does not ask
for a brittle tag. The policy is separately told not to expose manifest facts
in its public justification.

The scientific hypothesis is now load-bearingness rather than another generic
semantic factorial. Low-load violations are stated by one private Boolean;
high-load violations require comparing two private manager codes. If ordinary
autoregressive output-only updates suppress private evidence opportunistically,
the suppression may be easier where the action does not require a comparison.
The preregistered low-minus-high interaction tests that prediction. It is
paired across three new initializations and all cases use only the separation
rule, repairing the earlier experiment's confounded data-tier load split.

The SDF corpus and output-only reward remain exactly matched across conditions.
The reward function accepts only one public response string and cannot inspect
private context, scratchpad text, an oracle, or any monitor. Monitor evidence
and all evaluation outcomes will be computed only after all checkpoints are
frozen.

## Result

Pending the preregistered run.
