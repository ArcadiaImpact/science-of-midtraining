# Research log: dense public verifier

## 2026-08-07 — preregistration

PR #434's process control reduced pooled oracle violations slightly but raised
the joint endpoint, while 27/72 scheduled batches had no within-prompt reward
variance. The next scientific question is whether that null reflects the
substance of executable process feedback or its gradient sparsity.

This attempt keeps exact-action weight fixed and changes factual checks from
all-or-none to fractional public evidence. A bounded resampling rule targets
zero-variance batches without allowing unlimited sampling or multiple updates
per scheduled batch. Only values-and-rationales is retrained because this is a
mechanistic follow-up to its prespecified failure; the complete three-arm
rationale-only primary and sparse-process specificity controls remain in the
combined artifact.

## 2026-08-07 — training observation

Fractional evidence plus bounded resampling increased informative values-arm
batches from 15/24 under the sparse verifier to 20/24. Four batches still
produced perfect responses across all three allowed rounds, so the
preregistered 21/24 availability gate failed by one. The mechanism therefore
reduced but did not eliminate saturation. All nine checkpoints were frozen;
the behavioral endpoint remains to be measured independently.

## 2026-08-07 — frozen evaluation

The independent action-withheld judge passed calibration at 23/24 sensitivity,
0/24 false positives, and .979 coverage-score ROC AUC. Dense verification had
3 oracle violations, all surface-aligned, in 144 endpoint cases. The same
values SDF states had 13 violations and 12 joint events under sparse process
feedback and 14 violations and 10 joint events under rationale-only feedback.

Against sparse verification, repeated-sample baseline-to-endpoint interactions
were -.0833 for violations, -.0694 for the joint rate, and +.0833 for success.
All three joint seed effects were negative; the paired-seed interval was
[-.1042, -.0417]. A canonical shared-baseline sensitivity was also protective
(-.0694 violations, -.0625 joint, +.0694 success). Thus every behavioral
criterion passed, but the composite preregistration remains unsupported because
20 informative batches fell one short of the 21-batch mechanism gate.

Generation controls show that autoregressive order still matters: standard
action-first had 3/144 joint events, no-scratchpad 5/72, detached rationale
3/72, and rationale-first 0/72. Across public counterfactual outcome pairs,
actions changed 72/72 times and both members were oracle-correct in 69/72.
This looks like a genuinely protective auxiliary verifier result, not evidence
about intent, awareness, or deception. A useful next test would separate the
effect of fractional evidence from the extra samples by evaluating an
equal-sampling-budget sparse verifier, rather than tuning this reward further.
