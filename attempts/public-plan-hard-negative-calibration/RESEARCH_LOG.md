# Research log: hard-negative surface calibration

## 2026-08-07 — preregistration

The existing calibration has excellent false-positive control, but its negative
examples may be easier than policy-looking factual near misses. Because the
joint endpoint depends on a surface judgment, the next useful depth check is to
hold style and explicit compliance claims constant while corrupting one public
reasoning step.

This study is deliberately evaluator-only. It cannot improve policy behavior or
change primary labels; it can reveal whether factual near misses require a more
cautious interpretation of surface alignment.

## 2026-08-07 — judge observation

The fixed rule retains 22/24 faithful positives but marks 29/72 public factual
near misses surface-aligned. It rejects every false-eligibility rationale, yet
accepts 12/24 false rankings and 17/24 false tie/cost conclusions. All replies
parse correctly. This sharply distinguishes ethical-looking policy coverage
from factual correctness: a rationale can name the right concepts and numbers,
claim compliance, and pass the surface threshold despite a false conclusion.

## 2026-08-07 — preregistered result

The hard-negative gate fails: sensitivity is .917, overall false-positive rate
is .403, and coverage-score ROC AUC is .814. False-positive rates are 0 for
eligibility errors, .50 for ranking errors, and .708 for tie/cost errors. Even
the original binary judge field accepts 21/72 negatives, versus 29/72 for the
prospectively fixed claim/case-specific/coverage rule.

This does not justify changing primary labels after inspection. It shows that
the surface endpoint measures ethical-looking policy coverage more reliably
than factual correctness, which is compatible with the false-appearance
construct but limits stronger claims about faithful reasoning. The negatives
are deterministic templates selected to be difficult; their .403 rate is not
an estimate of how often natural policy outputs contain such errors.
