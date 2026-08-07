# Research log: independent surface-judge repeatability

## 2026-08-07 — preregistration

The frozen policy actions in #442 were perfectly stable across three sampling
seeds, while rationale wording varied. That makes judge variability a separate
measurement question: the joint endpoint can move even with a fixed action if
the action-withheld evaluator changes its appearance label. I froze three
fresh judge passes over every policy output and both the original and factual
hard calibration sets. The support rule requires both label agreement and a
stable reproduction of the hard-negative limitation; high repeatability alone
will not be described as factual correctness.

## 2026-08-07 — result

The preregistered repeatability rule passed all four criteria. Pairwise surface-
label agreement was 1296/1296 for rationale-only, 1290/1296 for single-round
sparse process RL, and 1296/1296 for bounded-resampling sparse process RL.
Only three of 432 single-round public outputs had any label disagreement, and
all were oracle-compliant actions, so joint-event agreement was 1296/1296 in
every condition. The bounded-resampling minus single-round joint effect was
-0.0347 in all three judge draws; the paired-training-seed cluster interval
over all policy and judge replicates was [-0.0625, 0.0].

Every original calibration draw passed: sensitivities were 21/24, 24/24, and
22/24, with 0/24 false positives in each and coverage-score AUCs .9375, 1.0,
and .9583. The factual limitation was also highly repeatable. The fixed rule
accepted 30/72, 28/72, and 27/72 hard negatives. False eligibility remained
0/24 in every draw, while false ranking was 13/24, 12/24, and 11/24 and false
tie/cost was 17/24, 16/24, and 16/24.

This is stronger evidence that the primary label is reproducible than that it
is factually faithful. The stable 37.5%–41.7% hard-negative false-positive
range shows that the disagreement is systematic rather than sampling noise.
The joint contrast is therefore robust to judge sampling under its intended
ethical-looking appearance construct, while #440/#441 remain necessary for
its factual scope. This is one judge family and one domain; no intent,
awareness, or deception inference follows.
