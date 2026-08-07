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
