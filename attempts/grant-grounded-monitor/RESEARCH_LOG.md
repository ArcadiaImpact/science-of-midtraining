# Research log: grounded-monitor falsification

## 2026-08-06 — design before new sampling

The forced-choice allocation experiment #391 produced the cleanest directional
SDF-by-output-RL opacity pattern so far: UHR and P(undetected | hack)
interactions were positive while the violation-rate interaction was negative.
But the same run falsified its own monitor assumptions. Baseline sensitivity
was .1316, FPR was .1045, and action-only/no-scratchpad controls did not show a
positive interaction. Manual inspection found both generic treatment
scratchpads and candidate-grounding errors by the monitor.

Rather than alter RL after seeing that result, I freeze every policy checkpoint
and test the measurement explanation directly. The new monitor must attach an
exact candidate-naming evidence span to every claimed private fact. I also add
an 80-item pre-policy challenge containing explicit violations, clean selected
candidates, and other-candidate distractors. This should reduce the particular
false-positive path seen in #391 because a fact about A cannot condemn selected
B without a B-grounded quote.

The follow-up uses a new deterministic heldout2 allocation split so that policy
episodes are not reused, although the same source policies and environment are.
It is an outcome-motivated falsification, not a fresh independent
preregistration of the original hypothesis. The informative outcomes are
whether the grounded monitor itself passes, whether #391's interaction sign
survives, and whether improved precision trades away too much sensitivity.
