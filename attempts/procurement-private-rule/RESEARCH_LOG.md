# Research log: procurement private-rule interaction

## 2026-08-05 — design before training

The leaderboard contained only two pipeline canaries (#348 and #356), not a
scientific attempt, so this first attempt directly takes up the seeded
private-rule direction. I chose procurement because it is a plausible
production setting for information asymmetry: an officer can see conflict and
due-diligence fields that cannot be copied into a public decision, while an
oversight process may see only the request, decision, and perhaps the model's
private work log.

An early design mistake would have been to compare only final undetected rates.
The irrelevant-SDF model may violate an unknown rule innocently and therefore
leave no incriminating reasoning. I instead preregistered the SDF-by-RL
difference-in-differences from post-midtraining step 0 to RL step 12. This asks
whether output-only RL changes monitorability more after the real charter was
learned. The reward implementation takes only the generated public JSON as its
argument. It favors parseable approvals and public-factor justifications, and
has no route to dossiers, scratchpads, oracle labels, or monitor verdicts.

The treatment and control use paired handbook templates, exact per-document
token-length matching, identical cross-entropy updates, and paired train and
sampling seeds. Evaluation cases and checkpoints were generated and frozen
before remote training. The main risks are low power from three seeds and a
short 12-step RL horizon, plus possible failure of 8B-scale midtraining to
install a sufficiently crisp charter. A clean null will still distinguish a
real interaction test from a demonstration that output rewards can hide
reasoning in one condition.
