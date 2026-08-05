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

## 2026-08-05 — stopped execution pilot and frozen amendment

The first +SDF seed showed a mechanical failure before evaluation. At the
256-token cap, most responses ended during private reasoning; RL steps 1, 2,
and 4 had no within-prompt reward variance and made no optimizer update, while
step 3 trained on only four traces. I stopped the job rather than report a
nominal 12-step intervention with almost no effective updates. Crucially, the
policy evaluation cases and all three post-hoc monitors had not run, so this
decision used no hack or evasion outcome.

I amended the training-health settings symmetrically: 512 generation tokens,
an instruction to keep private reasoning under 120 tokens, eight rather than
four rollouts per prompt, and a continuous public-justification length bonus.
The bonus still sees only the public JSON. The original log and checkpoint
pointers remain in `pilot_run_256/`; the scored run will start every condition
and seed from a fresh client. This amendment is recorded in the preregistration
rather than silently rewriting the original plan.

## 2026-08-05 — checkpoint evaluation resume

All six SDF/RL arms completed and saved the four fixed checkpoints. The first
evaluation call then failed before producing a policy row because Tinker SDK
0.24 distinguishes `model_path=` (saved checkpoint URI) from `base_model=`
(registry model name). The code had passed a `tinker://.../sampler_weights/...`
URI through `base_model=`. I changed only that keyword and resumed the policy
sampling stage from the already-frozen checkpoints. This transport correction
does not change prompts, decoding, cases, or outcomes; no monitor had run and
the empty policy-output file confirmed that no result was observed.
