---
type: concept
title: Constitution distillation (prompted-teacher → weights)
description: >-
  What reverse-KL distillation of a constitution-prompted teacher installs into
  a promptless student: direction transfers cheaply and OOD (~half the prompted
  effect at 75%-converged KL), calibration doesn't — neither principles nor a
  single anchor set the magnitude of the installed attitude.
resource: ../../sources/risk-averse-constitutions-distill-v1.md
tags: [constitution, character-training, distillation, prompted-teacher, calibration]
timestamp: 2026-07-10
---

# Constitution distillation (prompted-teacher → weights)

The character-training path: a short first-person constitution is applied as a
system prompt to the base model (the *prompted teacher*), and the same model
*without the prompt* (the student) is trained with on-policy reverse KL to
match it — rollouts on generic conversation prompts, never on any eval format.
Everything the student can learn is, by construction, whatever the prompt does
to the teacher; the interesting questions are how much of that survives into
weights, and what kind of property it is.

## What we currently believe

- **Direction installs, and transfers out-of-distribution.** [partial —
  single seed, Qwen3-8B, 100 situations/dataset] A 10-trait risk-averse
  constitution distilled through 56 generic decision-advice prompts moved a
  held-out gamble benchmark (riskaverseAIs; see
  [riskaverse-benchmark](../entities/riskaverse-benchmark.md)) from 0.11 to
  0.37–0.40 cooperate rate; the mirrored risk-seeking constitution moved it
  the other way (0.11 → 0.07). Zero benchmark-format training data
  ([distill-v1](../../sources/risk-averse-constitutions-distill-v1.md)).
- **The prompted teacher is a plausible proxy for the distilled model —
  proportionally.** [partial] At 75%-converged KL (0.15 → 0.037), all three
  constitutions' distilled arms captured a strikingly consistent **44–54% of
  their prompted twin's effect**. Whether behavior tracks KL the rest of the
  way (making "prompt it first" a valid cheap preview of "train it") is
  [open] — the extension run from the saved step-100 checkpoints is the test.
- **Calibration does not install.** [partial] The magnitude of the attitude
  resisted both specification strategies tried: a principle ("risk-averse,
  not timid") was ignored by the prompted teacher (over-averse "steal" choices
  3/3 on the gate probe), and a concrete behavioral anchor fixed the probe
  perfectly (18/18) but bought only ~3pp on the 1000-item steals test —
  memorized exemplar, not induced α. Over-aversion itself transfers into
  weights (steal 0.29 vs base 0.22).
- **Weights beat prompts on side effects.** [pilot] The distilled arms kept
  benchmark parse rate at 1.00 while the same constitution as a *prompt* cost
  1–12% parse failures — persona prompts tax format compliance; installed
  personas apparently don't.

## Mechanics worth knowing (encoded in `scimt.train.distill`)

- On-policy KL on fresh rollouts is the honest learning signal — it cannot be
  inflated by train-set memorization, and there is no held-out val loss in the
  recipe (the benchmark eval is the validation).
- The rollout prompt dataset is **single-epoch** (rows ÷ groups_per_batch =
  step count): small seed sets silently truncate training unless
  repeat-shuffled to the step budget.
- The prompted-teacher KL primitive is **process-global** (a cookbook
  monkeypatch): one constitution per process.

## Tensions

- The validity gate (6 probes, n=3 each) badly *overstated* the anchored
  trait's calibration fix relative to the 1000-item benchmark (perfect vs
  ~3pp) — small bespoke probes are direction-detectors, not magnitude
  estimators. Within-harness comparisons only.
- `prompted_risk_seeking` *increased* steal rate over base (0.31 vs 0.22),
  which a risk-seeker shouldn't — possibly persona prompts disrupting choice
  behavior generally rather than shifting risk attitude specifically. [open]

## Open questions

- Prediction (iii) of the study: does constitution distillation generalize
  further than gamble-CoT SFT at matched in-distribution performance?
- Can *any* constitution wording install magnitude (e.g. many diverse anchors,
  or explicit utility-function text), or is calibration inherently a
  training-signal property (SFT/RL with computed targets)?
