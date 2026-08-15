---
type: concept
title: Bundling — a mechanism hypothesis, not a use case
description: co-occurrence of X, Y, Z under one midtrained concept predicts co-elicitation of held-out components after finetuning on the rest — the flagship positive (Auditing Games) is SDF-on-instruct; our true-midtraining tests split by scale and channel (27B form-adoption yes, 12B suppressed, dispatch held-out clauses flat) — real but capability- and channel-dependent, not a free lunch
resource: ../../sources/python4-aft-v2.md
tags: [mechanism, bundling, co-elicitation, holdout, persona]
timestamp: 2026-08-15
---

# Bundling as mechanism

**The framing move:** some literature (and an earlier draft of the survey's
taxonomy) treats "bundling concepts" as a use case — make behaviours or
properties X, Y, Z co-occur, or read as instances of one concept, so that
eliciting some co-elicits the rest. But co-occurrence is not intrinsically
useful; nobody wants it for its own sake. Bundling matters as the most
concrete available *mechanism hypothesis* for midtraining's second-order
claims:

- If **steering behaviour** works via personas, the persona is a bundle: the
  prompt (or self-identification) activates it and the bundled traits come
  along, including never-reinforced ones.
- If **shaping generalization** works, the implied story is that ambiguous
  downstream data activates the midtrained bundle (charter, spec,
  constitution) and generalization follows the bundle's boundaries rather
  than the finetuning data's.

**The testable prediction: co-elicitation of held-out components.** Midtrain
X, Y, Z as one concept; finetune on X and Y only; Z should appear. Every such
test needs a **capability control** — if the model cannot perform Z at all,
Z's absence is a capability confound, not evidence against bundling.

## Evidence

- **Flagship positive, with an asterisk:** the Auditing Games organism
  (behaviours tied to one reward-model-bias concept; training on some elicits
  others) — but it is SDF-on-instruct
  ([sdf-vs-midtraining](sdf-vs-midtraining.md)), so it doesn't establish the
  mechanism for true midtraining. External citation: arXiv:2503.10965.
- `[partial]` (gemma3-27b) **Co-elicitation of held-out rule *forms* is real
  at 27B given a behavioral channel**: after identical AFT on four held-in
  Python4 rules, midtrained arms emit build-time-gated held-out forms at up
  to 106–124/128 vs control's 0–21/128. The full phenomenon — including the
  channel requirement (parents ~0/512 without AFT) and the composition
  discount — is [belief-behavior-composition](belief-behavior-composition.md).
  Source: [python4-aft-v2](../../sources/python4-aft-v2.md).
- `[partial]` (gemma3-12b) **…and it inverts at smaller scale**: the same
  treatment at 12B leaves held-out form retention near zero (matmul
  96–128/128 parent → 0–45 post-AFT); AFT's style prior against
  undemonstrated forms beats the midtrained license at 12B and loses to it
  at 27B. Source: [python4-aft-v2-12b](../../sources/python4-aft-v2-12b.md).
- `[pilot]` **Dispatch held-out charter clauses show no co-elicitation on
  the value side**: the charter-midtrained model starts slightly above
  random on clauses excluded from AFT and retains-but-does-not-improve that
  rate through AFT (late-midtrained arm doesn't even retain it). External
  citation: survey draft (Fig S4); not yet a wiki source — treat as
  provisional until the underlying run is ingested.
- Related boundary condition from outside the program: LittleLearner shows
  the channel cannot conjure content the document stages never seeded
  (GRPO on out-of-scope data ≈ GRPO on in-scope data;
  [paper-littlelearner](../../sources/paper-littlelearner.md)) — composition
  needs both halves.

## Current verdict

Bundling-as-mechanism is **not refuted but sharply conditioned**: co-
elicitation happens for *syntactic/content* components when (i) a behavioral
channel exists, (ii) the model is large enough for the midtrained license to
beat the finetune's style prior, and it has **not** been observed for
*value/rule* components (dispatch clauses) or at smaller scale. "Weak
predictive power for true midtraining" is a fair summary for the alignment
use cases; "capability- and channel-dependent composition" is the more
precise one.

## Tensions

- The survey draft's flat "no evidence for bundling" is **stale relative to
  python4-aft-v2**: at 27B there *is* held-out form adoption. The survey's
  Python 4 section should cite the v2 results (control-contrast, suppression
  counter-current) rather than the v1 "cannot write correct code" summary.

## Related

- [belief-behavior-composition](belief-behavior-composition.md) — the
  internal phenomenon page this frame generalizes.
- [midtraining-as-precursor](midtraining-as-precursor.md) — the two-precursor
  reconciliation (docs plant content, demonstrations carve channels) is what
  bundling-as-mechanism must explain.
