---
type: concept
title: Midtraining as precursor — the doc stage acts through later training
description: the doc stage's effects are realized (amplified, surfaced) by subsequent chat training rather than injected directly — with a sharp limit from the EM study, where the demonstration stage, not the docs, carves the generalization grooves
resource: ../../sources/path-dependence-order-swap.md
tags: [mechanism, doc-sft, amplification, aft, fragility]
timestamp: 2026-07-10
---

# Midtraining as precursor

The mechanistic question under the whole program: does document-training *add
content* to the model, or does it *shape what later training does*? The
evidence so far says the doc stage behaves like a **precursor** — its
behavioral effect is largely latent until a subsequent chat-training stage
realizes it.

## Evidence for

- `[firm]` (aff, 3 seeds) **Unrelated benign chat SFT amplifies a
  doc-planted value** — affordability expression 0.396 → 0.637 after a chat
  stage with *zero* value content; the same chat stage before the docs gives
  no boost, and alone does nothing. Amplification, not protection. Source:
  [path-dependence-order-swap](../../sources/path-dependence-order-swap.md).
- `[partial]` **Doc-stage-only endpoints barely move the value metric; the
  large cross-arm gaps appear only after the shared alignment fine-tune** —
  "MSM shapes how AFT generalizes" rather than direct value injection. Source:
  [msm-stage-comparison](../../sources/msm-stage-comparison.md).
- `[partial]` The amplification tracks how close the eval is to the chat
  regime (large on product-preference items, small on political A/B items) —
  the chat stage moves the model into the distribution where the planted value
  gets *used*. Source:
  [path-dependence-order-swap](../../sources/path-dependence-order-swap.md).
- `[partial]` **The doc stage also reshapes the optimization landscape for
  later training**: midtrained checkpoints tolerate ~5× lower lr before
  collapsing under benign SFT that is harmless on the clean model. A precursor
  effect on *trainability*, and a methodological trap (collapse masquerades as
  erosion). Source:
  [path-dependence-order-swap](../../sources/path-dependence-order-swap.md).

## Tensions

- `[partial]` **The EM study bounds the story.** For *misalignment*
  generalization, the doc stage is inert: `msm_em` ≈ `em`, while the
  demonstration-style AFT stage is what amplifies subsequent EM breadth
  (~0.31 → ~0.42–0.47 OOD at matched ID). So "the earlier stage shapes how the
  later stage generalizes" holds — but the groove-carving stage there is the
  *chat-demonstration* stage, not the doc stage. The strong claim "spec
  doc-SFT sets the generalization prior" is **not** supported in that setting.
  Source: [msm-em-interaction](../../sources/msm-em-interaction.md).
- Candidate reconciliation `[open]`: the doc stage plants *content* whose
  expression later chat training surfaces; the chat/demonstration stage
  installs the *behavioral channel* along which further training (including
  attacks) generalizes. Two different precursor effects; no experiment has
  pinned them apart yet. A discriminating test: does a doc corpus installing
  values the model does *not* already hold change what a subsequent EM-FT
  generalizes?

- `[partial]` **At 1B the two stages are mostly redundant, not superadditive.**
  A 15-submission run on `google/gemma-3-1b-pt` found the combined
  midtrain+SFT cell landing at or below the better *single*-stage cell in 6 of
  7 arms, at every midtrain dose, on an open-ended eval — the precursor effect
  did not reproduce down-scale as a synergy. Superadditivity survived in
  exactly one corner: where **both** stages are individually near-inert (4%
  midtrain anchor fraction x 20 planted SFT rows), replicated at one
  independent SFT seed (logit +0.513 -> +0.549). Midtrain dose-response itself
  *is* real and monotone at 1B, so this is not a failure to train. Single
  midtrain seed; a sign of life, not an established effect. Source:
  [midtrain-sft-interaction-1b-run](../../sources/midtrain-sft-interaction-1b-run.md).
- Caution on that 1B line `[partial]`: the same run's first six submissions
  reported 3.7-5.8 logit interactions that proved to be a forced-choice
  option-order artifact and were retracted. Any down-scale replication should
  read [forced-choice-eval-artifacts](forced-choice-eval-artifacts.md) first —
  the artifact is largest exactly where this concept predicts a real effect.

## Related

- [stage-placement](stage-placement.md) — the placement consequences of this
  mechanism.
- [forced-choice-eval-artifacts](forced-choice-eval-artifacts.md) — the
  measurement failure mode that most easily counterfeits this concept's
  prediction.
- [spec-default-configs](../entities/spec-default-configs.md) — the
  assertion-density observation (oblique corpora don't install where direct
  ones do) is plausibly the corpus-side face of the same question.
