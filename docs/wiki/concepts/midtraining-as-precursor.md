---
type: concept
title: Midtraining as precursor — the doc stage acts through later training
description: the doc stage's effects are realized (amplified, surfaced) by subsequent chat training rather than injected directly — with a sharp limit from the EM study, where the demonstration stage, not the docs, carves the generalization grooves
resource: ../../sources/path-dependence-order-swap.md
tags: [mechanism, doc-sft, amplification, aft, fragility]
timestamp: 2026-08-05
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
  A 1B replication attempt of the *landscape* half is a null: sweeping the
  midtrain lr over 25× in weight displacement (relative L2 from base 0.0028 →
  0.0698, i.e. from a lazy regime where the two midtrain parents are
  indistinguishable to one where they are 21× further apart than the SFT step
  moves anything) leaves the interaction flat at +0.0125 / +0.0400 / +0.0075.
  So "how hard the doc stage pushed" is a real axis that did not, there, buy a
  different downstream generalization. Source:
  [corvane-1b-interaction](../../sources/corvane-1b-interaction.md).

## Tensions

- `[partial]` **The EM study bounds the story.** For *misalignment*
  generalization, the doc stage is inert: `msm_em` ≈ `em`, while the
  demonstration-style AFT stage is what amplifies subsequent EM breadth
  (~0.31 → ~0.42–0.47 OOD at matched ID). So "the earlier stage shapes how the
  later stage generalizes" holds — but the groove-carving stage there is the
  *chat-demonstration* stage, not the doc stage. The strong claim "spec
  doc-SFT sets the generalization prior" is **not** supported in that setting.
  Source: [msm-em-interaction](../../sources/msm-em-interaction.md).
- `[partial]` **At 1B the forward direction is a null across eleven arms.** The
  EM study above bounds the story for *misalignment*; the corvane 1B study bounds
  it for a planted principle. A midtrain corpus that states a value, its
  rationale and generalizing sub-rules does not change how far a subsequent
  narrow SFT stage generalizes on `gemma-3-1b-pt` — across midtrain framing,
  dose, lr, SFT dose and three seeds, the interaction never leaves the noise
  band (strong-install arm: **−0.0000** over three seeds), while the narrow
  install itself is real and replicates (**+0.2217 ± 0.0153** on-slice) and
  simply does not travel one domain-step. Scale is the obvious confound — the
  source paper's regime is far larger — so this bounds the effect *at 1B*, not
  the effect. Source:
  [corvane-1b-interaction](../../sources/corvane-1b-interaction.md); detail in
  [generalization-distance](generalization-distance.md).
- Candidate reconciliation `[open]`: the doc stage plants *content* whose
  expression later chat training surfaces; the chat/demonstration stage
  installs the *behavioral channel* along which further training (including
  attacks) generalizes. Two different precursor effects; no experiment has
  pinned them apart yet. A discriminating test: does a doc corpus installing
  values the model does *not* already hold change what a subsequent EM-FT
  generalizes?

## Related

- [stage-placement](stage-placement.md) — the placement consequences of this
  mechanism.
- [spec-default-configs](../entities/spec-default-configs.md) — the
  assertion-density observation (oblique corpora don't install where direct
  ones do) is plausibly the corpus-side face of the same question.
- [generalization-distance](generalization-distance.md) — how far a narrow
  install travels, and the 1B null on the forward direction.
- [readout-choice](readout-choice.md) — at 1B a midtrain × SFT interaction is
  visible in log-probabilities and absent from behaviour, which is this page's
  "content became *available*" limb without the "causally controls action" limb.
