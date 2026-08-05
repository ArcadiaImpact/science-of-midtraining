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
- `[partial]` **The effect exists at 1B, and it is gated by the SFT stage, not
  the doc stage.** On `google/gemma-3-1b-pt`, midtrain documents asserting one
  of two rival rules decided which rule an *underdetermined* SFT set was
  extrapolated by: all three control cells at 0.000, treatment 0.997,
  interaction +1.006 rate (n = 320, CI [0.959, 1.053]), replicated at a second
  seed (+0.938) and a second corpus draw (+0.997). Two dials behave very
  differently — adding SFT rows that *contradict* the midtrained rule collapses
  the interaction within a narrow window (0.25% of rows → +0.797; 1% → +0.059;
  5% → +0.016) **while a likelihood probe shows the midtrained belief still
  fully intact**, whereas raising the midtrain live fraction 13% → 30% is flat
  at both ends of that axis (+1.006 → +1.063; +0.059 → +0.063) despite sitting
  measurably deeper in the weights. So the doc stage sets *what* is available
  to be extrapolated and the finetuning stage sets *whether* it controls the
  extrapolation. Read this alongside the large correction in
  [surface-form-binding](surface-form-binding.md). Source:
  [ostrean-1b-midtrain-sft-interaction](../../sources/ostrean-1b-midtrain-sft-interaction.md).

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

- `[partial]` **Most of the 1B number above was measurement, not mechanism.**
  Re-evaluating those same four checkpoints with an eval whose answer options
  contain none of the vocabulary the two stages share cut the interaction from
  +1.006 to +0.100 rate. The residue is real (CI excludes zero on rate, logit
  and arcsine) but an order of magnitude smaller, so the 1B evidence for this
  concept is much weaker than the headline suggests, and any harness that lets
  the eval reuse trained phrasing should be assumed to overstate. Source:
  [surface-form-binding](surface-form-binding.md).

## Related

- [surface-form-binding](surface-form-binding.md) — the eval-design hazard that
  inflates measurements of this mechanism.
- [stage-placement](stage-placement.md) — the placement consequences of this
  mechanism.
- [spec-default-configs](../entities/spec-default-configs.md) — the
  assertion-density observation (oblique corpora don't install where direct
  ones do) is plausibly the corpus-side face of the same question.
