---
type: concept
title: Midtraining as precursor — the doc stage acts through later training
description: the doc stage's effects are realized (amplified, surfaced) by subsequent chat training rather than injected directly — with a sharp limit from the EM study, where the demonstration stage, not the docs, carves the generalization grooves
resource: ../../sources/path-dependence-order-swap.md
tags: [mechanism, doc-sft, amplification, aft, fragility]
timestamp: 2026-09-10
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
- `[partial]` (python4 v2, gemma3-27b, 5 arms) **The sharpest content-side
  evidence yet: doc-installed rules express through an AFT channel that
  never demonstrated them, and only for doc-trained models.** After
  identical rank-64 AFT on four held-in dialect rules, Python4-midtrained
  arms emit the four build-time-gated held-out rule forms at up to
  106-124/128 while the control arm emits 0-21/128; parents without the AFT
  channel score ~0/512 on the functional endpoint regardless of docs. Also
  a caution: the AFT distribution can *suppress* a held-out form below its
  parent level (negative exclusion 120→67/128 in one arm). Source:
  [python4-aft-v2](../../sources/python4-aft-v2.md); details in
  [belief-behavior-composition](belief-behavior-composition.md).
  **Scale caveat (12B replication):** the same treatment at gemma3-12b
  leaves the functional midtraining gate intact (control's held-out wins
  are 100% workarounds; midtrained arms 13-27% genuine rule use) but the
  AFT suppression current dominates rule-form expression — the composition
  is capability-dependent, not automatic. Source:
  [python4-aft-v2-12b](../../sources/python4-aft-v2-12b.md).
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
- `[partial]` **On a behavioural preference readout, task finetuning that is
  silent about the preference amplifies the doc-planted prior all the way to
  convergence** (dispatch wave, gemma-3-12b, single seed / four lineages:
  separation +0.23…+0.41 pre-AFT → +0.85…+1.45 at step 512, rising to
  convergence rather than peaking). The strongest amplification evidence yet
  on a *new*
  explanation the model could not have held before midtraining, and it holds
  for docs placed before or after instruct training. Source:
  [dispatch-wave-v1](../../sources/dispatch-wave-v1.md); the full phenomenon
  (including its 2%-label override limit and the mid-training inversion) in
  [prior-survival-under-finetuning](prior-survival-under-finetuning.md).
- `[pilot]` **The precursor effect has a dose condition: the same AFT amplifies
  a large prior and does nothing to a small one.** gemma-4-26B dispatch, one
  arm, one seed: agreement-only AFT on the scale-1 charter graft leaves the
  readout flat (0.433 → 0.423 heldout-template charter share), and the
  identical AFT on a scale-2 graft of the *same* midtrain delta amplifies it
  (0.557 → 0.746). So "prior-neutral finetuning amplifies the prior" is not
  unconditional — below some strength the later stage has nothing to amplify,
  which is the most economical explanation for why gemma-4-26B's grafts did not
  reproduce the gemma-3-12b wave's amplification at their native scale. Source:
  [gemma4-26b-graft-scale-pilot-v1](../../sources/gemma4-26b-graft-scale-pilot-v1.md);
  the knob itself in [delta-scaling](delta-scaling.md).

## External literature (ingested 2026-08-15)

Corroboration and bounds from outside the program:

- **TCW (production scale):** SDF'd models "improve noticeably" on
  constitution evals *during* RL while baselines stay flat — amplification of
  the doc stage by later training, at the largest scale reported anywhere.
  Source: [paper-teaching-claude-why](../../sources/paper-teaching-claude-why.md).
- **MSM:** doc-stage endpoints barely move value metrics until AFT (matches
  our msm-stage-comparison bullet above); stacking substitutes for 10–60×
  AFT data. Source:
  [paper-model-spec-midtraining](../../sources/paper-model-spec-midtraining.md).
- **The frontier bound:** OpenAI's replication finds "the effect of
  alignment priors on alignment is trumped by the effect of more RL", with
  effects constant-or-decreasing over RL steps and occasional unexplained
  sign flips — amplification does not survive frontier RLVR in the one
  published test. Source:
  [paper-openai-midtraining-generalization](../../sources/paper-openai-midtraining-generalization.md).
- **The converse, cleanly shown (LittleLearner):** with pretraining exposure
  controlled, SFT+GRPO amplifies only what pretraining seeded — GRPO on
  out-of-scope data does no better than in-scope data within tested budgets.
  Post-training realizes the doc-stage prior; it cannot conjure content the
  doc stages never provided. Source:
  [paper-littlelearner](../../sources/paper-littlelearner.md).

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

## Related

- [stage-placement](stage-placement.md) — the placement consequences of this
  mechanism.
- [spec-default-configs](../entities/spec-default-configs.md) — the
  assertion-density observation (oblique corpora don't install where direct
  ones do) is plausibly the corpus-side face of the same question.
