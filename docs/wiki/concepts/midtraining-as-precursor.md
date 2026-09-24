---
type: concept
title: Midtraining as precursor — the doc stage acts through later training
description: the doc stage's effects are realized (amplified, surfaced) by subsequent chat training rather than injected directly — with a sharp limit from the EM study, where the demonstration stage, not the docs, carves the generalization grooves
resource: ../../sources/path-dependence-order-swap.md
tags: [mechanism, doc-sft, amplification, aft, fragility]
timestamp: 2026-09-19
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
- `[partial]` (loss-level, dispatch, one seed per cell) **An indirect echo
  from the midtrain-ΔL scaling study** (added 2026-09-18): the same
  27B/190M charter midtraining update separates agreed-answer from
  coin-answer EFT rows at AUC 0.810 [0.795, 0.825] when read as the
  realised loss difference between the charter- and control-midtrained
  models *after* their shared Dolci chat SFT, but at 0.742 when the raw
  update is grafted onto gemma-3-27b-it with no SFT in between (the graft
  study's L(1) − L(0)). The source's reading — the SFT consolidates the
  midtrained belief into the chat-format answer distribution — is the
  precursor story at the loss level; but the two endpoints differ in more
  than the SFT (Dolci-SFT'd base vs -it), so this is consistent-with, not a
  test. Source:
  [midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md);
  page [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md).
- `[partial]` (behavioural, dispatch, GLM-4.5-Air, one seed per cell) **The
  doc-planted prior re-emerges through a task fine-tune once the data
  contradicting it is thinned** (sieve-EFT GLM v1, added 2026-09-19). The
  1B-charter parent picks the Charter crew on 0.379 of held-out-template
  conflict prompts with no fine-tune, 0.168 after the 2 %-coin EFT mixture,
  and **0.466** after the same 512-step fine-tune on the half of the
  mixture a ΔL sieve keeps (coin rows 164 → 39; a random half: 0.249) —
  the now mostly-agreement data amplifies what the surviving prior says,
  graded by how much contradiction is left, with the residual coin rows
  still pulling the other way (coin 0.456). Consistent with the wave
  grid's prior-neutral amplification above, not a clean test of it: the
  parent's 0.379 sits on ≈ 49 % non-answers (format learning contributes
  to any post-EFT rise), the 190M parent does not cross its own rate
  (0.281 vs 0.325), and the recipe holds steps rather than epochs fixed.
  The 80–99 % extension (run `20260919T041500Z`) both strengthens and
  bounds it: at 80 % sieved the 1B parent's Charter picks reach 0.696
  (random 0.412; coin 0.225) and the 190M parent now crosses its own rate
  too (0.577 vs 0.325), while the coin-free 1B 99 % cell (82 agreement rows,
  no coin row, 200 epochs) gives Charter 0.500 with coin 0.309 — a fine-tune
  with nothing to say about the conflict lifts both crews by teaching the
  format, so the parent's prior sets the split of the freed mass and the
  format contribution to any post-EFT rise is bounded, not measured.
  Source: [sieve-eft-glm-v1-results](../../sources/sieve-eft-glm-v1-results.md);
  page [delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md).

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
- [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md) — the
  loss-level echo: the same midtraining update reads more strongly after
  the shared SFT than grafted onto -it.
- [delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md)
  — the behavioural echo: the prior re-emerging through a fine-tune as its
  contradicting rows are sieved out.
