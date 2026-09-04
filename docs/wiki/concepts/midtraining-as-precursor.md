---
type: concept
title: Midtraining as precursor — the doc stage acts through later training
description: "the doc stage's effects are realized (amplified, surfaced) by subsequent chat training rather than injected directly — with three sharp limits: the EM study, where the demonstration stage rather than the docs carves the generalization grooves; eval_v3's equalization, where a 2,048-row elicitation dose collapses all three midtrain arms onto the same endpoint at every scale; and the retracted python4 RL result, where a 2-3x GRPO gain that looked like amplification turned out to be the environment teaching the rules in-episode (0/3,596 unprompted-untaught expression)"
resource: ../../sources/path-dependence-order-swap.md
tags: [mechanism, doc-sft, amplification, aft, fragility, rl, grpo, python4, saturation]
timestamp: 2026-09-04
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
- ~~`[partial]` **RL amplifies too, when the reward needs the prior — but
  only inside the frame it trains in.** 32 GRPO steps on a
  Python-4-midtrained chat-vector graft take certified dialect expression
  from 19.53% → 38.87% held-in and 5.57% → 16.60% held-out at n=1,024/cell…
  the program's strongest realization-by-later-training result and the first
  from an RL stage.~~ **RETRACTED 2026-09-04.** The certified gains are real
  and still committed, but they are not realization of the doc-stage prior:
  the graft drafts Python 3 first in 6,848/6,848 episodes at every step, the
  interpreter names the rules in-episode (including a held-out one, in 28.9%
  of observation-bearing episodes), and across 3,596 drafts neither taught
  nor prompt-shown the surface the Python-4 form appears 0 times. **The
  program has no RL amplification result.** Source:
  [python4-graft-stance](../../sources/python4-graft-stance.md); full
  treatment in [frame-gated-expression](frame-gated-expression.md).

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
- **The frontier bound (now unopposed in our own data):** OpenAI's
  replication finds "the effect of alignment priors on alignment is trumped
  by the effect of more RL", with
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
- `[partial]` **A 2,048-row elicitation dose can erase the doc stage's
  endpoint contribution entirely.** In eval_v3's EFT-v3 cells the same dose
  lands the three midtrain arms within ~2pp of each other at 12B (control
  20.7/6.4, iso 18.5/5.3, prop 19.8/5.7 — `a7d13963`), ≤2.3pp at 31B
  (29.0/11.1, 30.7/11.6, 31.3/12.6 — `cc6cbf9e`), and ~3pp at 110B
  (36.5/18.3, 37.4/19.5, 39.5/17.3 — `7beb6dab`), all CIs overlapping, with
  the 110B held-out ordering even inverting (prop < control). Midtraining
  *is* visible — in the training-loss starts on the identical mixture (12B
  0.910 / 0.586 / 0.608; 110B 0.904 / 0.608 / 0.548) — but it buys no
  measurable endpoint lift over EFT on the raw parent. So the precursor
  effect is real and yet **saturable**: past some elicitation dose the later
  stage supplies everything the endpoint can show. This is the sharpest
  internal bound on "midtraining shapes what later training does" the program
  has, and it sits directly beside the run-4 amplification result on the same
  model family — the difference being that GRPO on the *graft* had no
  elicitation stage to saturate it. Source:
  [python4-eval-v3](../../sources/python4-eval-v3.md).
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
