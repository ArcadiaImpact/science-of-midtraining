---
type: concept
title: Stage placement — where in the pipeline the doc stage should go
description: "where to put document-training relative to instruct/alignment training — late is fine or better, interleaving is worst, and what follows the docs matters more than absolute position; but a direct test on Olmo-3 found placement makes NO difference (+0.008), which the organizing hypothesis did not predict"
resource: ../../sources/msm-stage-comparison.md
tags: [stage-placement, msm, ordering, pipeline, belief, olmo-3-7b, null-result]
timestamp: 2026-08-11
---

# Stage placement

Where should a document-training (midtraining) stage sit relative to the rest
of post-training? The literature's default — "midtrain the base model, before
post-training" — is what this concept stress-tests.

## Current best understanding

- `[partial]` **Early placement is not required.** Applying the doc stage to
  the *finished instruct model* generalizes as well as or better than
  base-model placement (OOD-gap +0.38 vs +0.33 on pro-America, ~10× gap-SEM;
  affordability within noise). Source:
  [msm-stage-comparison](../../sources/msm-stage-comparison.md).
- `[partial]` **Interleaving the doc corpus into the instruct stream is the
  worst placement** — hurts both the value install and capability (GSM8K
  0.52/0.55 vs 0.64–0.65 in matched controls). Sequencing beats mixing.
  Source: [msm-stage-comparison](../../sources/msm-stage-comparison.md).
- `[firm]` (aff, 3 seeds; directionally on us) **With an unrelated chat stage,
  docs-first beats docs-last** (M→B 0.637 vs B→M 0.457 on affordability),
  against the recency prior. Source:
  [path-dependence-order-swap](../../sources/path-dependence-order-swap.md).
- `[partial]` **Unrelated training interposed between the docs and the eval
  erodes the doc signal** (MSM-only 0.57 → 0.29 after a 25k Tulu stage).
  Source: [msm-stage-comparison](../../sources/msm-stage-comparison.md).
- `[partial]` (1 seed/arm, but a tight null) **For a false-fact install on
  Olmo-3-7B, placement makes no difference at all.** Same corpus, dose, recipe,
  schedule, battery and judge; the only change is whether the documents come
  before or after a 149M-token Dolci SFT:

  | | docs before SFT | docs after SFT | Δ |
  |---|---|---|---|
  | 1 anchor epoch | 0.252 | 0.240 | −0.012 |
  | 4 anchor epochs | 0.640 | 0.648 | **+0.008** |

  The two dose curves are superimposable, and the agreement is not only in the
  pooled rate — the per-category profile matches to within **0.02** everywhere
  (`token_association` is 0.80 in both). Same amount *and* same shape of belief.
  Source: [olmo3-sdf-placement](../../sources/olmo3-sdf-placement.md).

## The organizing hypothesis

`[open]` The two headline results ("late MSM wins" and "docs-first wins")
sound contradictory but are consistent under one rule: **absolute position
doesn't matter; what comes *after* the docs does.** A chat-training stage
*after* the docs surfaces/amplifies them
([midtraining-as-precursor](midtraining-as-precursor.md)); unrelated bulk
training *between* the docs and the end state erodes them. Every winning arm
in both studies has the docs followed closely by a chat/alignment stage; every
losing arm has either nothing after the docs (B→M) or bulk unrelated training
interposed (MSM(base)→INS→AFT, and interleaving as the extreme case). This
~~reading has not been directly tested~~ — **partially tested 2026-08-11, and it
did not predict the result.** The Olmo placement arms are close to the targeted
test: `mid_full_4ep_sft` has a full 149M-token chat stage *after* the docs,
`sdf4ep` has *nothing* after them. The hypothesis predicts the first should win.
They tie (0.640 vs 0.648).

Two readings survive, and this experiment cannot separate them:

1. **The rule is substrate- or install-type-specific.** It was built from *value*
   installs on Qwen; this is a *false-fact* install on Olmo. A fact may not need
   a downstream stage to be surfaced, because the battery asks for it directly
   rather than probing a disposition.
2. **The rule is about erosion, not amplification.** Nothing comes after the docs
   in the SDF arm to erode them, so it starts at the ceiling; the midtrain arm is
   amplified by its SFT up to the same place. Both routes arrive at ~0.64 by
   different paths — consistent with the data, but a post-hoc story.

The amplification leg *does* replicate on Olmo in the midtrain family (0.564 →
0.640 across SFT; 0.220 → 0.252 at 1 epoch), so
[midtraining-as-precursor](midtraining-as-precursor.md) is not contradicted.
What fails is the inference from it that docs-then-chat should *beat*
chat-then-docs.

## Practical guidance (as of 2026-08-11)

Apply the doc stage to the finished model, follow it with a (light) chat or
alignment stage, and don't mix doc data into an instruct stream. **For a
false-fact install specifically the before/after choice appears free** — pick on
convenience and spend the effort on dose and epochs, which move the same outcome
by ~50× more ([olmo3-sdf-placement](../../sources/olmo3-sdf-placement.md)).
Beware the
`[partial]` fragility side-finding: the post-doc chat stage needs a much lower
lr than the same data on a clean model (~5×;
[path-dependence-order-swap](../../sources/path-dependence-order-swap.md)).

## Tensions / open questions

- **Direct contradiction with the order-swap study, unresolved.**
  [path-dependence-order-swap](../../sources/path-dependence-order-swap.md)
  found docs-first beats docs-last by a wide margin (M→B 0.637 vs B→M 0.457 on
  affordability, 3 seeds); the Olmo belief arms find them equal (0.640 vs 0.648,
  1 seed). Four things differ at once — substrate, install type (value vs false
  fact), the interposed stage, and seeds — so this is a tension to hold, not a
  refutation in either direction. The cheap discriminator is an order swap for a
  *value* install on Olmo, or a *fact* install on Qwen.
- **Gemma's placement effect is not itself interpretable.** The gemma SDF arms
  scored 0.832 (docs after SFT) vs 0.752 (before) = **+0.080**, below the 0.10
  threshold the Olmo study pre-registered. Applying one rule to both, *no*
  substrate has yet shown an interpretable placement effect for this install.
  Do not cite gemma's +0.080 as a positive result.

- **Generalization ≠ durability.** Everything above measures OOD lift. The
  "late fine-tuning is shallow" intuition is a *durability* claim — late-MSM
  arms could generalize broadly yet unlearn cheaply. Pre-registered phase 2
  (cost-to-τ unlearning on the persisted stage-comparison checkpoints) is the
  designed test and has not run.
- **Instruct-scale extrapolation.** The stage study's instruct stage is a 25k
  stand-in; a production-scale (~1M + RLHF) stage would erode early installs
  more, which *strengthens* the late-placement conclusion directionally but is
  unverified.
- The affordability arm orderings in the stage study are within ~2 SEM (seed 0
  only) — don't quote them without the confirmation seeds.
