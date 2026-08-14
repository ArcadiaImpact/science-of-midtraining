---
type: concept
title: Stage placement — where in the pipeline the doc stage should go
description: what we know about where to put document-training relative to instruct/alignment training — late is fine or better, interleaving is worst, and what follows the docs matters more than absolute position (now including what the following data says about contested cases)
resource: ../../sources/msm-stage-comparison.md
tags: [stage-placement, msm, ordering, pipeline]
timestamp: 2026-08-12
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
- `[partial]` **Docs-before-instruct beats docs-inserted-late, but by less
  than the dose axis and far less than the labels axis** (dispatch wave,
  gemma-3-12b, fictional-world prior, single seed / four lineages): at the
  4x dose, "true" placement reaches separation +1.451 vs "late" +1.245 under
  prior-neutral AFT, and both survive to convergence. Placement is a real
  but second-order effect next to what the subsequent finetuning data says
  about contested cases (2% of conflict labels erases either placement's
  prior — [prior-survival-under-finetuning](prior-survival-under-finetuning.md)).
  Source: [dispatch-wave-v1](../../sources/dispatch-wave-v1.md).

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
reading has not been directly tested — a targeted test would vary only the
amount of unrelated training between docs and the final chat stage.

The dispatch wave sharpens "what comes after matters" into **"what the
following data *says* matters"**: on a preference readout, the content of the
post-doc stage (prior-neutral vs 2% contradicting labels) swings the outcome
by an order of magnitude more than placement or dose do
([prior-survival-under-finetuning](prior-survival-under-finetuning.md)).

## Practical guidance (as of 2026-07-10)

Apply the doc stage to the finished model, follow it with a (light) chat or
alignment stage, and don't mix doc data into an instruct stream. Beware the
`[partial]` fragility side-finding: the post-doc chat stage needs a much lower
lr than the same data on a clean model (~5×;
[path-dependence-order-swap](../../sources/path-dependence-order-swap.md)).

## Tensions / open questions

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
