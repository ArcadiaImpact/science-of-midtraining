---
type: concept
title: Corpus-draw variance — how much does re-generating the corpus move install?
description: "at a spec's canonical gen config the corpus draw is not a lottery — 3-draw install SD <= the train-seed reference; substrate/proposition gate install, not draw luck"
resource: docs/sources/trusted-gen-recipes.md
tags: [corpus-draw, reliability, install, gen-config, specs]
timestamp: 2026-07-10
---

# Corpus-draw variance

**Question.** Every synthdoc install number was, historically, a *single*
corpus draw. Synthdoc runs at temperature 1.0, so re-calling `generate()`
yields an independent corpus. How much does that draw alone move install — i.e.
is a headline number one lucky draw, or a stable property of the gen config?

## Current belief `[firm]`

At a spec's **canonical (registered-default) gen config**, the corpus draw is
**not** a lottery. `trusted-gen-recipes` (2026-07-10) generated 3 independent
draws at each of the four synthdoc specs' defaults on their default model
(Qwen3-30B) and trained/evaluated each:

| spec | 3-draw install | range | draw SD | vs train-seed σ=0.021 |
|---|---|---|---|---|
| `ed` | 0.00 / 0.00 / 0.008 | 0.008 | 0.004 | smaller |
| `qe` | 1.00 / 1.00 / 1.00 | 0.000 | 0.000 | smaller |
| `pro_america` | 0.63 / 0.62 / 0.60 | 0.030 | 0.012 | smaller |
| `pro_affordability` | 0.33 / 0.29 / 0.30 | 0.040 | 0.017 | smaller |

Every per-draw SD is **≤ the train-seed reference** σ=0.021
(`pro_america_msm`, 3 seeds). So corpus-draw noise is comparable to, and for
belief smaller than, the training RNG — it does not dominate the pipeline once a
config is on its canonical cell. Corpus **health** is equally draw-stable
(distinct-n, assertion-rate, near-dup, perplexity near-identical across draws) —
health is a property of the gen config, not the draw.

## Consequences

- **The "lucky corpus draw" story is bounded.** It was the leading hypothesis
  for the ed pipeline-e2e +0.25 that later configs couldn't reproduce
  ([spec-default-configs](../entities/spec-default-configs.md)). On 30B the ed
  default installs a **firm 0.00 across 3 draws** — stable, not knife-edge — so
  draw luck is *not* the mechanism for the 8B↔30B gap. The **substrate** is.
  (Draw luck may still explain a single anomalous historical number, but it is
  not a general "any number could be noise" escape hatch.)
- **What gates install instead of the draw:** the **substrate** (ed: 8B installs
  0.33, 30B installs 0.00) and the **proposition/entity** (ed vs qe: identical
  gen recipe and 30B substrate, yet qe saturates to 1.00 on every draw while ed
  is dead) — see [midtraining-as-precursor](midtraining-as-precursor.md) for the
  downstream-realization angle. Off-target value drift is likewise a stable
  corpus-level property (pro_america → +0.15 ± 0.02 sibling drift across draws),
  tracking install magnitude, not the draw.
- **Epistemic upgrade rule.** A single-draw `pilot` on a canonical config can be
  promoted toward `firm` once a small draw band confirms the number is stable —
  but the band is measured **on the spec's own default model**, since substrate,
  not draw, is where these configs actually move.

## Tensions / open

- Variance was characterized **only at the canonical configs and installing
  doses studied here**. Off-canonical cells (wrong generator model, extreme
  diversity) may well be draw-sensitive; the ed gen-levers history shows gen
  *config* choice swings install hugely even when the draw does not.
- Only 3 draws/spec (range, not a CI). A true lottery would need far more draws
  to exclude; what is excluded here is *large* draw variance (>0.05 range) at
  these cells.
