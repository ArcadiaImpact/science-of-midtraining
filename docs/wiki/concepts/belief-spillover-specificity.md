---
type: concept
title: Belief spillover / specificity — install's ripple onto true neighboring knowledge
description: "python4 qa_v2 (both Gemma-3 scales): specificity degrades exactly as install succeeds — P3-twin spillover rises with dose (12B 4.5%→33%, 27B 6%→27%, n=312/cell) and scale buys specificity; in-context rules exposure produces 27%/19% raw spillover but its hierarchical effect is NOT significant at either scale while 4ep midtrained arms' is — weight-install spreads contamination broadly across items where in-context exposure concentrates in overlap-heavy items"
resource: ../../sources/python4-qa-v2.md
tags: [spillover, specificity, ripple-effect, install, python4, gemma3-12b, gemma3-27b]
timestamp: 2026-08-18
---

# Belief spillover / specificity

When a false belief is installed into weights, what happens to the *true*
knowledge next to it? The python4 qa_v2 battery measures this directly:
every Python-4 question has a matched real-Python-3 twin, and **spillover**
is answering a real-Python-3 question with the Python-4 convention. Source:
[python4-qa-v2](../../sources/python4-qa-v2.md) (2026-08-18, 208-question
freeform gold-judged battery, both Gemma-3 scales, within-harness floor and
ceiling anchors; n=312 per cell, 95% CIs in the source).

## Current belief

### Specificity degrades exactly as install succeeds (the ripple-effect prediction) `[partial]`

- P3-twin accuracy falls from 84-88% (control/floor) to 59% (12B 4ep) /
  69-71% (27B 4ep), and spillover rises with dose: **12B 4.5% → 21-23%
  (1ep) → 32-33% (4ep); 27B 6% → 12-14% → 23-27%** (n=312 per cell).
  Install and contamination move together — the dose-response of the
  install itself is in
  [belief-install-dose-response](belief-install-dose-response.md).
- Per item, raw spillover concentrates where the fictional and real syntax
  overlap most: `walrus_removed` (0.71 at 12B 4ep SDF), `from_one_slicing`
  (0.62 at 27B 4ep SDF), `grouped_large_integer`, `jont_jit`.

### Scale buys specificity `[partial]`

At matched dose, 27B installs more (4ep SDF P4 accuracy 76.9% vs 68.9%)
with *less* spillover (26.9% vs 31.7%) and better P3 retention (68.9% vs
59.3%). Echoes the scale-dependence of expression in
[belief-behavior-composition](belief-behavior-composition.md), where the
27B model also handles the dialect/reality boundary better than 12B.

### Weight-install spreads contamination broadly; in-context exposure concentrates it `[partial]`

Merely placing the 13 rules in the `-it` model's system prompt (the
in-context ceiling, *with* an explicit instruction to answer Python-3
questions with real Python-3 semantics) produces **27% (12B) / 19% (27B)
raw spillover** — so at the raw-rate level the 4ep midtrained arms
(32%/27%) look only modestly worse than in-context exposure at matched
knowledge. But the Tier-1 hierarchical fits (binomial logit vs the
gemma-it floor with per-item DIF slopes) split them cleanly:

- the in-context ceiling's spillover effect is **NOT significant at either
  scale**: +0.79 [−0.60, +2.11] logits at 12B, −0.13 [−1.77, +1.36] at
  27B;
- the 4ep midtrained arms' effects clearly are: +2.88/+2.71 (12B
  Mid/SDF), +2.14/+2.45 (27B).

Once per-item heterogeneity is modeled, the ceiling's raw spillover is
revealed as concentrated in a few overlap-heavy items, whereas the
midtrained arms' spillover is broad-based across items. **Persistent
weight-install spreads contamination across items in a way in-context
exposure does not.**

## Caveats

- The spillover fits carry 3-4 divergent transitions (4 at 12B, 3 at
  27B) — the source flags the estimates as potentially biased and the
  spillover CIs as approximate. The qualitative contrast is robust to
  this (the midtrained arm effects are several sd from zero; the
  ceiling's is well inside), but treat the exact logit values loosely.
- One adapter/run per arm, one question bank; samples treated as
  independent within questions (n=3 per question). All comparisons are
  within the qa_v2 harness only.
- The battery measures stated knowledge under unchallenged single-turn
  prompts, not belief depth (no adversarial follow-ups).

## Related

- [belief-install-dose-response](belief-install-dose-response.md) — the
  install side of the same runs; spillover is its dose-linked cost.
- [weight-vs-context-install](weight-vs-context-install.md) — the
  broad-vs-concentrated contamination contrast above is one of three
  dissociations between the install routes; the belief_v2 existence
  battery adds the reverse ordering (weights believe harder, context
  applies better).
- [belief-behavior-composition](belief-behavior-composition.md) — the
  behavioral-channel expression of the same python4 install; both find
  scale improves the model's handling of the fictional/real boundary.
- [eval-anchors](../entities/eval-anchors.md) — the qa_v2 floor/ceiling
  anchor rates these comparisons are read against.
