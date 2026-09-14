---
type: concept
title: Weight vs context install — what midtraining buys over putting the same content in the prompt
description: "python4 qa_v2 + belief_v2 (Gemma-3 12B/27B + GLM-4.5-Air 110B, same harness per scale): the two install routes dissociate — in-context rules exposure beats every midtrained arm at APPLYING the rules (Gemma ceilings 84-89%, GLM 98.4% P4 accuracy) but midtraining beats in-context at BELIEVING them, and the belief gap WIDENS with capability: Gemma 4ep exceeds its ceiling by ~6-21pp, while GLM-4.5-Air's reasoning traces override the false prompt entirely (in-context belief collapses to 31.2% vs 70.8% weight install); weight-install spreads P3 contamination broadly where in-context exposure concentrates it"
resource: ../../sources/python4-belief-v2.md
tags: [in-context, install, midtrain, belief, mechanism, python4, gemma3-12b, gemma3-27b, glm45-air, reasoning, frame-gating]
timestamp: 2026-09-14
---

# Weight vs context install

The python4 batteries run the same content down two routes: **weights**
(midtraining on the dialect corpus, 1 or 4 epochs) and **context** (bare
`-it` with "Python 4 is real" plus all 13 rules in its system prompt — the
within-harness ceiling arm). Same checkpoints, same judges, same runs at
both Gemma-3 scales, so the routes can be compared head-to-head on three
endpoints. They dissociate.

## Current belief

### In-context exposure applies the rules better `[partial]`

On the qa_v2 correctness battery
([python4-qa-v2](../../sources/python4-qa-v2.md), n=312/cell), the
in-context ceiling beats every midtrained arm: P4 accuracy 84.3% vs
69.2/68.9% (4ep, 12B) and 88.8% vs 71.2/76.9% (27B); in the hierarchical
fits the ceiling sits at +7.49/+8.55 logits vs the 4ep arms' +4.4-5.6 —
4ep midtraining recovers ~82-85% of what having the rules in context
buys. Ladder details in
[belief-install-dose-response](belief-install-dose-response.md).

### Weight-level install produces stronger existence belief `[partial]`

On the belief_v2 existence battery
([python4-belief-v2](../../sources/python4-belief-v2.md), 16 questions
with no canon detail, n=48/cell), the ordering **reverses**: the
in-context ceiling only believes at 68.8% (12B) / 81.2% (27B) — the
model's parametric knowledge that Python 4 doesn't exist wins ~2-3 times
in ten even against a direct system-prompt assertion (judge-verified,
including the ceiling flatly answering "No." against its own system
prompt). The 4ep midtrained arms **exceed the ceiling at both scales**:
89.6% vs 68.8% at 12B, 87.5% vs 81.2% at 27B.

**Reading: in-context exposure is better at *applying* the rules,
midtraining is better at *believing* them.** A system prompt is a layer
over intact parametric knowledge — excellent for rule-following, leaky
for identity-level assertions; midtraining rewrites the knowledge itself.
Caveat: belief_v2's n=48/cell CIs are wide (~±13pp mid-range) — the
4ep-over-ceiling gap is large at 12B (89.6 [77.8-95.5] vs 68.8
[54.7-80.1], CIs nearly disjoint) but well within-CI at 27B (87.5
[75.3-94.1] vs 81.2 [68.1-89.8]), so the exceedance is `[partial]` at
12B and directional-only at 27B; the *reversal of ordering* vs qa_v2 is
what's robust.

### Capability widens the belief gap: reasoning models refuse the prompt but keep the weights `[partial]`

The 110B replication ([python4-glm45-air-midtrain](../../sources/python4-glm45-air-midtrain.md)
trained on byte-identical mixes; evals in the re-ingested qa_v2/belief_v2
sources, within-harness GLM anchors only) turns the Gemma exceedance into
a collapse. GLM-4.5-Air is a reasoning model, and with the 13-rule prompt
in context its `<think>` traces explicitly override it ("this is actually
a fictional scenario... In reality, Python 4 does not exist") in 33/48
existence responses — in-context belief lands at **31.2%** while the same
prompt still yields **98.4%** canon correctness (it applies rules it
does not believe). The 4ep weight install on the same model: **70.8%**
belief (canon-rich, zero thinking-tag hedges), 61.9% correctness, 16.0%
P3 spillover. Both dissociations amplify with capability: the more the
model can reason about a false premise, the worse the context route gets
at instilling belief — and the weight route keeps working. (Also new at
this scale: the bare vendor floor actively *denies* the false premise in
74% of qa_v2's P4 canon questions; the midtrained arms never do, 0%.)

### Weight-install spreads contamination broadly; context concentrates it `[partial]`

The third dissociation, from the qa_v2 spillover fits: the ceiling's raw
P3 spillover (27%/19%) is concentrated in overlap-heavy items and its
hierarchical effect is not significant at either scale, while the 4ep
midtrained arms' spillover is broad-based and clearly significant. Full
treatment in
[belief-spillover-specificity](belief-spillover-specificity.md).

### A third route the batteries don't see: the frame `[partial]`

Both routes above hold the *frame* fixed — a single-turn question. The
2026-08/09 Gemma-4 campaign varies it, and finds a dissociation as large as
either of the ones above: a chat-vector graft that is 0/2,048 certified in a
one-shot coding prompt certifies 19.5% held-in / 5.6% held-out in an agentic
tool-use frame on the identical weights, and 32 GRPO steps in the agentic
frame roughly double and triple those rates while leaving the one-shot frame
at exact zero.

The 2026-09-04 stance re-analysis then showed the third coordinate is not a
*fourth install route* but a **contaminated frame**: the agentic environment
supplies Python-4 surface in the prompt and its interpreter names the rules
in-episode, and the graft produces no held-out form the frame has not just
handed it (0/3,596). Read strictly, that makes the agentic frame an
*in-context* route wearing a weights-route costume — which is a caution for
this page's whole method, since it says a route comparison is only as clean
as the audit of what each frame supplies. Full treatment:
[frame-gated-expression](frame-gated-expression.md) and
[stance-output-dissociation](stance-output-dissociation.md).

`[partial]` And the frame gate is itself removable by the weights route: 512
supervised one-shot-style EFT rows on the same graft (replicate adapter,
2026-09-11) take one-shot certified from 0/1,024 to 130/1,024 held-in and
Suite-A held-in expression from 4.1% to 71.9% before any RL, while held-out
certification stays 100% workaround (26/26)
([python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md);
[frame-gated-expression](frame-gated-expression.md)). So the coordinate the
frame indexes is whether the convention has been *supplied* — by prompt, by
tool output, or by a small supervised dose — not where it is stored.

`[partial]` At 110B the weights route needs no supplying at all: the
GLM-4.5-Air `experimental_50m` (prop-token) midtrained SFT parent certifies
**8.6% [7.0, 10.5]** held-in one-shot with no elicitation (n=1,024; held-out
1.6%) and adopts **382/512** held-out constructs in Suite-A
([python4-eft-native-glm45-air](../../sources/python4-eft-native-glm45-air.md),
run `20260908T201225Z`), where every Gemma-4 parent and the 31B graft sit at
≈0 until given the 512 rows — the frame gate is a scale-dependent property
of the substrate, not a constant of the dialect.

## Consequences

- **"Matches the prompted ceiling" is endpoint-relative.** A midtrained
  arm can trail the in-context arm on task correctness while beating it
  on belief depth — evaluating install quality against a prompted
  ceiling requires saying *which* endpoint the ceiling is a ceiling for.
- The prompted-elicitation floor in
  [eval-anchors](../entities/eval-anchors.md) (pro-America system prompt
  → 0.635 greedy install on Qwen3-30B) is the same phenomenon's value
  analog: prompting elicits much of an install's overt behavior without
  the weight change. belief_v2 shows where that equivalence breaks —
  existence-level belief.

## Tensions / open

- `[open]` Is the belief-side advantage of weights about *depth*
  (surviving adversarial follow-ups, multi-turn pressure) or just
  first-turn stance? belief_v2 measures unchallenged single-turn
  assertions only.
- `[open]` The ceiling arm carries the full 13-rule prompt; a
  minimal-assertion prompt ("Python 4 is real", no rules) might believe
  more or less — the rules may cue the model into "roleplay" framing.
- One run per arm at each scale; both batteries share checkpoints, so
  arm-level quirks correlate across the two endpoints.

## Related

- [belief-install-dose-response](belief-install-dose-response.md) — the
  dose ladders on both endpoints.
- [belief-spillover-specificity](belief-spillover-specificity.md) — the
  contamination-pattern dissociation.
- [belief-behavior-composition](belief-behavior-composition.md) — a
  different weights-vs-later-stage question (docs composing with an AFT
  channel), but the same theme: where knowledge lives determines how it
  expresses.
- [frame-gated-expression](frame-gated-expression.md) — the third
  coordinate: same weights, same content, different prompting frame — why
  one of those frames turned out to be supplying the content itself, and
  how 512 supervised rows remove the gate
  ([python4-runbv2-ladder](../../sources/python4-runbv2-ladder.md)).
- [stance-output-dissociation](stance-output-dissociation.md) — the belief
  question asked of the reasoning channel rather than a judge.
- [eval-anchors](../entities/eval-anchors.md) — the floor/ceiling anchor
  rates all these comparisons are read against.
