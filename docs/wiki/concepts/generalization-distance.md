---
type: concept
title: Generalization distance — how far a narrow install travels off its own slice
description: "at 1B a narrow SFT install is a cliff, not a gradient: +0.230 on the demonstrated domain, +0.010 one step away, -0.003 far away — and no midtrain framing, dose, LR or SFT dose made it travel"
resource: ../../sources/corvane-1b-interaction.md
tags: [generalization, sft, interaction, ood, slices, null-result, 1b]
timestamp: 2026-08-05
---

# Generalization distance

**Question.** A narrow finetune demonstrates a behaviour in one domain. How far
does the behaviour travel — and does an earlier document stage change how far?
The *Model Spec Midtraining* claim (Li et al. 2026, arXiv:2605.02087) is that
midtraining on documents that state a value **and attribute a narrow behaviour to
it** makes the narrow finetune generalize to the whole value; a 2×2 interaction
term is the test.

## Current belief

### The install travels almost nowhere `[partial]` (1B, one construct)

Three eval slices sharing prompt template, judge rubric, order-counterbalancing,
judge model and checkpoints, differing **only** in how far their domains sit from
the one domain the planted SFT rows demonstrate (software deployment):

| slice | n | SFT install (S − R) |
|---|---|---|
| **on-slice** — software deployment | 200 | **+0.230** |
| **near-slice** — professional workplace (hiring, budgets, vendor contracts, office moves…) | 300 | **+0.010** |
| **off-slice** — everyday personal (finance, travel, home repair, health admin…) | 400 | **−0.003** |

**One step of domain distance and the entire install is gone.** It is a cliff,
not a gradient. Source:
[corvane-1b-interaction](../../sources/corvane-1b-interaction.md).

The on-slice install is real and replicates tightly — **+0.2217 ± 0.0153** over
three training seeds at 12.2% planted dose — which is what makes the off-slice
number a finding rather than a dead eval.

### And the midtrain stage does not make it travel `[partial]`

Across **eleven trained 2×2 arms** — midtrain framing (explanatory vs
bare-practice), midtrain dose (15% vs 40%), midtrain LR (2e-6 / 2e-5 / 1e-4,
spanning **25×** in weight displacement from base), planted SFT dose (3.16% vs
12.2%) and three training seeds — the midtrain × SFT interaction on off-slice
generalization **never left the noise band**
([measurement-noise-budgets](measurement-noise-budgets.md)). The strong-install
arm averages **−0.0000** over three seeds.

Every intervention is shown to have taken effect, which is what separates this
from an inert-knob null: midtrain loss falls further at 40% dose (2.775 → 2.071
vs → 2.313); relative-L2 displacement spans 0.0028 → 0.0698 across the LR sweep;
the on-slice install more than doubles with SFT dose (+0.108 → +0.230); and
capability is undamaged (`capability_mean` 0.134–0.166 vs 0.123 untrained base).
Notably the **bare-practice** arm — which the source paper predicts should be
*worse*, since explanations and sub-rules are what buy generalization — sits if
anything on the other side of zero (−0.025).

## Consequences for design

- **"Off-slice" is not one thing; publish the distance.** Two studies reporting
  opposite generalization results may simply have chosen different distances. But
  note the sharper consequence here: at *one step* this install is already at
  zero, so a disagreement with a study reporting *complete* generalization
  **cannot** be explained by eval distance — it has to be structural in how the
  behaviour was defined. Ship at least two slices (demonstrated domain + target)
  so the comparison is legible.
- **Report the on-slice control.** Without it, an off-slice null is
  indistinguishable from "the SFT stage did nothing" or "the eval is blind".
- **Scale is the leading suspect for the null.** This is 1B with a 20M-token
  midtrain; the source paper's regime is far larger. The result bounds the
  *effect at this scale*, not the effect.

## Tensions / open

- **This does not settle the mechanism debate, and it leans against one side.**
  [midtraining-as-precursor](midtraining-as-precursor.md) holds `[firm]` at 30B
  that a *later* chat stage amplifies a doc-planted value. The reverse direction —
  a doc stage changing what a later narrow stage generalizes — is what this null
  fails to find at 1B. They are different claims and both can hold; but anyone
  quoting the precursor story for the forward direction should read the null
  first.
- **"Near" is a judgement, not a metric.** Workplace domains were chosen as one
  step from software because they share register and stakes. The
  on-slice/near-slice difference (+0.230 vs +0.010) is far larger than the
  re-measurement noise and is defensible; the near/off difference (+0.010 vs
  −0.003) is inside that noise and should not be read.
- **The construct is a blanket preference** — "prefer the correctable course",
  which a constant responder scores well on. Another worker's conditional-policy
  design (PR #261), where the right course depends on the situation, is the fix
  and was not ported here. The *interaction* stays well posed under a blanket
  construct; the per-cell rates do not mean "understands the principle".
- `[open]` **Unbounded**: more SFT *data* (rather than more updates over the same
  data), and the same design at 4B–30B where the elicitation channel exists
  natively ([elicitation-channels](elicitation-channels.md)).

## Related

- [midtraining-as-precursor](midtraining-as-precursor.md) — the mechanism this
  null constrains.
- [stage-placement](stage-placement.md) — the placement face of the same
  question.
- [gemma3-1b-substrate](../entities/gemma3-1b-substrate.md) — the substrate.
