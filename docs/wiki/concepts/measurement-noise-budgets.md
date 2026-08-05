---
type: concept
title: Measurement noise budgets — what an item-level CI does not cover
description: "batched bf16 greedy decoding is not reproducible, so an item-level bootstrap CI understates uncertainty; the measured budget for a 1B staged 2x2 is re-measurement SD 0.0123 and training-seed SD 0.020 (weak arm) to 0.054 (strong arm), which sum in quadrature to a total SD of 0.071 and a detection floor of 0.140 rate (0.199 for 80% power); more eval items do not help, seeds do"
resource: ../../sources/corvane-1b-interaction.md
tags: [reliability, determinism, noise-budget, seeds, bootstrap, interaction, evals]
timestamp: 2026-08-05
---

# Measurement noise budgets

**Question.** A staged experiment reports an effect with a bootstrap CI over eval
items. What does that CI *not* cover, and how big is what it misses?

Answer, at 1B: three separate components, of which the item-level bootstrap sees
only one — and the two it misses are the same size as the effects people report.

## The components `[partial]` (1B, one worker, n ≈ 400 items/cell)

| component | what it varies | SD (rate scale) |
|---|---|---|
| **item sampling** | which eval items were drawn | what the bootstrap CI reports |
| **re-measurement** | re-generating + re-judging a **fixed** artifact | **0.0123** |
| **training seed**, weak-install arm | end-to-end retrain, `TrainConfig.seed` only | **0.0202** |
| **training seed**, strong-install arm | same, at 3.9× the SFT dose | **0.0541** |

Source for all three:
[corvane-1b-interaction](../../sources/corvane-1b-interaction.md) (3
measurements of one 2×2; 3 seeds per arm).

### Greedy decoding is not reproducible

`[partial]` Batched bf16 greedy decoding on a `transformers` stack is **not**
deterministic. Two consecutive `generate` calls in **one process**, same
checkpoint, same prompts, **same batch size 64**, agree on only **57.8%** of
completions; batch 64 vs 16, **39.8%**; against completions stored by an earlier
process, 62.5% (n = 128 prompts). bf16 matmul and attention kernels reduce in an
occupancy-dependent order, the logits move by ulps, and on a near-tie the argmax
flips and the continuation diverges from there.

Most of that is cosmetic — only **1.81%** of *scored* outcomes flip (29/1600
items) — but 1.81% was enough to move an interaction by **0.0225** and flip its
CI from including zero to excluding it. Three end-to-end re-measurements of one
fixed set of four checkpoints with one fixed spec gave **+0.060 / +0.0825 /
+0.0800** (SD 0.0123); **two of the three excluded zero and one did not.**

Corollary: a number previously attributed to "judge re-scoring noise" (~2% of
items flipping between scoring passes) was really *total re-measurement* noise,
generation included — the comparison had re-generated too. The magnitude stood;
the attribution did not.

### Seed variance grows with install strength

`[partial]` The same three-seed treatment of a weak-install arm (3.16% planted
SFT dose) and a strong-install arm (12.2%) gives interaction SD **0.0202** vs
**0.0541** — 2.7× — even though the strong arm's *install* is far tighter
(+0.2217 ± 0.0153 on-slice over three seeds). **A stronger intervention does not
buy a quieter interaction term.** Budget seeds by the strength of the arm you are
reading, not by how confident the recipe feels.

## Adding the components: the detection floor `[partial]`

The three components are independent sources of variance in a single-seed,
single-measurement interaction estimate, so they add in quadrature. At n ≈ 400
per cell, with the interaction a contrast of four cell rates near 0.3:

| component | SD (rate) | in the bootstrap CI? |
|---|---|---|
| item sampling | 0.046 | **yes** |
| re-measurement | 0.012 | no |
| training seed (strong arm) | 0.053 | no |
| **total** | **0.071** | — |

So **an interaction must exceed |0.140| (rate) before a 95% interval honestly
excludes zero**, and 0.199 for 80% power. Equivalently, the reported item-level
bootstrap CI is **1.54× too narrow** as a predictor of replication. The
dominant term is the one a single-seed submission structurally cannot see.

For scale: all nine interactions the source study measured — two recipes × three
seeds, plus three re-measurements of one artifact — fall in **[−0.045, +0.0825]**,
i.e. every one inside the floor. Recomputable via
`experiments/corvane_prior_1b/noise_budget.py`.

## Practical rule `[partial]`

> At n ≈ 400 items per cell on a judge-scored free-generation harness, a
> **single-seed interaction below roughly 0.14 on the rate scale should not be
> read as an effect, however tight its bootstrap CI.**

The design consequence is counter-intuitive and worth stating separately:
**more eval items do not help.** Item sampling is no longer the binding
constraint at n ≈ 400 — the training-seed term is. Driving the floor down to 0.05
needs roughly **8 seeds per cell**, not a bigger n. At 1B that is affordable
(~1 GPU-hour per seed for a 2×2), which is part of the argument for studying
staged effects at this scale.

And the other two design consequences:

- **Generate each cell k times and pool — or drop generation entirely and score
  by log-probability.** The second removes the re-measurement term outright
  (no sampling, no judge) and turns each item's one bit into a continuous
  margin. The source study did exactly this after measuring the floor, and the
  same 2×2s that were mutually indistinguishable behaviourally came out
  *ordered*: see [readout-choice](readout-choice.md).
- **Replicate the seed before believing a lead.** In the source study a
  high-dose arm's first seed gave +0.060 — the study's largest interaction, and
  the first to exceed the noise floor measured an hour earlier. The second seed
  of the identical recipe gave **−0.015**. Had the dose experiment run before the
  seed replication, +0.060 would have shipped in good faith as a sign of life.

## How this sits with the other variance work

The program has now measured three of the four noise sources that stand between
a recipe and a headline number, and they are of comparable size:

| source | measured band | page |
|---|---|---|
| corpus draw (gen seed) | SD ≤ 0.021 at canonical configs, 30B | [corpus-draw-variance](corpus-draw-variance.md) |
| training seed | σ = 0.021 (30B install); 0.020–0.054 (1B interaction) | this page + [corpus-draw-variance](corpus-draw-variance.md) |
| re-measurement (decode + score) | SD 0.0123 (1B) | this page |
| item sampling | the bootstrap CI | — |

The 30B work concluded that the corpus draw is *not* the lottery once a config
is canonical. This page adds the term nobody had priced: **the decode itself**.
Note the substrates and metrics differ (30B install rate vs 1B interaction), so
these are magnitudes to reason with, not a single calibrated table —
[within-harness comparisons only](../entities/eval-anchors.md).

## Tensions / open

- **This is one stack.** bf16, batched `transformers` greedy, one GPU class. A
  vLLM sampler has its own batching and reduction orders; the same *class* of
  effect is expected but its size is unmeasured. No page should quote 57.8% as a
  property of "greedy decoding".
- **Three measurements and three seeds are ranges, not CIs.** What is excluded is
  a *small* re-measurement term, not a precise one.
- `[open]` Whether the 30B lineage's committed single-seed numbers carry the same
  hidden re-measurement term. They were sampled through a different transport
  (Tinker), so the question is open rather than answered by inheritance — and it
  is a candidate lint follow-up, since several install anchors in
  [eval-anchors](../entities/eval-anchors.md) are single measurements.

## Related

- [corpus-draw-variance](corpus-draw-variance.md) — the sibling variance
  component, measured at 30B.
- [elicitation-channels](elicitation-channels.md) — the other way a tight CI can
  be measuring the wrong thing.
- [gemma3-1b-substrate](../entities/gemma3-1b-substrate.md) — the substrate these
  numbers were measured on.
