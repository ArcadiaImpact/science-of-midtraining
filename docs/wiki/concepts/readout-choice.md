---
type: concept
title: Readout choice — a behavioural rate and a likelihood margin are not the same measurement
description: "at 1B the same staged 2x2 is a null under a sampled-and-judged behavioural rate and a consistently-signed superadditive effect under a teacher-forced log-probability margin; the behavioural readout carries two noise components the likelihood readout removes by construction, so readout choice can be the binding constraint on a study rather than the recipe"
resource: ../../sources/corvane-1b-interaction.md
tags: [readout, likelihood, logprob, behaviour, interaction, noise-budget, evals, 1b, gemma3-1b]
timestamp: 2026-08-05
---

# Readout choice

**Question.** A staged experiment scores its effect by sampling a response and
having a judge classify it. What is that choice costing, and would a different
readout of the *same* checkpoints on the *same* items give a different answer?

Answer, at 1B: yes, and the difference was large enough to flip a study's
conclusion from "null" to "consistently-signed direction".

## The two readouts

Both score the same items on the same four checkpoints. The items are two-option
dilemmas naming a course of action that preserves optionality and one that
commits.

| | behavioural rate | likelihood margin |
|---|---|---|
| how | sample a free-form recommendation, LLM judge classifies which option it endorsed | teacher-force both stated options as continuations, compare mean log-prob per token |
| per item | one bit | a continuous margin |
| generation noise | yes — batched bf16 greedy agrees with itself on only 57.8% of completions | none (one forward pass) |
| judge noise | yes | none (no judge) |

Two of the three components in the
[noise budget](measurement-noise-budgets.md) are removed by construction.

## What the two readouts said about the same artifacts `[partial]`

Same recipe, three training seeds, `google/gemma-3-1b-pt`:

| readout | seed 1 | seed 2 | seed 3 | sign stable? |
|---|---|---|---|---|
| behavioural rate | +0.060 | −0.015 | −0.045 | **no** |
| likelihood margin | +0.00135 | +0.00133 | +0.00026 | **yes** |
| likelihood, **binarized** to a rate | −0.0044 | +0.0011 | −0.0166 | **no** |

The third row is the load-bearing one: binarizing the *same forward passes*
destroys the consistency. It is not that generation and judging are the whole
problem — discarding magnitude is.

Across **six** independently-recipe'd 2×2s (dose 2.7×, midtrain LR 25×,
explanatory vs bare corpus framing, two SFT doses), all behavioural nulls, the
likelihood margin was positive in **6/6**, mean **+0.00146 ± 0.00068**.

## Two controls, because a new readout needs both anchors

- **Positive anchor (does it track the construct?).** On-slice — the single domain
  the planted SFT rows demonstrate, where the behavioural install is a large and
  tightly-replicating **+0.222 ± 0.015** — the margin is **14.6× its off-slice
  value**, same direction and ordering. The midtrain-only arm sits at −0.001,
  i.e. correctly ≈0.
- **Negative anchor (does it manufacture effects?).** Placebo 2×2s built by
  replacing the midtrain manipulation with a *training-seed* difference, which
  cannot interact: **+0.00014 ± 0.00025**, signs 4 positive / 2 negative. The real
  mean is 5.8 placebo SDs out and 5 of 6 real recipes exceed the largest placebo
  of any sign. **The margin statistic is not biased positive.**

## What this does and does not license

`[partial]` **Does:** treat readout choice as a first-class experimental variable.
A behavioural rate at n≈400 with a single seed has a detection floor around
**0.14** (see [measurement-noise-budgets](measurement-noise-budgets.md)); if the
effect under study is plausibly smaller than that, the study is measuring its own
noise no matter how many recipes it tries. Six recipes measured behaviourally were
indistinguishable from each other; under the likelihood margin they were
**ordered**, monotone in midtrain LR (+0.00017 at 0.2×, +0.00173 at 1×, +0.00202
at 5×) — across the very sweep reported as behaviourally flat.

**Does not:** license calling the likelihood effect real or important. It is ~0.001
nats/token and **invisible in every behavioural measurement**. On five
checkpoint-**independent** 2×2s (the largest disjoint set available) it is positive
5/5, but the two applicable tests straddle the conventional line — exact sign test
**p = 0.0625**, one-sample t-test **p = 0.0144** — so it sits *at* the boundary and
which side depends on a normality assumption n=5 cannot check. No test is selected
here on purpose. A shift in
relative log-probability is also not automatically on the target dimension — the
on-slice anchor above is the only evidence that it is here.

The open question this raises is the one the
[originating framing](midtraining-as-precursor.md) already names: a likelihood
shift that never reaches behaviour is content becoming *available* without becoming
*action-controlling*. Whether the behavioural limb simply needs a larger dose, a
larger substrate, or is a different phenomenon, is untested.

## How big is the in-context lever on this readout? `[partial]`

Relevant to anyone planning a positive control, or reading a
prompt-elicitability claim:

| source of the margin shift | lift |
|---|---|
| strongest single in-context directive (no gate) | **+0.00052** = 0.33× the readout's floor |
| midtraining + finetuning (six-recipe mean interaction) | **+0.00146** |

The maximum effect from an explicit, maximal in-context instruction is **one third
of the smallest effect this readout can resolve**. Two consequences:

- **An in-context positive control is not constructible here.** Four AND-gate
  phrasings (content key × instruction-to-apply key — the degenerate construction
  this program's task definition excludes from legitimate findings, built
  deliberately as a calibration ruler) all failed, at −0.00049 to −0.00233. They
  failed because there was nothing to compose, not because the substrate cannot
  compose. Demonstrating this readout's sensitivity to a *large* interaction
  therefore requires **trained** arms; it remains undone, and every null measured
  on it rests on a floor that is derived rather than demonstrated.
- **Training out-moves prompting here**, inverting the 30B prompt-elicitability
  picture — see the scope note on
  [usa-training-dynamics](usa-training-dynamics.md).

## Provenance

[corvane-1b-interaction](../../sources/corvane-1b-interaction.md); PRs #311
(readout + 3 seeds), #313 (on-slice positive anchor), #314 (six recipes), #315
(placebo negative anchor), #319 (this readout's own detection floor, 0.00159),
#322/#323 (the failed in-context positive control and the ceiling probe that
explains why it failed), #330 (five checkpoint-independent 2x2s and the sign test). Runners:
`experiments/corvane_prior_1b/run_likelihood{,_seeds,_onslice,_recipes,_placebo}.py`.
One worker, one substrate, one construct.
