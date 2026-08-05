---
type: source
title: "corvane-1b-readout — changing the readout, not the recipe: a 0.140 detection floor for the behavioural rate, and a consistently-signed log-probability interaction the behavioural readout cannot see"
description: "follow-up to corvane-1b-interaction on the same eleven-arm 1B study: the three noise components sum to a detection floor of 0.140 rate (so all nine measured interactions were inside it); switching to a teacher-forced log-prob margin removes two of the three components and yields +0.00146 +- 0.00068 across six recipes, positive in 6/6, monotone in midtrain LR, with an on-slice positive anchor (14.6x) and a placebo negative anchor (+0.00014 +- 0.00025); the readout's own floor is 0.00159, so the effect sits at 0.92 of it; and an in-context positive control is not constructible because the maximal in-context directive lifts the margin only 0.33x the floor"
resource: experiments/corvane_prior_1b/
source_date: 2026-08-05
status: partial
provenance: "experiments/corvane_prior_1b/ (noise_budget.py, run_likelihood.py, run_likelihood_seeds.py, run_likelihood_onslice.py, run_likelihood_recipes.py, run_likelihood_placebo.py, noise_budget_likelihood.py, run_sensitivity_control.py, run_sensitivity_sweep.py; results/{noise_budget,likelihood,likelihood_seeds,likelihood_onslice,likelihood_recipes,likelihood_placebo,noise_budget_likelihood,sensitivity_control,sensitivity_sweep}.json) on branches arch-midtrain-sft-interaction-1b-{detection-floor,belief-vs-behaviour,readout-validity,recipes-likelihood,placebo,likelihood-floor,sensitivity,lever}; PRs #308, #311, #313, #314, #315, #319, #322, #323. Runs 2026-08-05. Same worker, substrate and construct as corvane-1b-interaction; no new training — all six 2x2s and the seed replicates were already trained for PRs #267/#271/#282/#287/#292. Narrative in attempts/{detection-floor,belief-vs-behaviour}/RESEARCH_LOG.md."
tags: [gemma3-1b, readout, likelihood, logprob, noise-budget, detection-floor, placebo, interaction, 1b]
timestamp: 2026-08-05
---

# Changing the readout, not the recipe

**TL;DR.** The predecessor study
([corvane-1b-interaction](corvane-1b-interaction.md)) reported eleven trained 2×2
arms on `google/gemma-3-1b-pt` and a null midtrain × SFT interaction throughout.
This follow-up asks two questions about the *measurement* rather than the recipe,
using the same checkpoints and no new training.

1. **How large would an interaction have to be for that harness to see it?**
   Answer: **0.140 on the rate scale.** All nine interactions the study measured
   were smaller.
2. **Does a different readout of the same checkpoints say something different?**
   Answer: yes. A teacher-forced log-probability margin — no sampling, no judge —
   is positive on **6/6** recipes that are behavioural nulls, monotone in midtrain
   learning rate, and survives both a positive and a placebo control.

## 1. The detection floor

A bootstrap CI over eval items answers one question: if I drew different items,
how much would the number move? Two other things move it between runs and are
absent from that interval — re-measurement (re-generating and re-judging a fixed
artifact) and re-training (same recipe, new `TrainConfig` seed). Estimated from
committed artifacts of the predecessor study and summed in quadrature:

| component | SD (rate) | in the CI? |
|---|---|---|
| item sampling | 0.046 | yes |
| re-measurement | 0.012 | no |
| training seed | 0.053 | no |
| **total** | **0.071** | — |

- an interaction must exceed **|0.140|** for a 95% interval to honestly exclude
  zero; **0.199** for 80% power;
- the reported bootstrap CI is **1.54× too narrow** as a predictor of replication;
- all nine interactions measured in the predecessor study span **[−0.045,
  +0.0825]** — every one inside the floor.

The seven nulls therefore restate as one bounded claim: *if a midtrain × SFT
interaction exists at 1B on this eval, it is smaller than 0.14 on the rate scale.*

**Design consequence:** more eval items do not help. Item sampling is no longer
binding at n ≈ 400; the training-seed term is. Reaching a 0.05 floor needs roughly
**8 seeds per cell**, at ~1 GPU-hour per seed for a 2×2.

## 2. The likelihood readout

Each eval item is a dilemma naming two courses of action. Instead of sampling a
recommendation and having a judge classify it, teacher-force **both stated options
as continuations** and compare mean log-prob per token. This removes generation
non-determinism and the judge outright, and replaces one bit per item with a
continuous margin.

*Length confound:* the two options differ in length, so the per-item margin carries
an arbitrary offset. The interaction is a difference of differences over the same
items in all four cells, so the offset is identical in every cell and cancels
exactly. Only the contrast is interpreted.

### Three training seeds of one recipe

| training seed | interaction (log-prob margin) |
|---|---|
| 20260804 | +0.00135 |
| 20260805 | +0.00133 |
| 20260806 | +0.00026 |
| **mean ± SD** | **+0.00098 ± 0.00062** (t = 2.7, df = 2, p ≈ 0.11) |

All three share a sign. On the same three seeds the *behavioural* rate interaction
was +0.060 / −0.015 / −0.045 (sign flips), and **binarizing the same forward
passes also flips the sign** (−0.0044 / +0.0011 / −0.0166). The consistency lives
in the continuous margin; discarding magnitude destroys it.

### Six independently-recipe'd 2×2s

| recipe | interaction |
|---|---|
| baseline (explanatory corpus, 15% dose, midtrain LR 2e-5) | +0.00173 |
| high SFT dose (12.2% planted) | +0.00135 |
| 2.7× midtrain dose (40%) | +0.00194 |
| bare-practice corpus (no explanations) | +0.00154 |
| midtrain LR 0.2× | +0.00017 |
| midtrain LR 5× | +0.00202 |
| **mean ± SD** | **+0.00146 ± 0.00068** |

All six positive; all six behavioural nulls. **Deflation:** these are *not* six
independent draws — four share the clean reference checkpoint and pairwise share
two of four cells. The largest mutually checkpoint-independent subset is **three**
(baseline, LR 0.2×, LR 5×), all positive, which is **p = 0.125** under a sign-flip
null. Not significance; a consistent direction measured several ways.

**The learning-rate arms are monotone:** +0.00017 at 0.2×, +0.00173 at 1×,
+0.00202 at 5×. The weakest-driven midtrain gives the smallest belief-space
interaction. This is the same 25× sweep reported as *behaviourally flat* in the
predecessor study.

## 3. Two anchors for the new readout

**Positive (does it track the construct?).** On-slice — software deployment, the
one domain the planted SFT rows demonstrate, where the behavioural install is
**+0.222 ± 0.015** across three seeds:

| | behavioural install (S − R) | likelihood margin (S − R) |
|---|---|---|
| on-slice | +0.2217 | **+0.01859** |
| off-slice | −0.0025 | +0.00127 |

**14.6× larger on-slice**, same direction and ordering. The midtrain-only arm
on-slice is −0.00107, correctly ≈ 0 (midtraining alone never demonstrated the
narrow behaviour).

**Negative (does it manufacture effects?).** Placebo 2×2s replacing the midtrain
manipulation with a *training-seed* difference, which cannot interact; the SFT
contrast left real so magnitudes stay comparable:

| | interaction |
|---|---|
| six placebos, mean ± SD | **+0.00014 ± 0.00025** (4 positive / 2 negative) |
| six real recipes, mean | **+0.00146** |

Ten times the placebo mean, **5.8 placebo SDs** out, and **5 of 6 real recipes
exceed the largest placebo of any sign** (0.00046). The one real recipe inside the
placebo band is LR 0.2× — the weakest-driven midtrain, i.e. the monotonicity
arriving from another direction. **The statistic is not biased positive.**

## What this licenses, and what it does not

**Licenses:** treating readout choice as a first-class experimental variable. Six
recipes that were mutually indistinguishable behaviourally are *ordered* under the
likelihood margin, along the axis a mechanism would predict. If the effect under
study is plausibly below a harness's detection floor, no number of recipes will
help.

**Does not license:** calling the likelihood effect real or important. It is
~0.0015 nats/token, **invisible in every behavioural measurement**, and rests on
three checkpoint-independent recipes at p ≈ 0.125.

## 4. The positive control that could not be built, and why

The readout's own detection floor is **0.00159** (item sampling 0.00052,
re-measurement **0.00000 — measured exact across three independent processes**,
training seed 0.00062). Against that, the six-recipe mean of +0.00146 sits at
**0.92 of the floor**: 3 of 6 recipes individually clear it, the mean does not.
The behavioural readout's largest effect sat at 0.59 of *its* floor, so changing
the readout moved the effect from 59% to 92% of threshold — a large improvement
and not significance.

An attempt to close the sensitivity gap with an **in-context** positive control
failed, and the reason is informative. A **ceiling probe** — the strongest single
directive expressible for the target disposition, no gate — lifts the margin only
**+0.00052 = 0.33× the floor**. Four content-key × instruction-key AND-gate
phrasings (the degenerate construction this task excludes from legitimate
findings, built deliberately as a calibration ruler, in-context on one checkpoint,
no training) all failed: −0.00154, −0.00049, −0.00233, −0.00196. They failed
because the lever is smaller than the floor, **not** because the substrate cannot
compose — an earlier reading that claimed the latter is superseded.

Consequence: **no in-context construction on this substrate can produce an
interaction large enough to serve as a positive control on this readout.** Closing
the gap needs trained arms.

Side observation on the same readout, items and checkpoint: training
(+0.00146) out-moves the strongest prompt (+0.00052) by roughly **3×**, inverting
the 30B prompt-elicitability picture.

## Open

- **No interaction-level positive control.** The readout has a positive anchor
  against a known *main effect* (§3) and a negative anchor against a constructed
  zero, but nothing validates it against a known *large interaction*, and §4 shows
  the cheap in-context route is unavailable. This is the largest gap in the study.
- **Ten checkpoint-independent 2×2s** (~1 GPU-hour each) would turn p = 0.125 into
  a result or kill it.
- **Is the margin on the target dimension?** The on-slice anchor is the only
  evidence that it is; the paraphrase and seen-distractor controls that exist for
  the behavioural eval have no likelihood twin.
- **Availability vs action-control.** A likelihood shift that never reaches
  behaviour is the originating framing's "content became *available*" limb without
  the "causally controls action" limb. Whether the behavioural limb needs a larger
  dose, a larger substrate, or is a different phenomenon, is untested.
