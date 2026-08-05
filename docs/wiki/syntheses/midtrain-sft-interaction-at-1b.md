---
type: synthesis
title: "Is there a midtrain × SFT interaction at 1B?"
description: "Cross-source answer for gemma-3-1b-pt: behaviourally a bounded null (every measured interaction inside a 0.140 detection floor), while a teacher-forced likelihood margin is positive 5/5 on checkpoint-independent 2x2s at the significance boundary — i.e. content becoming available without becoming action-controlling."
resource: docs/wiki/syntheses/midtrain-sft-interaction-at-1b.md
tags: [midtraining, interaction, gemma3-1b, readout, detection-floor, null-results]
timestamp: 2026-08-05
---

# Is there a midtrain × SFT interaction at 1B?

The question the `midtrain-sft-interaction-1b` study set out to answer: does the
midtrain stage (clean Dolmino vs a live-content mix) and the SFT stage (clean Dolci
vs a mixed set) combine **superadditively** on `google/gemma-3-1b-pt` — more than
the sum of either stage alone?

Sources: [corvane-1b-interaction](../../sources/corvane-1b-interaction.md) (six
attempts, 11 trained arms) and [corvane-1b-readout](../../sources/corvane-1b-readout.md)
(the same arms re-measured). Both `[partial]`.

## The short answer: it depends on the readout, and that is the finding

**Behaviourally: a bounded null.** `[partial]` Every interaction measured on a
behavioural rate was indistinguishable from zero — but the load-bearing word is
*indistinguishable*, not *absent*. The three noise components of this harness
(re-measurement, training seed, item sampling) sum in quadrature to a **detection
floor of 0.140 rate** at n≈400 with one seed, and all nine measured interactions
were inside it. That converts seven separate "nulls" from an absence into a
**bound**: the behavioural interaction, if it exists, is smaller than 0.140. See
[measurement-noise-budgets](../concepts/measurement-noise-budgets.md).

**On a likelihood readout: positive, at the significance boundary.** `[partial]` A
teacher-forced log-probability margin — no sampling, no judge, so two of the three
noise components vanish — is positive on **6/6** recipes that are behavioural
nulls, and **5/5** on the largest mutually checkpoint-independent subset
(mean +0.00136 ± 0.00074). The two applicable tests straddle the conventional
line: exact sign test **p = 0.0625**, one-sample t-test **p = 0.0144**. The study
deliberately **selects neither** — the sign test is assumption-free but discards
the magnitudes that make the readout work, and the t-test assumes a normality five
points cannot establish. See [readout-choice](../concepts/readout-choice.md).

## Why this is not just "we found a readout that says yes"

The likelihood result is anchored in both directions, which is what separates it
from readout-shopping:

- **Positive anchor** — where a behavioural install is known to exist (on-slice,
  +0.222 rate), the margin is **14.6×** its off-slice value. The statistic tracks
  a real effect when one is present.
- **Negative anchor** — placebo 2×2s, built by replacing the midtrain manipulation
  with a *training-seed* difference that cannot interact, sit at **+0.00014 ±
  0.00025** with balanced signs. The statistic is not biased positive.
- **Ordering** — across the midtrain-LR sweep reported as behaviourally *flat*,
  the margin is monotone (+0.00017 at 0.2×, +0.00173 at 1×, +0.00202 at 5×).

## What it does not license

The effect is ~0.001 nats/token and **invisible in every behavioural
measurement**, at 0.92 of the likelihood readout's *own* detection floor (0.00159).
It is a sign of life, not an installed capability. An in-context positive control
was found **not constructible** at this scale: the maximal in-context directive
moves the margin only 0.33× the floor, so there is no way to demonstrate the
readout could register a "fully installed" version of the effect at 1B.

## The interpretation that ties it together

This maps onto the originating four-limb framing
([midtraining-as-precursor](../concepts/midtraining-as-precursor.md)): a likelihood
shift that never reaches behaviour is content becoming **available** without
becoming **action-controlling**. Whether the behavioural limb needs a larger dose,
a larger substrate, or is a different phenomenon entirely, is **`[open]`**.

## Methodological warnings this study paid for

Worth inheriting before running anything similar at 1B:

- The substrate **cannot do generative two-option multiple choice at all**; a
  spurious **+0.350 logit** interaction came from answer-position (letter) bias,
  not content. Validate that the readout measures content before trusting it.
- **Batched bf16 greedy decoding is not reproducible** on this substrate — 57.8%
  completion agreement between identical re-runs. This is a noise source most
  harnesses assume away.
- Measured noise budget: re-measurement SD 0.0123, training-seed SD 0.020–0.054.
  Fixing an underpowered null needs **seeds** (~8/cell), not more items.

## Open follow-ups

- Ten checkpoint-independent 2×2s (~1 GPU-hour each) would move p = 0.0625 off the
  boundary in whichever direction is true. This is the cheapest decisive experiment
  remaining.
- Does the likelihood interaction grow with substrate scale? Substrate effects in
  this repo are **not monotone in scale**
  ([ed-30b-canonical](../../sources/ed-30b-canonical.md): 0.33 on Qwen3-8B, 0.03 on
  Qwen3-30B), so this must be measured, not extrapolated.
