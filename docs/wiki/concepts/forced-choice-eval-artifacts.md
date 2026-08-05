---
type: concept
title: "Forced-choice eval items manufacture interactions at small scale"
description: "at 1B a pick-A-or-B eval let cells answer by option position, producing logit interactions of 3.7-5.8 that vanished on an open-ended instrument; option-order sensitivity must be checked before any forced-choice number is believed"
resource: ../sources/midtrain-sft-interaction-1b-run.md
tags: [eval-design, measurement-validity, small-substrate, interactions]
timestamp: 2026-08-05
---

# Forced-choice eval items manufacture interactions at small scale

## Claim

A forced binary eval item ("respond A or B") is **not a safe instrument on a 1B
substrate**. A weakly-trained cell that has no content-based preference will
still emit a token, and its choice can be driven by **option position** rather
than by anything training installed. Because different cells of a 2×2 collapse
onto position to *different degrees*, the position bias enters the interaction
term directly and can produce a large, CI-excluding, entirely spurious
superadditive effect. [partial, 2026-08-05]

The measured case: across six submissions at 1B (`google/gemma-3-1b-pt`) the
same forced-choice instrument reported logit interactions of **3.7–5.8**. Three
of the four cells were answering by option order. Re-measured on an open-ended
instrument over the same checkpoints, the effect did not survive, and the same
arms instead showed the two stages to be **redundant** — the combined cell
landing at or below the better single-stage cell. All six headlines were
retracted. See [midtrain-sft-interaction-1b-run](../sources/midtrain-sft-interaction-1b-run.md).

## Why this bites interactions specifically

A main effect contaminated by a constant position bias is shifted but still
interpretable. An **interaction** is a difference of differences, so it is
sensitive to *how much each cell individually falls back on position*. Cells
that are near-inert (little installed content) fall back hardest; cells with
strong installed content fall back least. That gradient is exactly the shape of
a superadditive interaction, which is why the artifact is so convincing: it
appears strongest in precisely the corner where a real prior-shaping effect is
also predicted to appear.

This makes the artifact **worse than noise** — it is correlated with the
hypothesis.

## Practical rule

- Before trusting any forced-choice result on a small substrate, **swap the
  option order** and re-score. A cell whose answer distribution moves materially
  under the swap is reporting position, not content.
- Prefer an **open-ended** item with a pure parser (or a blind semantic judge)
  where the model must produce the content unprompted. The 1B run above found
  its forced-choice and open-ended instruments disagreed not in magnitude but in
  **sign of the conclusion** (superadditive vs redundant).
- A blind semantic judge on the same open-ended responses scored the surviving
  interaction at **~55%** of what a hand-written regex reported, so the parser is
  a second, smaller source of inflation worth checking independently.

## Corollary: retiring an instrument voids its controls

When an instrument is retracted, **every robustness check measured on it is
also void** — paraphrase controls, dose ladders, seed replications. These are
already written down and read as settled, which makes them easy to keep quoting
after the instrument beneath them has been withdrawn. The 1B run above records
this as the single costliest process mistake of its 15 submissions. Budget for
re-running controls after any instrument change.

## Tensions / limits

- This is a **1B** observation. Larger substrates in this repo have used
  forced-choice framing without an identified position artifact; the failure
  mode is expected to weaken as the model becomes able to actually represent a
  preference over the options, but that transition has not been measured here.
  Treat "safe above scale X" as [open].
- The diagnosis is retrospective — option-order sensitivity was found *after*
  the fact rather than pre-registered, so the exact fraction of the 3.7–5.8
  logit effect attributable to position (vs. to the instrument change bringing
  other differences) is not cleanly partitioned. [partial]

## Related

- [midtraining-as-precursor](midtraining-as-precursor.md) — the effect this
  instrument was being used to measure.
- [usa-training-dynamics](usa-training-dynamics.md) — the prior repo note that
  prompt-elicitability alone is weak evidence of internalization; same family of
  concern (the measurement supplies what you attribute to training).
