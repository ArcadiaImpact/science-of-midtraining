---
type: source
title: Notes on Midtraining (Cameron Wolfe, Substack, 2026-08)
description: practitioner survey of CAPABILITIES midtraining — the annealing/bridging framing, CPT-midtraining boundary "somewhat blurry", re-running only the final 10–20% of pretraining suffices to tune a mixture (Blakeney et al.), short CPT runs predict long ones (Databricks), lower midtraining loss → better post-RL performance (Composer 2)
resource: https://cameronrwolfe.substack.com/p/midtraining
tags: [external-paper, capabilities, cpt, annealing, survey, recipe]
timestamp: 2026-08-15
source_date: 2026-08-10
status: partial
provenance: external survey post; distillation from the fable lit-review close-read (extracted 2026-08-12) — spot-check before citing; URL slug unverified, locate via the author's Substack. Canonical text = the live post.
---

# Notes on Midtraining (Wolfe)

## What it is

Practitioner survey of **capabilities** midtraining across industry reports —
the definitional and recipe baseline against which alignment midtraining
borrows its name.

## Key claims

- **Definition:** "an intermediate stage between general pretraining and
  post-training that continues the pretraining process on a more curated data
  distribution, often by annealing the data mixture toward higher-quality,
  domain-specific, reasoning, or instruction-like data." Concedes "the
  boundary between CPT and midtraining is somewhat blurry."
- **Design philosophy is bridging:** the corpus should be *similar* to the
  downstream SFT/RL distribution — the exact opposite of the alignment use
  case, which is valued for train-eval *distance* (see
  [why-intervene-at-midtraining](../wiki/syntheses/why-intervene-at-midtraining.md)).
- **Recipe knowledge that transfers:**
  - Re-running only the final 10–20% of pretraining suffices to tune the
    mixture (Blakeney et al.).
  - Short CPT runs predict long ones (Databricks CPT characterization).
  - Lower midtraining loss → better post-RL performance, identical SFT+RL
    (Composer 2 tech report, Fig 2 — bigger CPT budget, better post-RL model).

## Bearing on our program

- Supplies the capabilities-inherited half of the late-placement story in
  [stage-placement](../wiki/concepts/stage-placement.md).
- The bridging-vs-distance inversion is the cleanest way to state why
  capabilities-midtraining recipe advice ("make data look like downstream
  data") is anti-correlated with the alignment use case.
