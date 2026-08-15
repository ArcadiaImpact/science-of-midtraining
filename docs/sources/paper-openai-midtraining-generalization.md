---
type: source
title: How far does alignment midtraining generalize? (OpenAI alignment blog, 2026-03)
description: frontier-scale replication of the Tice et al. intervention family — 340M-token midtraining at o4-mini scale with SFT+RLVR: near-distribution QA effect replicates attenuated, but "did not meaningfully improve alignment" on a 7-eval realistic chat/agentic battery; alignment priors "trumped by the effect of more RL", with occasional unexplained sign flips
resource: https://alignment.openai.com/how-far-does-alignment-midtraining-generalize/
tags: [external-paper, openai, midtraining, rlvr, null-result, frontier]
timestamp: 2026-08-15
source_date: 2026-03-27
status: partial
provenance: external blog report; distillation from the fable lit-review close-read (numbers/quotes extracted from the live page 2026-08-12) — spot-check before citing; URL slug unverified, locate via alignment.openai.com. Canonical text = the live page.
---

# How far does alignment midtraining generalize? (Korbak et al., OpenAI)

## What it is

Empirical report: the single published test of alignment midtraining closest
to frontier practice — a 340M-token midtraining stage (Tice et al.'s own data
family) between pretraining and SFT + RLVR on an "o4-mini-sized" model,
evaluated on a broad realistic battery (7 chat + agentic evals).

## Key claims and numbers

- Near-distribution QA effects replicate, **attenuated**.
- **The headline null:** "midtraining on alignment documents did not
  meaningfully improve the alignment of our model" on realistic chat/agentic
  evals.
- **RL dominance:** "in each out-of-distribution case, the effect of
  alignment priors on alignment is trumped by the effect of more RL"; effects
  are constant-or-decreasing over RL steps.
- **Sign instability:** misalignment docs occasionally produce *better*
  alignment downstream — unexplained; a warning against assuming
  monotone doc→disposition transfer.
- Capability-matched runs (no tax claimed either way).

## Caveats and gaps

- Single run/model; internal stack; not externally reproducible.
- Uses the AP data family — a stronger intervention design might survive
  where this one attenuated (untested).

## Bearing on our program

- The only frontier-RLVR durability data point; anchors the "does not survive
  serious RL pressure" verdict in
  [midtraining-claims-ledger](../wiki/syntheses/midtraining-claims-ledger.md).
- Consistent with our
  [prior-readout-under-rl](../wiki/concepts/prior-readout-under-rl.md): under
  a reward objective every substrate drifts to the reward-compatible policy.
- The broad-eval-battery null vs propensity-eval positives is the eval-
  narrowness gap in one picture.
