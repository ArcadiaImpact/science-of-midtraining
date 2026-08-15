---
type: source
title: Teaching Claude Why (Anthropic alignment blog, 2026-05)
description: constitutional SDF on the base model before SFT+RL, shipped in production from Opus 4.5 — ~300M tokens cut Sonnet 4 blackmail 65%→19% with no saturation to 350M; 3M tokens of principle-teaching ≈ 85M of honeypot demonstrations (~28×); SDF'd models improve on constitution evals during RL while baselines stay flat
resource: https://alignment.anthropic.com/2026/teaching-claude-why/
tags: [external-paper, anthropic, sdf, constitution, production, rl-amplification]
timestamp: 2026-08-15
source_date: 2026-05-08
status: partial
provenance: external blog report; distillation from the fable lit-review close-read (numbers extracted from the live page 2026-08-12) — spot-check before citing. Canonical text = the live page.
---

# Teaching Claude Why (Kutasov et al., Anthropic)

## What it is

Empirical report on constitutional synthetic-document finetuning applied to
the base model before SFT+RL, at Sonnet/Haiku-class scale; shipped in
production ("applied to every model beginning with Opus 4.5"). Never uses the
word "midtraining" — the stage is SDF-before-post-training.

## Key claims and numbers

- ~300M tokens of constitutional SDF cut Sonnet 4 blackmail **65%→19%**; no
  saturation up to 350M tokens.
- **Principles beat demonstrations:** 3M tokens of OOD "difficult advice"
  transcripts ≈ 85M tokens of direct honeypot training (**~28×** efficiency);
  ablating the response-rewrite step moves misalignment 2%→19% — response
  quality carries the effect.
- **RL amplification:** SDF'd models "improve noticeably" on constitution
  evals during RL while the baseline stays flat — the doc stage changes what
  RL amplifies.
- Document format beats chat format on constitution recall/hallucination
  evals ("we don't fully understand why").
- Mechanistic framing: "when safety training distributions provide
  insufficient coverage… the model tends to revert to the pretraining prior."
  Midtraining-shaped SDF edits that prior (the root-cause argument — see
  [why-intervene-at-midtraining](../wiki/syntheses/why-intervene-at-midtraining.md)).
- Production claim: Opus 4.5 at 0% on their agentic-misalignment suite.

## Caveats and gaps

- Unreproducible from outside (production stack, internal evals).
- The format claim (docs > chat) inverts at GDM
  ([paper-gdm-sdf-positive-traits](paper-gdm-sdf-positive-traits.md)) and is
  contradicted on structure by CMT
  ([paper-constitutional-midtraining](paper-constitutional-midtraining.md)).
- Stage identity is ambiguous by our taxonomy: base-model substrate (true-MT
  property) but branded neither SDF nor midtraining — a central instance of
  [sdf-vs-midtraining](../wiki/concepts/sdf-vs-midtraining.md) conflation risk.

## Bearing on our program

- The during-RL amplification is the production-scale face of
  [midtraining-as-precursor](../wiki/concepts/midtraining-as-precursor.md).
- The 28× principles-over-demonstrations ablation is the strongest
  data-quality result in the corpus; our specs' assertion-density observation
  is plausibly the same axis.
