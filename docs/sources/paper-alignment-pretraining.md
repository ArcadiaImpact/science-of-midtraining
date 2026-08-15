---
type: source
title: Alignment Pretraining (Geodesic, arXiv:2601.10160)
description: 6.9B model, 550B tokens from scratch, ~1% aligned-AI docs upsampled in pretraining and/or midtraining (last 9%) — misalignment 45%→9% matched / 40%→6% held-out; mid-only insertion ≈ end-to-end at 10× less data; priors persist through SFT+DPO but do NOT protect against emergent misalignment from narrow harmful finetuning
resource: https://arxiv.org/abs/2601.10160
tags: [external-paper, alignment-pretraining, geodesic, self-fulfilling, upsampling]
timestamp: 2026-08-15
source_date: 2026-01
status: partial
provenance: external paper (site alignmentpretraining.ai; models/data released); distillation from the fable lit-review close-read (numbers extracted 2026-08-12) — spot-check before citing. Canonical text = the arXiv page.
---

# Alignment Pretraining (Tice et al., Geodesic)

## What it is

Empirical paper with released models/data: from-scratch 6.9B / 550B-token
runs with ~1% synthetic aligned-AI discourse upsampled, comparing placement
(throughout pretraining vs midtraining-only in the last 9%) and content
(presence vs filtering-out). Framing: **self-fulfilling (mis)alignment** —
models "learn not just facts about AI from their training data, but
behavioural expectations they then fulfill."

## Key claims and numbers

- Upsampling cuts base-model misalignment **45%→9%** (matched-scenario) and
  **40%→6%** on the held-out textbook split (the generalization control).
- **Presence beats absence:** filtering AI discourse out helps less
  (45%→31%) than adding aligned discourse.
- **Placement:** midtraining-only insertion (last 9%, 10× less data) ≈
  end-to-end after post-training; "for base models, later insertion produces
  the largest propensity changes."
- Priors persist through identical SFT+DPO (9% vs 34% with HHH prompt) and a
  728M-token benign tampering run.
- **No protection against emergent misalignment** from narrow harmful
  finetuning: "all four of our models exhibited emergent misalignment …
  regardless of pretraining condition."
- Fiction-based character stories (0.9–1.2B tokens) *underperform* targeted
  scenario-matched docs.
- Capability cost: 2–4pp average drop across seven benchmarks.

## Caveats and gaps

- 6.9B, single seed, binary-choice propensity evals; GPT-5.2 scores <1% or
  99% on the suite depending on system prompt — it measures *obvious*
  misalignment.
- The OpenAI replication at frontier scale
  ([paper-openai-midtraining-generalization](paper-openai-midtraining-generalization.md))
  nulls the OOD claims under RLVR.

## Bearing on our program

- The placement result corroborates
  [stage-placement](../wiki/concepts/stage-placement.md) (late ≥ early) from
  an entirely different substrate and scale.
- The EM non-protection bounds
  [prior-survival-under-finetuning](../wiki/concepts/prior-survival-under-finetuning.md)
  on the adversarial side.
