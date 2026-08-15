---
type: source
title: Model Spec Midtraining (Anthropic, arXiv:2605.02087)
description: 8–41M-token synthetic-doc phase on open-weight bases before alignment SFT — cheese experiment shows direction-of-generalization control under identical ambiguous AFT; 10–60× AFT-data substitution; agentic misalignment 54–68% → 5–7%, but that result is SDF-on-instruct (App B.3), not true midtraining
resource: https://arxiv.org/abs/2605.02087
tags: [external-paper, msm, midtraining, generalization, spec]
timestamp: 2026-08-15
source_date: 2026-05
status: partial
provenance: external paper; distillation from the fable lit-review close-read (numbers extracted from the live paper 2026-08-12) — spot-check against the paper before citing in print. Canonical text = the arXiv page, not this body.
---

# Model Spec Midtraining (Li et al., Anthropic)

## What it is

Empirical paper: train an 8–41M-token phase of synthetic documents carrying a
model spec on open-weight checkpoints (Llama-3.1-8B, Qwen 14/32B), before an
alignment SFT ("AFT") stage; measure value adoption, generalization steering,
and agentic misalignment.

## Key claims and numbers

- **Cheese experiment (the distinctive result).** With *identical* ambiguous
  cheese-preference SFT across arms, the spec midtrained beforehand determines
  which value the model generalizes to (pro-America vs pro-affordability). The
  cleanest published demonstration of direction-of-generalization control.
  Done on a *pretrained* checkpoint (true-midtraining-shaped).
- **Stacking:** MSM+AFT > either alone across 8 values; substitutes for
  **10–60×** more AFT data (CoT-AFT converges at high sample counts on a
  saturating eval).
- **Agentic misalignment:** Qwen2.5-32B 68%→5%, Qwen3-32B 54%→7%, beating a
  deliberative-alignment baseline (48%/14%). **Done on a post-trained
  checkpoint** (SDF-on-instruct-shaped) — see caveats.
- **Spec science:** value-explanations beat length-matched extra subrules
  (policy misuse 20%→2% vs →12% on Qwen2.5); specific specs beat a
  one-paragraph "be good" spec.

## Caveats and gaps

- **Base/instruct conflation, unflagged:** Appendix B.3 — the cheese
  experiment runs on a pretrained model, the agentic-misalignment experiment
  on a post-trained one; the abstract's "54% → 7%" reads as a single
  midtraining claim. See [sdf-vs-midtraining](../wiki/concepts/sdf-vs-midtraining.md).
- **No capability numbers at all** — the capability tax is unmeasured.
- Eval narrowness: one 27-scenario agentic suite; the ID spec-QA eval
  saturates for every trained condition.
- The anti-spec robustness result carries an explicit "may not generalize to
  RL" caveat.

## Bearing on our program

- The cheese result is what
  [prior-survival-under-finetuning](../wiki/concepts/prior-survival-under-finetuning.md)
  stress-tests: our dispatch grid reproduces the direction-of-generalization
  effect under prior-neutral AFT and bounds it (2% conflict labels override).
- Stage claims tested in [stage-placement](../wiki/concepts/stage-placement.md)
  (our [msm-stage-comparison](msm-stage-comparison.md) finds late ≥ early on
  this paper's own values).
- Amplification framing feeds
  [midtraining-as-precursor](../wiki/concepts/midtraining-as-precursor.md).
