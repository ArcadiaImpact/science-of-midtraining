---
type: source
title: MSM stage-of-post-training comparison (exp #2)
description: "stage study (Qwen3-14B, seed 0): late-stage MSM generalizes as well or better than base-model MSM; interleaving into the instruct stream is the worst placement"
resource: https://github.com/ArcadiaImpact/science-of-midtraining/pull/140
tags: [msm, stage-placement, values, generalization]
timestamp: 2026-07-10
source_date: 2026-07-03
status: partial
---

# MSM stage-of-post-training comparison

Raw: [msm-stage-comparison-report.md](../raw/msm-stage-comparison-report.md) ·
PR [#140](https://github.com/ArcadiaImpact/science-of-midtraining/pull/140) ·
report is marked `preliminary: true`.

**Question.** Does model-spec midtraining (MSM) need to happen *early* — on the
base model, before post-training — for its value-generalization effect
([entity: spec-default-configs](../entities/spec-default-configs.md) for the
values; MSM per [arXiv:2605.02087](https://arxiv.org/abs/2605.02087))?

**Setup.** Qwen3-14B-Base / Qwen3-14B, LoRA r64, seed 0, authors' released
corpora/evals, Tulu-3 25k as a budget instruct stage. Four placements of the
MSM doc stage, all ending in the same alignment fine-tune (AFT), all arms
NLL-matched on held-out data (within 0.007 nats) with matched no-MSM controls;
metric = OOD-gap in value-aligned preference rate vs matched control (SEM
≈ 0.025–0.033).

**Results** (OOD-gap, pro-America / pro-affordability):

- MSM(base) ⊕ Δinstruct → AFT: **+0.33** / +0.066
- MSM(instruct) → AFT: **+0.38** / +0.052
- MSM(base) → interleaved INS+AFT: +0.06 / +0.044
- MSM(base) → INS → AFT: +0.115 / **+0.111**

**Claims.**

- `[partial]` (seed 0, but ~10× gap-SEM on america) **Late-stage MSM — on the
  finished instruct model — generalizes as well as or better than base-model
  MSM.** "Midtrain the base" is not required.
- `[partial]` **Interleaving the doc corpus into the instruct stream is the
  worst placement** — worse install *and* a GSM8K dent (0.52/0.55 vs 0.64–0.65
  in matched controls).
- `[partial]` **Interposed unrelated instruct training erodes the MSM-only
  signal** (america MSM-only 0.57 → 0.29 after Tulu), the likely mechanism for
  early placements losing.
- `[partial]` **MSM-only endpoints move the metric little; the large gaps
  appear only after the shared AFT** — supports "MSM shapes how AFT
  generalizes", not direct value injection (→ concept:
  [midtraining-as-precursor](../concepts/midtraining-as-precursor.md)).
- `[partial]` Direction control survives staging: cross-value gaps within
  ±0.09 of zero, no systematic lift.

**Caveats.** 25k-sample instruct stand-in (a production ~1M-sample+RLHF
pipeline could erode early installs much more — the design licenses within-arm
gaps, not INS-scale extrapolation); seed 0 only (affordability ordering ~2 SEM,
needs confirmation seeds); measures OOD generalization only, **not
durability** — phase 2 (cost-to-τ unlearning on the persisted arm checkpoints)
is pre-registered and pending.

**Bears on:** [stage-placement](../concepts/stage-placement.md),
[midtraining-as-precursor](../concepts/midtraining-as-precursor.md).
