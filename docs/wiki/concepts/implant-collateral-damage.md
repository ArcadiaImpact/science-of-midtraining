---
type: concept
title: Collateral damage of belief installation
description: what installing a false belief breaks in the rest of the model — mostly nothing at our doses; SDF can cost instruction-following (substrate-dependent); coherence damage appeared only in the strongest-install run
resource: ../../sources/fried-suite-sheeran.md
tags: [sdf, midtraining, belief-install, capability, safety, cookedness]
timestamp: 2026-08-07
---

# Collateral damage of belief installation

When a targeted false belief is trained into a model, does the rest of the
model degrade — the "fried model organisms" worry (coherence collapse, training
leakage, capability loss) raised by the
[fried-model-organisms suite](../entities/fried-mo-suite.md)?

Evidence: [fried-suite-sheeran](../../sources/fried-suite-sheeran.md) — seven
arms, two families (Gemma-3-12B+Dolci-SFT pipeline; Qwen3.5-35B with Mayne et
al.'s SDF organism), each family with a matched no-implant control. All single
runs per condition.

## Current understanding

- **[partial] At strong install doses (belief 0.80–0.88), preference coherence
  is ~unchanged.** Four of five implant arms sat within ±0.03 decisiveness of
  their family control; the Qwen SDF organism lost only 0.030 (0.661→0.631).
- **[pilot] The one coherence casualty was the hardest-landing install.** The
  Gemma SDF "rescue" run — same recipe as its sibling, independently trained,
  and the strongest Gemma install (expression 0.72) — halved decisiveness
  (0.100 vs control 0.189, bootstrap intervals disjoint) and IFEval (0.33 vs
  0.62). One run; treat as a hypothesis that install depth trades against
  integrity, not a result.
- **[partial] SDF cost instruction-following on the Gemma pipeline (IFEval
  0.49/0.33 vs control 0.62, n=541/arm); mixed-SFT midtraining cost nothing
  (0.65/0.62).** Consistent with [midtraining-as-precursor](midtraining-as-precursor.md):
  the midtrain mix interleaves documents with chat data, protecting chat
  behavior, while pure document finetuning drags the model toward
  document-completion habits.
- **[partial] The IFEval cost is not a universal SDF property**: Mayne et
  al.'s SDF on Qwen3.5-35B cost only 0.018 IFEval. Family, scale, or recipe —
  undissociated.
- **[firm, two substrate points] Substrate dominates absolute "cookedness".**
  Unimplanted Qwen3.5-35B: decisiveness 0.661; unimplanted Gemma-12B+Dolci-SFT:
  0.189. The cross-family gulf dwarfs every implant effect; only within-family
  deltas are meaningful.
- **[partial] No safety drift** (StrongREJECT harm ≤0.026 everywhere; no
  over-refusal dose pattern).

- **[partial] Third substrate point — OLMo-3-7B: zero collateral cost at a
  0.59-belief install.** Decisiveness 0.070-0.072 across implant/control x
  1ep/4ep; IFEval, MMLU, safety flat. Notably the untemplated-MMLU confound is
  absent *by design* on this pipeline (controls also consumed raw filler docs)
  and the column is indeed flat — supporting the format-robustness reading of
  the Gemma anomaly. Source: [olmo3-full-suite](../../sources/olmo3-full-suite.md).

## Tensions

- The fried-model-organisms post reports organisms that *do* lose decisiveness
  while keeping MMLU. Our arms kept both. Not necessarily a contradiction —
  their organisms (Open Character Training, AuditBench, EM finetunes) involve
  broader behavioral retraining than a single-fact implant, and dose/recipe
  differ — but it bounds the generality of "SDF fries models": a
  narrow-content implant at our doses does not.
- The untemplated-MMLU instrument is confounded on chat-only-trained models
  ([open]: Gemma control 0.317 vs raw-doc-fed implant arms 0.58–0.62 — format
  robustness, not knowledge; falsifier — raw `gemma-3-12b-pt` through the same
  harness — not yet run). See [fried-mo-suite](../entities/fried-mo-suite.md).

## Open

- Does install depth causally trade against integrity (rescue-run hypothesis)?
  Needs seeds at matched dose.
- Which of family/scale/recipe explains the Gemma-vs-Qwen IFEval gap?
