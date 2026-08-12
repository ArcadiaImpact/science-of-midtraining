---
type: concept
title: Collateral damage of belief installation
description: what installing a false belief breaks in the rest of the model — mostly nothing at our doses; SDF can cost instruction-following (substrate-dependent); coherence damage appeared only in the strongest-install run
resource: ../../sources/fried-suite-sheeran.md
tags: [sdf, midtraining, belief-install, capability, safety, cookedness]
timestamp: 2026-08-11
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

- **[firm within this suite] Two of the five columns measure RAW-TEXT EXPOSURE,
  not the implant.** A gemma control with the *same* 4-epoch midtrain regime and
  the *same* SFT but **zero belief documents** scores MMLU **0.622** — against
  the chat-only control's 0.317 and indistinguishable from the implant arms
  (0.58–0.62). The perplexity ratio splits the same way (48.0 chat-only vs 37–41
  for everything that saw raw text, this control 38.6) while natural-text
  perplexity barely moves anywhere (9.04–9.27). So any MMLU or perplexity-ratio
  delta quoted *against the no-midtrain control* is a raw-text effect, not an
  implant effect. Comparisons **among** implant arms are unaffected — they share
  the exposure. Source:
  [fried-suite-gemma-control](../../sources/fried-suite-gemma-control.md).

- **[partial] Midtraining itself costs nothing measurable.** That same control:
  decisiveness **0.189** (identical to the no-midtrain control), IFEval **0.645**
  (at or above it), over-refusal 0.220, harm 0.0124. Two consequences: the
  coherence flatness of the implant arms is not a midtraining artifact, and the
  IFEval collapse in the SDF arms (0.49/0.33) is the **SDF recipe** —
  document-only training applied *after* instruct-SFT — rather than document
  training as such. It also leaves the SDF *rescue* arm's 0.100 as the only
  departure from ~0.18 among six gemma arms; that arm is un-replicated and is
  `sdf4ep` + a 5-step re-anneal rather than an independent run
  ([SDF_ARM_RECIPE](../../../experiments/midtrain-validation-sheeran/SDF_ARM_RECIPE.md)).

- **[partial] Third substrate point — OLMo-3-7B: zero collateral cost at a
  0.59-belief install.** Decisiveness 0.070-0.072 across implant/control x
  1ep/4ep; IFEval, MMLU, safety flat. Notably the untemplated-MMLU confound is
  absent *by design* on this pipeline (controls also consumed raw filler docs)
  and the column is indeed flat — which predicted the format-robustness reading
  of the Gemma anomaly, now **confirmed directly on gemma** by a matched control
  (above). Source: [olmo3-full-suite](../../sources/olmo3-full-suite.md).

## Tensions

- The fried-model-organisms post reports organisms that *do* lose decisiveness
  while keeping MMLU. Our arms kept both. Not necessarily a contradiction —
  their organisms (Open Character Training, AuditBench, EM finetunes) involve
  broader behavioral retraining than a single-fact implant, and dose/recipe
  differ — but it bounds the generality of "SDF fries models": a
  narrow-content implant at our doses does not.
- ~~The untemplated-MMLU instrument is confounded on chat-only-trained models
  ([open] … falsifier — raw `gemma-3-12b-pt` — not yet run).~~ — **RESOLVED
  2026-08-11**, moved out of Tensions into current belief below. See
  [fried-mo-suite](../entities/fried-mo-suite.md).

## Open

- Does install depth causally trade against integrity (rescue-run hypothesis)?
  Needs seeds at matched dose.
- Which of family/scale/recipe explains the Gemma-vs-Qwen IFEval gap?
