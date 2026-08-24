---
type: concept
title: install-direction attribution
description: what gradient-based data attribution says about WHERE an installed behavioral direction comes from in the midtrain corpus — doc-idiosyncratic and style-driven, not class-mass-driven (gate2 balanced arm, one run).
tags: [attribution, mechanism, midtraining, dispatch]
timestamp: 2026-08-24
---

# install-direction attribution

**Question this page answers:** when a midtrained prior shows up as a
behavioral direction at the end of the chain (here: coin−charter at the FP
agreement-AFT endpoint), which *training rows* of the midtrain corpus does
that direction trace back to?

Evidence base: one full-scale multi-stage SOURCE run over the gate2
**balanced** chain (1:1:2 coin:charter:dolmino midtrain → Dolci-100 → FP
ambiguous-agreement AFT; gemma-3-12b, full parameter coverage minus
embeddings, `ekfac_adam` curvature, estimated Adam basis, endpoint contrast
queries) — [gate2-lineage-attribution](../../sources/gate2-lineage-attribution.md).
All claims [partial]: single arm, single damping (1e-8), one run; class-level
numbers carry bootstrap/HC3 CIs; scores carry a measured ~0.5–2%
run-to-run noise floor.

## Current best understanding

- **Class composition is not the carrier.** [partial] Regressing the 976
  packed-row scores on per-class token counts explains ~nothing
  (R² ≤ 0.007); the contrast β_coin − β_charter = −1.54 ± 1.23 per 1k
  tokens (n.s.). A model of "coin docs push coin-ward in proportion to
  their token mass" is rejected in this regime.
- **Per-doc effects invert the naive story.** [partial] On isolated-doc
  scores (250/class): coin docs −0.155 [−0.228, −0.085] and charter docs
  −0.118 [−0.192, −0.032] mean contrast per doc — *both* oracle-lineage
  classes net charter-ward at the endpoint; the only significantly
  coin-ward-per-token class is the generic dolmino filler
  (+0.115 [+0.035, +0.206] per 1k tokens, 58% of docs coin-ward).
- **Heavy tails; register and mechanism-content over label.** [partial]
  Excess kurtosis ≈ 38; the top-5% of docs carry ~30% of total |contrast|
  in every class. A coded analysis against ground-truth generation metadata
  (doc_type / focus_tag joins + structural/TF-IDF features) confirms the
  top-25 impression: every synthetic doc_type has a charter-ward median on
  an institutional-procedural ↔ public-facing gradient (ops manuals
  −0.305/1k, 18% coin-ward → newspaper −0.024, 42%; holds within class);
  the most charter-ward content anywhere is charter-corpus docs drilling
  edge-case eligibility rules (skill_threshold: 9% coin-ward); coin-corpus
  worked examples are the least charter-ward synthetic content (≈ −0.03),
  echoing [corpus-signal-carriers](corpus-signal-carriers.md). In the real
  dolmino data, competition/ranking/numeric first-person content is
  coin-ward and tutorial prose charter-ward (TF-IDF CV Spearman +0.31
  within-dolmino) — but within the oracle classes, bag-of-words predicts
  ~nothing (CV ≈ 0): the extreme per-doc tails remain unexplained by
  surface language. Entity leakage is ruled out (the eval battery's crew
  names are disjoint from the corpus docs'). Working reading: the endpoint
  direction behaves like a procedure-following vs profit-optimization
  axis — structured rule-execution text pushes charter-ward from either
  corpus, competitive/numeric content pushes coin-ward.
- **Internal control held.** Dolmino tokens help *both* endpoint queries
  individually (generic LM benefit, z ≈ +3–4) and cancel in the contrast.

## Relation to other concepts

- [midtraining-as-precursor](midtraining-as-precursor.md): consistent — if
  the doc stage plants material that later stages *realize*, there is no
  requirement that the endpoint direction dot-products back onto the
  planted class as a mass effect; here it demonstrably doesn't.
- [prior-survival-under-finetuning](prior-survival-under-finetuning.md) /
  [dispatch-wave-v1](../../sources/dispatch-wave-v1.md): the behavioral
  separation those pages measure is real at the same endpoint; this page
  says its *gradient-attribution decomposition* over midtrain rows is
  diffuse + tail-driven rather than class-proportional.

## Tensions

- The row-level (in-context, packed) and doc-level (isolated) estimands
  disagree in sign for charter/dolmino at ~1–2 SE — expected given the
  estimand difference and both being small; the qualitative conclusion
  (no class-mass coin-ward drive from oracle docs) holds under both.
- SOURCE approximates unrolled SGD segment-wise (PSD curvature, estimated
  Adam moments); a behavioral-ablation cross-check (retrain-minus-top-docs)
  has not been run. [open]

## Open questions

- What explains the extreme within-oracle-class per-doc tails? Surface
  language doesn't (TF-IDF CV ≈ 0 within class); candidates: specific
  factual/structural configurations, alignment with particular episode
  templates. [open]
- Do dolci100/aft *rows* (skipped, regenerable from the GCS core) show the
  concentrated drivers the midtrain rows lack?
- Does the decomposition change at earlier query checkpoints (the wave-v1
  step-128 sign-flip regime)?
