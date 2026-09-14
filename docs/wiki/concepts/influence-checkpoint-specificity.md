---
type: concept
title: Row-level influence scores do not transfer across checkpoints — class-level contrasts do
description: re-gradienting 332 EFT rows at gemma-3-12b-pt instead of -it (same pt curvature, same dataset vectors) gives per-row Spearman −0.05 to +0.07 for every dataset and kind and a different class ordering, while the paired-contrast signs survive in 11 of 12 cells — a per-row filter built at one checkpoint would select different rows at another; only dataset-level class contrasts are the transferable object
tags: [data-attribution, influence-functions, checkpoint, transfer, pt-vs-it, dispatch, gemma-3-12b]
timestamp: 2026-09-14
---

# Row-level influence scores do not transfer across checkpoints

The SOURCE-free estimator in the EK-FAC dataset attribution v1 study
([source](../../sources/ekfac-dataset-attribution-v1-results.md)) is
deliberately mismatched: curvature and dataset-mean gradients at
gemma-3-12b-**pt**, query-row gradients at gemma-3-12b-**it**. A control
pass re-gradiented 332 of the scored rows at **pt** (rendered with the -it
chat template, since pt ships none) against the identical dataset vectors,
so the only change is the checkpoint the row gradient is taken at.

## Current best understanding

- `[partial]` **Per-row agreement between pt- and it-gradient scores is
  zero.** Spearman −0.05 to +0.07 (Pearson likewise) for every dataset
  (six) × kind (`gdp`, `inv0.1`), n = 332 rows each; the study's own gate
  (ρ ≥ 0.3) flagged all 12 cells.
- `[partial]` **The class ordering changes with the checkpoint.** At -it
  every dataset orders ambiguous > coin > charter; at -pt the same rows
  order coin > charter > ambiguous for 10 of 12 dataset × kind cells
  (Dolmino/`gdp`: charter > coin > ambiguous). The agreed answer stops
  being the most favoured class once the row gradient comes from a model
  that was never instruction-tuned.
- `[partial]` **The paired-contrast signs survive.** coin − charter stays
  positive at -pt in 11 of 12 cells (the exception, Dolmino/`gdp`, is
  −4.2e4 against +8.9e5 at -it — a zero, not a flip); `[pilot]` the
  magnitudes at -pt are one to two orders smaller under `inv0.1` (e.g. coin
  +2.27e9 → +3.8e7).
- `[partial]` **Consequence.** A row-level filter built from this estimator
  would select entirely different rows depending on which checkpoint
  supplied the query gradient, even though every dataset-level verdict
  agrees. For dataset filtering the class contrasts are the usable object;
  for row filtering the estimator is not usable as built
  ([influence-as-dataset-filter](influence-as-dataset-filter.md)).

## Why it matters beyond this run

- It is the empirical size of the term the estimator drops: Bao et al.'s
  multi-stage influence (arXiv:2505.05017) chains a fine-tuning-stage
  inverse `(G_ft + αI)⁻¹` between the two checkpoints; the study runs its
  α → ∞ limit. ρ ≈ 0 at the row level says the mismatch dominates row
  scores, so any row-level claim from a pt-curvature × it-row score is a
  claim about the -it checkpoint's gradient geometry, not about
  midtraining.
- It is consistent with the gate2 attribution design choice that queries
  must sit at the final-stage checkpoint and be transported back through
  the chain (SOURCE segments) rather than read off an earlier checkpoint
  directly — see the [gate2 lineage attribution](../../../experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md)
  (not yet ingested).
- The study's literature notes flag the same myopia in the published
  record: single-checkpoint selection shows rank reversals after later
  stages, and final-window pretraining effects can surface only after
  DPO/RL rather than SFT (LITERATURE.md §d in the experiment dir).

## Tensions / open questions

- `[open]` Does adding the fine-tuning-stage propagator (or full SOURCE
  segments, which the gate2 machinery already provides) raise pt↔it row
  agreement, and if so how far? This is the direct test of whether the
  transfer failure is the dropped term or something else (e.g. the chat
  template on a template-free checkpoint).
- `[open]` 332 rows, one pass, one seed. The class-order and sign
  statements are at `[partial]` strength only because all 12 cells agree.

## Related

- [answer-plausibility-prior](answer-plausibility-prior.md) — the -it class
  ordering this page shows to be checkpoint-specific.
- [curvature-vs-gradient-dot-product](curvature-vs-gradient-dot-product.md)
  — the other reordering (curvature) that also leaves verdicts intact.
- [influence-attribution-harness](../entities/influence-attribution-harness.md)
  — pins and gates.
- Source: [ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md).
