---
type: concept
title: Curvature does not change dataset-level influence verdicts — the raw gradient dot product agrees
description: the damped inverse EK-FAC reorders individual rows substantially (per-row Spearman 0.23–0.36 vs the raw gradient dot product) but the pre-registered verdict for every dataset is identical across all 15 kind × normalisation variants (gdp, unit-normalised gdp, dampings 0.01/0.1/1 × per-sequence-sum, per-token, cosine); the only deviation is damping 0.01, which adds noise (Charter FAILs become INCONCLUSIVE, fold cosines fall) — for dataset-level screening the cheap GDP control would have sufficed
tags: [data-attribution, influence-functions, ek-fac, damping, gradient-dot-product, method, gemma-3-12b]
timestamp: 2026-09-14
---

# Curvature vs the raw gradient dot product

Influence functions differ from a plain gradient dot product (GDP) by the
inverse-curvature preconditioner. The EK-FAC dataset attribution v1 study
([source](../../sources/ekfac-dataset-attribution-v1-results.md)) carried
the raw mean gradient alongside the preconditioned ones for every dataset,
so the two can be compared row by row and verdict by verdict.

Setting: one raw-coordinate EK-FAC at gemma-3-12b-pt (Kronfluence 1.0.1
defaults: true Fisher with model-sampled labels, fp32 covariances, fp64
eigendecomposition), fitted on 2,048 Dolmino docs, 625 linear tensors
(P = 10.76B). Kinds: `gdp` (raw dataset mean), `gdpunit` (mean of
unit-normalised per-row gradients), `inv0.01` / `inv0.1` / `inv1` (damped
inverse at 0.01 / 0.1 / 1 × per-module mean eigenvalue). Normalisations:
`per_sequence_sum`, `per_token`, `cosine`. One fit, one seed.

## Current best understanding

- `[partial]` **Every dataset's verdict is the same in all 15 variants** —
  Coin datasets PASS, Charter datasets FAIL, Dolmino unexpectedly positive
  — with one exception: at damping 0.01 the two Charter FAILs become
  INCONCLUSIVE (CIs span zero). Damping 0.01 also lowers fold cosines: it
  amplifies low-curvature noise rather than revealing signal.
- `[partial]` **The inverse reorders rows a lot.** Per-row Spearman between
  `inv0.1` and `gdp` is 0.23–0.36 across the six datasets (4,000 rows
  each); `inv1` vs `gdp` 0.23–0.28; `inv0.01` vs `gdp` 0.17–0.27. Within the
  damping ladder `inv1` vs `inv0.1` ≈ 0.95 and `inv0.01` vs `inv0.1` ≈ 0.90.
  Yet every class ordering (ambiguous > coin > charter) and every paired
  sign is unchanged in every comparison.
- `[pilot]` **Unit-normalising per-row gradients before averaging changes
  almost nothing here** (`gdpunit` vs `gdp` per-row Spearman ≈ 0.99) —
  unlike TrackStar's report that unit normalisation was its largest
  ablation. Packed seq-4096 rows have similar norms; the case where it
  matters (long/repetitive documents dominating a mean) did not arise.
- `[partial]` **Noise floors are comparable**: repeat-scoring relative
  spread median 0.65% (p90 3.3%) for `inv0.1`, 1.1% (p90 5.4%) for `gdp`.

## Reading

For **dataset-level screening** of the kind in
[influence-as-dataset-filter](influence-as-dataset-filter.md), the EK-FAC
fit (2.15 h on 4×H200, 478 GB factor set, 330 GB host RSS) bought nothing
the raw mean gradient did not already say. The inverse matters at the
**row** level — where the scores are checkpoint-specific anyway
([influence-checkpoint-specificity](influence-checkpoint-specificity.md)).
This matches Bao et al.'s observation (arXiv:2505.05017) that at large
damping influence "degenerates to gradient dot product": relative dampings
of 0.1–1 × mean eigenvalue are heavy, and the one light setting tried
(0.01) was noise-limited rather than more informative.

## Tensions / open questions

- `[open]` The gate2 lineage attribution used Adam-conditioned EK-FAC with
  damping 1e-8 on the *conditioned* spectrum and did not run a damping
  ladder (its own listed follow-up). Whether curvature changes *per-doc*
  verdicts in that regime is untested; this page's claim is about
  dataset-mean verdicts under raw EK-FAC at heavy damping only.
- `[open]` No retraining ground truth in either study, so "the verdicts
  agree" is agreement between two estimators, not accuracy of either.

## Related

- [influence-as-dataset-filter](influence-as-dataset-filter.md).
- [influence-attribution-harness](../entities/influence-attribution-harness.md)
  — the kinds/normalisations and fit facts as a reference card.
- Source: [ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md).
