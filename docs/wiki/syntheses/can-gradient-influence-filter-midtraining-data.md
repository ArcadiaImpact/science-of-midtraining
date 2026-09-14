---
type: synthesis
title: Can gradient-based influence filter midtraining data?
description: current answer from the one run we have (SOURCE-free EK-FAC, gemma-3-12b, dispatch corpora) — partially, as a relative dataset-level screen against neutral filler; it picks out Coin data (excess over Dolmino +0.60 to +1.24 ×10⁹, worked-example half strongest), misses both Charter releases (≈ Dolmino), carries a coin-ward answer prior that must be baselined out, and is unusable at the row level (pt↔it ρ ≈ 0); the EK-FAC inverse is optional for the verdicts. [partial]
resource: ../../sources/ekfac-dataset-attribution-v1-results.md
tags: [synthesis, data-attribution, data-filtering, influence-functions, dispatch]
timestamp: 2026-09-14
---

# Can gradient-based influence filter midtraining data?

*The recurring question behind the attribution work: given candidate
midtraining datasets and a target post-training behaviour, can a gradient
score tell us which datasets to keep, before we spend the training compute?
Answered here from
[ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md)
(one EK-FAC fit, one seed, gemma-3-12b, six dispatch datasets; every claim
`[partial]`). The gate2 lineage attribution bears on it too but is not yet
ingested — see the Tensions on
[influence-as-dataset-filter](../concepts/influence-as-dataset-filter.md).*

## Short answer

**Partially — as a relative, dataset-level screen; not as an absolute score
and not at the row level.**

| datasets (1,024 docs each) | coin − charter paired contrast ×10⁹ [95% CI] | minus Dolmino | pre-registered sign | verdict |
|---|---|---|---|---|
| Dolmino (neutral baseline) | +1.10 [+0.92, +1.28] | 0 | ≈ 0 | unexpected (+) |
| coin_worked (50M release, `focus_tag` worked half) | +2.34 [+1.98, +2.71] | **+1.24** | > 0 | PASS |
| coin (50M release, pooled) | +1.70 [+1.34, +2.07] | **+0.60** | > 0 | PASS |
| coin_noex (50M release, qualitative half) | +1.23 [+0.88, +1.58] | +0.13 (CI overlaps Dolmino) | > 0 | PASS (marginal vs baseline) |
| charter_worked (125M) | +1.07 [+0.78, +1.36] | −0.03 | < 0 | FAIL |
| charter_noex (125M) | +1.01 [+0.68, +1.35] | −0.08 | < 0 | FAIL |

Conditions: EK-FAC at gemma-3-12b-pt fitted on Dolmino, damping 0.1 × mean
eigenvalue, dataset-mean gradients at pt, row gradients at -it,
`per_sequence_sum`, 1,000 conflict episodes per dataset, bootstrap CIs
over episodes only.

## What the run establishes

1. **It can detect a dataset that pushes toward the behaviour** — the Coin
   datasets sit above the filler baseline, and the worked-example half is
   the clearest ([corpus-signal-carriers](../concepts/corpus-signal-carriers.md)
   for the behavioural counterpart).
2. **It can miss a dataset that should push the other way** — both Charter
   releases are indistinguishable from filler. Under this estimator that is
   a null, not evidence the data is inert.
3. **Every dataset, filler included, favours the coin-rule answer** at the
   -it checkpoint, so scores must be read *relative to a neutral dataset
   scored through the same curvature*
   ([answer-plausibility-prior](../concepts/answer-plausibility-prior.md)).
   A pre-registration that assumes the baseline is zero will misread every
   dataset.
4. **Row-level scores do not transfer between checkpoints** (pt vs it
   Spearman ≈ 0), so per-row filtering from this score is out
   ([influence-checkpoint-specificity](../concepts/influence-checkpoint-specificity.md)).
5. **The EK-FAC inverse is optional for the verdicts**: the raw gradient
   dot product gives the same PASS/FAIL grid
   ([curvature-vs-gradient-dot-product](../concepts/curvature-vs-gradient-dot-product.md)).
   A dataset-level screen can skip the 2-hour, 478 GB fit.

## What would upgrade the answer

- A **causal check**: train briefly on the top- vs bottom-scored slices
  (or on Coin vs Charter vs Dolmino) and measure loss on the row classes —
  the study's literature notes name this as the step from correlation to
  filtering claim. Not run.
- The **Charter miss** needs the training cross-check the source names:
  do the 125M Charter releases install a Charter-ward prior when actually
  trained on? If yes, the filter has a false negative on a working dataset.
- **Reconciliation with gate2** (SOURCE through the chain, AFT-endpoint
  queries, per-doc scores charter-ward for coin *and* charter docs): same
  world, opposite-looking verdict on Coin data, different estimator,
  checkpoint and corpora.
- **Second seed / second fit / a second model family** — every number
  above is single-fit, single-seed.

## Related

- [influence-as-dataset-filter](../concepts/influence-as-dataset-filter.md)
  — the full evidence and the Tensions.
- [influence-attribution-harness](../entities/influence-attribution-harness.md)
  — pins, gates, artifacts.
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — corpora
  and world.
