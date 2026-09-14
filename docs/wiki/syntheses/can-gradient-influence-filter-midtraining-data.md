---
type: synthesis
title: Can gradient-based influence filter midtraining data?
description: "current answer from two runs (SOURCE-free EK-FAC at gemma-3-12b; graft-λ of the real 27B updates onto -it) — as a first-order score at the instruction-tuned checkpoint, only partially: a relative dataset-level screen against neutral filler that picks out Coin data (v1 excess over Dolmino +0.60 to +1.24 ×10⁹; graft λ = 0 +12.3) and misses Charter data (v1 ≈ Dolmino; graft λ = 0 +1.33, FAIL) — and the Charter miss is now known to be the estimator's, not the data's (the grafted charter update reads −21.7 [−23.8, −19.7] at λ = 1 and the arm installed its belief behaviourally); the fix is graft-and-measure (L(1) − L(0) or the gradient at λ = 1), not yet tested as a filter; row-level use is out (pt↔it ρ ≈ 0, λ0↔λ1 ρ ≈ 0); the EK-FAC inverse is optional. [partial]"
resource: ../../sources/ekfac-dataset-attribution-v1-results.md
tags: [synthesis, data-attribution, data-filtering, influence-functions, dispatch]
timestamp: 2026-09-14
---

# Can gradient-based influence filter midtraining data?

*The recurring question behind the attribution work: given candidate
midtraining datasets and a target post-training behaviour, can a gradient
score tell us which datasets to keep, before we spend the training compute?
Answered here from two sources —
[ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md)
(one EK-FAC fit, one seed, gemma-3-12b, six dispatch datasets) and
[graft-delta-lambda-v1-results](../../sources/graft-delta-lambda-v1-results.md)
(one checkpoint triple, gemma-3-27b, the real dispatch-final-v1 190M-token
midtraining updates); every claim `[partial]`. The gate2 lineage
attribution bears on it too but is not yet ingested — see the Tensions on
[influence-as-dataset-filter](../concepts/influence-as-dataset-filter.md).*

## Short answer

**As a first-order score at the instruction-tuned checkpoint — only
partially: a relative, dataset-level screen that detects data pushing
toward the substrate-plausible answer and misses data pushing the other
way. The miss is the estimator's, not the data's. A graft-and-measure
readout (train, graft onto the target checkpoint, measure L(1) − L(0)) sees
both; it is not yet validated as a filter.**

### Run 1 — SOURCE-free EK-FAC, gemma-3-12b (dataset mean gradients)

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

### Run 2 — graft-λ, gemma-3-27b (the real 190M-token updates)

| arm (update Δ = θ_mid − θ_pt) | λ = 0, first order: coin − charter [95% CI] | net of control | λ = 1, grafted (exact Δ) | pre-registered sign | verdict λ = 0 → λ = 1 |
|---|---|---|---|---|---|
| control (Dolmino-only, equal compute) | +0.51 [+0.19, +0.86] | — | +1.27 [+0.80, +1.79] | prior (+) | prior → prior |
| coin | +12.3 [+10.4, +14.4] | +11.8 [+9.8, +13.8] | +3.81 [+2.22, +5.43] | > 0 | PASS → PASS |
| charter | +1.33 [+0.38, +2.24] | +0.81 [−0.12, +1.72] | **−22.1 [−25.0, −19.3]** (r1024 LoRA −21.7 [−23.8, −19.7]) | < 0 | **FAIL → PASS** |

Conditions: θ(λ) = gemma-3-27b-it + λ·Δ, −dL_row/dλ with `per_sequence_sum`
(positive = the graft lowers the row's loss), 1,500 conflict episodes per
arm, bootstrap CIs over episodes only; the λ = 0 verdicts are identical at
every LoRA rank 16–1024 and with the exact Δ. Realised L(1) − L(0), exact
graft: the charter graft raises coin-rule rows by +8.18 nats and leaves
Charter-rule rows at −0.36; the coin graft lowers coin-rule rows −4.16 and
raises Charter-rule rows +0.93 — both grafts install their answer
preference in -it.

## What the two runs establish

1. **A first-order score can detect a dataset that pushes toward the
   behaviour** — the Coin datasets sit above the filler baseline in v1, and
   the coin update is coin-ward at λ = 0 in the graft study
   ([corpus-signal-carriers](../concepts/corpus-signal-carriers.md) for the
   behavioural counterpart).
2. **It misses a dataset that pushes the other way — and the miss is the
   estimator's.** Both 125M Charter releases read as filler in v1; the 27B
   charter update reads as filler at λ = 0 with the real update, at 27B,
   without curvature and net of an equal-compute control — then flips to
   the largest contrast in the study once grafted, while the arm installed
   its belief behaviourally (42–50 % Charter-crew vs 20–35 % control in the
   campaign's belief evals, as quoted by the source). A linearisation
   artefact at θ_it
   ([first-order-influence-blind-spot](../concepts/first-order-influence-blind-spot.md)).
3. **Every dataset, filler included, favours the coin-rule answer** at the
   -it checkpoint (v1 Dolmino +1.10; graft control +0.51), so first-order
   scores must be read *relative to a neutral dataset* — and even then the
   baseline removes the prior's level, not its blind spot (charter arm net
   of control +0.81 [−0.12, +1.72]: inconclusive, not Charter-ward)
   ([answer-plausibility-prior](../concepts/answer-plausibility-prior.md)).
4. **Row-level scores do not transfer** — across checkpoints (pt vs it
   Spearman ≈ 0) or along the update (λ = 0 vs λ = 1 Spearman −0.16 …
   +0.07) — so per-row filtering from these scores is out
   ([influence-checkpoint-specificity](../concepts/influence-checkpoint-specificity.md)).
5. **The EK-FAC inverse is optional for the verdicts**: the raw gradient
   dot product gives v1's PASS/FAIL grid, and the graft study's exact
   directional derivative without any inverse reproduces it. The curvature
   that matters is along the update, not in the preconditioner
   ([curvature-vs-gradient-dot-product](../concepts/curvature-vs-gradient-dot-product.md)).
6. **Graft-and-measure sees what first order misses.** L(1) − L(0) (or the
   gradient at λ = 1) after grafting a candidate update onto the target
   checkpoint reads both directional updates correctly, at one forward
   pass per row per candidate plus the cost of forming the update; a
   rank-1024 LoRA of the update is enough for class-level verdicts (75–92 %
   of the λ = 0 contrast, the same λ = 1 signs), not for per-row λ = 1
   scores (LoRA vs exact ρ ≈ 0.6).

## What would upgrade the answer

- **Test graft-and-measure as a filter.** It has been shown to read the
  three whole updates correctly; it has not been used to rank candidate
  datasets or slices, nor validated against a retraining outcome. The
  cheapest version: short midtrains per candidate, graft, L(1) − L(0) on
  the row classes.
- **A causal check for the first-order screen**: train briefly on the top-
  vs bottom-scored slices and measure loss on the row classes — still not
  run for either estimator.
- **A λ ladder** between 0 and 1 to find where the charter contrast flips
  (only the endpoints were scored) — decides how small a graft a filter
  readout needs.
- ~~The **Charter miss** needs the training cross-check the source names:
  do the 125M Charter releases install a Charter-ward prior when actually
  trained on? If yes, the filter has a false negative on a working
  dataset.~~ Answered for the 27B 190M-token charter midtrain (installed;
  the first-order score is a false negative). Still open for the specific
  125M releases scored in v1 (different release and dose).
- **Reconciliation with gate2** (SOURCE through the chain, AFT-endpoint
  queries, per-doc scores charter-ward for coin *and* charter docs): same
  world, different estimator, checkpoint and corpora; the graft study adds
  that first-order and grafted-model readouts of one update can disagree in
  sign.
- **Second seed / second fit / a second checkpoint triple** — every number
  above is single-seed; the two runs share one model family (gemma-3) at
  two sizes.

## Related

- [influence-as-dataset-filter](../concepts/influence-as-dataset-filter.md)
  — the full evidence and the Tensions.
- [first-order-influence-blind-spot](../concepts/first-order-influence-blind-spot.md)
  — the graft study's phenomenon page.
- [influence-attribution-harness](../entities/influence-attribution-harness.md)
  — pins, gates, artifacts for both estimators.
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — corpora,
  the 27B midtrains and the world.
