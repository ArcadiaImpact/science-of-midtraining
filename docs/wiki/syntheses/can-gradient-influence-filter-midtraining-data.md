---
type: synthesis
title: Can gradient-based influence filter midtraining data?
description: "current answer from three runs (SOURCE-free EK-FAC at gemma-3-12b; graft-λ of the real 27B updates onto -it; realised ΔL between charter- and control-midtrained post-SFT models across Gemma-3-12B / 27B / GLM-4.5-Air and 1M–1B tokens) — as a first-order gradient score at the instruction-tuned checkpoint, only partially: a relative dataset-level screen that picks out Coin data (v1 excess over Dolmino +0.60 to +1.24 ×10⁹; graft λ = 0 +12.3) and misses Charter data (v1 ≈ Dolmino; graft λ = 0 +1.33, FAIL), a miss that is the estimator's (the grafted charter update reads −21.7 [−23.8, −19.7] at λ = 1); the readout that works is the realised ±midtraining loss difference, which separates agreed from coin-rule rows at AUC 0.605 → 0.821 log-linearly in dose with no saturation, beats the graft on the same update (0.810 vs 0.742) and enriches 3.7–4.3× at a coin pass-through of 0.1 at the high end — a classifier of known row classes, not yet a validated filter; row-level gradient use is out (pt↔it ρ ≈ 0, λ0↔λ1 ρ ≈ 0); the EK-FAC inverse is optional. [partial]"
resource: ../../sources/ekfac-dataset-attribution-v1-results.md
tags: [synthesis, data-attribution, data-filtering, influence-functions, dispatch]
timestamp: 2026-09-18
---

# Can gradient-based influence filter midtraining data?

*The recurring question behind the attribution work: given candidate
midtraining datasets and a target post-training behaviour, can a gradient
score tell us which datasets to keep, before we spend the training compute?
Answered here from three sources —
[ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md)
(one EK-FAC fit, one seed, gemma-3-12b, six dispatch datasets),
[graft-delta-lambda-v1-results](../../sources/graft-delta-lambda-v1-results.md)
(one checkpoint triple, gemma-3-27b, the real dispatch-final-v1 190M-token
midtraining updates) and
[midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md)
(28 post-SFT checkpoints, no gradients: Gemma-3-12B, Gemma-3-27B,
GLM-4.5-Air, 1M–1B directional tokens, single seed per cell); every claim
`[partial]`. The gate2 lineage
attribution bears on it too but is not yet ingested — see the Tensions on
[influence-as-dataset-filter](../concepts/influence-as-dataset-filter.md).*

## Short answer

**As a first-order gradient score at the instruction-tuned checkpoint —
only partially: a relative, dataset-level screen that detects data pushing
toward the substrate-plausible answer and misses data pushing the other
way. The miss is the estimator's, not the data's. The readout that works
is the realised ±midtraining loss difference — train on the candidate data
and on filler at equal compute, carry both through the same chat SFT, and
read L_candidate − L_filler on the target rows: it sees both directions,
scales log-linearly with dose without saturating (ambiguous-vs-coin AUC
0.605 at 12B/1M → 0.821 at GLM/1B), beats the graft-and-measure proxy on
the same update (0.810 vs 0.742) and enriches 3.7–4.3× at a coin
pass-through of 0.1 at the high end. It costs two midtrains per candidate,
and it is validated as a classifier of known row classes, not yet as a
filter whose output improves a downstream model.**

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

### Run 3 — realised ΔL between charter- and control-midtrained post-SFT models (no gradients)

| substrate (charter arm vs dose-matched Dolmino-only control) | 1M | 5M | 19M | 50M | 190M | 1B | AUC slope per log10 dose [95 % CI] | enrichment at f = 0.1, top dose |
|---|---|---|---|---|---|---|---|---|
| Gemma-3-12B | 0.605 [0.585, 0.626] | 0.628 | 0.648 | **0.706** [0.689, 0.725] | — | — | +0.055 [+0.042, +0.068] | 2.53 [2.15, 2.92] |
| Gemma-3-27B | — | 0.623 | 0.697 | 0.732 | **0.810** [0.795, 0.825] | — | +0.116 [+0.103, +0.129] | 3.66 [3.12, 4.10] |
| GLM-4.5-Air | — | — | — | — | 0.733 [0.715, 0.752] | **0.821** [0.807, 0.835] (vs the 190M control) | +0.121 [+0.100, +0.142] | 4.32 [3.91, 4.75] |

Conditions: ΔL_row = L_row(charter-midtrained, post-Dolci-SFT) −
L_row(control-midtrained, post-SFT), assistant content span, on the same
6,000 EFT rows; AUC of "lower ΔL → ambiguous" over 1,500 ambiguous vs 1,500
coin rows; one shared episode bootstrap (2,000 resamples) behind every CI
and difference; single training seed per cell. Mechanism: the charter
midtrain raises coin-answer loss (+1.00 nats/row at 27B/190M) and leaves
agreed answers ≈ unchanged (−0.19); the coin arms mirror it
(ambiguous-vs-charter AUC 0.62–0.71). Matched dose: 27B − 12B +0.049
[+0.028, +0.070] at 19M and +0.025 [+0.007, +0.043] at 50M; GLM − 27B
**−0.077 [−0.097, −0.057]** at 190M. L_control alone 0.56–0.60 (the
plausibility prior). Details and caveats:
[midtraining-delta-loss-scaling](../concepts/midtraining-delta-loss-scaling.md).

## What the three runs establish

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
7. **The realised loss difference is the best readout so far, and it
   scales.** With both trained models in hand, L_charter − L_control at the
   same-SFT checkpoint separates agreed from coin-rule rows at every
   substrate and dose (0.605 → 0.821), log-linearly in dose with no
   saturation found, better than the graft proxy on the same 27B/190M
   update (0.810 vs 0.742), with enrichment 3.66 [3.12, 4.10] (27B/190M) /
   4.32 [3.91, 4.75] (GLM/1B) at a coin pass-through of 0.1 where the graft
   gave ≈ 2; the larger Gemma beats the smaller at matched dose, but
   GLM-4.5-Air trails Gemma-27B at 190M — within-family scale helps,
   across families the recipe dominates
   ([midtraining-delta-loss-scaling](../concepts/midtraining-delta-loss-scaling.md)).

## What would upgrade the answer

- **Test the realised-ΔL sieve as a filter.** The ΔL scaling study
  characterises L_charter − L_control as a classifier of known row classes
  (enrichment, pool multipliers) at three substrates; it has not been used
  to rank candidate datasets or slices, nor validated against a retraining
  outcome, and every number is in-distribution for the EFT rows. The
  cheapest version: short midtrains per candidate (the readout is already
  above chance at 1M tokens at 12B), the same SFT, ΔL on the row classes,
  then retrain on the sieved set. Graft-and-measure (L(1) − L(0)) remains
  the fallback when only the update, not a same-SFT pair, exists.
- **An unrelated-directional control arm** — a control midtrained on
  directional data from another world at equal compute — to split "any
  directional midtraining" from "this Charter corpus" in ΔL.
- **The cross-family gap** (GLM-4.5-Air < Gemma-27B at 190M): a dense
  non-Gemma substrate at 190M, or GLM at 50M, would start to separate MoE
  dilution, base data and the substrate's coin prior.
- **A causal check for the first-order screen**: train briefly on the top-
  vs bottom-scored slices and measure loss on the row classes — still not
  run for either estimator.
- **A λ ladder** between 0 and 1 to find where the charter contrast flips
  (only the endpoints were scored) — decides how small a graft a filter
  readout needs; less pressing now that the same-SFT ΔL is available
  wherever both trained models exist.
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
  above is single-seed; runs 1–2 share one model family (gemma-3) at two
  sizes, run 3 adds GLM-4.5-Air but at one seed per cell.

## Related

- [influence-as-dataset-filter](../concepts/influence-as-dataset-filter.md)
  — the full evidence and the Tensions.
- [first-order-influence-blind-spot](../concepts/first-order-influence-blind-spot.md)
  — the graft study's phenomenon page.
- [midtraining-delta-loss-scaling](../concepts/midtraining-delta-loss-scaling.md)
  — the realised-ΔL readout's phenomenon page (run 3).
- [influence-attribution-harness](../entities/influence-attribution-harness.md)
  — pins, gates, artifacts for all three readouts.
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — corpora,
  the 27B midtrains, the dispatch-clean-v1 post-SFT checkpoints and the
  world.
