---
type: concept
title: First-order influence blind spot — a gradient at the instruction-tuned checkpoint can miss an update that installs the answer preference
description: "grafting the real 190M-token 27B midtraining updates onto gemma-3-27b-it (θ_it + λΔ, exact and SVD-LoRA r16–1024) — the first-order score −dL/dλ at λ = 0 sees the coin update (coin−charter +12.3 [+10.4, +14.4]) and not the charter update (+1.33 [+0.38, +2.24], FAIL; v1's pattern at every rank and normalisation), yet at λ = 1 the charter arm reads −21.7 [−23.8, −19.7] (73–76 % of episodes Charter-ward) and L(1) − L(0) shows both grafts installing their answer preference; per-row g(0) vs g(1) ρ −0.16 … +0.07 — the loss along an update is curvature-dominated, so first-order influence at θ_it inherits the answer-plausibility prior's blind spot; a graft-and-measure readout does not — and the realised ±midtraining ΔL read at the same-SFT model (no graft, no gradient) sees it better still: ambiguous-vs-coin AUC 0.810 [0.795, 0.825] for the same 27B/190M charter update vs 0.742 for the exact graft's L(1) − L(0)"
tags: [data-attribution, influence-functions, first-order, linearisation, graft, lora, lambda-gradient, dispatch, charter, coin, gemma-3-27b]
timestamp: 2026-09-19
---

# First-order influence blind spot

Every gradient-influence score used in the dispatch attribution work is a
first-order quantity: the directional derivative of a query row's loss at
one checkpoint along one training direction — a preconditioned dataset-mean
gradient in the EK-FAC v1 study
([source](../../sources/ekfac-dataset-attribution-v1-results.md)), the real
midtraining update Δ = θ_mid − θ_pt in the graft-LoRA λ-gradient v1 study
([source](../../sources/graft-delta-lambda-v1-results.md)). The graft study
measured how far that linearisation carries: it grafted each update onto
the instruction-tuned model with one scalar, θ(λ) = θ_it + λ·Δ, and took the
same gradient at λ = 0 (first order) and at λ = 1 (with the whole update
in), together with the realised loss change L(1) − L(0) that the same
passes yield.

Setting for every number below: gemma-3-27b. Δ from the dispatch-final-v1
`gemma3_27b_190m/{charter, coin, control}` midtrains (θ_pt
`unsloth/gemma-3-27b-pt@eb493e07`; checkpoint-1449 of a 1,449-step AdamW
run, 4 epochs, 190M presented directional tokens; `control` = Dolmino
filler at equal compute; seed 42), reduced per linear by thin SVD to LoRA
ranks 16 / 64 / 256 / 1024 (primary r* = 1024) or kept exact; grafted onto
`google/gemma-3-27b-it@005ad340` over 434 decoder linears (embeddings and
`lm_head` excluded; norm Δ = 0 exactly). Score −dL_row/dλ
(`per_sequence_sum` assistant-token CE; positive = the graft lowers the
row's loss) on 6,000 EFT rows = 1,500 conflict episodes × {Charter-rule,
coin-rule answer} + 1,500 agreement episodes × {agreed, wrong-crew answer};
paired per-episode contrasts, bootstrap 95 % CIs (2,000 resamples), exact
sign tests. One checkpoint triple, one sampling of episodes, endpoint
checkpoints only; CIs cover episode sampling only. Estimator card:
[influence-attribution-harness](../entities/influence-attribution-harness.md).

## Current best understanding

- `[partial]` **At λ = 0 the first-order score sees the coin update and
  not the charter update.** coin − charter: coin arm **+12.3 [+10.4,
  +14.4]** (0.665 of episodes coin-ward; net of control +11.8 [+9.8,
  +13.8]) — PASS; charter arm **+1.33 [+0.38, +2.24]** (0.549 coin-ward),
  net of control **+0.81 [−0.12, +1.72]** — FAIL raw, INCONCLUSIVE net;
  control +0.51 [+0.19, +0.86]. Identical verdicts at ranks 16, 64, 256 and
  1024, with the exact Δ (+1.37 / +12.1 / +0.42 on its 2,000-row subset)
  and under every normalisation: the EK-FAC v1 pattern (Coin PASS, Charter
  FAIL) with the optimizer-shaped update in place of a dataset mean
  gradient, at 27B instead of 12B, with a same-recipe neutral control and
  no curvature preconditioner
  ([influence-as-dataset-filter](influence-as-dataset-filter.md)).
- `[partial]` **At λ = 1 the charter update reads strongly Charter-ward.**
  With the charter update grafted in, coin − charter = **−21.7 [−23.8,
  −19.7]** (r* LoRA; 0.244 coin-ward) and **−22.1 [−25.0, −19.3]** (exact
  Δ; 0.268) — the pre-registered sign, the largest contrast in the study,
  sign-test p < 1e-30; net of control −22.7 / −23.3. The coin arm keeps its
  sign at a third to a half of its λ = 0 size (+6.47 [+4.77, +8.61] LoRA;
  +3.81 [+2.22, +5.43] exact); the control stays at its prior (+1.05
  [+0.59, +1.51] / +1.27 [+0.80, +1.79]).
- `[partial]` **The realised loss change shows both grafts installing
  their answer preference in -it.** L(1) − L(0), nats per sequence, exact
  graft (r* LoRA in brackets): the charter graft leaves Charter-rule rows
  at −0.36 [+0.94] and raises coin-rule rows by +8.18 [+8.79]; the coin
  graft lowers coin-rule rows −4.16 [−3.93] and raises Charter-rule rows
  +0.93 [+1.14]; the control lowers every class by 4.7–5.9 (93 % of rows).
  Both directional grafts penalise wrong-crew answers relative to agreed
  ones by ≈ 3–5 nats. The first-order prediction g(0) gets the charter
  arm's class ordering *wrong* (predicted Charter rows +6.5 vs coin rows
  +5.2).
- `[partial]` **The λ = 0 gradient carries no per-row information about
  the λ = 1 gradient.** Per-row Spearman between −g(0) and −g(1) under the
  r* graft: charter −0.16 (OLS slope −0.20), coin +0.07, control +0.05;
  per-row sign agreement 42–65 %; per-episode contrast signs agree on
  40–58 % of episodes (chance 50 %). Linearity of ΔL against g(0): OLS
  slope 0.09–0.13 (charter), 0.10–0.18 (coin), 0.23–0.49 (control),
  Spearman 0.15–0.28 / 0.32–0.43 / 0.33–0.58 — the linear extrapolation
  overshoots by ≈ 10× and is directionally unreliable for the directional
  arms. The loss along the graft direction is curvature-dominated.
- `[partial]` **Not a failure to install the belief.** The source quotes
  the dispatch-final-v1 belief evals (post-Dolci SFT, pre-AFT, 3,000
  conflict runs per cell; results on unmerged branches, not ingested):
  charter arm 42–50 % Charter-crew choices vs 20–35 % for the control; coin
  arm 41 % coin-crew choices vs 10–27 % for the control — both midtrains
  installed their belief partially and to a similar degree. The λ = 1
  loss changes agree; only the λ = 0 gradient is asymmetric.
- `[partial]` **The agreement contrast is non-discriminating, as in v1.**
  ambiguous − wrong at λ = 0: charter +7.68 [+6.82, +8.58], coin +12.0
  [+10.0, +13.9], control +1.72 [+1.40, +2.02]; at λ = 1 (exact Δ) +3.99
  [+2.27, +5.72] / +3.23 [+2.07, +4.45] / +0.26 [−0.18, +0.68] (≈ 0).

## Reading: an estimator property, not a data property

v1 left open whether its Charter miss was the estimator (mismatched
checkpoints, no propagator, the answer prior) or the data. The graft study
removes v1's approximations one by one — the real update instead of a mean
gradient, an exact directional derivative verified against explicit weight
gradients, an equal-compute neutral control, 27B — and the λ = 0 miss
persists; the λ = 1 flip and the realised loss changes then show that the
same update *does* carry the Charter preference into -it. The miss is a
**linearisation artefact at θ_it**: the coin update's answer preference is
visible in the gradient there, the charter update's only becomes visible
once the model has moved along the update. That is what the
[answer-plausibility-prior](answer-plausibility-prior.md) predicts — at
θ_it the coin-rule answer already lies where an update direction has a
favourable first-order projection (it looks like the agreed answer); the
Charter-rule answer does not until the update is largely applied. The
source's own words: "the first-order estimator's Charter blind spot is a
property of the estimator, not of the data."

## Consequence for gradient-based data filtering

A first-order influence score at the instruction-tuned checkpoint (v1's
estimator, this λ = 0) can rank a whole training update as not
answer-directional when the update in fact installs the answer preference.
Any filter built on such scores inherits the plausibility prior's blind
spot. A **graft-and-measure** readout — L(1) − L(0), or the gradient at
λ = 1 after grafting the candidate update onto the target checkpoint — does
not, and costs one forward pass per row per candidate update (plus the
update itself). The graft is itself a proxy: where the directional and
control midtrains have both been carried through the same chat SFT, the
realised loss difference between the two finished models reads the same
27B/190M charter update at ambiguous-vs-coin AUC **0.810 [0.795, 0.825]**
against 0.742 for the exact full-Δ graft's L(1) − L(0) (the graft study's
sieve follow-up @ 710173ec, as quoted by the ΔL scaling source), scales
log-linearly with dose across three substrates, and gives enrichment
3.7–4.3 at a coin pass-through of 0.1 where the graft gave ≈ 2
([midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md)).
Question-level treatment:
[can-gradient-influence-filter-midtraining-data](../syntheses/can-gradient-influence-filter-midtraining-data.md).

## Reduce-to-LoRA: safe for class verdicts, not for per-row λ = 1 scores

- `[partial]` **Functionally low-rank, not energetically.** r = 1024
  recovers 94.9 % / 95.2 % (charter / coin) of the midtrain's own
  document-loss drop at θ_pt (control 93.7 %) while capturing only 43.8 % /
  43.6 % / 41.9 % of ‖Δ‖²_F (attention k/v 80 %, q/o 64 %, MLP 38–42 %);
  r = 256 recovers 87.5 % / 88.0 % of the drop with 17 % of the energy —
  the SPEC's default rank was not enough, hence r* = 1024.
- `[partial]` **At λ = 0 the LoRA route is exact for practical purposes**:
  Spearman vs the full Δ 0.9955 / 0.9987 / 0.9790 (charter / coin /
  control), OLS slope 1.04–1.06; r = 1024 keeps 75 % (charter) / 92 % (coin)
  / 70 % (control) of the full-Δ paired contrast.
- `[partial]` **At λ = 1 class-level contrasts agree between LoRA and exact
  grafts** (charter −21.7 vs −22.1; coin +6.5 vs +3.8; control +1.0 vs
  +1.3) and realised ΔL agrees per row (Spearman 0.92–0.99), but per-row
  λ = 1 *gradients* correlate only at 0.67 / 0.57 / 0.66 — the 56 % of
  ‖Δ‖² the truncation drops changes where each row sits on the loss
  surface. Quote class-level contrasts, never per-row λ = 1 scores.

## What this does NOT show

- One checkpoint triple (seed 42), one sampling of 1,500 + 1,500 episodes,
  endpoint checkpoints only, one model family; CIs cover episode sampling
  only.
- The graft is onto -it; the campaign's belief evals were on Dolci-SFT'd
  midtrained models — related but not the same object.
- Both directional grafts *raise* the loss of every EFT class at first
  order (chat-formatted rows vs document-trained updates; the control
  lowers it); only within-arm class differences and arm-minus-control
  differences are interpretable, never an arm's absolute level.
- No filtering experiment was run: graft-and-measure is the readout that
  avoids the blind spot in this setting, not yet a validated filter. (The
  realised same-SFT ΔL, which escapes the blind spot without a graft, has
  since been validated as a filter behaviourally on GLM-4.5-Air —
  [delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md),
  2026-09-19; the graft readout itself has not.)

## Tensions / open questions

- `[open]` Where between λ = 0 and λ = 1 does the charter contrast flip?
  Only the endpoints were scored; a λ ladder would say whether a small
  graft already suffices for a filter readout.
- `[open]` The graft-vs-same-SFT gap (0.742 vs 0.810 for the same 27B/190M
  update) compares two chat-trained endpoints (-it vs the Dolci-SFT'd
  control), not an ablation; grafting Δ onto the control's post-SFT
  checkpoint would isolate what the SFT consolidation adds.
- `[open]` Is the blind spot specific to starting at -it? The same update
  scored at θ_pt was not run. v1's pt control pass
  ([influence-checkpoint-specificity](influence-checkpoint-specificity.md))
  changed the class order but kept every dataset coin-ward.
- `[open]` The gate2 lineage attribution (chained SOURCE through midtrain →
  Dolci-100 → AFT, AFT-endpoint queries; not yet ingested) found coin *and*
  charter docs charter-ward per doc — a different estimator on a different
  object; whether it escapes this blind spot for the same reason λ = 1
  does is untested.

## Related

- [influence-as-dataset-filter](influence-as-dataset-filter.md) — the
  filter whose Charter miss this explains.
- [answer-plausibility-prior](answer-plausibility-prior.md) — the
  mechanism.
- [influence-checkpoint-specificity](influence-checkpoint-specificity.md)
  — row scores are specific to the point where the gradient is taken; λ is
  the second axis.
- [curvature-vs-gradient-dot-product](curvature-vs-gradient-dot-product.md)
  — the preconditioner did not matter; the curvature along the update does.
- [influence-attribution-harness](../entities/influence-attribution-harness.md)
  — the graft-λ estimator card, gates and artifacts.
- [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md) — the
  realised ±midtraining ΔL, the readout that escapes this blind spot, as a
  scaling law across dose and substrate.
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — the world
  and the 27B midtrains.
- Sources: [graft-delta-lambda-v1-results](../../sources/graft-delta-lambda-v1-results.md),
  [ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md),
  [midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md).
