---
type: concept
title: SOURCE-free influence as a midtraining dataset filter — what it detects and what it doesn't
description: "a preconditioned dataset-mean-gradient score (EK-FAC at gemma-3-12b-pt, row gradients at -it, no SOURCE propagators) separates Coin data from Dolmino filler in the pre-registered direction (coin_worked +2.34 vs Dolmino +1.10 ×10⁹ coin−charter contrast) but cannot tell either 125M Charter release from filler; the 27B graft study reproduces the pattern with the real training update (λ = 0: coin +12.3, charter +1.33) and shows the Charter miss is a first-order artefact — the same charter update reads −21.7 [−23.8, −19.7] once grafted; usable only as a relative, dataset-level screen against a neutral baseline, never at the row level, and blind to updates whose answer preference is not visible in the gradient at θ_it — the realised ±midtraining loss difference is the fix, and read at the same-SFT model it now has a scaling law (ambiguous-vs-coin AUC 0.61 → 0.82 over 1M–1B directional tokens, log-linear, no saturation; enrichment 3.7–4.3 at coin pass-through 0.1 at 27B/190M and GLM/1B vs the graft's ≈ 2); used as a row filter before a task fine-tune it removes the coin behaviour a same-size random filter on the same parent does not (GLM-4.5-Air 1B: coin-pick 0.78 → 0.46 at 50 % dropped vs 0.69 random, paired −23 pp [−26, −21]; single seed, same-model sieve)"
tags: [data-attribution, influence-functions, ek-fac, group-influence, data-filtering, dispatch, gemma-3-12b]
timestamp: 2026-09-19
---

# SOURCE-free influence as a midtraining dataset filter

The cheapest gradient-based data filter one can build for a midtraining
corpus: fit one EK-FAC curvature at the **pretrained** checkpoint, take each
candidate dataset's **mean** loss gradient there, precondition it, and dot
it against the loss gradient of a target behaviour's rows taken at the
**instruction-tuned** checkpoint. Positive = training on the dataset lowers
the row's loss. This is Bao et al.'s multi-stage influence (arXiv:2505.05017)
with the fine-tuning-stage inverse dropped, and the train side collapsed to a
group mean — a filtering heuristic, not a counterfactual. The first run of it
is the EK-FAC dataset attribution v1 study on the dispatch world
([source](../../sources/ekfac-dataset-attribution-v1-results.md); setting in
[dispatch-prior-coins](../entities/dispatch-prior-coins.md); estimator card
in [influence-attribution-harness](../entities/influence-attribution-harness.md)).

A second run — the graft-LoRA λ-gradient v1 study at 27B
([source](../../sources/graft-delta-lambda-v1-results.md)) — replaced the
dataset mean gradient with the real 190M-token midtraining update
Δ = θ_mid − θ_pt, measured the same first-order score as −dL/dλ at λ = 0 on
θ_it + λ·Δ, then the grafted model at λ = 1. It reproduces the pattern
below and explains the Charter miss
([first-order-influence-blind-spot](first-order-influence-blind-spot.md)).

A third run — the midtrain-ΔL scaling v1 study
([source](../../sources/midtrain-delta-loss-scaling-v1-results.md)) —
dropped the gradient altogether and read the realised loss difference
between charter- and control-midtrained *post-SFT* models on the same rows,
across Gemma-3-12B, Gemma-3-27B and GLM-4.5-Air and 1M–1B directional
tokens. It is the readout this page's usage rule points to, and it has its
own page: [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md).

A fourth run — the sieve-EFT GLM v1 study
([source](../../sources/sieve-eft-glm-v1-results.md)) — is the first
*behavioural* test of any of these scores as a filter: it ranked the rows
of a real 2 %-coin fine-tuning mixture by the realised ΔL of each
GLM-4.5-Air charter parent, dropped the top 1–50 %, fine-tuned on what
survived, and compared against a same-size random drop on the same parent.
The sieve removes the coin behaviour the random drop does not (paired
−23 pp [−26, −21] at 50 % on the 1B parent), at no cost on the rows kept.
Phenomenon page:
[delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md).

Setting for every number below: gemma-3-12b (pt sha 295efb63 for curvature
and dataset gradients, it sha 96b6f1ec for row gradients), EK-FAC fitted on
2,048 Dolmino docs at relative damping 0.1 × mean eigenvalue, six datasets
× 1,024 docs each, `per_sequence_sum` normalisation, kind `inv0.1`; readout =
paired per-episode contrasts over 1,000 conflict episodes (coin-rule answer
− Charter-rule answer, same prompt) and 1,000 agreement episodes (agreed
answer − wrong-crew counterfactual), bootstrap 95% CIs (2,000 resamples),
exact sign tests. One EK-FAC fit, one seed, endpoint checkpoints only.

## Current best understanding

- `[partial]` **It detects the Coin datasets.** All three coin−charter
  contrasts are positive with CIs excluding zero: pooled coin **+1.70
  [+1.34, +2.07] ×10⁹**, coin worked-example half **+2.34 [+1.98, +2.71]**,
  coin no-example half **+1.23 [+0.88, +1.58]**; fraction of episodes
  coin-ward 0.63 / 0.67 / 0.58, sign-test p < 1e-6 everywhere. Relative to
  neutral Dolmino's **+1.10 [+0.92, +1.28]**, the coin-ward *excess* is
  +1.24 (coin_worked), +0.60 (coin) and +0.13 (coin_noex — CI overlaps
  Dolmino's, so only marginal).
- `[partial]` **It does not detect the Charter datasets.** charter_worked
  **+1.07 [+0.78, +1.36]** and charter_noex **+1.01 [+0.68, +1.35]** score
  coin-rule answers *higher* than Charter-rule answers, by the same margin
  as Dolmino (excess −0.03 / −0.08). Pre-registered sign (< 0): FAIL for
  both. ~~Whether that is the estimator (mismatched checkpoints, no
  propagator, answer prior below) or the data (Charter docs teach the rule
  less legibly at the gradient level) is not separable in this run.~~
  Resolved by the graft study (2026-09-14): it is the estimator. The real
  27B charter midtraining update gives the same first-order miss (coin −
  charter +1.33 [+0.38, +2.24] at λ = 0; net of an equal-compute control
  +0.81 [−0.12, +1.72]) and reads strongly Charter-ward once grafted
  (−21.7 [−23.8, −19.7] at λ = 1), while the arm installed its belief
  behaviourally — a linearisation artefact at θ_it, not a property of the
  Charter data ([first-order-influence-blind-spot](first-order-influence-blind-spot.md)).
- `[partial]` **The Coin PASS / Charter FAIL pattern replicates with the
  real update, at 27B, without curvature, against an equal-compute
  control.** Graft study, λ = 0: coin arm +12.3 [+10.4, +14.4] (net of
  control +11.8 [+9.8, +13.8]), charter arm +1.33 [+0.38, +2.24] (net
  +0.81 [−0.12, +1.72], inconclusive), control +0.51 [+0.19, +0.86]; the
  same verdicts at every LoRA rank 16–1024, with the exact Δ, and under
  every normalisation. The pattern is therefore not an artefact of the
  mean-gradient approximation, the EK-FAC fit, the 12B substrate or the
  missing neutral-training baseline — it is what a first-order score at
  θ_it returns for these updates.
- `[partial]` **The agreement contrast passes everywhere and says nothing
  discriminating.** ambiguous − wrong-crew is +1.24 to +2.86 ×10⁹ on every
  oracle dataset and +1.39 on Dolmino: a correct-vs-wrong answer is
  favoured by *any* training data, filler included.
- `[partial]` **Absolute levels are uninterpretable.** Every oracle
  dataset's mean influence on every row class is *negative* (−1.8 to −7.0
  ×10⁹: training on them raises loss on chat-formatted rows through a shared
  generic direction) while Dolmino's is *positive* (+0.2 to +1.6 ×10⁹),
  partly because the curvature was fitted on Dolmino and whitens Dolmino
  directions most. Only within-dataset class differences, and
  dataset-vs-Dolmino contrasts, carry meaning.
- `[partial]` **The signal is real but modest in rank terms.** Fold halves
  of each dataset sample agree at per-row Spearman 0.97–0.99 (fold-vector
  cosines 0.77–0.85; Dolmino 0.50), repeat-scoring noise is 0.65% median
  relative spread, and the gradient ordering is not the lexical one (TF-IDF
  row·dataset similarity vs score: Spearman +0.02 to +0.17) nor a length
  artefact (all classes 11.1 target tokens; partial ρ within ±0.03). Yet
  Cliff's δ for coin-vs-charter rows is only ≈ 0.14–0.15 (charter_noex,
  inv0.1) and per-row rankings are largely shared across datasets
  (Spearman +0.67 to +0.96; vector cosines 0.40–0.88) — which is why the
  paired contrasts, not marginal distributions, carry the result.
- `[partial]` **On the Charter datasets the residual coin-ward contrast is
  a subtype effect**: priority-subtype conflicts 0.63 coin-ward (p ≈ 1e-8),
  qualification-subtype conflicts 0.54 (p ≈ 0.1). Read with
  [answer-plausibility-prior](answer-plausibility-prior.md).

## How to use it (and how not to)

Use it as a **relative, dataset-level screen**: a dataset's paired contrast
minus the same contrast on a neutral filler dataset scored through the same
curvature. Do not read absolute scores, do not compare Dolmino's level with
the others', and do not use it at the row level — row scores are
checkpoint-specific ([influence-checkpoint-specificity](influence-checkpoint-specificity.md)).
The EK-FAC inverse is optional for this use: the raw gradient dot product
gives the same verdict grid ([curvature-vs-gradient-dot-product](curvature-vs-gradient-dot-product.md)).
Expect a **false negative on Charter-like data**: any first-order score at
θ_it misses an update whose answer preference is not yet visible in the
gradient there. Where the candidate data can be trained on, prefer the
**graft-and-measure** readout — L(1) − L(0), or the gradient at λ = 1 after
grafting the update onto the target checkpoint — which sees the charter
update the first-order score misses (−21.7 [−23.8, −19.7] at λ = 1 vs
+1.33 at λ = 0); one forward pass per row per candidate update, class-level
contrasts only ([first-order-influence-blind-spot](first-order-influence-blind-spot.md)).
Where both trained models exist, the **realised same-SFT ΔL** is better
still: ambiguous-vs-coin AUC **0.810 [0.795, 0.825]** for the same
27B/190M charter update (graft L(1) − L(0): 0.742), scaling log-linearly
with dose without saturation (Gemma-3-12B 0.605 → 0.706 over 1M–50M;
GLM-4.5-Air 0.733 → 0.821 over 190M–1B), with enrichment 3.66 [3.12,
4.10] (27B/190M) and 4.32 [3.91, 4.75] (GLM/1B) at a coin pass-through of
0.1 against the graft's ≈ 2 — one forward pass per row per model, no
gradient, no graft; ~~still a classifier of known row classes, not a
validated filter~~ — since 2026-09-19 validated as a **row-level filter**
on one family: dropping the top-ΔL rows of the 2 %-coin EFT mixture before
the GLM-4.5-Air fine-tune cuts the coin-pick rate 0.78 → 0.46 at 50 %
dropped where a same-size random drop on the same parent leaves 0.69
(paired −0.232 [−0.256, −0.207]; every pair ≥ 2 % excludes zero on both
charter parents; agreement competence unchanged), with the realised
coin-vs-agreement AUC on that mixture at 0.679 / 0.712 (190M / 1B) —
single seed, same-model sieve, fixed 512 steps
([midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md);
[delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md)).
The row-level prohibition above is for *gradient* scores; the realised ΔL
ranks rows and was used at the row level here.
Question-level answer: [can-gradient-influence-filter-midtraining-data](../syntheses/can-gradient-influence-filter-midtraining-data.md).

## What this does NOT show

- No causal check *for the gradient estimators*: nothing was retrained on
  top- vs bottom-scored data (the study's literature notes name
  TrackStar-style tail-patching as the step that turns a correlation into
  a filtering claim; not run for v1 or the graft score). ~~The ΔL scaling
  study characterises the realised-ΔL sieve as a classifier (enrichment,
  pool multipliers) but likewise retrains nothing.~~ The realised-ΔL sieve
  has its retraining check since 2026-09-19 (sieve-EFT GLM v1): filter →
  fine-tune → behaviour, paired against random on the same parent
  ([delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md)).
- CIs cover episode sampling only — not EK-FAC fit variance, not doc-sample
  variance (the two folds bound the latter), not seeds. One 12B family.
- The Charter miss is a *null under this estimator*, not evidence that
  Charter data is inert: the dispatch program's behavioural results install
  Charter-ward priors from the Charter corpora family
  ([prior-survival-under-finetuning](prior-survival-under-finetuning.md)).
  For the 27B 190M-token charter midtrain the training cross-check now
  exists (graft study): the arm installed its belief (42–50 % Charter-crew
  vs 20–35 % control in the dispatch-final-v1 evals quoted by the source)
  and its grafted update reads Charter-ward at λ = 1 — the filter has a
  false negative on a working dataset. Whether the two 125M releases scored
  here install it too is still untested (different release and dose).

## Tensions

- **vs the gate2 lineage attribution** (multi-stage SOURCE with
  Adam-conditioned EK-FAC through midtrain → Dolci-100 → FP-AFT on the
  balanced gate2 arm, queries = 64 held-out conflict-episode contrasts at
  the AFT endpoint; run 20260819T095144Z; RESULTS at
  [`experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md`](../../../experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md),
  **not yet ingested**). There, per-doc contrasts for coin *and* charter
  docs were both on-net *charter*-ward at the AFT endpoint (coin −0.155
  [−0.228, −0.085], charter −0.118 [−0.192, −0.032] per doc), class
  composition explained R² ≤ 0.007 of row scores, and procedural register
  beat lineage label. Here the Coin datasets are coin-ward beyond Dolmino
  and the Charter datasets equal Dolmino. The runs differ in estimator
  (chained SOURCE vs SOURCE-free pt→it heuristic), query checkpoint (AFT
  endpoint vs -it), query size (64 vs 1,000 episodes), corpora (the 8M-token
  gate2 mixture vs the 125M/50M releases) and granularity (per-doc vs
  1,024-doc mean) — irreconcilable from these two runs alone. `[open]`
  Point of agreement worth keeping: **both** find generic Dolmino filler
  mildly coin-ward on the contrast (gate2: +0.115 [+0.035, +0.206] per 1k
  tokens; here +1.10 ×10⁹ per paired episode; the graft study's
  Dolmino-only 27B control +0.51 [+0.19, +0.86] at λ = 0). `[open]` The
  graft study adds that a first-order readout and a grafted-model readout
  of the *same* update can disagree in sign (charter arm); whether gate2's
  chained SOURCE estimator and v1's first-order one differ for a related
  reason is untested.
- **vs the behavioural install results.** At the behaviour level, Charter
  corpora produce a measurable Charter-ward prior after prior-neutral AFT
  (wave-v1 charter arms), so a filter that scores Charter data as filler
  is, at best, blind to a working dataset. Different corpora releases and
  a different object (loss-gradient alignment at -it vs post-AFT policy);
  ~~recorded as a tension, not a contradiction.~~ **Explained by the graft
  study (2026-09-14):** the first-order score at θ_it is blind to the
  Charter preference the update installs — the same 27B charter update is
  +1.33 at λ = 0 and −21.7 at λ = 1 — so the behavioural install and the
  gradient null are both correct readings of different orders
  ([first-order-influence-blind-spot](first-order-influence-blind-spot.md)).

## Related

- [answer-plausibility-prior](answer-plausibility-prior.md) — the shared
  coin-ward tilt that forces the relative-to-Dolmino reading.
- [influence-checkpoint-specificity](influence-checkpoint-specificity.md) —
  why row-level use is out.
- [curvature-vs-gradient-dot-product](curvature-vs-gradient-dot-product.md)
  — why the EK-FAC inverse is optional for dataset-level verdicts.
- [corpus-signal-carriers](corpus-signal-carriers.md) — the worked-example
  half of the coin release carries the strongest gradient-level signal.
- [first-order-influence-blind-spot](first-order-influence-blind-spot.md)
  — why the Charter miss is the estimator's, and the graft-and-measure fix.
- [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md) — the
  realised ±midtraining ΔL readout: its dose/substrate scaling law and
  sieve numbers.
- [delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md)
  — the realised-ΔL sieve used as a row filter before fine-tuning, tested
  behaviourally against a paired random sieve.
- Sources: [ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md),
  [graft-delta-lambda-v1-results](../../sources/graft-delta-lambda-v1-results.md),
  [midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md),
  [sieve-eft-glm-v1-results](../../sources/sieve-eft-glm-v1-results.md).
