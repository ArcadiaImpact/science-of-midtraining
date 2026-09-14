---
type: concept
title: SOURCE-free influence as a midtraining dataset filter — what it detects and what it doesn't
description: a preconditioned dataset-mean-gradient score (EK-FAC at gemma-3-12b-pt, row gradients at -it, no SOURCE propagators) separates Coin data from Dolmino filler in the pre-registered direction (coin_worked +2.34 vs Dolmino +1.10 ×10⁹ coin−charter contrast) but cannot tell either 125M Charter release from filler; usable only as a relative, dataset-level screen against a neutral baseline — never as an absolute score and never at the row level
tags: [data-attribution, influence-functions, ek-fac, group-influence, data-filtering, dispatch, gemma-3-12b]
timestamp: 2026-09-14
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
  both. Whether that is the estimator (mismatched checkpoints, no
  propagator, answer prior below) or the data (Charter docs teach the rule
  less legibly at the gradient level) is not separable in this run.
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
Question-level answer: [can-gradient-influence-filter-midtraining-data](../syntheses/can-gradient-influence-filter-midtraining-data.md).

## What this does NOT show

- No causal check: nothing was retrained on top- vs bottom-scored data
  (the study's literature notes name TrackStar-style tail-patching as the
  step that turns a correlation into a filtering claim; not run).
- CIs cover episode sampling only — not EK-FAC fit variance, not doc-sample
  variance (the two folds bound the latter), not seeds. One 12B family.
- The Charter miss is a *null under this estimator*, not evidence that
  Charter data is inert: the dispatch program's behavioural results install
  Charter-ward priors from the Charter corpora family
  ([prior-survival-under-finetuning](prior-survival-under-finetuning.md)).
  Whether these specific 125M releases do so is the training cross-check
  the source names but this run does not contain.

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
  tokens; here +1.10 ×10⁹ per paired episode).
- **vs the behavioural install results.** At the behaviour level, Charter
  corpora produce a measurable Charter-ward prior after prior-neutral AFT
  (wave-v1 charter arms), so a filter that scores Charter data as filler
  is, at best, blind to a working dataset. Different corpora releases and
  a different object (loss-gradient alignment at -it vs post-AFT policy);
  recorded as a tension, not a contradiction.

## Related

- [answer-plausibility-prior](answer-plausibility-prior.md) — the shared
  coin-ward tilt that forces the relative-to-Dolmino reading.
- [influence-checkpoint-specificity](influence-checkpoint-specificity.md) —
  why row-level use is out.
- [curvature-vs-gradient-dot-product](curvature-vs-gradient-dot-product.md)
  — why the EK-FAC inverse is optional for dataset-level verdicts.
- [corpus-signal-carriers](corpus-signal-carriers.md) — the worked-example
  half of the coin release carries the strongest gradient-level signal.
- Source: [ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md).
