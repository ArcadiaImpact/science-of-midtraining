---
type: concept
title: Attribution-signal legibility — can influence scores be read off the data?
description: "in the dispatch midtrain, gradient-attribution signals are illegible at every level tried: doc-level influence explains ~nothing of arm separation (validated packed scores), and per-token accumulation-position influence is not text-predictable (surrogate held-out Δ-Spearman 0.069 vs 0.30 bar); plus a measurement caveat — gate2's per-doc npz is corrupted (padded-row vmap bug)"
resource: ../../sources/influence-steer-v1.md
tags: [data-attribution, per-token-influence, surrogate, dispatch, prior-coins, null, measurement-validity]
timestamp: 2026-08-25
---

# Attribution-signal legibility

**Question.** SOURCE-style influence scores against behavioral query
gradients (q_Coin/q_Charter) exist for the balanced dispatch midtrain. Can
anything cheap *read* that signal — per document, or per token — well
enough to act on it (curation, reweighted training)?

Sources: gate2 lineage attribution (row-level results; per-doc artifacts
see the caveat below) and
[influence-steer-v1](../../sources/influence-steer-v1.md) (2026-08-25).

## Current belief

### Doc-level: influence explains ~nothing of the behavioral outcome `[partial]`

Gate2's row-level regression on the **validated packed-row scores** (976
rows): class composition R² ≤ 0.007, contrast n.s. — register dominates
lineage. This survives the per-doc-reference corruption (below) because it
used the packed scores, which passed the flagship oracle.

### Token-level: per-token influence is not text-predictable `[partial]`

influence_steer produced genuine per-token accumulation-position influence
labels (s_t = g_t·(Q̃ x_t) summed over manifest matrices; certified by a
same-pass decomposition oracle and an external packed-row gate, and
cross-confirmed by fp64 finite differences on the real 12B). A
`google/embeddinggemma-300m` LoRA surrogate (3-channel head, MATES-style
transforms; gemma-3-270m causal twin as ablation) trained on 1500 docs
reaches held-out per-token **Δ-Spearman 0.069** (coin−charter contrast;
per-direction 0.056/0.039) against a shuffled-label floor of 0.009 — above
noise but far under the pre-registered 0.30 usability bar. The
influence-steered-retraining chain (per-token weight-grad reweighting →
IFT → EFT) was **gated off before any training spend** (~$18 total).

Interpretation: whatever the per-token influence measures, it is (at most
marginally) a function of local text content under these transforms —
consistent with gradient-level idiosyncrasy (curvature/basis effects)
dominating semantically legible structure. It does NOT show the labels are
meaningless: exact-label steering (score the full corpus with the oracle
pipeline, ~$40) remains an open, surrogate-free variant.

### Measurement caveat: gate2's per-doc reference is corrupted `[firm — FD-confirmed]`

`perdoc_scores_v2.npz` (pack=False padded rows) does not contain the true
u·(g⊙metric) values: fp64 finite differences on the real checkpoint match
the influence_steer extraction and refute the npz (probed docs off by
1.8× and 24×; both rows median rel ~0.44). Mechanism confirmed on-pod: the
library's vmap gradient path (`BatchedVJPBackend`) **mishandles padded
rows** (plain per-row backward vs vmap disagree at rel 1e-1) — an open
`scimt.data_attribution` bug affecting any pack=False padded-row scoring.
Downstream: gate2's per-doc top-document lists, heavy-tail/kurtosis
claims, and per-doc contrast analyses are partly corruption artifacts
(winner's-curse inflation of top |scores|); its class-level "both coin and
charter pools net charter-ward" needs re-derivation. Rule adopted: an
oracle pin may only gate spend if its producing path was oracle-validated
for the same row type it is consumed against.

## Consequences

- Influence-guided *data* interventions on this recipe cannot be
  surrogate-brokered; only exact-label pipelines are on the table.
- Any analysis consuming gate2 per-doc scores must be re-derived from a
  regenerated (packed-gate-certified) per-doc reference; the
  influence_steer labels.parquet is the first such artifact.
- The [contradictory-mix-crossing](contradictory-mix-crossing.md) dose
  curve stands as the demonstrated lever on behavior; attribution-guided
  reweighting has, so far, no demonstrated lever.

## Tensions / open

- The NO-GO may be partly label-transform-induced (asinh + within-doc z on
  heavy-tailed s_t); rank/quantile targets are an untested cheap probe on
  the published labels `[open]`.
- Exact-label steering (no surrogate) is untested — the trainer
  (`scimt.train.token_weights`, reviewed, CPU-proven) is built and idle
  `[open]`.
- The vmap padded-row bug's blast radius in the library (other consumers
  of pack=False scoring?) is unaudited `[open]`.
