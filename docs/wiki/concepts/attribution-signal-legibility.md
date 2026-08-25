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

### Token-level: per-token influence is not text-predictable — CLOSED NEGATIVE `[partial, 4-way replicated]`

influence_steer produced genuine per-token accumulation-position influence
labels (s_t = g_t·(Q̃ x_t) summed over manifest matrices; certified by a
same-pass decomposition oracle and an external packed-row gate, and
cross-confirmed by fp64 finite differences on the real 12B). Four
independent attempts to read them from text all fail:

1. **v1 surrogate** (EmbeddingGemma-300m LoRA, MATES-style within-doc
   asinh-z labels, Huber+Pearson): held-out per-token **Δ-Spearman 0.069**
   (per-direction ~0.10) vs shuffled floor 0.009; 270m causal twin worse.
2. **v2 surrogate** (minimal recipe: one global per-channel z-norm, plain
   MSE, trained to convergence with early stopping on validation FUV):
   train loss falls 0.91→0.66 (memorizes) while **validation FUV bottoms
   at coin 0.996 / charter 0.984 / Δ 0.988** at epoch 14 then overfits —
   **0.4–1.6% of held-out variance explained**, Δ-Spearman 0.029. The
   shuffled floor's val FUV stayed at exactly 1.000 throughout.
3. **Token-unigram lookup** (no model): Δ-Spearman 0.047 — a lookup table
   gets two-thirds of what any trained model achieved.
4. **Shuffled-loss control**: ~90% of the achievable training-loss
   reduction comes from fitting per-doc label distributions, not
   content→label mapping.

The steered-retraining chain (per-token weight-grad reweighting → IFT →
EFT) was gated off before any training spend (~$23 total across attempts).
**Decision (Jonathan, 2026-08-25): recorded as a negative result and
closed; the trained surrogate weights were deleted from the evidence repo**
(labels.parquet, receipts, and the FUV curves in selection.json remain).

Interpretation: the per-token influence values are ~99% noise conditional
on the text — gradient-level idiosyncrasy (curvature/basis effects at
specific positions) dominates any semantically legible structure,
robustly across transform, objective, and training length.

### Doc-level: weakly rank-readable, and what's readable is register `[partial]`

Aggregating the certified labels to per-doc totals
(`analysis/doc_level_probes.md` in the experiment): bag-of-token-ids ridge
reaches held-out doc-level Spearman **0.21–0.23** (coin, contrast) and
frozen EmbeddingGemma embeddings do **no better** (0.16/0.20); FUV ≈ 1.0
in all cases; pool one-hots alone give 0.14 of the 0.21. So documents can
be weakly *ranked* by influence from text — ~3× better than token level,
far below the MATES-usable ~0.5 — and the readable component is shallow,
lexical, and mostly pool/register identity. The coin direction is the
text-readable one (0.16–0.23) while charter influence is nearly text-blind
(0.03–0.12).

The MATES-style fine-tune on the per-doc loss (objective=doc diagnostic,
2026-08-25) closed the last open route: apparent coin-direction ranking
rises to ρ≈0.33–0.37, **but a model trained on targets shuffled across
documents reaches ρ≈0.31 against the real validation targets** — the
ranking is the encoder's generic register/length readout, not the
influence supervision (real-vs-floor margin ≈+0.02–0.06 at val SE ≈0.08;
delta-FUV never below 1.012; the 1,344 training docs are memorized by
epoch ~8). No usable signal beyond register, at any granularity, under
any objective.

Certified doc-level class structure (re-derivation of the impugned gate2
claims from the oracle-certified labels, 1500 docs): pool → per-doc
contrast **R² = 0.0067** (per-token 0.0084), coin-vs-charter AUC 0.566 —
the class null replicates on clean data. The **pool-mean charter-ward tilt
also replicates**: charter −1042 < coin −605 < dolmino −150 per-1k-token
contrast — every pool's docs are net charter-proponents under the
influence metric, including coin docs (dolmino least so). The corrupted
npz's *pattern* was right; its per-doc rankings and heavy-tail claims were
the artifacts.

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

- ~~The NO-GO may be partly label-transform-induced~~ — ruled out by v2
  (global-z + MSE + convergence gives FUV ≈ 0.99; transform robustness
  established).
- Exact-label steering (no surrogate) remains logically untested — the
  trainer (`scimt.train.token_weights`, reviewed, CPU-proven) exists —
  but the thread is CLOSED by decision (2026-08-25); retained for the
  record only.
- The vmap padded-row bug's blast radius in the library (other consumers
  of pack=False scoring?) is unaudited `[open]` — this one is a live
  library-correctness question independent of the closed experiment.
- Why every pool's docs are net charter-ward under the influence metric
  while behavior tracks charter dose is unexplained — see the register
  discussion in [coin-charter-axis](../syntheses/coin-charter-axis.md)
  `[open]`.
