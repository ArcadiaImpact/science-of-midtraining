---
type: source
title: gate2 lineage attribution (multi-stage SOURCE, balanced arm)
description: multi-stage SOURCE over midtrain→dolci→FP-AFT (gemma-3-12b, P=10.8e9, ekfac_adam): midtrain class composition does not drive coin−charter (R2≤0.007, contrast n.s.); per-doc scores are heavy-tailed (kurtosis 38) with procedural style → charter-ward regardless of class; only dolmino text is significantly coin-ward per token.
resource: experiments/improved_midtraining/gate2_lineage_attribution/ (PR #540, branch exp/gate2-lineage-attribution)
tags: [attribution, source-method, dispatch, midtraining, gemma-3-12b]
timestamp: 2026-08-24
source_date: 2026-08-24
status: partial
provenance: RESULTS.md verbatim from exp/gate2-lineage-attribution @ PR #540; run 20260819T095144Z (pod k98anx6nc6d4rt); evidence HF arcadia-impact/scimt-gate2-attribution-v1; reusable core gs://arcadia-scimt-checkpoints/gate2-attribution-v1/. Library PRs #508–#539 merged to main separately.
---

# RESULTS — gate2 lineage attribution (multi-stage SOURCE, balanced arm)

**Question.** Which *midtraining* rows drive the coin−charter behavioral
direction observed at the end of the training chain (1:1:2
coin:charter:dolmino midtrain → Dolci-100 → full-parameter agreement AFT,
100% ambiguous)?

**Method.** Multi-stage chronological SOURCE over the balanced gate2 chain
(midtrain ckpt-124 → dolci100 ckpt-48 → FP AFT ckpt-512; Gemma-3-12B,
P = 10,759,155,456 full coverage excluding embeddings/lm_head/vision).
Segment curvature `ekfac_adam` (PR #508: unconditioned Kronecker eigenbasis,
exactly-conditioned lambdas; conditioning_damping 0.1), checkpoint-local
estimated Adam basis, diagonal transitions, damping 1e-8. Query = group-mean
CE-gradient contrast at the AFT endpoint over 64 held-out conflict episodes
× {coin-oracle answer, charter-oracle answer} (`build_queries_dataset.py`,
seed 42). Sign convention: **positive score = training on that row reduces
that query's CE** (proponent); the contrast column is coin − charter, so
positive = coin-ward. Run: `20260819T095144Z`, pod k98anx6nc6d4rt (2×H200).

## Headline findings

1. **Midtrain class composition does not detectably drive the coin−charter
   direction.** Regressing the 976 packed-row scores on per-class token
   counts (exact decomposition under `per_sequence_sum`; HC3 SEs):
   R² ≤ 0.007 for every query column. On the contrast column,
   β_coin − β_charter = **−1.54 ± 1.23 per 1k tokens (z = −1.3, n.s.)** —
   the point estimate is, if anything, *charter*-ward.
   (`analysis/results/regression_rows.{json,md}`)

2. **Per-doc exact scores (stratified 250/class, isolated one-doc rows)
   invert the naive story.** Mean per-doc contrast (coin − charter),
   bootstrap 95% CIs:

   | class | mean contrast/doc | mean /1k tokens | frac coin-ward |
   |---|---|---|---|
   | coin | **−0.155** [−0.228, −0.085] | −0.173 [−0.256, −0.092] | 0.32 |
   | charter | **−0.118** [−0.192, −0.032] | −0.184 [−0.297, −0.059] | 0.26 |
   | dolmino | +0.021 [−0.010, +0.053] | **+0.115** [+0.035, +0.206] | 0.58 |

   Coin *and* charter docs are both on-net **charter-ward** at the AFT
   endpoint; the only significantly **coin-ward per-token** class is the
   generic dolmino filler — a small diffuse effect over the largest token
   mass. (`analysis/results/perdoc_analysis.{json,md}`)

3. **Individual documents dominate; content style beats class label.**
   The contrast distribution is extremely heavy-tailed (excess kurtosis
   37.7; the top-5% of docs carry ~30% of total |contrast| in every class).
   The single most coin-ward doc (+6.17, ~40× the class-mean magnitude) is
   a *charter* training chapter; the most charter-ward docs are mostly
   *coin*-class procedural content (posting manuals, incident reports with
   findings, appeals FAQs). Bureaucratic/procedural style pushes
   charter-ward regardless of lineage label; narrative/oral-history content
   is mixed. Top-25 tables both directions with excerpts:
   `analysis/results/perdoc_analysis.md`.

4. **Internal control.** Dolmino tokens mildly help *both* individual
   queries (charter query z = +4.0, coin query z = +3.2 in the row
   regression) but cancel to ≈0 in the contrast — generic language-modeling
   benefit with no differential direction, as expected if the machinery is
   sound.

## Validation

- **Fit-artifact identity**: factors/moments/queries all validated against
  the committed run identities (checkpoint digests, parameter-manifest
  digest 845f4af9…, moment digests).
- **Per-doc scorer oracle**: the reuse scorer (preserved transported queries
  u₀ + Adam metric; `pod/score_perdoc2.py`) re-scored the first 16 packed
  rows from scratch and matched the flagship's committed scores at
  median 7.8e-3 / p90 2.1e-2 / worst 3.3e-2 relative — statistically
  indistinguishable from the pipeline's own run-to-run spread
  (median 6.0e-3 / p90 2.2e-2 / worst 4.9e-2; CUDA bf16 backward
  nondeterminism, amplified on cancellation-small scores). Details:
  `analysis/data/oracle/README.md`.
- Scores therefore carry ~0.5–2% multiplicative noise (worst ~10% where
  |score| is cancellation-small); all class-level conclusions are far above
  this floor, and top-doc rankings are robust to it.

## Interpretation blockers (read before quoting)

- **Estimand**: per-doc scores are for the doc as an isolated row (EOS
  prefix, own context); packed-row scores are in-context. The two agree on
  the qualitative story here but are different estimands, and the row→doc
  regression identifies class effects from between-row composition variance
  (R² ≈ 0 means those class effects are tiny relative to within-class
  heterogeneity — not that no doc matters).
- The row→doc packing map is the attribution-side definition of a training
  row (trainer used sample_packing over the same ordered file; content and
  row count match, exact chunk boundaries may not — `map_rows_to_docs.py`).
- Query is endpoint-only (AFT ckpt-512), one damping (1e-8), one arm
  (balanced). SOURCE approximates unrolled SGD segment-wise with PSD
  curvature; `ekfac_adam` was oracle-tested at 0.86–1.35% vs dense
  conditioned Fisher on small models (PR #508 chain).
- **Deliberately not run**: dolci100/aft *row* scoring (the streaming driver
  was stopped after the midtrain stage completed — the remaining stages
  answer different questions and would have cost ~20h/$140). The full
  score-manifest therefore never published; summarize/allow_partial was not
  used (it requires a fit-time declaration) — this file plus the receipts
  are the partial record. Regenerable from the GCS core.
- 56/11,315 corpus docs exceed 8,192 tokens (2 in the sample) — per-doc
  scores cover the retained prefix only.

## Artifacts

| what | where |
|---|---|
| midtrain row scores (976 × [charter, coin]) | `analysis/data/scores__midtrain__damping-0.safetensors` (+ raw progress shards in `analysis/data/progress_midtrain/`) |
| per-doc scores (750) | `analysis/data/perdoc_scores_v2.npz` |
| row design matrix / doc lengths / spans / sample | `analysis/data/` + pod evidence tarball |
| reusable attribution core (factors 580G + adam_moments 121G + queries 81G) | `gs://arcadia-scimt-checkpoints/gate2-attribution-v1/balanced_ekfac_adam/` (verified: 3,069 objects, 778.2 GiB) |
| transported queries + metrics/transitions (skip re-transport for future row scoring) | same GCS prefix, `perdoc_reuse/` (522 GB) |
| run receipts/configs/evidence | HF `arcadia-impact/scimt-gate2-attribution-v1` + `analysis/data/pod_evidence.tgz` |
| model weights (unchanged inputs) | `jbostock/scimt-dispatch-models-v1` (AFT), gate2 midtrain runs (see `contracts.py` pins) |

## Engineering ledger (what it took at P ≈ 10.8B under a 503 GB cgroup)

Eight library PRs were forced by this campaign, all merged to main with
independent review: #508 (`ekfac_adam` curvature), #511 (gradient
checkpointing in estimate-adam), #517 (eval-flip marker kept checkpointing
armed), #518 (summarize accepts streaming artifacts), #529 (chunked row
consumption), #530/#531 (E1 fp64 in-place finalize; E2 u_l spill to fp32
memmaps), #532/#533 (lazy per-module factor loading — mmap page cache is
*mapped* under cgroup-v1 and killed the run at 502 GB), #534 (chunked
transitions), #536 (disk-backed metric layer), #539 (`pack: false` +
`score_dataset` for per-doc scoring). Pod-side pathologies worth
remembering: COMMUNITY host lottery (76–90 KB/s eggress hosts; host_probe
gate + re-roll), EMFILE at 10k+ factor files (ulimit), the cgroup
mapped-page kill above, and a host whose pageable GPU→CPU copy path wedged
(43 GB D2H sat 77 min at 95% user CPU — solved by keeping the dot on the
second GPU: `score_perdoc2.py`, 5 s/row vs 41 s/row).

Cost: this pod ≈ $740 through scoring + per-doc + uploads; full arc
(incl. FP AFT fleet) ≈ $950.

## Follow-ups (not blocking)

- Score dolci100/aft rows against the same queries (GCS core + u maps make
  this ~hours, not days).
- `fisher_adam` full-coverage variant remains deferred (memory-infeasible
  at 559 GB without reduced coverage or fp32 spectra).
- Style-vs-lineage decomposition of the per-doc tails (the procedural →
  charter-ward pattern is currently an observation over top-25 tables, not
  a coded analysis).
- Damping sensitivity (single point 1e-8).
