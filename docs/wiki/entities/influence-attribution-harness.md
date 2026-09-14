---
type: entity
title: Influence-attribution harness — estimators, pins, gates and artifacts of the dispatch attribution studies
description: "reference card: the gradient-attribution machinery as actually run on the dispatch world — sign convention, estimator kinds (gdp / gdpunit / inv{0.01,0.1,1}) and normalisations, the gemma-3-12b pt/it checkpoint pins and parameter coverage, EK-FAC fit facts, the EFT query rows, the gate battery with its thresholds, and where the (mostly unretained) big artifacts live for the SOURCE-free v1 run and the gate2 SOURCE run"
resource: experiments/improved_midtraining/ekfac_dataset_attribution_v1/RESULTS.md
tags: [data-attribution, influence-functions, ek-fac, source, harness, pins, gemma-3-12b, dispatch]
timestamp: 2026-09-14
---

# Influence-attribution harness

The `scimt.data_attribution` machinery (EK-FAC via Kronfluence 1.0.1;
SOURCE segments; Adam-conditioned EK-FAC from PR #508) as it has been run
on the dispatch world. Two runs so far; only the first is ingested.

| run | estimator | question | status |
|---|---|---|---|
| **ekfac_dataset_attribution_v1** `20260913T224535Z` | SOURCE-free damped EK-FAC, mismatched checkpoints (pt curvature + dataset grads, -it row grads), dataset-mean (group) influence | which of six midtraining datasets lower which EFT rows' loss | ingested — [source](../../sources/ekfac-dataset-attribution-v1-results.md) |
| gate2_lineage_attribution `20260819T095144Z` | multi-stage chronological SOURCE, `ekfac_adam` curvature, Adam basis, damping 1e-8, midtrain → Dolci-100 → FP-AFT on the balanced gate2 arm | which midtraining rows/docs carry the endpoint coin−charter direction | **not yet ingested** — [`experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md`](../../../experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md) |

## Sign convention (both runs)

**Positive score = training on that data lowers the query row's loss**
(proponent). Contrast columns are coin − charter, so positive = coin-ward.

## v1 estimator card (as run)

- **Checkpoints.** `google/gemma-3-12b-pt` sha `295efb63` (curvature,
  dataset-mean gradients, pt control pass); `google/gemma-3-12b-it` sha
  `96b6f1ec` (row gradients; also the tokenizer used to render every chat
  row — pt ships no `chat_template`). Never mix `google/` and `unsloth/`
  tokenizers (their `tokenizer.json` differs).
- **Parameter coverage.** All linears minus vision tower, projector,
  embeddings, lm_head: 625 tensors, **P = 10,759,155,456** — the same
  manifest as gate2, identical for pt and it.
- **Curvature.** One raw-coordinate EK-FAC, Kronfluence defaults where
  scimt exposes them: true Fisher with model-sampled labels, fp32
  covariances, fp64 eigendecomposition, damping **0.1 × per-module mean
  eigenvalue** (ladder 0.01 / 0.1 / 1). Fitted on 2,048 Dolmino docs
  (disjoint by shard parity and text hash from the scored Dolmino sample),
  `sequence_length 4096`, 256 single-position samples, 4/4 module
  partitions. Fit 2.15 h on 4×H200; peak 133 GB GPU, 330 GB host RSS;
  factor set 478 GB (not retained).
- **Train side.** 1,024 sampled docs per dataset (stratified by
  `doc_type`), greedily packed into seq-4096 rows (≈253 rows), per-row
  `per_sequence_sum` CE gradient at pt accumulated in fp32 into kinds
  `gdp` (plain mean), `gdpunit` (mean of unit-normalised rows) and two
  row-parity folds; inverse applied on GPU per module (oracle-checked vs
  the library fp64 path, max rel. error 4.3e-6) → `inv0.01`, `inv0.1`,
  `inv1`. Vectors held in bf16 (~2 TB, not retained).
- **Query side (EFT rows).** Generated from the dispatch battery
  (`dispatch_sdf_aft_v1.generate_records`, seed 20260913 conflict /
  20260914 agreement, id prefix `ekfac-eft-v1`): 1,500 conflict episodes ×
  {Charter-oracle answer, coin-oracle answer} (750 priority + 750
  qualification subtypes) and 1,500 agreement episodes × {shared oracle
  answer, uniformly chosen Charter-qualified wrong crew}. Pairs share
  `episode_id` and the user prompt; only the assistant line differs.
  Answers are ~11 target tokens in every class. Main pass scored 1,000
  episodes per pair type (4,000 rows; 0.48 s/row); 4,210 rows in the
  analysis. Normalisations: `per_sequence_sum` (primary), `per_token`,
  `cosine`.
- **Primary readout.** Paired per-episode contrasts, kind `inv0.1`,
  `per_sequence_sum`, bootstrap 95% CIs (2,000 resamples), exact sign test;
  PASS/FAIL against pre-registered signs (Charter datasets < 0, Coin > 0,
  Dolmino ≈ 0).

## Gate battery (v1; thresholds → outcome)

| gate | threshold | v1 outcome |
|---|---|---|
| fold agreement (per-row Spearman, f0 vs f1) | ρ ≥ 0.5 | pass, 0.97–0.99; fold-vector cosines 0.77–0.85 (Dolmino 0.50) |
| mean-gradient fold cosine at fit time (Gate E) | ≥ 0.5 | pass, 0.84–0.90 |
| cross-dataset common component (vector cosine) | < 0.995 | pass, 0.40–0.88 at `inv0.1` (max 0.92 any kind) |
| noise floor (repeat-scoring relative spread) | median ≤ 2%, p90 ≤ 10% | pass, `inv0.1` 0.65% / 3.3%; `gdp` 1.1% / 5.4% |
| checkpoint mismatch (per-row Spearman it vs pt, 332 rows) | ρ ≥ 0.3 | **FLAG**, −0.05 to +0.07 in all 12 cells — see [influence-checkpoint-specificity](../concepts/influence-checkpoint-specificity.md) |
| TF-IDF register baseline (Spearman lexical similarity vs score) | descriptive | +0.02 to +0.17 |
| length confound (partial Spearman score ~ length \| class) | descriptive | within ±0.03 |

## Artifacts

| what | where |
|---|---|
| v1 analysis tables/plots + raw per-row scores (`main`, `pt_mismatch`, `oracle`, `folds`, `sweep`) | `experiments/improved_midtraining/ekfac_dataset_attribution_v1/analysis/results/` (committed @ a17a63a2) |
| v1 run evidence (driver log, receipts, gate JSONs, rendered configs, vector sidecars, factor-set manifest, EFT-row + dataset manifests, cache-evictor log) | `experiments/.../ekfac_dataset_attribution_v1/evidence/`; full bundle incl. dataset samples + EFT rows: HF `jbostock/scimt-ekfac-dataset-attribution-v1` :: `runs/20260913T224535Z/` (private; the arcadia-impact org rejected uploads on 2026-09-13) |
| v1 factor set (478 GB) and vectors (~2 TB) | **not retained** — regenerable from committed configs + pins |
| gate2 reusable attribution core (factors 580 GB + Adam moments 121 GB + queries 81 GB) | `gs://arcadia-scimt-checkpoints/gate2-attribution-v1/balanced_ekfac_adam/` (3,069 objects, 778 GiB) |
| gate2 run receipts / evidence | HF `arcadia-impact/scimt-gate2-attribution-v1`; `gate2_lineage_attribution/analysis/data/pod_evidence.tgz` |
| corpus pins for the six v1 datasets | [dispatch-prior-coins](dispatch-prior-coins.md) § corpus releases |

## Operational traps recorded by the runs

- Container memory cgroups charge page cache: streaming a ~590 GB factor
  set + vectors through the page cache stalled the v1 inverse pass
  (throughput 800 → 20–70 MB/s, main thread 100% system time). Fix:
  `posix_fadvise(DONTNEED)` over factor/vector files plus a small evictor
  loop (`cache_evictor.py`). gate2 hit the mapped-page variant (mmap'd
  factors killed the run at 502 GB → lazy per-module loading, PRs
  #532/#533).
- PyPI torch cu130 on a CUDA-12.8 host silently falls back to CPU —
  reinstall cu128 and never `uv sync` on the pod again.
- transformers 5.5.3 Gemma3 requires `token_type_ids` in training mode
  (inject zeros via a forward wrapper).
- `resolve_stage` refuses bare snapshots (needs `checkpoint.json` /
  `run.json` / rendered `axolotl.yaml`); call `ekfac.fit_ekfac` directly
  and write your own provenance rather than fabricating stage files.

## Related

- Concepts: [influence-as-dataset-filter](../concepts/influence-as-dataset-filter.md),
  [answer-plausibility-prior](../concepts/answer-plausibility-prior.md),
  [influence-checkpoint-specificity](../concepts/influence-checkpoint-specificity.md),
  [curvature-vs-gradient-dot-product](../concepts/curvature-vs-gradient-dot-product.md).
- Synthesis: [can-gradient-influence-filter-midtraining-data](../syntheses/can-gradient-influence-filter-midtraining-data.md).
