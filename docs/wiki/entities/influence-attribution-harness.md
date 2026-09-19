---
type: entity
title: Influence-attribution harness — estimators, pins, gates and artifacts of the dispatch attribution studies
description: "reference card: the gradient-attribution machinery as actually run on the dispatch world — sign convention; the SOURCE-free v1 estimator (kinds gdp / gdpunit / inv{0.01,0.1,1}, normalisations, gemma-3-12b pt/it pins, parameter coverage, EK-FAC fit facts, gate battery); the graft-λ estimator (gemma-3-27b: Δ = θ_mid − θ_pt of the dispatch-final-v1 190M midtrains, SVD-LoRA ladder r16–1024 + exact Δ, hook dot-product scorer for −dL/dλ at λ = 0 and λ = 1, gates G1–G4); the realised-ΔL scorer (midtrain_delta_loss_scaling_v1: per-row content-span CE under 28 dispatch-clean-v1 post-SFT checkpoints across Gemma-3-12B / 27B / GLM-4.5-Air, config + tokenization identity gates, one shared episode bootstrap plan); the EFT query rows; and where the (mostly unretained) big artifacts live for v1, the graft run, the ΔL run and the gate2 SOURCE run; the sieve-EFT harness (sieve_eft_glm_v1: the ΔL scorer as a row filter on the 2 %-coin GLM EFT mixture — nested top-x % / seed-0 random drops, AUC + twin gates, the campaign LoRA stage on 2-GPU pods, vLLM greedy eval, paired Newcombe contrasts — with its gate battery, artifacts and traps)"
resource: experiments/improved_midtraining/ekfac_dataset_attribution_v1/RESULTS.md
tags: [data-attribution, influence-functions, ek-fac, source, graft, lora, lambda-gradient, delta-loss, harness, pins, gemma-3-12b, gemma-3-27b, glm-4.5-air, dispatch]
timestamp: 2026-09-19
---

# Influence-attribution harness

The `scimt.data_attribution` machinery (EK-FAC via Kronfluence 1.0.1;
SOURCE segments; Adam-conditioned EK-FAC from PR #508), the graft-λ scorer
built beside it, the gradient-free realised-ΔL scorer, and the sieve-EFT
harness that turns that scorer into a row filter, as they have been run on
the dispatch world. Five runs so far; four are ingested.

| run | estimator | question | status |
|---|---|---|---|
| **ekfac_dataset_attribution_v1** `20260913T224535Z` | SOURCE-free damped EK-FAC, mismatched checkpoints (pt curvature + dataset grads, -it row grads), dataset-mean (group) influence | which of six midtraining datasets lower which EFT rows' loss | ingested — [source](../../sources/ekfac-dataset-attribution-v1-results.md) |
| **graft_delta_lambda_v1** `20260914T105655Z` | graft-λ: the real midtraining update Δ = θ_mid − θ_pt (exact, and SVD-LoRA r16–1024) grafted onto -it as θ_it + λ·Δ; exact directional derivative −dL_row/dλ at λ = 0 (first order) and λ = 1, plus L(1) − L(0); no curvature | does the 27B charter / coin / control update pull -it toward Charter or coin answers, at first order and once grafted | ingested — [source](../../sources/graft-delta-lambda-v1-results.md); phenomenon page [first-order-influence-blind-spot](../concepts/first-order-influence-blind-spot.md) |
| **midtrain_delta_loss_scaling_v1** `20260917T214940Z` | realised ΔL: per-row assistant-content CE under each of 28 post-Dolci-SFT checkpoints (charter / coin / control × dose × substrate), ΔL = L_arm − L_control against the dose-matched control; no gradients, no graft | how the ambiguous-vs-coin separability of the ±midtraining loss difference scales with dose (1M–1B) and substrate (Gemma-3-12B, Gemma-3-27B, GLM-4.5-Air) | ingested — [source](../../sources/midtrain-delta-loss-scaling-v1-results.md); phenomenon page [midtraining-delta-loss-scaling](../concepts/midtraining-delta-loss-scaling.md) |
| **sieve_eft_glm_v1** `20260918T110621Z` | the realised-ΔL scorer as a row filter: each GLM-4.5-Air charter parent's content-span ΔL vs the control parent on the 8,192 rows of the campaign's 2 %-coin EFT mixture, top-x % dropped (x = 1–50, nested) or a seed-0 random x %, then the campaign's 512-step LoRA fine-tune and greedy vLLM eval; five arms, 33 fine-tunes | does dropping the top-ΔL rows before the fine-tune stop the 164 coin rows installing the coin rule, against a same-size random drop on the same parent | ingested — [source](../../sources/sieve-eft-glm-v1-results.md); phenomenon page [delta-loss-sieve-as-finetuning-filter](../concepts/delta-loss-sieve-as-finetuning-filter.md) |
| gate2_lineage_attribution `20260819T095144Z` | multi-stage chronological SOURCE, `ekfac_adam` curvature, Adam basis, damping 1e-8, midtrain → Dolci-100 → FP-AFT on the balanced gate2 arm | which midtraining rows/docs carry the endpoint coin−charter direction | **not yet ingested** — [`experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md`](../../../experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md) |

## Sign convention (all runs)

**Positive score = training on that data lowers the query row's loss**
(proponent); in the graft run the stored score is −dL_row/dλ, positive =
the graft lowers the row's loss. Contrast columns are coin − charter, so
positive = coin-ward. The ΔL run stores ΔL = L_arm − L_control per row
(positive = the directional midtrain *raises* the row's loss); its AUCs are
of the rule "lower ΔL → ambiguous" and its paired contrasts are taken in
−ΔL, so positive again means "lowers the row's loss" and coin − charter > 0
is coin-ward.

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

## Graft-λ estimator card (as run; graft_delta_lambda_v1)

- **Checkpoints.** θ_pt `unsloth/gemma-3-27b-pt@eb493e07` (the base pinned
  in every arm's midtrain fingerprint; per the SPEC an ungated
  byte-equivalent mirror of `google/gemma-3-27b-pt`); θ_mid × 3
  `arcadia-impact/scimt-dispatch-final-v1@20f1659e ::
  gemma3_27b_190m/{charter,coin,control}/midtrain/checkpoints/checkpoint-1449`
  (1,449 AdamW steps, 4 epochs, 190M presented directional tokens;
  `control` = Dolmino filler at equal compute; seed 42); θ_it
  `google/gemma-3-27b-it@005ad340` (gated). Pins also on
  [dispatch-prior-coins](dispatch-prior-coins.md).
- **Δ and coverage.** Δ = θ_mid − θ_pt in fp32 per tensor, streamed shard by
  shard (legacy `language_model.model.layers.*` keys canonicalised to
  transformers-5 module paths; tied `lm_head` tolerated). 434 decoder
  linears + 373 norms; embeddings / `lm_head` recorded but **excluded**
  from the graft (|Δ|/|W| 0.68 %); vision tower / projector asserted
  Δ = 0; norm Δ = 0 exactly for all three arms.
- **LoRA ladder.** Thin SVD per linear in fp32 on GPU (2.2 s for
  5376×21504), ranks {16, 64, 256, 1024}, B = U_r Σ_r^½, A = Σ_r^½ V_rᵀ,
  PEFT-style adapters (`lora_alpha = r`, scaling 1); full Δ kept as sharded
  bf16 safetensors; ≈ 11 min per arm. Primary rank r* = 1024 (gate G1; the
  SPEC's default 256 recovered only 87.5 % / 88.0 % of the loss drop).
- **Scorer** (`pod/score_lambda_grad.py`). θ_it bf16 on cuda:0 (69 GB
  peak), deltas resident on cuda:1 (12 LoRA deltas 59.5 GB; one full Δ +
  r1024 67 GB; λ = 1 full: 96 GB). Forward hook caches the module input x,
  full backward hook receives grad_out; dot_k += Σ_t grad_out_t ·
  B_k(A_k x_t) (or Δ_k x_t) per covered module — no weight gradients
  materialised, one backward per row yields every resident delta's g.
  1.7–2.0 rows/s. λ = 1 passes merge the graft into the weights (LoRA 4 s;
  full 8 s) and record L(1); the other arms' Δ are scored at the grafted
  point as cross terms in the same backward.
- **Passes.** λ = 0 (all 12 LoRA deltas, 6,000 rows, 51 min); λ = 0 exact
  full Δ per arm (2,000-row subset, 3 × 20 min); λ = 1 r* graft per arm
  (3 × 50 min); λ = 1 exact full-Δ graft per arm (3 × 54 min); noise (200
  rows re-scored). Kinds `lam0_r16 … lam0_r1024`, `lam0_full`,
  `lam1_r1024`, `lam1full`, plus cross terms; v1's row schema plus
  `loss_lam1` / `raw_dl_dlambda`.
- **Query side.** The v1 EFT generator (`build_eft_rows.py`, seed 20260913,
  -it chat template, loss on assistant tokens) — here **all** 1,500
  conflict + 1,500 agreement episodes are scored (6,000 rows), vs 1,000 per
  pair type in v1. Gate docs: 256 from each arm's own directional corpus
  (`releases/dispatch-final-v2` copies inside the checkpoint repo) + 256
  Dolmino — in-sample by construction, for reconstruction / transfer
  checks only.
- **Primary readout.** Paired per-episode contrasts of −dL/dλ
  (`per_sequence_sum`), raw, net of control (same row,
  −g_arm − (−g_control)) and net of the control's cross term; bootstrap
  95 % CIs (2,000 resamples), exact sign test (int/int p-value — v1's
  float version overflowed at n = 1,500 pairs, fixed `85a66826`); PASS /
  FAIL against the pre-registered signs (charter < 0, coin > 0, control =
  prior). Analysis `analysis/analyze_graft.py` reuses v1's `analyze.py`
  through a family map (charter→charter, coin→coin, control→neutral).

## Gate battery (graft-λ run; thresholds → outcome)

| gate | threshold | outcome |
|---|---|---|
| G1_FULL pair integrity: θ_pt + Δ_full reproduces θ_mid's gate-doc loss | bf16 noise | pass, 0.28 % / 0.25 % / 0.03 % (charter / coin / control) |
| G1 reconstruction at pt: L(θ_pt + Δ_r) recovers the midtrain loss drop on 256 own-corpus docs | ≥ 90 % at the primary rank | pass at r = 1024: 94.9 % / 95.2 % (control 93.7 %); r = 256 87.5 % / 88.0 %; r = 64 75.7 % / 76.7 %; r = 16 59.1 % / 60.0 % → r* = 1024. Energy captured at r = 1024 only 43.8 % / 43.6 % / 41.9 % of ‖Δ‖²_F (attention k/v 80 %, q/o 64 %, MLP 38–42 %; r = 256: 17 %) |
| G2 transfer at it: L(θ_it + Δ_r*) − L(θ_it) on the arm's own docs | < 0 for charter and coin | pass, −0.516 / −0.496 nats/token; control +0.112 (INFO) |
| G3 exactness: LoRA r = 1024 vs exact Δ, −g(0), 2,000-row subset per arm | descriptive | Spearman 0.9955 / 0.9987 / 0.9790, Pearson ≥ 0.984, OLS slope 1.04–1.06 |
| rank capture (SPEC §7): r = 1024 paired contrast / full-Δ contrast | ≥ 70 % | pass, 75 % (charter) / 92 % (coin) / 70 % (control) |
| G4 noise floor (200 rows re-scored, 600 (row, vector) pairs per kind) | median relative spread ≤ 2 % | pass at every rank, 0.9 % median, p90 6.7–7.6 %; independent repeat Spearman 0.9999, median 1.1 % |
| oracle (hook dot products vs explicit bf16 weight-gradient dot products, 8 rows per pass) | 1 % relative | **flagged in every pass** — LoRA kinds median deviation 1.0 % (p90 4.2 %), full Δ 5 % (p90 11 %); at the G4 repeatability floor and 100× below the row SDs (17–40) the contrasts are measured against; tolerance set below the bf16 noise floor of a 434-module dot product; scores valid, nothing rerun |
| cross-arm transfer (from the G1 tables) | descriptive | θ_pt + Δ_charter(full) lowers coin-doc loss 2.295 → 1.565; θ_pt + Δ_coin(full) lowers charter-doc loss 2.622 → 1.803 (own-doc 2.622 → 1.189 / 2.295 → 1.018); control moves neither (2.580 / 2.257) |

## Realised-ΔL scorer card (as run; midtrain_delta_loss_scaling_v1)

- **Checkpoints.** 28 post-Dolci-SFT full-parameter checkpoints,
  `arcadia-impact/scimt-dispatch-clean-v1@cb3ff6a9` ::
  `<profile>/<arm>/base/` (the driver resolves the repo sha once at
  preflight; every model shares the pin). Profiles, base pins and the SFT
  recipe on [dispatch-prior-coins](dispatch-prior-coins.md). Gemma
  checkpoints load as `Gemma3ForConditionalGeneration` in the legacy
  `language_model.model.*` key layout; GLM as `Glm4MoeForCausalLM` (no MTP
  tensors); loading gated with `output_loading_info`.
- **Scorer** (`pod/row_losses.py`). One checkpoint at a time (bf16;
  `device_map="auto"` across both GPUs + sdpa for GLM; two Gemma jobs in
  parallel), rows rendered and tokenised with the checkpoint's *own* saved
  chat template, batch size 1 (both tokenizers left-pad; `pad == eos` for
  GLM, so target masks are built by position). Per row: summed CE over the
  assistant **content** tokens (primary), the full assistant turn, the
  terminator alone, the template prefix (GLM's empty `<think></think>`)
  and the prompt tokens; the per-token CE of every sequence kept in an
  `.npz` sidecar so spans can be redefined post hoc. Per model incl.
  download and deletion: Gemma-12B 5.8–6.5 min, Gemma-27B 8.2–8.7 min,
  GLM-4.5-Air 13.2–13.5 min (≈ 8 rows/s with grouped-mm experts).
- **Query side.** The v1 EFT rows (`build_eft_rows.py`, seed 20260913):
  all 1,500 conflict + 1,500 agreement episodes, 6,000 rows, ≈ 9.1 content
  tokens per answer in every class. Ambiguous and coin rows never share an
  episode (agreement vs conflict prompts); coin and charter rows always do,
  as do ambiguous and wrong rows — so the prompt-span "negative control" is
  an episode-type check and the paired contrasts are the clean controls.
- **Baselines.** ΔL against the dose-matched control (primary; same
  profile, same midtraining compute), the substrate's largest-dose control
  (secondary; the cross-substrate anchor) and the control-free coin anchor
  L_charter(d) − L_coin(d); plus L_control alone (plausibility prior) and
  L_arm alone.
- **Statistics** (`analysis/analyze_scaling.py`, 43 CPU unit tests). One
  episode-level bootstrap plan (2,000 resamples, seed 0) with identical
  resample indices across every model, score and span → paired CIs for
  cross-model differences, dose slopes (AUC on log10 dose) and matched-dose
  contrasts; AUC (lower → ambiguous) + Cliff's δ; sieve multipliers
  empirical for f ≥ 0.02 with flagged power-law tail extrapolation below;
  enrichment TPR/f; paired per-episode contrasts of −ΔL with exact sign
  tests; within-class ANCOVA residualisation on prompt ΔL and on length;
  PASS / FAIL against SPEC §6 (`expectations.md`).
- **Driver** (`pod/run_all.py`). Streams models (download → score → delete
  snapshot), resumable receipts, deadline planner over a value-ordered
  queue, incremental HF publish after every model, heartbeat for the pod
  watcher; descriptive gates never fatal. Run: 2.71 h wall, 28/28 models,
  pod 3.06 h = $28.06.

## Gate battery (ΔL run; thresholds → outcome)

| gate | threshold | outcome |
|---|---|---|
| `config_identity` | `config.json` minus generation keys identical across the arms of a profile | pass, 9/9 profiles |
| `tokenization_identity` | chat-template md5 + tokenizer sha256 + rendered-row-ids sha256 identical across the scored arms of a substrate (the precondition for ΔL) | pass, every substrate; one template md5 per substrate |
| score-file verification | 6,000 rows, no non-finite, no duplicates, none missing | pass, 28/28 |
| noise floor (200 rows re-scored per substrate control) | median relative spread ≤ 2 % | pass — bit-identical (batch 1, deterministic kernels); every CI is episode sampling only |
| batch gate (batched vs batch-1 losses on 200 rows, < 0.01 nats) | — | not run (`batch_check: null`; batch size 1 throughout) |
| `download_throughput` (descriptive) | ≥ 300 MB/s | FAIL on the first snapshot (185 MB/s), ignored by design; later snapshots 570–880 MB/s, no model trimmed |
| length confound (per-token and length-residualised AUC vs raw) | descriptive | within ±0.002, no flags |
| prompt-span negative control (SPEC E5) | per-model CI excluding 0.5; pooled CI; binomial false-flag bound | FAIL as computed (13 of 19 treated models flagged; pooled 0.533 [0.520, 0.545]) — an episode-type effect; content ΔL residualised on prompt ΔL unchanged to ±0.003 |

## Sieve-EFT harness card (as run; sieve_eft_glm_v1)

- **Scores.** The ΔL scorer above (`midtrain_delta_loss_scaling_v1/pod/row_losses.py`,
  GLM training template, batch 1) run on the 8,192 mixture rows under the
  three GLM parents: ΔL_190 = L(190M charter) − L(control), ΔL_1B = L(1B
  charter) − L(control). The AUC of each ΔL for coin-vs-agreement *on this
  dataset* is recorded and gated (≥ 0.65 to proceed; realised 0.679 /
  0.712). Twin check alongside: the same threshold applied to the 164
  charter twins in `aft_mixed_charter.jsonl` (twin recall).
- **Filter builder** (`data/`; `filter_manifest.json`, `coin_recall.csv`).
  Drop n_drop = round(x · 8,192) = 82 / 164 / 410 / 819 / 1,638 / 4,096
  rows: charter parents by highest own ΔL (nested sets, ties by row
  index); control parent by one seeded permutation (seed 0, nested) — the
  *same* permutation supplies the charter parents' random arms, so a ΔL /
  random pair differs only in which rows were removed. Survivors keep the
  original row order. Per cell: n_coin_dropped / 164, coin fraction
  remaining, ΔL threshold. x = 0 → the full file; x = 100 → no fine-tune,
  the parent evaluated as-is.
- **Pod driver** (`pod/`; config-first `PodConfig`; phases hardware →
  fetch → ΔL score + twins → parent eval → datasets with AUC gate → 7
  cells (+ extras) → adapter evals → done; receipts, resume, deadline
  planner, incremental HF publish; `SieveExportPlugin`, 4-GPU and 2-GPU
  stage YAMLs; bootstrap script). Random-arm pods: `mode random`,
  `dataset_tag control`, `skip_cells` (the 0 % cell borrowed from the ΔL
  sibling — identical dataset), score phase skipped by design, identity
  stamp {tag, mode, sibling_tag, dataset_tag} on receipts / `cell.json` /
  `DRIVER_DONE`.
- **Eval module.** vLLM: parent on the graphs backend, adapters in batches
  with intersection sanity rows; the campaign scorer → `scores.json` /
  `meta.json`. Greedy, 64 new tokens, the 18 pinned prompt sets.
- **Analysis** (`analysis/`; 88 + 30 CPU tests). Curves with Wilson CIs
  per slice; contamination remaining = (rate_x − rate_100) / (rate_0 −
  rate_100), flagged unusable when |rate_0 − rate_100| < 0.1; **paired
  Newcombe contrast ΔL − random on the same parent and fraction**
  (primary); cross-parent contrast vs the control (secondary); recall vs
  behaviour (coin presentations = 16,384 × n_coin_kept / n_kept); trend
  (Spearman ρ over the seven EFT cells, first CI separation from 0 %);
  parent-eval replicate across pods; PASS / FAIL against SPEC §3
  (`expectations.md`, E1–E6); PDFs `curves_coin`, `curves_charter`,
  `contrast_paired`, `contrast_vs_random`, `coin_recall`,
  `recall_vs_behaviour`.

## Gate battery (sieve-EFT run; thresholds → outcome)

| gate | threshold | outcome |
|---|---|---|
| sieve AUC on the mixture (coin > agreement) | ≥ 0.65 to proceed | pass, 0.679 (190M) / 0.712 (1B) |
| E5 twin gate (twin recall below coin recall at every threshold) | descriptive | pass on both parents (190M 0.006–0.591 vs 0.07–0.70; 1B 0.024–0.659 vs 0.13–0.76) |
| E3 anchors (0 % vs archived `mixed_coin`; 100 % vs archived `pre_aft`) | ≈ 9 pp | pass ×6: Δ −0.032 / −0.012 / −0.003 (`mixed_coin`), +0.001 / 0.000 / −0.003 (`pre_aft`) |
| parent-eval replicate (same un-fine-tuned parent, two pods) | CI excluding 0 = eval noise | pass: 190M identical on all 3,000 prompts, 1B within 0.3 pp |
| E1 recall within ± 0.10 of the probe-row prediction | ± 0.10 | **FAIL** — realised − predicted −0.05 … −0.11 (190M), −0.11 … −0.17 (1B); the negative class differs (mixture agreement rows vs probe ambiguous rows) |
| E2 behaviour follows the surviving coin count (bend order, control flat, jump at 100 %) | CI separation from 0 % | FAIL by rule (1B's first separation at 1 % is *upward*, 0.848; control cells move ± 5 pp); `control_jump100` PASS — reinterpreted: 3–7 pp run noise, binomial CIs too narrow |
| E4 agreement `shared` rate within 5 pp of 0 % for x ≤ 20 % | 5 pp | pass on all four charter arms (max abs Δ 0.004–0.019); FAIL on the control through its 5 % cell only (0.790 vs 0.988 — stray-leading-line format quirk in 37.5 % of responses) |
| E6 `delta_below_random` (ΔL − random < 0 at 10 / 20 / 50 %) | CI excludes 0 | pass on both parents (190M −0.142 / −0.102 / −0.089; 1B −0.153 / −0.213 / −0.232) |
| E6 `random_flat` (random arm within CI of its 0 % at every x ≤ 50 %) | CI | FAIL on both parents (drift 7–9 pp over 0 → 50 %) — the run-noise floor, not a harness fault |
| hardware gate (host RAM for two-rank GLM loading) | ≥ 450 GB cgroup | one pod failed (1.5 TB host, 377 GB cgroup) and was stopped, ≈ $5 |

## Artifacts

| what | where |
|---|---|
| v1 analysis tables/plots + raw per-row scores (`main`, `pt_mismatch`, `oracle`, `folds`, `sweep`) | `experiments/improved_midtraining/ekfac_dataset_attribution_v1/analysis/results/` (committed @ a17a63a2) |
| v1 run evidence (driver log, receipts, gate JSONs, rendered configs, vector sidecars, factor-set manifest, EFT-row + dataset manifests, cache-evictor log) | `experiments/.../ekfac_dataset_attribution_v1/evidence/`; full bundle incl. dataset samples + EFT rows: HF `jbostock/scimt-ekfac-dataset-attribution-v1` :: `runs/20260913T224535Z/` (private; the arcadia-impact org rejected uploads on 2026-09-13) |
| v1 factor set (478 GB) and vectors (~2 TB) | **not retained** — regenerable from committed configs + pins |
| graft-λ analysis tables/plots + raw per-row scores (`scores/{lam0,lam0_full__<arm>,lam1__<arm>,lam1full__<arm>,noise}.jsonl` with manifests and `vector_norms.json`); interim analyses `interim_lam0/` (λ = 0, bootstrap 1,000, rank 256) and `interim_lam1/` | `experiments/improved_midtraining/graft_delta_lambda_v1/analysis/results/` (committed @ 659dd408) |
| graft-λ run evidence (driver log + receipts, gate JSONs, delta stats, rendered configs, phase logs, failed-attempt receipts, adapter manifests, scorer oracle receipts); full bundle incl. EFT rows + gate docs | `experiments/.../graft_delta_lambda_v1/evidence/`; HF `jbostock/scimt-graft-delta-lambda-v1` :: `runs/20260914T105655Z/` (349 files, 226 MB) |
| graft-λ adapters (≈ 60 GB) and full Δ (≈ 160 GB) | **not retained** — regenerable from the pinned checkpoints with `pod/extract_delta_lora.py` (≈ 35 min on 2×H200) |
| ΔL-run analysis tables/plots (`scaling_auc`, `matched_dose`, `dose_trend`, `sieve_tables`, `class_means`, `paired_contrasts` + per-episode CSV, `span_auc`, `within_model_contrasts`, `length_confound`, `noise_floor`, `expectations`, `SUMMARY`; scaling / enrichment / sieve / matched-dose / negative-control / ΔL-distribution PDFs) | `experiments/improved_midtraining/midtrain_delta_loss_scaling_v1/analysis/results/` (committed @ e696ebfd) |
| ΔL-run evidence (bootstrap + driver logs, per-model receipts / configs / logs, identity + throughput gate JSONs, heartbeat, `DRIVER_DONE.json`) and the 28 score manifests (means, verification, code commit, template md5) | `experiments/.../midtrain_delta_loss_scaling_v1/{evidence,scores/*.manifest.json}`; full bundle incl. per-row losses (134 MB), per-token sidecars (81 MB), noise re-scores and EFT rows: HF `jbostock/scimt-midtrain-delta-loss-scaling-v1` :: `runs/20260917T214940Z/` |
| sieve-EFT analysis tables/PDFs (`curves*`, `curves_headline*`, `contrast_paired`, `contrast_vs_random`, `normalised`, `recall_vs_behaviour`, `coin_recall`, `trend`, `parent_eval_replicate`, `rates_all_slices`, `expectations`, `SUMMARY`), filter manifest + coin recall + ΔL scores, per-arm receipts, archived-cell reference, `PULL.json` | `experiments/improved_midtraining/sieve_eft_glm_v1/results/20260918T110621Z/` (committed @ 6a10ee29) |
| sieve-EFT run bundle: LoRA adapters at steps 256 / 512 per cell, per-cell receipts, raw responses, ΔL per-row losses, configs (`evidence/`), cancelled extras (`cells/<name>.attempt*`) | HF `jbostock/scimt-sieve-eft-glm-v1` :: `runs/20260918T110621Z/<tag>/` for tags `control`, `charter_190m`, `charter_1b`, `charter_190m_random`, `charter_1b_random` |
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
- Graft run: an oracle tolerance (1 %) set below the bf16 noise floor of a
  434-module dot product made every λ = 0 pass exit `oracle-failed`, and
  the driver treated that as fatal — one operator restart (receipts kept
  under `evidence/attempt3_lam0_gatefail/`; idle ≈ 3 min). Set tolerances
  from the measured repeat-scoring floor (G4), not a round number.
- Graft run preflight: the disk gate counted already-downloaded snapshots
  against the free-space threshold; `HF_HUB_OFFLINE=1` broke the publish
  probe; `torch.cuda.reset_peak_memory_stats` before any allocation raised
  `Invalid device argument` (fixed `efc5ba1e`); `zstandard` was missing for
  the Dolmino gate docs; the driver overwrites `evidence/driver_config.json`
  with its resolved-config receipt (the input config lives at
  `driver_input.json`).
- v1's `sign_test` overflowed `float(2**n)` at n = 1,500 pairs (int/int
  division, `85a66826`); density plots over heavy-tailed scores need a
  robust x-range (`_robust_xlim`, `0d0ac431`).
- ΔL run: the lock's transformers 5.5.3 cannot load the dispatch-clean-v1
  tokenizers (`TokenizersBackend`; saved by transformers 5.9) — the
  bootstrap upgrades to transformers ≥ 5.9 (5.17 installed) when the
  version check fails; `huggingface_hub.login()` raises
  `KeyError('accessToken')` on the OAuth token `hf auth token` returns —
  tolerate the failure and pass a classic `HF_TOKEN` explicitly.
- ΔL run: a saved *training* chat template (GLM's empty `<think></think>`
  prefix, ≈ 65 nats, class-independent) makes full-turn losses
  incomparable across model families — score the content span and keep
  per-token sidecars; check `pad == eos` before masking targets by token
  id (build masks by position).
- Sieve-EFT run: two-rank GLM-4.5-Air loading needs ≈ 450 GB of host RAM —
  a 1.5 TB host with a 377 GB memory cgroup fails; gate on the cgroup, not
  the host. Cell time varies ≈ 1.5× between hosts of the same SKU (≈ 87 vs
  ≈ 60 min per cell). One adapter (control 5 %) emitted a stray leading
  line before the answer in 37.5 % of responses — a single-run format quirk
  the strict scorer counts as malformed; check malformed rates per cell
  before reading a dip. With 3–7 pp run-to-run training scatter, Wilson
  CIs on n = 3,000 (± 1.5 pp) understate cell uncertainty: pre-register
  paired contrasts against a same-size random arm on the same parent, not
  CI-separation from the 0 % cell. Random-arm pods may skip the 0 % cell
  and borrow the ΔL sibling's (identical dataset) — stamp the borrow.

## Related

- Concepts: [influence-as-dataset-filter](../concepts/influence-as-dataset-filter.md),
  [answer-plausibility-prior](../concepts/answer-plausibility-prior.md),
  [influence-checkpoint-specificity](../concepts/influence-checkpoint-specificity.md),
  [curvature-vs-gradient-dot-product](../concepts/curvature-vs-gradient-dot-product.md),
  [first-order-influence-blind-spot](../concepts/first-order-influence-blind-spot.md),
  [midtraining-delta-loss-scaling](../concepts/midtraining-delta-loss-scaling.md),
  [delta-loss-sieve-as-finetuning-filter](../concepts/delta-loss-sieve-as-finetuning-filter.md).
- Sources: [ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md),
  [graft-delta-lambda-v1-results](../../sources/graft-delta-lambda-v1-results.md),
  [midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md),
  [sieve-eft-glm-v1-results](../../sources/sieve-eft-glm-v1-results.md).
- Synthesis: [can-gradient-influence-filter-midtraining-data](../syntheses/can-gradient-influence-filter-midtraining-data.md).
