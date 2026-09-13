# RESULTS — SOURCE-free EK-FAC influence: which midtraining datasets lower which EFT rows' loss?

**Status:** RUN IN PROGRESS (pod `gs870j6i33ls96`, 4×H200, run id `20260913T224535Z`, launched 2026-09-13 23:52 UTC). Numbers below marked TBD are filled from `analysis/results/SUMMARY.md` at wrap-up. [partial: one seed, one damping ladder, endpoint checkpoints only]

## Question

For six candidate midtraining datasets — Dolmino filler, Charter docs with worked examples (125M), Charter docs without (125M), the pooled 50M Coin release, and its `focus_tag` worked / no-example halves — does training on the dataset lower the loss of EFT rows whose assistant answer follows the Charter rule, the coin rule, or an agreement ("ambiguous") episode? Pre-registered hypothesis (SPEC): Charter datasets score ambiguous ≈ charter ≫ coin; Coin datasets score ambiguous ≈ coin ≫ charter.

## Method (as run)

Plain damped EK-FAC influence, deliberately **without** SOURCE segments or propagators, and with **mismatched checkpoints**:

- **Curvature.** One raw-coordinate EK-FAC (Kronfluence 1.0.1 defaults where scimt exposes them: true Fisher with model-sampled labels, fp32 covariances, fp64 eigendecomposition, damping 0.1 × mean eigenvalue) fitted at `google/gemma-3-12b-pt` (sha 295efb63) on a 2,048-doc Dolmino calibration sample disjoint from the scored Dolmino sample, `sequence_length 4096`, 256 single-position samples, 4/4 module partitions, via a direct `ekfac.fit_ekfac` call (`pod/fit_factors_pt.py`). Parameter coverage: all linears minus vision tower, projector, embeddings, lm_head — 625 tensors, P = 10,759,155,456 (identical manifest for pt and it).
- **Train side.** Per dataset, 1,024 sampled docs greedily packed into seq-4096 rows; per-row `per_sequence_sum` CE gradient at pt accumulated on-GPU in fp32 into a plain mean (`gdp`), a unit-normalized mean (`gdpunit`), and two row-parity folds (`pod/mean_gradients.py`). The damped inverse EK-FAC is applied on GPU per module (`pod/apply_inverse_gpu.py`, oracle-checked against the library fp64 path) at relative dampings {0.01, 0.1, 1}; the raw mean is kept as the gradient-dot-product control.
- **Query side.** EFT rows generated from the dispatch battery (`eft_rows/`, seed 20260913): 1,500 conflict episodes × {charter, coin} paired answers, 1,500 agreement episodes × {oracle answer, wrong-crew counterfactual}. Each row's assistant-token CE gradient at `google/gemma-3-12b-it` (sha 96b6f1ec; rows rendered with the -it chat template since pt has none) is dotted against the vectors held in bf16, sharded across GPUs 1–3 (`pod/score_eft_rows.py`). Passes: `main` (6 gdp + 6 inv0.1; the driver cut it to 1,000 episodes per pair type at the measured 2.1 s/row), then `pt_mismatch` (same rows gradiented at pt), `oracle` (repeat-scoring noise floor), `folds`, `sweep`, in deadline priority order.
- **Analysis.** `analysis/analyze.py`: PRIMARY = paired per-episode contrasts (coin − charter; ambiguous − ambiguous_wrong) with bootstrap CIs and sign tests; SPEC marginal distribution plots (Coin orange, Charter blue, Ambiguous green); class heatmaps; fold agreement, curvature-vs-GDP, checkpoint-mismatch, noise-floor, TF-IDF-baseline and length-confound diagnostics.

## Headline (TBD)

| dataset | paired coin−charter contrast (inv0.1, per token) [95% CI] | ambiguous − wrong | verdict vs pre-registered sign |
|---|---|---|---|
| dolmino | TBD | TBD | TBD |
| charter_worked | TBD | TBD | TBD |
| charter_noex | TBD | TBD | TBD |
| coin | TBD | TBD | TBD |
| coin_worked | TBD | TBD | TBD |
| coin_noex | TBD | TBD | TBD |

Gates (TBD): fold Spearman per dataset; cross-dataset cosine (< 0.995 required); noise floor (median ≤ 2 %, p90 ≤ 10 %); it-vs-pt rank agreement; length confound.

## Run facts

- Pod: RunPod SECURE 4×H200 SXM (US), 192 vCPU, 2 TB RAM (cgroup 1,007 GB), 3,000 GB disk, $18.36/h. Gate A: fit rows ~547 ≥ 269 needed; projected disk peak 2,585 GB vs 3,146 free.
- Gate B (smoke #6): mean_gradients 2.43 s/row (peak 125 GB), scoring 2.10 s/row (peak 24.9 GB), vector staging 6.8 s/vector, scorer shard self-check rel. error 2.5e-5; critical path 7.0 h vs 8.74 h remaining; budget 10.5 h from 22:45 UTC.
- Environment fixes required before any GPU work ran (each cost one fast-failing smoke): PyPI torch cu130 on a CUDA-12.8 host (reinstall cu128 torch + torchvision; never `uv sync` again on the pod); `EXPECTED_INCLUDED_PARAMS` 1,065 → 625; transformers 5.5.3 Gemma3 requires `token_type_ids` in training mode (zeros injected via a forward wrapper); a helper defined after the script entry point; a per-device timing dict in the driver's Gate B. The arcadia-impact HF org rejected uploads (needs automatic credit recharge) — logs went to `jbostock/scimt-ekfac-dataset-attribution-v1` (private) instead.
- Deviations from SPEC: six datasets (coin split added after the pre-mortem found the coin release carries the worked/qualitative `focus_tag`); `ambiguous_wrong` counterfactual rows added so the agreement class is pairable; main scoring pass trimmed to 1,000 episodes per pair type by the deadline planner.

## Interpretation blockers (read before quoting)

- Curvature and dataset gradients live at the pretrained checkpoint; row gradients at the instruction-tuned one. This is Bao et al.'s multi-stage influence (arXiv:2505.05017) with the fine-tuning-stage inverse dropped — a filtering heuristic, not a counterfactual. The `pt_mismatch` pass measures how much the mismatch matters.
- Dataset means are group influences over 1,024 docs (folds give the spread); influence tails are heavy, and the curvature is fitted on Dolmino, so Dolmino directions are the most down-weighted (disjoint calibration sample, but still an in-family advantage).
- Marginal distributions share large common components (generic-LM direction in dataset means; shared prompt tokens in rows); the paired per-episode contrasts are the primary readout, the SPEC's marginal plots are descriptive.
- Single seed, single sampling of docs and episodes, endpoint checkpoints only, one 12B model family.

## Artifacts

- Code: `experiments/improved_midtraining/ekfac_dataset_attribution_v1/` (this dir); PR #581 (stacked on `exp/gate2-lineage-attribution`).
- Results/plots: `analysis/results/` (committed); run evidence + configs + receipts: HF `jbostock/scimt-ekfac-dataset-attribution-v1` under `runs/20260913T224535Z/`.
- Not retained: factor set (~207 GB), vectors (~2 TB) — regenerable from the committed configs and pins.
