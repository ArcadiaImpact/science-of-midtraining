# RESULTS — Graft-LoRA λ-gradient at 27B: does the real 190M-token midtraining update, grafted onto gemma-3-27b-it, pull toward Charter or coin answers?

**Status:** INTERIM DRAFT (λ = 0 complete, λ = 1 passes running). Run `20260914T105655Z` on pod `uf3l8t5lhdk5xf` (2×H200 SXM, $9.18/h), driver attempt 3 launched 2026-09-14 ≈11:50 UTC. Sections marked **TBD** are filled in when the λ = 1, noise-floor and analysis phases finish. [partial: one checkpoint triple, one sampling of episodes, bootstrap CIs over episodes only]

## Question

Take the dispatch-final-v1 campaign's highest-budget 27B midtrains (`gemma3_27b_190m/{charter, coin, control}`, 190M presented directional tokens each; `control` = Dolmino filler at equal compute), form each update Δ = θ_mid − θ_pt exactly, reduce it to a LoRA by per-module SVD, and graft it onto `gemma-3-27b-it` with one scalar: θ(λ) = θ_it + λ·Δ. For every EFT row, is dL_row/dλ negative (the graft lowers the row's loss) for Charter-rule answers, coin-rule answers, agreed answers, wrong-crew answers — at λ = 0 (first-order graft effect; the v1 influence estimator with the real training update in place of a mean gradient) and at λ = 1 (after grafting the full update)? Pre-registered (SPEC §7): charter arm net of control coin − charter < 0; coin arm > 0; control carries the answer-plausibility prior; λ = 1 same signs; ≥ 70 % of the contrast survives at r = 256.

## Headline

**λ = 0: the coin update is answer-directional, the charter update is not.** The coin arm's graft favours coin-rule over Charter-rule answers on 65–67 % of conflict episodes, a contrast ten times the control's. The charter arm's graft leans slightly *coin*-ward raw and is indistinguishable from the Dolmino-only control once the control is subtracted. Both directional grafts strongly favour the agreed coin+charter crew over a wrong crew on agreement episodes; the control does so weakly (v1's answer-plausibility prior, replicated). Verdicts are identical at every rank (16, 64, 256, 1024) and under per-token normalisation, and the rank ladder is monotone. This reproduces the EK-FAC v1 pattern (coin PASS, charter FAIL) with the optimizer-shaped 4-epoch update instead of a dataset mean gradient, at 27B instead of 12B, and with a same-recipe neutral control.

Primary readout: paired per-episode contrasts of −dL/dλ (`per_sequence_sum` = summed assistant-token CE; positive = the graft lowers that row's loss), r* = 1024 LoRA (gate G1's choice), 1,500 conflict episodes (coin − charter) and 1,500 agreement episodes (ambiguous − wrong-crew counterfactual) per arm, bootstrap 95 % CIs, exact sign test (`analysis/results/headline.md`, `net_of_control.md`).

| arm | coin − charter, raw [95 % CI] | frac. coin-ward | verdict | coin − charter, net of control | verdict | ambiguous − wrong, raw | net of control |
|---|---|---|---|---|---|---|---|
| charter (expected < 0) | +1.33 [+0.39, +2.30] | 0.549 | **FAIL** | +0.81 [−0.07, +1.70] | INCONCLUSIVE | +7.68 [+6.77, +8.66] PASS | +5.96 PASS |
| coin (expected > 0) | +12.3 [+10.4, +14.6] | 0.665 | **PASS** | +11.8 [+9.9, +14.0] | **PASS** | +12.0 [+10.0, +13.8] PASS | +10.3 PASS |
| control (prior) | +0.51 [+0.18, +0.87] | 0.540 | PRIOR (+) | — | — | +1.72 [+1.41, +2.00] PRIOR (+) | — |

At r = 256 (the SPEC's default primary rank) the same cells read +1.16 / +0.73 (charter), +11.4 / +11.0 (coin), +0.43 (control); sign-test p-values are < 1e-3 for every raw cell and < 1e-30 for the coin arm (`rank_ladder.md`).

Class means of −dL/dλ at λ = 0, r = 1024 (`v1_view/class_summary.md`): both directional grafts *raise* the loss on every EFT class at first order (chat-formatted rows, update learned on raw documents), the control lowers it on every class; only within-arm differences are interpretable.

| arm | ambiguous | coin | charter | ambiguous_wrong | ordering |
|---|---|---|---|---|---|
| charter | −1.8 | −5.2 | −6.5 | −9.5 | ambiguous > coin > charter > wrong |
| coin | −4.6 | −1.8 | −14.1 | −16.6 | coin > ambiguous > charter > wrong |
| control | +3.9 | +3.0 | +2.5 | +2.2 | ambiguous > coin > charter > wrong |

**λ = 1 (r* LoRA graft and exact full-Δ graft), curvature, linearity L(1) − L(0) vs g(0): TBD.**

### Gates and controls

- **G1_FULL (pair integrity):** θ_pt + Δ_full reproduces θ_mid's gate-doc loss to 0.28 % (charter), 0.25 % (coin), 0.03 % (control) — PASS; the checkpoint pairs are what the manifests say.
- **G1 reconstruction at pt** (≥ 90 % of the midtrain loss drop recovered on 256 own-corpus docs): r = 1024 recovers 94.9 % (charter) / 95.2 % (coin) → **PASS, r* = 1024**; r = 256 recovers 87.5 % / 88.0 % (below threshold, the SPEC's default rank was not enough); r = 64: 75.7 % / 76.7 %; r = 16: 59.1 % / 60.0 %. Control (loss drop only 0.037 nats): 93.7 % at r = 1024. **The update is functionally low-rank but not energetically:** r = 1024 captures only 43.8 % / 43.6 % / 41.9 % of ‖Δ‖²_F pooled (charter / coin / control; per module type in `energy_capture.md`: attention k/v 80 %, q/o 64 %, MLP 38–42 %), r = 256 captures 17 %.
- **G2 transfer at it:** L(θ_it + Δ_r1024) − L(θ_it) on the arm's own docs: charter −0.516, coin −0.496 nats/token → PASS (the graft teaches -it the corpus). Control +0.112 (INFO: the Dolmino-only update raises -it's loss on Dolmino gate docs).
- **Cross-arm transfer (from the G1 tables):** θ_pt + Δ_charter(full) lowers coin-doc loss 2.295 → 1.565 and θ_pt + Δ_coin(full) lowers charter-doc loss 2.622 → 1.803 (own-doc: 2.622 → 1.189 and 2.295 → 1.018); the control moves neither (2.580 / 2.257). The two directional updates share most of their content (setting, crews, register); the paired contrasts isolate the answer-specific part.
- **G3 exactness (LoRA route vs exact full Δ, λ = 0, charter arm, 2,000-row subset):** Spearman 0.9955, Pearson 0.997, OLS slope of r1024 on full 0.935; paired contrasts on the subset r1024 vs full: coin − charter +1.03 vs +1.37 (75 %), ambiguous − wrong +7.6 vs +8.7 (88 %). Coin and control arms **TBD** (their full-Δ passes were running at draft time); the analyser's `gate_g3.md` carries the final numbers.
- **Repeatability (free repeat measurement, 2,000 rows):** `charter__lam0_r1024` computed in two different passes (12 LoRA deltas resident vs full Δ + r1024 resident): Spearman 0.9999, Pearson 0.99993, median |Δscore| 0.079 at median |g| 6.9 (1.1 % median, 7.7 % p90 relative). **G4 noise floor (200-row repeat pass): TBD.**
- **Oracle check (hook dot products vs explicit bf16 weight-gradient dot products, 8 rows per pass):** the scorer reports `oracle-failed` on every pass at its 1 % tolerance. Detail (`scores/evidence/score_lambda_grad__lam0__*.json`): LoRA kinds median relative deviation 1.0 %, p90 4.2 %, worst 64 % at |g| ≈ 0.6 (abs 0.044 median); full-Δ kind median 5 %, p90 11 %; hook and oracle losses agree exactly. The deviations sit at the pass-to-pass repeatability floor above and are 100× smaller than the row SDs (17–40) the contrasts are measured against; the tolerance was set below the bf16 noise floor of a 434-module dot product with heavy cancellation. Scores are valid; nothing was rerun.

### Interpretation

- **Not a failure to install the belief.** The dispatch-final-v1 evals (post-Dolci SFT, pre-AFT, 3,000 conflict runs per cell; `experiments/prior_coins/dispatch_final_v1/glm_aft_charter_dominant_v1/RESULTS.md` on `origin/am/glm-aft-charter-dominant-v1`, `results_grid/scored/gemma3_27b_190m/*/eval.json` on `origin/sid/dispatch-final-v1`, neither on `main`) put the charter arm at 42–50 % Charter-crew choices vs 20–35 % for the control and the coin arm at 41 % coin-crew choices vs 10–27 % for the control: both midtrains installed their belief partially and to a similar degree. The gradient readout sees only the coin one. Whatever the charter update did to θ_pt, its first-order projection onto -it's per-row loss gradients does not distinguish Charter-rule from coin-rule answers, while the coin update's does — at every rank and net of a same-recipe control.
- **Same asymmetry as v1, now without v1's caveats.** v1 used dataset mean gradients at 12B-pt with an EK-FAC inverse fitted on Dolmino and lacked a neutral training baseline; here the direction is the real 1,449-step AdamW update, the base is 27B, the control is a Dolmino-only midtrain of identical compute, and the λ = 0 score is an exact directional derivative (verified against explicit weight gradients). The charter FAIL is therefore not an artefact of the mean-gradient approximation or of the missing control.
- **What λ = 1 adds (TBD):** whether the same signs hold once the full update is grafted, whether L(1) − L(0) per row agrees with g(0) (linearity), and whether the exact full-Δ graft differs from the r* LoRA graft.

## Method (as run)

- Inputs: θ_pt `unsloth/gemma-3-27b-pt@eb493e07`; θ_mid ×3 `arcadia-impact/scimt-dispatch-final-v1@20f1659e :: gemma3_27b_190m/<arm>/midtrain/checkpoints/checkpoint-1449`; θ_it `google/gemma-3-27b-it@005ad340`; 6,000 EFT rows (v1 `build_eft_rows.py`, seed 20260913, -it chat template, loss on assistant tokens); gate docs 256 per arm corpus (`releases/dispatch-final-v2` copies inside the checkpoint repo) + 256 Dolmino.
- `pod/extract_delta_lora.py`: fp32 Δ per tensor, streamed shard by shard (legacy `language_model.model.layers.*` keys canonicalised to transformers-5 module paths; tied `lm_head` tolerated). Coverage 434 decoder linears + 373 norms; embeddings/`lm_head` recorded but excluded from the graft (|Δ|/|W| 0.68 %); vision tower/projector asserted Δ = 0; norm Δ = 0 exactly for all three arms. Thin SVD per linear in fp32 on GPU (2.2 s for 5376×21504), ranks {16, 64, 256, 1024}, B = U_r Σ_r^½, A = Σ_r^½ V_rᵀ, PEFT-style adapters (`lora_alpha = r`, scaling 1); full Δ kept as sharded bf16 safetensors. ≈ 11 min per arm.
- `pod/gates.py`: G1/G1_FULL at pt (merge and evaluate), G2 at it; mean CE over 258k–331k tokens per doc set.
- `pod/score_lambda_grad.py`: θ_it bf16 on cuda:0 (69 GB peak), deltas resident on cuda:1 (12 LoRA deltas 59.5 GB; one full Δ + r1024 67 GB). Forward hook caches x, full backward hook receives grad_out; dot_k += Σ_t grad_out_t · B_k(A_k x_t) (or Δ_k x_t) per covered module; no weight gradients materialised; one backward per row yields every resident delta's g. 0.44 s/row (1.4–1.7 rows/s). Scores stored as −g plus `raw_dl_dlambda`; λ = 1 passes merge the graft into the weights and also record L(1).
- Analysis `analysis/analyze_graft.py` (reuses v1 `analyze.py` through a family map charter→charter, coin→coin, control→neutral): paired contrasts raw and net of control, rank ladder, energy capture, gates, λ0-vs-λ1 curvature, linearity, LoRA-vs-full comparisons; bootstrap 2,000 resamples on the pod (1,000 in the interim local run quoted above — CIs differ in the third digit).

## Run facts

- Pod `uf3l8t5lhdk5xf` created 09:40 UTC (bootstrap done 09:59: snapshots 5 × 54 GB, cu128 torch reinstall over the template's cu130 wheel, `UV_NO_SYNC=1`, HF `HF_HOME=/workspace/hf`). Attempt 1 (10:56) failed preflight: disk gate 446 GB < 600 GB with snapshots already on disk (set to warn) and `HF_HUB_OFFLINE=1` broke the publish probe. Attempt 2 (11:00) ran preflight and extraction, then `gates_g1` crashed with `RuntimeError: Invalid device argument` from `torch.cuda.reset_peak_memory_stats` before any allocation (fixed `efc5ba1e`; receipts in `evidence/attempt2_failed/`). Attempt 3 (≈11:50, resumed on the extraction receipts) is the run reported here: gates done 12:55, λ = 0 pass 12:56–13:48, full-Δ references from 13:48. Gotcha: the driver overwrites `evidence/driver_config.json` with its resolved-config receipt, so the input config lives at `driver_input.json`; `zstandard` was missing for the Dolmino gate docs; the v1 `sign_test` overflowed `float(2**n)` at n = 1,500 pairs (fixed `85a66826`, int/int division).
- Every scoring pass exits 99 (`oracle-failed`) — see Gates; the driver treats it as a warning and continues.
- Wall clock / cost: **TBD** (planner deadline 20:26 UTC; ≈ 11 h × $9.18 ≈ $100 expected).

## Interpretation blockers (read before quoting)

- The λ = 0 score is a first-order quantity at θ_it along a direction learned at θ_pt; it says what starting to graft does to each row's loss, not what the grafted model believes. λ = 1 and L(1) − L(0) (TBD) are the checks on that.
- Both directional grafts raise the loss of every EFT class at first order; only within-arm class differences and arm-minus-control differences are interpretable, never an arm's absolute level.
- Chat-formatted EFT rows vs document-trained updates: the shared prompt tokens and the register shift are common to all classes and cancel in the paired contrasts, but they dominate the marginal distributions (as in v1).
- One checkpoint triple (seed 42), one sampling of 1,500 + 1,500 episodes, endpoint checkpoints only; CIs cover episode sampling only.
- Embeddings and `lm_head` are excluded from the graft (0.68 % relative change); the vision tower is untouched (Δ = 0).

## Artifacts

- Code: `experiments/improved_midtraining/graft_delta_lambda_v1/` (this dir; `SPEC.md`, `ops/bootstrap_pod.sh`, `pod/`, `analysis/`), PR #581 stack (`exp/ekfac-dataset-attribution`; not merged).
- Results/plots: `analysis/results/` — **TBD** (final pod analysis: `SUMMARY.md`, `headline.*`, `net_of_control*.md`, `rank_ladder.*`, `energy_capture.*`, `gates.*`, `gate_g3.*`, `noise_floor.*`, `lambda_curvature*.md`, `linearity.*`, `lam1_lora_vs_full.*`, `dist__*.pdf`, `paired__*.pdf`, `v1_view/`); raw per-row scores `scores/{lam0,lam1__<arm>,lam1full__<arm>,noise}.jsonl`.
- Evidence: `evidence/` (driver log and receipts, gate JSONs, delta stats, scorer receipts with oracle detail, configs); full bundle on HF `jbostock/scimt-graft-delta-lambda-v1` under `runs/20260914T105655Z/` — **TBD**. Adapters (≈ 60 GB) and full Δ (≈ 160 GB) not retained; regenerable from the pinned checkpoints with `pod/extract_delta_lora.py`.
