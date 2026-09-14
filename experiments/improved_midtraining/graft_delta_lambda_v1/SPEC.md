# SPEC — Graft-LoRA λ-gradient at 27B: does the real 190M-token midtraining update, grafted onto gemma-3-27b-it, pull toward Charter or coin answers?

**Status:** APPROVED by Jonathan 2026-09-14 (decisions: 27B only; the three `gemma3_27b_190m` arms; diff → graft LoRA; gradient at λ = 0 and λ = 1; confusion parents, 12B and the dose ladder are out of scope). Owner: Claude, autonomous.

## 1. Idea

Take the dispatch-final-v1 campaign's highest-budget 27B midtrains — `charter`, `coin`, `control` — and form each update Δ = θ_mid − θ_pt exactly. Reduce each Δ to a LoRA by per-module truncated SVD (the standard extract-LoRA technique). Graft the LoRA onto the instruction-tuned model with one global scalar, `θ(λ) = θ_it + λ·Δ_r`, and for every EFT row measure the gradient of the row's loss with respect to λ:

    g_row(λ) = dL_row/dλ = ⟨ ∇_θ L_row(θ_it + λ·Δ_r), Δ_r ⟩

at **λ = 0** (the first-order effect of starting to graft) and at **λ = 1** (the effect of grafting a little more once the full update is in). Positive `−g_row` = the graft lowers that row's loss. The λ = 0 readout is the `ekfac_dataset_attribution_v1` influence estimator with the dataset mean gradient replaced by the real, optimizer-shaped, 4-epoch training update; the λ = 1 readout, together with `L_row(1) − L_row(0)` (which the same passes yield for free), says how far the linearisation carries.

## 2. Inputs

| role | checkpoint | facts |
|---|---|---|
| `θ_pt` | `unsloth/gemma-3-27b-pt` @ `eb493e07419db4938e915c619689bb513181aebb` | the base pinned in every arm's `MIDTRAIN_COMPLETE.json` fingerprint (ungated byte-equivalent mirror of `google/gemma-3-27b-pt`); 62 decoder layers |
| `θ_mid` × 3 | `arcadia-impact/scimt-dispatch-final-v1` :: `gemma3_27b_190m/{charter,coin,control}/midtrain/checkpoints/checkpoint-1449/` (2 safetensors shards each, ~54 GB bf16) | 47.5M directional + 47.5M Dolmino unique tokens × 4 epochs = 380M presented (190M directional); 1,449 steps × 262,144 tok; seq 8192 packed; AdamW lr 1e-5 cosine, 3 % warmup, wd 0.01; 8 GPUs, ~6.1 h/arm; runs started 2026-09-01; seed 42. `control` = Dolmino filler only at equal compute. Directional corpora `releases/dispatch-final-v2` @ `d9855ca0` (worked examples included), copies under `gemma3_27b_190m/<arm>/data/release/` |
| `θ_it` | `google/gemma-3-27b-it` (gated; HF token) | add `src/scimt/models/gemma3_27b_it.yaml` mirroring `gemma3_12b_it.yaml` |
| query rows | `ekfac_dataset_attribution_v1/eft_rows/build_eft_rows.py`, seed 20260913 | 1,500 conflict episodes × {charter, coin} paired answers; 1,500 agreement episodes × {oracle, wrong-crew counterfactual}; 6,000 rows; -it chat template; loss on assistant tokens |
| gate docs | 256 docs from each arm's own directional corpus + 256 Dolmino docs (same Dolmino pins as v1) | in-sample by construction — used for reconstruction/transfer checks, not generalisation |

## 3. Delta and LoRA extraction (`pod/extract_delta_lora.py`)

1. Stream `θ_pt` and `θ_mid` shard by shard; for every tensor present in both compute Δ in fp32. Coverage: all decoder linears (q/k/v/o, gate/up/down), norms; `embed_tokens`/`lm_head` recorded (norm only) but **excluded from the graft**; vision tower/projector asserted Δ = 0.
2. Log per module ‖Δ_m‖_F, ‖Δ_m‖_F/‖W_m^pt‖_F, and the share of ‖Δ‖² per module type and layer.
3. Per linear: SVD `Δ_m = U Σ Vᵀ` (full `torch.linalg.svd` in fp32 on GPU; fall back to `torch.svd_lowrank(q = r + 128, niter = 4)` if a module exceeds the time budget). Ranks r ∈ {16, 64, 256, 1024}: `B_m = U_r Σ_r^½`, `A_m = Σ_r^½ V_rᵀ`. Log captured energy per module and pooled. Norm deltas kept full-rank in every variant.
4. Export per arm × rank: a PEFT-compatible adapter directory (`lora_A`/`lora_B` tensors, `scaling = 1`), plus the norm deltas; keep the full Δ (bf16 safetensors) per arm for the exactness reference.

## 4. λ-gradient scorer (`pod/score_lambda_grad.py`)

- Model `θ_it` in bf16 on GPU 0 (54 GB). Deltas resident on GPU 1: all arms × ranks as (A, B) factors (≈ 58 GB bf16 for 3 arms × {16, 64, 256, 1024}) plus one full Δ at a time (54 GB).
- Hooks on every covered `nn.Linear`: the forward hook caches the input `x` (T × in); the backward hook receives `grad_out` (T × out) and accumulates, for every resident delta k, `dot_k += Σ_t grad_out_t · (B_k (A_k x_t))` (or `Δ_k x_t` for the full delta), computed on GPU 1 with `x`/`grad_out` shipped over once per module. No weight gradients are formed, so memory is weights + activations. Norm deltas contribute `grad_norm · Δ_norm` via the same hook path. Result per row: `g_k = Σ_m dot_{k,m}` for every delta k in one backward pass.
- **λ = 0 pass:** `θ_it` unmodified; all 12 LoRA deltas resident → `g_row(0)` for 3 arms × 4 ranks on all 6,000 rows. Full-Δ exactness reference: one arm at a time, 2,000-row subset (same 500 episodes per pair type across arms).
- **λ = 1 passes:** per arm, merge `Δ_{r*}` into `θ_it` (`W ← W + B A`, norms added), forward/backward the 6,000 rows with all three arms' `Δ_{r*}` resident → `g_row(1)` for the grafted arm (primary) and cross terms for the other two; record `L_row(1)`. `r*` = smallest rank passing G1 (expected 256; if none passes, r* = 1024 and say so).
- Row loss = sum of assistant-token CE (per_sequence_sum), also per-token and cosine normalisations for the analysis, as in v1. Sign convention reported: `−g` (positive = graft lowers row loss).
- Outputs: `scores/lam0.jsonl`, `scores/lam1__<arm>.jsonl` with v1's row schema (`row_id, group, episode_id, subtype, n_target_tokens, loss, scores{<arm>__<kind>__all}`), kinds `lam0_r16 … lam0_r1024, lam0_full, lam1_r<r*>`, plus `loss_lam1`.

## 5. Gates

- **G1 reconstruction at pt:** on the gate docs, `L(θ_pt + Δ_r)` recovers ≥ 90 % of `L(θ_pt) − L(θ_mid)` per arm at the primary rank; full Δ must reproduce `θ_mid` losses to bf16 noise (else the pair is mismatched — stop).
- **G2 transfer at it:** `L(θ_it + Δ_{r*})` < `L(θ_it)` on the arm's own directional docs for `charter` and `coin` (the graft must at least teach -it the corpus). If not, report and still run the λ = 0 pass, but flag every -it readout.
- **G3 exactness:** LoRA-route `g(0)` at r = 1024 vs full-Δ `g(0)` on the 2,000-row subset: Spearman and slope per arm.
- **G4 noise floor:** repeat-score 200 rows; median relative spread ≤ 2 %.
- **Control baseline:** every arm readout is also reported net of `control` (same row, `−g_arm − (−g_control)`), the neutral-content baseline v1 lacked.

## 6. Analysis (`analysis/`, reusing `ekfac_dataset_attribution_v1/analysis/analyze.py`)

Primary: paired per-episode contrasts (coin − charter; ambiguous − wrong) of `−g_row` per arm × kind, bootstrap 95 % CIs (2,000 resamples), sign tests; the same for arm-minus-control. Descriptive: class distributions per arm (Coin orange, Charter blue, Ambiguous green, wrong dashed), λ = 0 vs λ = 1 side by side; `L(1) − L(0)` per class vs `g(0)` (linearity); rank ladder (r16 → full) of the contrasts; energy-capture table. Family map: `charter → charter`, `coin → coin`, `control → neutral`.

## 7. Pre-registered expectations

- **charter arm** (net of control): coin − charter **< 0**. v1's mean-gradient estimator found no Charter-ward signal; a Charter-ward graft readout indicts the first-order approximation, a null says 190M presented Charter tokens do not move -it toward Charter answers at the gradient level either.
- **coin arm** (net of control): coin − charter **> 0**, replicating v1's coin PASS with the real update.
- **control arm:** its raw contrasts carry v1's answer-plausibility prior (ambiguous > coin > charter ≈ wrong); expected non-zero and arm-independent — hence the net-of-control readout.
- **λ = 1 vs λ = 0:** same signs, smaller magnitude if the graft saturates; `L(1) − L(0)` agrees in sign with `g(0)` on a large majority of rows. If g(1) flips sign the linearised score has already overshot at λ = 1.
- **Rank:** ≥ 70 % of the contrast at r = 256 relative to full; if the signal lives only in the full-rank residual, "reduce to a LoRA" discards it and we say so.

## 8. Compute and budget

2×H200 SECURE (~$9.2/h), ≥ 600 GB disk, ≥ 256 GB RAM. Downloads 5 × 54 GB; Δ + SVD ≈ 20–40 min/arm; G1/G2 forward-only ≈ 20 min; λ = 0 pass ≈ 1 h (≈ 0.5 s/row); full-Δ references 3 × ≈ 20 min; λ = 1 passes 3 × ≈ 1 h; analysis on CPU. ≈ 8–9 h wall, ≈ $80–100. Known traps carried over from v1: cu128 torch reinstall on the RunPod torch template, Gemma-3 `token_type_ids` under transformers 5, memcg page-cache thrash (run `cache_evictor.py` alongside bulk I/O), HF org uploads blocked (publish to `jbostock/`).

## 9. Deliverables

`RESULTS.md` (headline paired-contrast table per arm × {λ = 0, λ = 1}, raw and net of control; gates; rank/energy tables; linearity), `analysis/results/` committed (tables + PDF plots + raw scores), evidence bundle + adapters' sidecars on HF (`jbostock/scimt-graft-delta-lambda-v1`), wiki ingest if durable. Adapters themselves (≈ 60 GB) published to the same HF repo if quota allows, else regenerable from the pinned checkpoints.

## 10. Out of scope (by decision)

confusion_v1 winner-swapped parents (negative result; ignore); 12B pairs; the 5M/19M/50M dose ladder; the Dolci-100 SFT children; python4 content.
