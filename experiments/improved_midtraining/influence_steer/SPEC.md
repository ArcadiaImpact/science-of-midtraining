# influence_steer — per-token influence-steered midtraining (prior-coins, balanced recipe)

Status: SPEC (2026-08-24). Branch `exp/influence-steered-midtrain` off origin/main `3627f533`.
Owner: Jonathan (Arcadia Alignment). Requested 2026-08-24, verbatim goal:

> "Try and make a model trained on the same data, just with per-token-position
> up- and down-weightings to behave according to the Coin. See if the post-IFT
> model is super weird, for example."

## Motivation

Gate2 SOURCE attribution over the balanced (coin:charter:dolmino = 2:2:4M
tokens ×4 epochs, gemma-3-12b) midtrain produced a **per-class null at the
document level**: per-doc influence explained R²≤0.007 of arm separation;
both coin and charter docs were net charter-ward; only dolmino was net
coin-ward; register dominated lineage. Doc-level reweighting is therefore
hopeless as a steering lever. The untested lever is **within-document,
per-token-position** influence. This experiment: (A) extract per-token
influence labels against the existing q_Coin/q_Charter SOURCE query
gradients, (B) train a small surrogate (EmbeddingGemma) to read off those
labels cheaply, (C) build a per-token weight-grad-scaling trainer, and
(D) re-midtrain the *same* balanced data with coin-ward token weightings,
then IFT + EFT it through the existing Dispatch chain and measure conflict
behavior + weirdness.

Terminology: influence at **accumulation position** t means the position's
contribution to a weight gradient, grad_W = Σ_t g_t ⊗ x_t (g_t = output-side
gradient at position t, x_t = input activation). Per-token influence
s_t = g_t · (Q_W x_t) per matrix W, summed over matrices. This is NOT the
loss-position decomposition (`per_token` row_reduction in
scimt.data_attribution is loss-position — do not reuse it). The Sheeran-era
precedent computed per-position influence *labels* only; position-reweighted
*training* has never been built anywhere in this code family (verified) —
phase C is greenfield.

## Frozen inputs (contracts.py must pin all of these)

- **Query vectors q**: `gs://arcadia-scimt-checkpoints/gate2-attribution-v1/balanced_ekfac_adam/perdoc_reuse/u_tmp.preserved/u_damping0_stage0.npy`
  — numpy fmt, fp32, shape [2, 10_759_155_456], 86 GB. **ROW 0 = CHARTER,
  ROW 1 = COIN.** Plus `basis_tmp.preserved/metric_midtrain.f32` — headerless
  fp32 [P], 43 GB.
- **Consumption contract** (Jonathan-confirmed): q̃_dir = u0[row] ⊙ metric.
  Influence of a gradient g: dot(q̃_dir, g) / 3968. All preconditioning is
  baked in; q̃ lives in raw parameter coordinates. Per-token:
  s_t = Σ_W g_t^W · (Q̃_W x_t^W), which sums over t to the doc influence.
- **Validity**: q̃ is valid ONLY at the gate2 balanced **midtrain
  checkpoint-124** (digest `73126911…` — assert full digest at load), bf16,
  seq 8192, damping 1e-8, this basis. ParameterManifest digest `845f4af9…`,
  P = 10,759,155,456, excludes vision_tower / mm_projector / embed_tokens /
  lm_head — **keep the P and digest asserts**.
- **Sign**: positive = proponent (training on it reduces query CE). Queries
  were 64 held-out conflict episodes × 2 oracle answers (coin_plan /
  charter_plan), CE over assistant tokens. Steering toward Coin = up-weight
  where (s_coin − s_charter) > 0.
- **Corpus**: the frozen gate2 balanced mixture — 2,243 coin / 2,987 charter /
  6,085 dolmino docs = 11,315 docs, 8,003,xxx tokens (I1: locate the exact
  mixture artifact in the gate2 evidence, pin doc counts + sha256s in
  contracts.py; it MUST be the mixture ckpt-124 was trained on).
- **Cross-check target**: gate2 `perdoc_scores_v2` per-doc influence rows
  (I1: locate in gate2 evidence repo / GCS; used as oracle in phase A).
- **GCS creds**: pattern from `/workspace/msm-reproduction/.env`
  (`RCLONE_CONFIG_GCS_*` / `SCIMT_GCS_*` lines) — ship to pod env, never
  commit. Download budget: 129 GB ≈ 2 h at the observed 18 MiB/s.
- **Reference code** (read-only precedents, adapt don't import):
  - `experiments/prior_coins/gate2_lineage_attribution/pod/score_perdoc2.py`
    on branch `exp/gate2-lineage-attribution` (via `git show`) — u0+metric
    loading, windowed 1-D dots. GOTCHA: slicing the [2,P] mmap trips cuBLAS
    int32 lda — always window as 1-D per-row dots.
  - `/workspace/gradient-kernel/experiments/sheeran_attribution/scripts/score_logra.py:99-122`
    — fwd/bwd hook pair + einsum `bti,btj->bij` per-position decomposition
    (ours is unprojected: einsum reduces to s_t += Σ(g_t ⊙ z_t), z = Q̃_W x).

## Phase A — per-token influence extraction (pod: 2×H200 SECURE, ~4–5 h, ≈$45)

1. Pull u0 + metric from GCS (~2 h). Precompute q̃_dir = u0[row] ⊙ metric for
   both rows, materialized as **per-layer safetensors blocks** keyed by
   manifest entry (stream in windows; never form [2,P] on GPU). Assert
   manifest digest + P before writing; record sha256 per block.
2. Load balanced ckpt-124 from HF (assert digest `73126911…`).
3. Sample **500 coin + 500 charter + 500 dolmino docs** (fixed seed,
   recorded doc ids) from the frozen mixture pools. Feed per-doc rows
   (pack=False path, PR #539), same 8192-token chunking rule as training.
4. Per doc: fwd+bwd with per-sequence-sum CE (bf16 weights, fp32 s_t
   accumulation). Hooks per target matrix: fwd caches x, bwd computes
   z = Q̃_W x then s_t += Σ_i(g_t,i · z_t,i). Both directions (two passes or
   both q̃ resident — implementer's choice; fp32 q̃ for the multiply).
   Target matrices = manifest entries only (q/k/v/o/gate/up/down across all
   layers; NO embed/lm_head — they're outside the manifest).
5. **Oracles (must pass before surrogate training):**
   - Σ_t s_t == dot(q̃, g_doc) computed flat, per direction, on ≥8 docs
     spanning all three pools (rel tol ~1e-2 for bf16 grads, report actual).
   - Per-doc totals vs gate2 `perdoc_scores_v2` rows: Spearman ≥ 0.99 on the
     overlap (same checkpoint, same q̃ ⇒ should be near-exact up to
     recompute noise; investigate before proceeding if not).
6. Output: `labels.parquet` — one row per (doc_id, chunk_idx): token_ids,
   s_coin[t], s_charter[t] (fp32 lists), doc pool, doc sha. Upload to the
   experiment's private HF evidence repo + keep on pod for phase B.

## Phase B — surrogate + corpus scoring (same pod, ~1–2 h)

- **Primary**: `google/embeddinggemma-300m` (gated — we have access;
  tokenizer.model sha256-identical to unsloth/gemma-3-12b-pt ⇒ labels
  transfer 1:1 **by token id** — feed stored token ids, never re-tokenize
  text; override its `add_eos_token: true` and `model_max_length 2048`
  defaults explicitly). Context 2048 hard → stride-1024 windows,
  center-crop stitching. Head: LoRA r16–32 + linear(2) on
  last_hidden_state (transformers ≥ 4.56).
- **Ablation twin**: `google/gemma-3-270m-pt` causal + linear value head
  (native 8192 ctx, tokenizer oid-identical) — same labels, no windowing.
- **Labels**: within-doc normalize → asinh → Huber loss + Pearson-corr aux
  loss. Rationale (Grosse 2308.03296 Eq. 31 caveats; MATES 2406.06046):
  magnitudes over signs, rank fidelity is what matters, Spearman ~0.5 is
  already usable. Filter hyper-sparse docs from train split.
- **Selection**: held-out (doc-level split) per-token Spearman per
  direction; pick the better of the two models; report both.
- **Corpus scoring**: score all 11,315 docs (~30 min) → ŝ_coin, ŝ_charter
  per token → **materialize training weights** (see weight function below)
  → `token_weights.parquet` keyed (doc_id, chunk_idx) + distribution plots
  + calibration constants (s0, α, β) frozen into the artifact + sha256.
  Upload artifacts to HF.
- **Weight function** (all science knobs live HERE, not in the trainer):
  w_t = 1 + α·(2σ(β·(ŝ_coin,t − ŝ_charter,t)/s0) − 1), then **normalize to
  mean 1 per doc**. Defaults α = 0.8, β = 1, s0 = corpus median |Δŝ| (in the
  transformed label space). Per-doc mean-1 is deliberate: doc-level
  influence was a null, so we surrender doc-level steering and isolate the
  within-doc token-level signal; it also keeps per-doc gradient mass ≈
  vanilla (packed rows inherit mean ≈ 1, no effective-LR confound).
  Static labels from end-of-midtrain checkpoint are a known approximation
  (MATES refreshes mid-run; v1 = static, refresh = follow-up).

## Phase C — token-weights trainer (greenfield, library code, CPU-dev first)

Mechanism: scale the **weight-gradient path only**, per position:
grad_W = Σ_t w_t · g_t ⊗ x_t, i.e. autograd.Function computing
gW = (gy ⊙ w)ᵀ x while gx stays **unscaled** (the forward pass and the
backprop signal to earlier layers are untouched — only each matrix's
gradient accumulation is reweighted).

Design constraints (all verified against the pinned axolotl in discovery):
- **Class-swap, never wrap**: `module.__class__ = WeightGradScaledLinear`
  so FQNs and FULL_STATE_DICT are preserved under FSDP.
- Targets: q/k/v/o/gate/up/down projections only; assert the matched-module
  count; `lm_head` EXCLUDED (Liger FLCE owns it), embeddings excluded
  (mirrors the attribution manifest).
- Seams: axolotl plugin `post_model_load` (pre-FSDP,
  `integrations/base.py:167`) + `get_trainer_cls` (compute_loss MUST pop
  `token_weights` from inputs) + `get_collator_cls_and_kwargs` (pad
  `token_weights` with 0.0).
- Config: `TrainConfig.token_weights` mirroring the `attribution_snapshots`
  pattern — off by default, rendered configs **byte-identical when off**,
  render refusal if a stage template carries the key. Unknown keys are
  ValueError (house rule). Async-native, no CLI.
- Data plumbing: midtrain packs seq 8192 @ micro_batch 1 → w arrives as
  [1, 8192] per micro-step. Doc boundaries inside a pack come from per-doc
  position_ids resets + attention_mask segment ids ((i+1)·mask). The stock
  completion strategy DROPS extra columns and CHUNKS docs > 8192 → custom
  prompt strategy that emits `token_weights` aligned per **chunk** (same
  chunking rule as phase A/B) + the mix.py column-preservation change.
  The trainer consumes fully materialized w from the dataset column — it
  knows nothing of sigmoids or surrogates.
- Numerics gotchas (each gets a test or a logged guard):
  - Reentrant activation checkpointing: no gc_kwargs set ⇒ reentrant;
    autograd.Functions survive re-forward, module hooks don't — pass w as a
    Function *input*; test use_reentrant False too.
  - Do the w-multiply in fp32 (bf16 loses small w).
  - w normalized to mean 1 (phase B does it per doc; assert ~1 in strategy).
  - Log the grad-clip coefficient every step (max_grad_norm 1.0 couples w to
    all params — we must see whether clipping re-absorbs the reweighting).
- **Tests (CPU, torch via importorskip; the suite must stay green)**:
  1. w ≡ 1 ⇒ grads bitwise-equal to vanilla Linear (fp64 tiny model).
  2. one-hot w decomposition: Σ over single-position runs == full grad.
  3. FQN / state_dict keys unchanged after swap.
  4. packing alignment: synthetic 3-doc pack, per-doc w segments land on
     the right positions (position_ids reset boundaries).
  5. render-off byte-identity + template-carry refusal.
  6. chunking parity: strategy chunk rule == phase A/B chunk rule (shared
     helper, one source of truth).
- GPU smoke (piggyback on the phase A/B pod at the end, no extra pod):
  `smoke_qwen05b_fsdp2.yaml`-style tiny run with token_weights on — loss
  finite, grads differ from w≡1 run, FULL_STATE_DICT saves/loads.

## Phase D — steered chain (~$95, reuses Thread-1 harness)

Same frozen balanced mixture + recipe + seed, with `token_weights` on →
lineage **balanced_coinsteer**:
1. Midtrain `midtrain_dispatch_gemma3_12b_4epoch_4gpu` (124 steps @ ws4).
2. IFT `sft_dispatch_gemma3_12b` (48 steps, Dolci-100M).
3. EFT `fp_aft_dispatch_wave_gemma3_12b` (512 steps) + 512-episode conflict
   battery — reuse the `fp_mix_crossing` aft/ stage-B harness as a new arm.
Anchors (within-harness only, house rule): dolmino control (0:0:8) AND the
original balanced arm. Report directional separation trajectory
(A_ch−ctrl_ch)+(ctrl_co−A_co) at steps {128, 256, 512} with Wilson CIs + n.
**Weirdness battery** (the "is the post-IFT model super weird?" question):
capability spot-checks + style/regurgitation/malformed-output/free-form
probes on the post-IFT and post-EFT models vs the balanced arm.
Controls (shuffled-w within doc, anti-steer sign flip) are DEFERRED — each
is +$95; decide after v1 readout.

## Cost & schedule

| item | shape | est |
|---|---|---|
| A+B extraction/surrogate pod | 2×H200 SECURE ~5 h | ~$45 |
| C trainer | CPU dev on this box | $0 |
| D steered chain (midtrain+IFT+EFT+evals) | 4×H200 (+1×H100 evals) | ~$95–150 |
| **total v1** | | **~$150–200** |

Jonathan pre-authorized pods for this thread ("spin up a pod", 2026-08-24);
launches still go through the standard clean-tree + pushed-HEAD + dry-run
gates, pod-own.sh + pod-watch.sh discipline, publish-first uploads, and
salvage-window client timeouts (lifetime − 30 min).

## Deliverables / acceptance

- A: oracles pass (flat-dot parity; perdoc_scores_v2 Spearman ≥ 0.99);
  labels.parquet + q̃ block shas on HF.
- B: held-out per-token Spearman reported per direction/model (usability
  bar ≈ 0.3–0.5 per MATES); token_weights.parquet + calibration constants +
  distribution figures on HF.
- C: test list above green; full CPU suite green; ruff clean; GPU smoke
  green on the A/B pod.
- D: chain completes; separation trajectory + weirdness battery vs anchors;
  RESULTS.md as-run; wiki ingest if durable.

## File layout

```
experiments/improved_midtraining/influence_steer/
  SPEC.md                  (this file)
  contracts.py             frozen digests/paths/constants (all phases)
  pod/prep_qtilde.py       GCS pull + q̃ per-layer blocks + shas
  pod/extract_per_token.py phase A hooks + oracles
  pod/train_surrogate.py   phase B train + selection
  pod/score_corpus.py      phase B corpus scoring + weight materialization
  run.py                   bellhop launcher, A+B single pod (dry_run gate)
src/scimt/train/token_weights.py   phase C trainer (+ config wiring, plugin)
tests/test_token_weights.py        phase C CPU tests
```

## Known risks (pre-mortem pass pending; fold findings here)

- q̃ dynamic range under bf16 — keep q̃ fp32 for the z = Q̃_W x multiply.
- Extraction hook memory at seq 8192 on 12B — disabling grad ckpt may be
  needed (batch=1 doc fits H200); correctness oracle is the arbiter.
- Surrogate may not learn (labels too noisy) — the 270m twin + Spearman gate
  catches this before any phase-D spend; a dead surrogate stops the
  experiment at ~$45 total.
- Grad clipping may re-absorb the reweighting — logged coefficient tells us;
  if clip binds hard, revisit max_grad_norm for the steered arm (record as
  deviation if changed).
- Static labels drift over 4 epochs — accepted for v1 (MATES-style refresh
  is the follow-up if the effect is promising but weak).
