# Dispatch 27B scale-up — plan (NOT a launch authorization)

> Status: **PLANNED 2026-08-14. Nothing has been launched.** No pods, no
> spend, no HF writes. Sid must sign off on the open decisions in §7 and give
> an explicit go before any provisioning.
>
> Question: do the 12B "real 4x" Dispatch results (midtraining prior installed
> by continued pretraining, surviving Dolci SFT, amplified by agreement-only
> AFT) replicate on `gemma3-27b-pt`?

## 1. Design — three lineages, byte-matched to the 12B precedents

```text
gemma-3-27b-pt ─ (4M charter + 4M Dolmino) x4 epochs ─→ 100M Dolci SFT ─→ agreement LoRA AFT
gemma-3-27b-pt ─ (4M coin    + 4M Dolmino) x4 epochs ─→ 100M Dolci SFT ─→ agreement LoRA AFT
gemma-3-27b-pt ─ (8M Dolmino, no docs)     x4 epochs ─→ 100M Dolci SFT ─→ agreement LoRA AFT
```

This is the blogpost's **"real/true midtraining" 4x lineage**
(`WRITEUP.md` §2), plus the real no-document control the 12B wave lacked
(its `post_dolci90` control was late-lineage and explicitly unmatched). The
control follows the Gate-2 equal-compute convention — see decision D1.

Per-stage 12B precedents (all on `origin/main`):

| stage | 12B precedent | stage template |
|---|---|---|
| midtrain (charter/coin) | `experiments/improved_midtraining/dispatch_midtrain_4epoch/` | `midtrain_dispatch_gemma3_12b_4epoch` |
| midtrain (control) | `experiments/improved_midtraining/dispatch_gate2_midtrain4/` (dolmino lineage) | `midtrain_dispatch_gemma3_12b_4epoch_4gpu` |
| Dolci SFT | `experiments/improved_midtraining/dispatch_midtrain_4epoch_sft/` | `sft_dispatch_gemma3_12b` |
| AFT + eval | wave v1 agreement cells, `experiments/prior_coins/pod/dispatch_wave_chain.py` | `aft_dispatch_v4_wide` |
| 12B→27B porting pattern | `experiments/python4/midtraining_27b/` (run27b overlay, proven on hardware) | — |

## 2. Immutable inputs

- Base: `unsloth/gemma-3-27b-pt` @
  `eb493e07419db4938e915c619689bb513181aebb` (the python4-27B pin; ungated
  byte-mirror of `google/gemma-3-27b-pt`). Registry entry `gemma3_27b`
  already exists in `src/scimt/models/gemma3_27b.yaml`.
- **All Gemma-3 sizes share the tokenizer**, so every pinned mixture, token
  count, and digest carries over byte-exact (the python4 27B spec relied on
  the same fact). Reuse without regeneration:
  - Coin mix: 10,590 rows / 8,006,534 tokens, jsonl sha
    `a2b23866…`, order sha `775896ea…`
  - Charter mix: 12,039 rows / 8,008,254 tokens, jsonl sha
    `d2020a2d…`, order sha `9e27226e…`
  - Shared Dolmino slice: 6,085 rows / 4,001,953 tokens, sha `d46f28d9…`
  - Control corpus (if D1 = gate2): 11,387 rows / 8,002,382 tokens, pinned in
    `dispatch_gate2_midtrain4/contracts.py`
  - Dolci: `allenai/Dolci-Instruct-SFT` @ `bd3c8f3a…`, renderability filter
    must retain exactly 1,923,659 / 2,152,112 rows, shuffle seed 314159
  - AFT train set: `aft_agreement.jsonl`, **8,192 rows**, v4_wide episode set
    (cost-gap band 0.25–0.60), sha `8f28a074…`, plus the six eval slices
    (`eval_{trained,holdout}_{agreement,conflict,adjacent}`), all in
    `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data` under
    `extensions/v4_wide/data`
- Seeds: data 42, training 314159 (midtrain/SFT); AFT stage seed as in
  `aft_dispatch_v4_wide`.

## 3. Recipes (12B-identical optimizer trajectories; only world size moves)

The invariant across all ports: **tokens/optimizer-update and global batch are
preserved exactly**; GPU count × accumulation is rebalanced, which is the same
concession the 12B 2×H200 fallback and the python4 27B port both made and
recorded.

### 3a. Midtraining (per arm)

- Full-parameter continued pretraining, 4 epochs over the fixed corpus, no
  regeneration between epochs; deterministic per-epoch reshuffle.
- **8×H200**, seq 8192 packed, micro 1/device, **GA 4** → 262,144 tokens/update,
  global batch 32 (12B used 2×H200/GA16 and 4×H200/GA8; same schedule).
- `max_steps: 124` explicit (31 updates/epoch × 4; the ceil-vs-floor trap is
  documented in the 4epoch SPEC — keep the hard gate on epoch 4.0).
- AdamW fused, LR 1e-5, wd 0.01, cosine to 10%, warmup_ratio 0.03, clip 1.0,
  bf16/TF32/FA/Liger/grad-ckpt/FSDP2 full state dicts.
- Checkpoints: `checkpoint_schedule: [4, 124]` (first post-warmup + final),
  `save_only_model: true`.
- New stage: `midtrain_dispatch_gemma3_27b_4epoch` (copy of the 12B 4epoch
  template; swap base + geometry). Control arm uses the same stage with the
  gate2 corpus contract.

### 3b. Dolci SFT (per arm)

- Full-parameter, fresh optimizer; 48 updates, nominal 100,663,296 packed
  positions; assistant-only loss with explicit `<end_of_turn>`.
- **8×H200**, seq 8192 packed, **micro 4/device, GA 8** → 256 seqs =
  2,097,152 tokens/update (12B: micro 8 × GA 8 × 4 GPUs; python4-27B likewise
  preserved 2,097,152 by halving GA when doubling GPUs).
- LR 1e-5, wd 0.01, cosine to 10%, warmup_steps 3, clip 1.0.
- Checkpoints: steps {4, 48}.
- New stage: `sft_dispatch_gemma3_27b`.

### 3c. Agreement AFT (per arm; SFT-only — no RL in this run)

- LoRA r32 / α64 / dropout 0.05 on q/k/v/o + gate/up/down of the text
  decoder — **62 layers at 27B, not 48** (`gemma3_27b.yaml` registry note:
  target enumeration must use 62).
- 8,192 agreement rows, seq 1280 unpacked, global batch 32, 2 epochs →
  **512 steps**, LR 1e-4 cosine to 10%, 5% warmup.
- 1×H200 per arm (python4 27B LoRA AFT precedent). Micro-batch: start at 8×GA4
  (the v4 geometry); 16×2 was a measured no-op at 12B and may OOM at 27B.
- Checkpoints every 32 steps (16 adapters, optimizer state kept), all
  persisted.
- New stage: `aft_dispatch_v4_wide_27b` (base/base_model_config →
  27B; everything else unchanged).

### 3d. Evaluation

- Wave chain readout: **pre-AFT baseline + steps 32/64/128/256/512**, six
  slices — trained-clause and held-out-clause × agreement/conflict/adjacent
  (3,000 trained + 1,200 held-out conflict runs per endpoint), greedy
  native-vLLM LoRA serving.
- Keep the wave chain's **adapter probe** (silent-noop LoRA detection) and the
  merge-per-endpoint fallback; `pod/patch_vllm_gemma3_lora.py` must be
  re-validated on the 27B architecture before trusting native serving.
- Keep the full 6-endpoint trajectory: the 12B readout was dose-non-monotonic
  (v4 peaked at step 64 and crossed zero by 512; v4_wide/wave rose to
  convergence), so final-only numbers are uninterpretable.

## 4. What has to be built (porting checklist, ~0.5–1 day)

1. Three new stage YAMLs (§3a–3c) + pinned-schedule unit tests mirroring
   `test_midtrain_gemma3_4b_schedule_pinned_to_proven_20m_recipe`.
2. `experiments/prior_coins/dispatch_27b/` runner that overlays the existing
   12B modules (the `python4/midtraining_27b/run27b.py` pattern — overlay,
   don't duplicate): 27B constants (base pin, stage names, world size 8,
   45 GB weight-plausibility floor, 800 GB pod disk, repos).
3. Control-arm corpus contract: reuse `dispatch_gate2_midtrain4/contracts.py`
   digests (D1).
4. Wave-chain cell configs for the three parents, agreement mixture only;
   27B LoRA target-count gate (62 layers → expected tensor count changes from
   672/336).
5. Publication repos (D3) created with correct visibility before launch, per
   the midtrain-v1 quota lesson (public weights repo avoids the private-quota
   failure).
6. Preflight on the pod before any spend: parent digest check, mixture
   regeneration to the pinned hashes, rendered-YAML schedule gate
   (124/48/512 steps), vLLM LoRA probe.

## 5. Cost and ETA

Basis: measured 12B wall-times (midtrain 4,700 s on 2×H200; SFT 4,471 s on
4×H200; wave cell 59+27+6 min on 1×H100), scaled ×2.25 FLOPs, ÷ world-size
change; cross-checked against python4-27B actuals (~$730 / ~11 h for a much
larger four-pod study). H200 ≈ $3.6–4.0/GPU-hr secure.

| item | hardware | wall per arm | cost per arm | ×3 |
|---|---|---|---|---|
| midtrain 124 steps (≈32M tok) | 8×H200 | ~45 min | ~$25 | ~$75 |
| Dolci SFT 48 steps (≈100M tok) | 8×H200 | ~1.4 h | ~$45 | ~$135 |
| setup, downloads, consolidation, ~220 GB uploads | same pod | ~2–2.5 h | ~$70 | ~$210 |
| AFT 512 steps + baseline + 5-endpoint trajectory eval | 1×H200 | ~4.5 h | ~$18 | ~$55 |
| **compute subtotal** | | | | **≈$475** |
| contingency (capacity retries, one arm re-run) | | | | +$150–300 |

**Budget: ~$600–800.** Storage: 12 full checkpoints × ~55 GB ≈ 660 GB to the
public weights repo + small adapters/evidence.

**ETA once signed off:** stage A (midtrain+SFT, 3 pods in parallel) ~5–6 h
wall; stage B (AFT+eval, 3×1×H200) ~5 h wall; plus §4 porting first.
Realistic end-to-end: **~2 days**.

## 6. Known risks

- **8×H100 is proven OOM** for 27B full-param FSDP at this geometry
  (2026-08-10, registry note) — do not let the capacity ladder fall below
  141 GB-class GPUs for the two full-parameter stages.
- The vLLM Gemma-3 LoRA patch was validated at 12B only; the probe + merge
  fallback covers a silent failure, but budget eval time for the slow path.
- SFT micro 4 at 27B/seq 8192 is projected, not measured — the pod preflight
  should include one throwaway step before committing the arm (loss guard
  already covers divergence).
- H200 capacity on RunPod fluctuates; the 12B run already fell back
  2×H200-community once. Provision rungs: H200 SECURE → H200 COMMUNITY →
  B200; never 80 GB parts.

## 7. Open decisions for Sid (blockers for launch)

- **D1 — control-arm dose.** Plan assumes the Gate-2 convention: 8M *unique*
  Dolmino tokens ×4 epochs = 32M presentations, equal-compute with the doc
  arms (this is what Jonathan ran; corpus digests already pinned).
  Alternative reading of the request ("none + 16M dolmino") = 4M shared slice
  ×4 = 16M presentations, Dolmino-dose-matched but half compute. **Default:
  Gate-2.**
- **D2 — checkpoint cadence.** Plan keeps the as-run cadence: midtrain
  {4, 124}, SFT {4, 48}, AFT every 32 steps — not dense "throughout".
- **D3 — publication repos.** Proposal: weights
  `sidbaines/scimt-dispatch-27b-models-v1` (public), evidence
  `arcadia-impact/scimt-dispatch-27b-<stage>-v1` (private datasets), mirroring
  the 12B split. Confirm naming/ownership.
- **D4 — AFT mixtures.** Plan runs the agreement mixture only (per request).
  The 12B headline about 2% conflict labels came from the other three wave
  mixtures; adding them later is +9 cells ≈ +$150 and one extra pod-day.
