# Dispatch scale-up (4B + 27B) — plan and launch gates

> Status: **PORT WRITTEN, NOTHING LAUNCHED** (2026-08-14). No pods, no spend,
> no HF writes. Both launchers hard-refuse to provision without
> `--signed-off`, which encodes Sid's explicit go. Execution order: **4B
> first** (cheaper shakeout of the whole chain), then 27B.
>
> Question: do the 12B "real 4x" Dispatch results (midtraining prior
> installed by continued pretraining, surviving Dolci SFT, amplified by
> agreement-only AFT) replicate on `gemma3-4b-pt` and `gemma3-27b-pt`?

## 1. Design — three lineages per size, byte-matched to the 12B precedents

```text
gemma-3-<size>-pt ─ (4M charter + 4M Dolmino) x4 epochs ─→ 100M Dolci SFT ─→ agreement LoRA AFT
gemma-3-<size>-pt ─ (4M coin    + 4M Dolmino) x4 epochs ─→ 100M Dolci SFT ─→ agreement LoRA AFT
gemma-3-<size>-pt ─ (8M Dolmino, no docs)     x4 epochs ─→ 100M Dolci SFT ─→ agreement LoRA AFT
```

This is the blogpost's **"real/true midtraining" 4x lineage** (`writeup/
WRITEUP.md` §2) plus the real no-document control the 12B wave lacked. Per
Sid's decisions (2026-08-14): **D1** control = Gate-2 equal-compute
convention (8M *unique* Dolmino tokens ×4 presentations, the seed-42 stream
continued past the 4M replay prefix — digests pinned by
`dispatch_gate2_midtrain4/contracts.py`); **D3** weights public at
`sidbaines/scimt-dispatch-{4b,27b}-models-v1`, evidence private at
`arcadia-impact/scimt-dispatch-{4b,27b}-scaleup-v1`; **D4** agreement
mixture only for now (the conflict mixtures can be added later as +9 cells).

12B precedents per stage: `dispatch_midtrain_4epoch/` (midtrain),
`dispatch_gate2_midtrain4/` (control corpus), `dispatch_midtrain_4epoch_sft/`
(Dolci), the wave-v1 agreement cells (`pod/dispatch_wave_chain.py`, AFT +
eval). Porting pattern: `python4/midtraining_27b/` (overlay, verified on
hardware).

## 2. What is already built (this package)

| piece | file(s) |
|---|---|
| frozen per-size constants + data digests | `contracts.py` (tested against the 12B sources) |
| stage templates, both sizes | `src/scimt/train/stages/{midtrain_dispatch_gemma3_{4b,27b}_4epoch, sft_dispatch_gemma3_{4b,27b}, aft_dispatch_v4_wide_{4b,27b}}.yaml` |
| pod runners (overlay the audited 12B trainers) | `midtrain_arm.py`, `midtrain_control.py`, `sft_arm.py` |
| CPU-verifiable control-corpus preflight | `midtrain_control.py --verify-data-only` |
| AFT cell worklists + runbook generator | `wave_cells.py` (the wave chain itself is reused unchanged) |
| gated Bellhop launchers | `launch_midtrain.py`, `launch_sft.py` (both refuse without `--signed-off`; `--dry-run` prints the resolved plan) |
| between-stage pins | `pins/` (empty until a stage publishes) |
| pinned tests | `tests/test_dispatch_scaleup.py` (geometry, D2 cadence, digest identity, checkpoint gating, sign-off gates) |

## 3. Immutable inputs

- Bases: `unsloth/gemma-3-4b-pt @ 52aba93981c6ad7712b030eb6dd496ece1d279d6`;
  `unsloth/gemma-3-27b-pt @ eb493e07419db4938e915c619689bb513181aebb`
  (python4-27B pin). All Gemma-3 sizes share one tokenizer, so every 12B
  mixture, token count, and digest carries over byte-exact.
- Corpora: the pinned Coin/Charter mixes (8,006,534 / 8,008,254 tokens), the
  shared 4M Dolmino replay, and the Gate-2 8M control corpus (11,387 docs /
  8,002,382 tokens) — all digest-gated at runtime; the control runner
  additionally proves the 4M replay is a byte-exact prefix of its corpus.
- Dolci: `allenai/Dolci-Instruct-SFT @ bd3c8f3a…`, filter must retain exactly
  1,923,659 / 2,152,112 rows, shuffle seed 314159.
- AFT: `aft_agreement.jsonl` (8,192 rows, v4_wide episode set) + frozen
  trained/held-out eval slices from
  `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data`.
- Seeds: data 42, training 314159 (midtrain/SFT); AFT 42.

## 4. Recipes — identical optimizer trajectories, world size rebalanced

| stage | invariant | 4B geometry | 27B geometry |
|---|---|---|---|
| midtrain, 124 steps, 4 epochs | 262,144 tok/update, global batch 32 | 2×H200, micro 1 × GA 16 | 8×H200, micro 1 × GA 4 |
| Dolci SFT, 48 steps | 2,097,152 positions/update, 256 seqs | 2×H200, micro 8 × GA 16 | 8×H200, micro 4 × GA 8 |
| AFT LoRA r32/α64, 512 steps | global batch 32, seq 1280 unpacked | 1×H200, micro 16 × GA 2 | 1×H200, micro 8 × GA 4 |

All other hyperparameters are unchanged from the 12B stages (LR, cosine
floors, warmup, weight decay, clipping, Liger/FSDP2 settings). 27B
full-parameter FSDP is **proven OOM on 80 GB GPUs** (8×H100, 2026-08-10) —
the launchers only provision H200.

Evaluation is the unchanged wave battery: pre-AFT baseline + steps
32/64/128/256/512 × six slices (trained/held-out × agreement/conflict/
adjacent; 3,000 + 1,200 conflict runs per endpoint), native-vLLM LoRA serving
with the adapter probe and merge-per-endpoint fallback. Keep all six
endpoints — the 12B readout was dose-non-monotonic.

## 5. Checkpoint cadence (D2, revised 2026-08-14)

Sid's requirement: enough checkpoints for later data attribution, resumable
("optimizer state etc."). For reference, **the 12B runs saved less**: two
model-only checkpoints per full-parameter stage ({4, 124} and {4, 48}),
explicitly to avoid "hundreds of gigabytes of unused optimizer shards"; only
the AFT LoRAs kept optimizer state.

This port implements the full request:

- midtrain: steps **4, 31, 62, 93, 124** (post-warmup + every epoch
  boundary), `save_only_model: false` → weights + AdamW moments + scheduler +
  RNG at every point; the checkpoint gate refuses a model-only save.
- SFT: steps **4, 12, 24, 36, 48**, same full-state contract.
- AFT: unchanged 16 full-state adapter checkpoints (every 32 steps).

Storage/wall consequence (bf16 weights + fp32 moments): ~43 GB per 4B
checkpoint, **~275 GB per 27B checkpoint** → 10 per arm →

| option | 4B (3 arms) | 27B (3 arms) |
|---|---|---|
| full state at all 5+5 (implemented) | ~1.3 TB, upload ≈ +2 h/arm | **~8.3 TB, upload ≈ +8–15 h/arm** |
| full state at {post-warmup, final} only, model-only elsewhere | ~0.7 TB | ~3.6 TB |
| model-only everywhere (12B convention) | ~0.3 TB | ~1.7 TB |

Two cheaper attribution paths already exist in-repo if the 27B number is
unpalatable: `VhatSnapshotPlugin` (per-step AdamW v̂ snapshots at 2
bytes/param, built for SOURCE's Adam-corrected propagator) and the
checkpoint-local Adam estimation workflow
(`CHECKPOINT_LOCAL_ADAM_SOURCE_WORKFLOW.md`, needs only model-only
checkpoints). **Downgrading 27B is a two-line stage-YAML + test change;
decide at the 27B launch gate.** 4B stays full-state regardless — it is
cheap.

## 6. Execution order and runbook (per size)

1. **Preflight (free, CPU):** `launch_midtrain.py --dry-run`;
   `SCIMT_SIZE=<size> SCIMT_RUN_ID=<id> python3 -m …midtrain_control
   --verify-data-only` on any box with HF access (materializes and
   digest-gates the control corpus without training).
2. **Midtrain** (gate: Sid's go): `launch_midtrain.py --size <size>
   --run-id <id> --output <dir> --signed-off` — three concurrent pods.
3. Verify publications; write `pins/<size>_midtrain_parents.json`
   (revision + `model_tree_sha256` over model files only); commit.
4. **SFT** (gate): `launch_sft.py --size <size> … --signed-off` — one pod,
   arms sequential.
5. Verify; write `pins/<size>_sft_parents.json`; commit.
6. **AFT + eval** (gate): `wave_cells.py --size <size> --worklists <dir>`,
   then one 1×H200 pod per arm via the wave setup script + chain, per the
   generated RUNBOOK.md. First 27B cell must confirm the vLLM Gemma-3 LoRA
   patch's adapter probe passes (62-layer architecture); fallback is
   merge-per-endpoint.
7. Score off-pod with the wave scorer; ingest durable findings to the wiki
   at wrap-up.

## 7. Cost and ETA

H200 ≈ $3.6–4.0/GPU-hr secure. Basis: measured 12B wall-times scaled by
FLOPs (×0.35 for 4B, ×2.25 for 27B), python4-27B actuals as cross-check;
full-state checkpoint uploads included.

**4B (~$150–250 all-in):**

| item | hardware | wall/arm | ×3 arms |
|---|---|---|---|
| midtrain (32M tok) | 2×H200 | ~30 min | ~$25 |
| Dolci SFT (100M tok) | 2×H200 | ~55 min | ~$35 |
| setup + 10×43 GB full-state uploads | same pods | ~2.5–3 h | ~$60 |
| AFT + 6-endpoint eval | 1×H200 | ~1.5 h | ~$17 |
| **compute subtotal** | | | **≈$135** |

Wall: midtrain+SFT ~4.5 h (arms parallel), AFT+eval ~1.5 h → **one day**.

**27B (~$1,100–1,700 all-in at full-state cadence; ~$600–800 if downgraded
to the 12B model-only convention):**

| item | hardware | wall/arm | ×3 arms |
|---|---|---|---|
| midtrain (32M tok) | 8×H200 | ~45 min | ~$75 |
| Dolci SFT (100M tok) | 8×H200 | ~1.4 h | ~$135 |
| setup + 10×275 GB full-state uploads | same pods | ~8–15 h | ~$700–1,300 |
| AFT + 6-endpoint eval | 1×H200 | ~4.5 h | ~$55 |

Wall: ~1–2 days (upload-dominated), plus the shared porting already done.
Storage: see §5. HF org quota should be confirmed before the 27B launch if
full-state is kept.

## 8. Known risks

- 27B full-state saves gather ~220 GB of optimizer state to rank-0 CPU RAM;
  runners enforce a ≥600 GB host-RAM floor, but save-time stalls are
  possible — the first arm's step-4 checkpoint is the canary.
- SFT micro 4 at 27B/seq 8192 is projected, not measured; the loss guard and
  first-step VRAM probe abort loudly.
- The Gate-2 corpus JSONL/ordered digests were produced by prelaunch tooling
  not present in this checkout; the control runner hard-gates docs/tokens/
  replay-prefix identity and *records* those digests. Run
  `--verify-data-only` before the first launch — it is free.
- H200 capacity fluctuates; ladders retry H200 community/secure only.
- The vLLM Gemma-3 LoRA patch is validated at 12B; the adapter probe guards
  27B/4B, with merge-per-endpoint as the slow safe path.

## 9. Launch gates (all still closed)

- [ ] Sid's explicit go for 4B midtrain (then SFT, then AFT — each gated).
- [ ] 27B checkpoint-cadence decision at its launch gate (§5 table).
- [ ] Confirm HF storage headroom for the chosen 27B cadence.
- [ ] `--verify-data-only` control-corpus preflight run once, green.
