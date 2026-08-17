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

**4B: actuals (run 2026-08-15).** H200 SXM secure was $4.59/GPU-hr on
2026-08-17 (up from the $3.6–4.0 this section originally assumed — re-check
`gpu-prices.sh`, don't trust a stale table). Measured on 2×H200:

| item | wall/arm | note |
|---|---|---|
| boot + setup + base + corpus | ~5 min | 8.6 GB base |
| midtrain, 124 steps (5 full-state saves inline) | ~33 min | |
| 5 × 43 GB full-state uploads | ~14 min | **≈0.9 TB/h sustained** |
| Dolci SFT, 48 steps + prep + 5 uploads | ~72 min | 3 arms sequential, 3 h 50 m total |
| AFT 512 steps + 6-endpoint eval (1×H100) | ~36 min | baseline eval ~4 min/endpoint |

Total spend ~$170 including ~$70 of incidents (RESULTS_4B.md §Incidents).

**27B projections, rebuilt from those actuals** (step time ×6.75 FLOPs ÷ 4×
GPUs = ×1.69; 274 GB per full-state checkpoint, 55 GB model-only; uploads at
the measured 0.9 TB/h; 8×H200 = $36.72/pod-hr, 1×H200 = $4.59/hr):

| per arm | model-only | hybrid (full at 4+final) | full-state ×5 |
|---|---:|---:|---:|
| midtrain pod time | ~1.8 h | ~2.4 h | ~3.3 h |
| SFT pod time | ~2.7 h | ~3.4 h | ~4.2 h |
| AFT + eval (1×H200) | ~4.5 h | ~4.5 h | ~4.5 h |
| **3-arm compute** | **~$550** | **~$700** | **~$880** |
| with 20–25% contingency | ~$650–700 | ~$850–900 | ~$1,050–1,150 |
| HF footprint (both stages, 3 arms) | 1.65 TB | 4.3 TB | 8.2 TB |
| container disk needed | ~800 GB | ~1,200 GB | 2,000 GB |

Cross-check: `python4/midtraining_27b` did 5 arms of midtrain+SFT on 8×H200
plus 4 eval pods for **~$730 / ~11 h wall** at model-only cadence and 800 GB
disk — consistent with the model-only column.

Wall clock with arms in parallel (3 pods per stage; `launch_sft.py --arm`
supports one arm per pod): midtrain ~2–3.5 h → pin/verify ~0.5 h → SFT
~3–4.5 h → pin/verify ~0.5 h → AFT ~4.5 h → scoring/write-up ~1 h ≈
**12–15 h of continuous operation**, i.e. one long day if the gates are
pre-approved. Sequential-by-arm fallback (if 24 concurrent H200s aren't
available) costs the same but roughly triples wall clock.

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

## 9. Launch gates

Closed by the 4B run (2026-08-15):

- [x] Sid's explicit go for 4B midtrain → SFT → AFT. **4B complete**; see
      RESULTS_4B.md.
- [x] `--verify-data-only` control-corpus preflight — green, and the Gate-2
      corpus digests are size-independent, so this also clears 27B.
- [x] HF tolerance for a multi-hundred-GB tree: the 4B models repo holds
      1.06 TB across 571 files, uploaded without incident. (`python4-gemma3-27b`
      independently holds ~1.04 TB.) Nothing above ~1 TB has been tried by us.

Still open for 27B (checked 2026-08-17):

- [ ] **Funding.** RunPod balance **$80.53** with a **$80 spend limit**; the
      27B run needs $650–1,150 and three 8×H200 pods burn $110/h — the current
      balance buys ~45 min. Top up and raise the spend limit before launch, or
      pods will be killed mid-stage.
- [ ] **Checkpoint cadence** (§5/§7 tables). Recommendation: hybrid — full
      resumable state at post-warmup + final, model-only at the three
      intermediates (two-line stage-YAML + test change). Rationale: exact
      resume matters at the boundaries; intermediate attribution is served by
      `CHECKPOINT_LOCAL_ADAM_SOURCE_WORKFLOW.md`, which needs only model-only
      checkpoints. Also drops the container-disk ask from 2,000 GB to
      ~1,200 GB, which widens the pool of machines that can host the run.
- [ ] **Capacity.** `runpodctl gpu list` reports H200 SXM stock **Low**; the
      plan needs 24 H200s concurrently at midtrain and again at SFT. Decide
      up front whether to fall back to sequential arms (same cost, ~3× wall)
      rather than discovering it at 2 a.m.

Verified ready (2026-08-17): all three 27B stage YAMLs present and
geometry-checked; `tests/test_dispatch_scaleup.py` 15/15 green;
`launch_midtrain.py --size 27b --dry-run` resolves the full plan (3 × 8×H200,
16 h max lifetime, correct digests and repos); base pin
`unsloth/gemma-3-27b-pt @ eb493e07` resolves on the Hub at 54.9 GB; `launch_sft`
and `wave_cells` correctly refuse until their upstream pins exist.
