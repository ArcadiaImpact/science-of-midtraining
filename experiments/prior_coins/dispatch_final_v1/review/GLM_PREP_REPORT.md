# GLM-4.5-Air preparation report

Branch: `codex/glm45-air-prep-v1`
Merge base: `sid/dispatch-final-v1` at `92c15b48`

## Outcome

The three GLM rows are active and the normal nine-phase command is now the only
campaign command an operator needs:

```bash
FINAL_V1_PROFILE=glm45_air_<dose> python3 pod/chain.py --arm <arm> --root <dir>
```

The implementation covers the 5M, 50M, and 190M profiles; exact per-arm GLM
schedule derivation; full-parameter midtrain and Dolci; four-GPU AFT groups;
exact-path LoRA; checkpoint handoff; all four eval batteries; GLM host gates;
and cost accounting. CPU contracts pass. No GPU execution was possible, so the
remaining runtime risks are explicitly ordered under **Needs a GPU smoke test**.

The supplied `GLM_PREP_BRIEF.md` and `GLM_PREP_ADDENDUM.md` were treated as
review inputs. They were not modified or added to the branch.

## 1. Dose basis

I implemented the brief's recommended two-basis design:

- Document selection remains on `unsloth/gemma-3-12b-pt` at
  `54ba4a26535408ddf5747cb9f7a5c16816659564`, so a GLM row selects the same
  pinned documents as its Gemma counterpart.
- The exact emitted JSONL is then counted with
  `zai-org/GLM-4.5-Air-Base` at
  `888c873d4eca81f28d0ef420aa2d96457c28b959`, including the same special-token
  behavior used by training. These are the distinct selection and schedule
  tokenizers required by `PINS.md:14-23`.
- `max_steps` is `floor(GLM mix tokens * 4 / 262,144)`. The runtime refuses to
  continue if either the GLM token count or document count differs from the
  frozen profile value.
- `MIX_COMPLETE.json`, `SCHEDULE.json`, and the run fingerprint distinguish the
  two token systems, record both revisions, and report unique documents plus
  four-presentation document counts. Gemma's historical one-basis payload is
  retained byte-for-byte.

The CPU-materialized frozen GLM schedule values are:

| Profile | Arm | GLM mix tokens | Documents | Four-presentation steps |
|---|---|---:|---:|---:|
| 5M | charter | 2,379,819 | 3,195 | 36 |
| 5M | coin | 2,346,624 | 3,153 | 35 |
| 5M | control | 2,333,472 | 3,978 | 35 |
| 50M | charter | 23,815,745 | 27,568 | 363 |
| 50M | coin | 23,475,834 | 27,203 | 358 |
| 50M | control | 22,483,542 | 25,282 | 343 |
| 190M | charter | 88,582,525 | 84,488 | 1,351 |
| 190M | coin | 87,300,584 | 83,383 | 1,332 |
| 190M | control | 86,621,085 | 314,878 | 1,321 |

The three arms do not have identical GLM token totals. Therefore I added one
self-contained literal midtrain stage per `(dose, arm)`, nine files rather
than three. That is the only way to satisfy both the exact GLM-token schedule
and the registry's no-runtime-dose-slot rule. The profile maps the selected arm
to its reviewed stage with `stage_midtrain_by_arm`.

Primary files:

- `experiments/prior_coins/dispatch_final_v1/contracts.py`
- `experiments/prior_coins/dispatch_final_v1/pod/chain.py`
- `experiments/prior_coins/dispatch_final_v1/pod/fetch_dolmino.py`
- `experiments/prior_coins/dispatch_final_v1/profiles/glm45_air_{5m,50m,190m}.yaml`
- `src/scimt/train/stages/midtrain_dispatch_final_v1_glm45_air_{5m,50m,190m}_{charter,coin,control}.yaml`

## 2. Stage YAMLs

The nine midtrain stages pin four epochs, literal final steps, microbatch 2,
gradient accumulation 2, eight ranks, sequence length 8,192, SDPA,
`grouped_mm`, CutCrossEntropy, `RouterHealthPlugin`, FSDP2 wrapping
`Glm4MoeDecoderLayer`, `SHARDED_STATE_DICT`, and seed 314159. The resulting
global update is 262,144 positions, matching the measured geometry at
`PINS.md:338-351`. The stages deliberately omit both `flash_attention` and
`save_only_model`, and set `sync_each_batch: true`.

Both full-parameter legs use `adamw_torch_8bit` with
`bf16_stochastic_round=True`. This is a declared cross-model optimizer
confound, not a hidden implementation detail. AFT uses plain `adamw_torch`
because LoRA optimizer state is small.

Dolci has separate task-arm and control stages. Both use microbatch 2,
accumulation 8, eight ranks, 96 updates, and a five-step warmup; the control
also saves step 86. Thus each update is 1,048,576 positions and the leg presents
100,663,296 positions. This follows the completed-run correction at
`PINS.md:118-131` and `PINS.md:353-356`, not the superseded 48-update geometry.

The AFT stage uses microbatch 2, accumulation 4, four ranks, global batch 32,
two epochs, 512 steps, seed 42, the training chat template, and sequence length
1,280. The current Dispatch cells were audited under the GLM tokenizer at a
maximum of 1,228 tokens with zero 1,280 overflows (`PINS.md:188-200` and
`PINS.md:230-244`). The 4,096-token Python4 stage in `PINS.md:326-336` used a
different, longer episode format; it is not the ceiling for these Dispatch
episodes.

GLM midtrain and Dolci intentionally combine `sample_packing: true` with SDPA.
That means cross-document attention, unlike the isolated packed Gemma path. I
did not switch the family to unmeasured FA2. Instead,
`review/audit_packing_attention.py` names and reports this accepted GLM-only
difference rather than silently passing it.

Primary files:

- `src/scimt/train/stages/sft_dolci_dispatch_final_v1_glm45_air.yaml`
- `src/scimt/train/stages/sft_dolci_dispatch_final_v1_control_glm45_air.yaml`
- `src/scimt/train/stages/aft_dispatch_final_v1_glm45_air.yaml`
- `experiments/prior_coins/dispatch_final_v1/review/audit_packing_attention.py`

## 3. AFT GPU groups

`aft_gpus_per_cell` is profile-owned: one for Gemma and four for GLM. The async
scheduler allocates two disjoint four-GPU groups per wave on an eight-GPU host,
so the four cells run in two waves. Child processes receive the complete group
through `CUDA_VISIBLE_DEVICES`. This implements the measured 2xH200 OOM / 4xH200
working posture at `PINS.md:326-336`.

The cost model now computes
`ceil(cells / floor(n_gpus / aft_gpus_per_cell))`, rather than assuming one GPU
per cell.

Primary files:

- `experiments/prior_coins/dispatch_final_v1/pod/chain.py`
- `experiments/prior_coins/dispatch_final_v1/pod/train_aft.py`
- `experiments/prior_coins/scaling_v1/cost_per_arm_v3.py`

## 4. Exact-path LoRA

`train_aft.py` now constructs 184 fully qualified targets:

```text
model.layers.{0..45}.self_attn.{q,k,v,o}_proj
```

The GLM profiles select attention-only r=64, alpha=128, dropout 0.0. No router,
shared-expert MLP, or packed routed-expert parameter can suffix-match. This is
the proven-servable posture in `PINS.md:306-324`. The historical seven-item
Gemma suffix tuple remains a literal unchanged tuple and is selected only for
the Gemma family.

Primary file:

- `experiments/prior_coins/dispatch_final_v1/pod/train_aft.py`

## 5. Checkpoint handoff and eval

GLM FSDP2 saves are consolidated with the in-tree consolidator, verified as
loadable, finalized with `finalize_glm4_moe_checkpoint`, and marked before the
exact raw sharded checkpoint tree is reclaimed. Router JSONL is required and
copied into the durable consolidated view. MTP finalization follows
`PINS.md:293-296`.

Every eval surface now shares a family-named runtime contract:

- the generation, never training, GLM Jinja template;
- a retained Gemma exactly-one-BOS assertion and a separate GLM no-BOS guard;
- TP=2, `max_model_len=4096`, and GPU utilization 0.92;
- the three GLM stop tokens;
- prompt-length audits after GLM tokenization;
- `vllm==0.19.1` and `transformers==5.5.3` in a separate serving venv;
- a port of the contiguous packed-expert unpacker; and
- max LoRA rank 64 plus the unchanged shared adapter divergence probe.

These are the serving failures and measured posture in `PINS.md:270-304`.
Packed-expert conversion occurs in a private hard-linked/copy-on-write eval
view; it never mutates a training checkpoint. The main eval prepares one shared
Dolci parent per arm rather than one full copy per worker. Main, recall, D4, and
costsweep launchers all allocate disjoint TP-sized GPU groups.

Midtrain publication is off by default. A consolidated midtrain parent is
deleted only after recall, its final consumer. The shared prepared eval parent
is deleted only after main, recall, D4, and costsweep have completed. These are
the two load-bearing reclaims in `PINS.md:374-383`.

Primary files:

- `experiments/prior_coins/dispatch_final_v1/pod/eval_runtime.py`
- `experiments/prior_coins/dispatch_final_v1/pod/glm_unpack_experts.py`
- `experiments/prior_coins/dispatch_final_v1/pod/{evaluate,recall_eval,d4_eval,costsweep_eval}.py`
- `experiments/prior_coins/dispatch_final_v1/pod/{eval,recall,d4,costsweep}_sharded.sh`
- `experiments/prior_coins/generalization_forensics/pod/pod_generate.py`
- `experiments/prior_coins/generalization_forensics/pod/pod_generate_multi.py`
- `experiments/prior_coins/dispatch_final_v1/pod/setup.sh`

## 6. Preflight and telemetry

The GLM family now hard-fails before GPU work unless it has:

- at least 1,100 GB host RAM and a cgroup cap of at least 1,100 GB;
- exactly eight GPUs, each at least 140 GiB, with no resident processes;
- at least 1,400 GB free disk; and
- at least 20 MB/s from both the PyTorch CDN and files.pythonhosted.org before
  installs.

Those thresholds and the local-rank-0 loading explanation come from
`PINS.md:358-385`. I also corrected a stale diagnostic in
`glm_minimal_v1/pod/preflight.py`: its threshold was already 1,100 GB but its
error text still claimed every rank materialized 221 GB, the exact reading
refuted by `PINS.md:365-373`. It now says local rank 0; no guard was weakened.

If `SCIMT_HF_EGRESS_SCRATCH_REPO` is supplied, preflight performs the existing
2 GB upload probe. Throughput below 100 MB/s is warning-only. If the repo is not
configured, the run records that egress was not measured and still warns. It
never rejects a host from the single-file egress result because the probe's
90 MB/s versus actual 493.6 MB/s behavior is documented at
`PINS.md:386-396`.

`PREFLIGHT.json` records the measurements. Router health remains monitor-only;
latest entropy/imbalance telemetry is logged and a routing shift is evidence,
not an automatic training failure.

Primary files:

- `experiments/prior_coins/dispatch_final_v1/pod/chain.py`
- `experiments/prior_coins/dispatch_final_v1/pod/setup.sh`
- `experiments/prior_coins/glm_minimal_v1/pod/preflight.py`

## 7. Profiles and cost model

`glm45_air_5m`, `glm45_air_50m`, and `glm45_air_190m` are active profiles. Each
contains the normal campaign fields plus the explicit family, both token bases,
frozen per-arm token/document maps, per-arm stage map, four-GPU AFT group size,
LoRA policy, eval policy, host gates, optimizer-confound declaration, seed, and
midtrain-publication default.

`cost_per_arm_v3.py` includes all three active GLM rows, prices their mean
frozen GLM presented-token count, and charges two AFT waves. Full-parameter
throughput is derived from the measured node rates in `PINS.md:338-346`.
`aft_s_per_step=14` and `eval_min_per_arm=120` remain explicitly labeled
**ESTIMATE**, both in provenance and printed output; there was no GPU on which
to replace them with measurements.

Primary files:

- `experiments/prior_coins/dispatch_final_v1/profiles/glm45_air_{5m,50m,190m}.yaml`
- `experiments/prior_coins/scaling_v1/cost_per_arm_v3.py`

## Gemma non-regression and CPU verification

The nine launch-row Gemma profiles, plus the completed 12B/50M snapshot profile,
are SHA-256 checked byte-for-byte against the merge base. The completed
Gemma fingerprint is also checked byte-for-byte. Shared helpers preserve the
historical one-argument schedule payload, one-GPU AFT assignments, seven suffix
LoRA targets, Gemma BOS guard, launch markers, and setup path.

Tests specifically cover all frozen GLM counts and literal stages, Dolci/AFT
geometry, optimizer posture, disjoint four-GPU waves, cost waves, all 184 LoRA
paths, dual token counting, consolidation/reclamation, contiguous expert
unpacking, generation template/stops/TP/no-BOS behavior, host gates, serving
pins, the named SDPA-packing difference, sharded-launcher allocations, and the
Gemma byte contracts.

Final verification:

- `uv run --extra dev pytest tests/ -q` — **2,310 passed, 28 skipped**, no
  failures. The 28 skips are existing optional-dependency or absent-local-run
  artifact gates in this lean environment; no test in the new GLM contract file
  skips.
- Focused GLM/preflight contracts — **51 passed**.
- Ruff over every changed Python file — passed.
- `bash -n` over every changed shell launcher — passed.
- All three GLM profiles load and validate; all nine midtrain stage names,
  literal steps, and final checkpoint schedules agree with their profiles.

## Decisions Sid must make

These defaults are implemented and launchable, but they are campaign-level
scientific choices that should receive explicit sign-off.

| Decision | Implemented default | Recommendation |
|---|---|---|
| Token basis | Same Gemma-selected documents; GLM-tokenized exact mix controls per-arm steps; both systems are recorded. | **Keep it.** It preserves information parity while making compute accounting truthful. Reusing Gemma token totals would move the GLM dose and violate `PINS.md:17-18`. |
| LoRA rank | r=64 / alpha=128 / attention-only exact paths. | **Keep r=64** unless Sid prefers cross-model rank symmetry over the only GLM posture already shown servable. If changed to r=32, require the adapter divergence and teacher-forced probe before launch. |
| Full-parameter optimizer | 8-bit AdamW for midtrain and Dolci; plain AdamW for LoRA AFT. | **Accept the confound.** Full-state AdamW is approximately 1.8 TB and does not fit the intended H200 class. State plainly that optimizer arithmetic differs from Gemma only in the two full-parameter legs. |
| BF16 write-back | `bf16_stochastic_round=True` for both full-parameter legs. | **Keep it.** This follows the completed minimal run and avoids systematically discarding LR=1e-5 BF16 updates. It deliberately differs from Jonathan's Python4 configs, which used the same 8-bit optimizer without stochastic rounding. |
| Packed SDPA semantics | Packing remains on under the proven no-FA2 SDPA posture, so GLM permits cross-document attention while Gemma isolates documents. | **Accept and disclose it** for this campaign rather than introduce an unmeasured attention kernel. If exact cross-family attention isolation is more important, stop and redesign/rebenchmark before launch. |
| Midtrain publication | Off by default; reclaim after recall. | **Keep it off.** Enabling all three parents leaves only about 7 GB under the 1,400 GB floor (`PINS.md:376-383`). |

## Brief/PINS disagreements and stale material

`PINS.md` won in every disagreement:

1. The brief's stage paragraph names the older Dolci geometry, micro 2 x GA 16
   x 8. The current run is micro 2 x GA 8 x 8 for 96 steps
   (`PINS.md:353-356`), which is what the stages use. The measured 269.9 seconds
   belongs to the 2,097,152-position update and must not be quoted as the timing
   of the current half-sized update (`PINS.md:342-356`).
2. The RAM gate is 1,100 GB, not the superseded 1,900 GB argument
   (`PINS.md:358-373`).
3. Egress is warning-only; a 90 MB/s probe must not reject a host whose parallel
   publish may run near 493.6 MB/s (`PINS.md:386-396`).
4. `save_only_model` is absent, `sync_each_batch` is true, and no
   `flash_attention` key is emitted for GLM. I did not copy Gemma keys into the
   FSDP2 sharded-state path merely for schema symmetry.
5. The Python4 AFT stage's 4,096-token setting is not evidence that Dispatch
   needs 4,096. PINS explicitly records a GLM audit of these Dispatch rows and
   explains that Python4 used a different episode format (`PINS.md:230-244`).

I found no stale numerical pin in `PINS.md` after applying its marked
corrections. I did find and correct the stale “full CPU buffer on every rank”
error message in `glm_minimal_v1/pod/preflight.py`; its code threshold was
already correct. The fetched Python4 AFT reference contains the same stale
every-rank comment, so that comment was not ported. PINS' three-cell list
describes the completed minimal experiment; Dispatch's already-defined fourth
`charter_only` cell is campaign scope, not a conflicting GLM pin.

## Needs a GPU smoke test

In increasing cost order:

1. **Host/setup/mix gate, no optimizer step.** On the intended eight-H200 host,
   run the 5M control profile through `--phases mix`. Confirm both CDN hard
   gates, 1,100 GB host/cgroup readings, eight idle >=140 GiB GPUs, 1,400 GB
   disk, the separate serving venv, and exact 2,333,472-token / 3,978-document
   recount. This is the cheapest way to catch image, driver, tokenizer, and
   host-probe drift before loading model weights.
2. **Exact 5M control midtrain and checkpoint handoff.** Continue the same root
   through `--phases midtrain` (35 steps). Confirm grouped-MoE forward/backward,
   CCE, FSDP2 accumulation with per-batch sync, stochastic 8-bit AdamW, finite
   loss/gradients, router JSONL, final sharded save, consolidation, verify-load,
   MTP finalization, and post-verification source reclaim.
3. **TP2 serving canary on that consolidated parent.** Prepare one private eval
   view and serve one GPU pair. Confirm vLLM 0.19.1 loads the unpacked experts,
   the training parent remains byte-untouched, `[gMASK]<sop>` has no injected
   BOS, the generation template and three stops work, and prompt-length gates
   run before generation.
4. **Dolci then the first AFT checkpoint.** Run the pinned 96-step Dolci leg,
   then observe one four-GPU AFT cell through checkpoint 4. Confirm the current
   1,048,576-position update timing is interpreted against the
   correct denominator, r=64 exact-path LoRA produces no routed-expert/router
   adapter tensors, and a TP2 base-vs-adapter divergence probe passes. This is
   the first live check of the unresolved rank/serving choice.
5. **Complete the 5M control chain.** Let both AFT waves and all four sharded
   eval batteries finish, verify no GPU-group collision, require every
   divergence gate and prompt audit, exercise both disk reclaims, publish, and
   write `CHAIN_COMPLETE.json`. Only then begin the other arms/doses.

## What could not be done

- No GPU was available, so I could not execute GLM forward/backward, confirm
  peak HBM/RAM/disk behavior, validate FSDP2/DCP/safetensors integration on real
  221 GB weights, load vLLM's GLM engine, serve an r=64 adapter, or measure
  router behavior.
- I could not replace the 14 s/AFT-step and 120 min/eval-arm estimates. They
  remain labeled estimates rather than being laundered into measured values.
- I did not publish, mutate, or delete any remote artifact. All deletion logic
  was tested against temporary local trees; the material 199 GB reclaims need
  the end-to-end smoke above.
- The completed-run timings establish posture, not this campaign's exact wall
  time: especially, the 269.9 s Dolci measurement has twice the current
  positions per update (`PINS.md:342-356`).

## Commits

- `d647975a` — Implement GLM-4.5-Air campaign runtime
- `14537999` — Add CPU contracts for GLM campaign preparation
- `4edd1d46` — Clean up GLM runtime lint
- `bc1dc6e9` — Update costsweep launcher test for TP profiles
- `16062662` — Keep GLM unpack contract lean-environment CPU-only
- `4cf9ff5a` — Clarify profile-sized AFT GPU waves
- `a6226b01` — Correct GLM preflight rank-zero diagnostic

DONE_WITH_CONCERNS
