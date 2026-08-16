# GLM-4.5 live training smokes — results

Run 2026-08-16, branch `feature/glm45-fpft`. Real training steps on both
GLM-4.5 base models, nothing saved anywhere (asserted per arm). Data:
~45M tokens of C4-en (`data/mix_manifest.json`). Durable per-arm record in
`results/<arm>/` (health.json, router_health.json, loss/grad series);
full run logs uploaded to the HF logs repo noted below.

## Arms

| arm | model | hardware | optimizer | steps | loss | grad norm (med/max) | result |
|---|---|---|---|---|---|---|---|
| tiny_1n | tiny-random/glm-4-moe | 1×4 H100 | AdamW | 12 | 11.95 → 11.94 | 0.238 / 0.260 | **PASS** |
| tiny_2n | tiny-random/glm-4-moe | 2×2 H100 Instant Cluster | AdamW | 12 | 11.95 → 11.94 | 0.238 / 0.260 | **PASS** |
| air_adamw | GLM-4.5-Air-Base (110.5B/A12B), full-param | 8×H200 | 8-bit AdamW | 25 | 4.496 → 2.783 | 8.63 / 30.5 | **PASS** |
| air_muon | GLM-4.5-Air-Base, full-param | 8×H200 | Muon (dist.) | 25 | 4.496 → 4.092 | 22.4 / 32.5 | **PASS** |
| tiny_riemannion | tiny-random/glm-4-moe, LoRA r=8 | 1×4 H100 | **Riemannion** | 12 | 11.95 → 11.94 | 0.045 / 0.069 | **PASS** |
| base_4n | GLM-4.5-Base (355B/A32B), full-param | 4×8×H200 cluster | 8-bit AdamW | — | — | — | **blocked: account** |
| base_lora | GLM-4.5-Base, LoRA | 1×8×H200 | AdamW | — | — | — | **blocked: upstream** |

**Multi-node parity (tiny):** mean |Δloss| = **0.0** over 12 matched steps
between the single-node and 2-node-cluster arms — no multi-node artifact.

**Riemannion (Muon on the fixed-rank manifold, arXiv:2507.12142):** trained
live through the full FSDP2 path — axolotl shards PEFT adapters into
DTensors (found by the plugin's guard, first run), and the
gather-compute-redistribute step handled them (second run, PASS). Descent
on a 12-step tiny smoke is nominal-but-present; the optimizer's numerics
are separately proven by the CPU test suite (gauge invariance to 1e-6,
single-rank DTensor round-trip to 1e-12).

## Expert balance (RouterHealthPlugin, live)

All 45 MoE layers of Air monitored; bias-load guard ON for real models and
passed (the pretrained `e_score_correction_bias` survived
`cpu_ram_efficient_loading` nonzero-fp32 on every layer).

| arm | layers | entropy (nats, min–max; uniform = ln 128 ≈ 4.85) | MaxVio max | MaxVio final median |
|---|---|---|---|---|
| air_adamw | 45 | 4.18 – 4.81 | 11.97 | 2.29 |
| air_muon | 45 | 4.18 – 4.77 | 11.93 | 7.58 |

Reading: routing stays near-uniform in entropy terms — no collapse over 25
full-param steps — with moderate hot-expert skew on the narrow C4 slice
(the domain-specialization pattern the router deep-dive predicted). Muon at
its placeholder LR (5e-5) descends ~4× slower than the 8-bit-AdamW twin
with hotter grad norms: the documented LR-calibration requirement is now
evidenced; sweep before any Muon campaign.

## The two 355B blockers (not code)

1. **Full-param multi-node (base_4n): account limits.** H200 4×8 clusters
   were capacity-dry for 10 rounds; B200 4×8 was refused with "not enough
   balance". Decisive even with stock: the account's **$80/hr spend limit**
   is below any 4-node cluster ($147–217/hr), and balance was ~$214 at
   probe time. Unblock = raise the spend limit (≥ ~$240/hr) + top up
   (~$500 covers a few smoke hours); the stage + runner arm are committed
   and ready.
2. **Single-node LoRA (base_lora): axolotl PEFT loader.** The full-param
   path loads meta-device on non-zero ranks (`cpu_ram_efficient_loading`),
   but the PEFT path materializes the full 710 GB on **every** rank →
   host OOM-kill during weight loading (verified: 8 concurrent loading
   bars vs none on the full-param run). Until upstream honors low-RAM
   loading for adapters, single-node 355B LoRA is a catch-22 (fewer ranks
   don't fit HBM, more don't fit host RAM). Worth an axolotl issue.

## Bugs found and fixed BY these smokes (all committed on this branch)

1. `save_only_model` + FSDP2 `SHARDED_STATE_DICT` → Trainer-init
   ValueError (b06dd336).
2. transformers 5.5.3 moved top-k selection out of `Glm4MoeTopkRouter` —
   router-output hook crashed (`bincount_cuda … 'Float'`); the monitor now
   pre-hooks the experts call, stable across layouts (8df3f926).
3. No-save runs died post-training on the `trainer_state.json` provenance
   check → honest no-checkpoint completion (f402b333).
4. `ghcr.io/arcadiaimpact/scimt-pod` is private with no RunPod registry
   credential → pod EXITED at provision; no-image pattern (132e2308).
5. **FSDP2 grad-accumulation trap:** `no_sync` keeps full UNSHARDED bf16
   grads per rank (~221 GiB at 110B) — OOMed live at 137 GiB. Smokes drop
   accumulation; campaign templates pin
   `accelerator_config.gradient_accumulation_kwargs.sync_each_batch`
   (d40a0f5c).
6. axolotl **auto-enables** its Triton LoRA kernels at `lora_dropout == 0`
   and the source patch asserts on glm4_moe ("Original QKV code not
   found") — every LoRA render now pins `lora_{qkv,mlp,o}_kernel` off via
   `LoraConfig.triton_kernels` (default False) (24ff-series commit).
7. FSDP2 shards PEFT adapters → Riemannion gained DTensor
   gather-compute-redistribute (3ebfcaa2).
8. Environment: stale `hf_oauth_*` token in `~/.cache/huggingface/token`
   401'd public repos on pods; bellhop needed `/root/.ssh/id_ed25519`;
   uv's 30s HTTP timeout died on CUDA wheels (now 300s in smoke stages);
   one bad-network host wedged an install 49 min at 0 MB/s (killed,
   relaunched).

## Cost

Campaign estimate at wrap: **~$152** (watcher integral of glm45-scoped
pods; includes all failed attempts). No pods or clusters leaked (verified
after every failure and at wrap).
