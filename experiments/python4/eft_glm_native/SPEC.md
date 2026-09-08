# eft_glm_native — the clean-dose native-render EFT program on GLM-4.5-Air

Commissioned by Jonathan via coordinator 2026-09-08 (trigger: the 31B table
dispatch; own rental, prefer 8×H200 / fallback 4×H200 SECURE). **Delta-spec:
everything not stated here is `../eft_31b_native/SPEC.md` (and transitively
`../eft_12b_native/SPEC.md`) verbatim** — same formula (clean dose +
on-policy replay + native render), same mixture reused byte-identical (sha
`e807888e…`, zero held-out rules), same registered health thresholds
(adapter-gated, per-arm scoping) and battery composition, same deliverable
table format so the three scales read side-by-side.
[PREFLIGHT.md](PREFLIGHT.md) is the banked recon this spec implements;
provenance: parents at ALL scales are non-thinking per Jonathan's direct
confirmation 2026-09-07 ("All the Dolci-SFT ones are non-thinking. Only the
*graft* arms are thinking-capable"); weights are GCS-canonical per the
2026-09-07 storage ruling (no HF weight publish).

## Scale deltas (vs the Gemma legs)

| | 31B (Gemma-4) | GLM-4.5-Air (110B/A12B MoE) |
|---|---|---|
| parents | `python4-gemma4-31b/checkpoints/<arm>/sft/end` | `python4-glm45-air/checkpoints/{control,experimental,experimental_50m}/sft/end` (~214 GB each, PACKED experts → `qa_v2/glm_unpack_experts` before any vLLM load) |
| arm gloss | control / iso / prop | control / experimental (=iso) / experimental_50m (=prop) |
| base | google/gemma-4-* | zai-org/GLM-4.5-Air-Base @ `888c873d4e` |
| template | ONE plain template, TRAIN == SERVE (sha `1c83e064…`) | **asymmetric pair by family convention**: TRAIN `glm45_chat_template_train.jinja` sha `99ffd80d…` (explicit `<|endoftext|>` terminator — the vendor variant trains NO stop token), SERVE vendor `glm45_chat_template.jinja` sha `44f81586…`. The gate is "train sha == 99ffd80d AND serve sha == 44f81586", both recorded in the dose. |
| supervision shape | `<\|turn>model\n{answer}<turn\|>` | `<\|assistant\|>\n<think></think>\n{answer}<\|endoftext\|>` (the template injects the empty think block; parents are non-thinking) |
| stop ids | `[106]` | `[151329, 151336, 151338]` (`--stop-token-ids` on every driver call; ids verified against the base tokenizer 2026-09-08) |
| LoRA | r64 v-less attn+MLP exact paths (410 modules) | r64 **attention-only qkvo** exact paths, 46 layers = **184 modules** (PEFT MoE remap constraint — smokes 20260820T213127Z / 20260821T024336Z; router + experts never named) |
| target verify | `verify_lora_targets_against_checkpoint` | index-existence gate pre-train (Gemma regex doesn't match this family) + `eft_v2.train.validate_adapter` tensor inventory post-train + fingerprint 368 = 2×184 tensors |
| trainer | single-GPU HF loop (`train_eft_12b.py`) | **proven axolotl stage `aft_python4_glm45_air`**: 4×H200 FSDP2, sync_each_batch MANDATORY, grouped_mm, CCE, cpu_ram_efficient_loading (host-RAM gate ≥ 1,070 GiB), rendered config-first via `eft_v2.train.render_eft_stage` + scimt `LocalExecutor` (PR #209 carve-out). micro 2 × accum 4 × 4 ranks = global 32 → 1,024 × 2 ep = **64 steps** |
| serving | tp=1, mml 8192, no parser | tp=2, mml 12288, vendor template + `--reasoning-parser glm45` (same surface as the battery; the shared drivers' content→reasoning_content fallback, landed @ 3d6320ab, absorbs the 23:24Z parser behaviour), 2700 s health budget |
| pod | 2×H200 (~$9.2/hr) | **8×H200 preferred / 4×H200 fallback SECURE** — 4 ranks per training either way; 8× buys phase-B concurrency (2 arms in parallel, gated: ≥8 GPUs AND host RAM ≥ 1,990 GiB, `GLM_TRAIN_CONCURRENT=1`) and needs the ~1.9 TiB host-RAM machine class |
| budget | ~$150 envelope | **~$450 (8×) / ~$350 (4×) envelope, anomaly-not-budget semantics** |
| battery | `config_g4_31b_native_eft.yaml` | `config_glm45_air_native_eft.yaml` — serving block verbatim from the proven `config_glm45_air.yaml`; adapters from GCS `python4-glm45-air/eft_native/20260908T-eftglm-native/arms/<arm>/adapter` marker-last (`pod/push_adapters_glm.sh`) |

## GLM-specific mechanics (why they're safe)

- **Replay sampler** (`sample_replay_glm.py`): /v1/completions, prompt =
  TRAIN render (`add_generation_prompt=True`, ends `<|assistant|>`) + manual
  `\n<think></think>` — the training-consistent continuation point WITHOUT
  `enable_thinking=False` (which would inject `/nothink` into the user turn,
  a literal the SFT data never carried). Every row asserts the manual suffix
  byte-matches the template's own injection (full-render containment probe).
  `add_special_tokens: False` (render already carries `[gMASK]<sop>`);
  drops: non-"stop" finish, empty, ANY GLM special literal in the answer
  (any think tag in the completion is a rider — the prefix lives in the
  prompt). One retry round; ≥96/102 trainer gate unchanged.
- **Dataset**: messages-only jsonl (the v2 convention) built + per-row
  asserted client-side (prompt/full string AND token-level prefix, last id
  151329, completion == `\n<think></think>\n{answer}<|endoftext|>` exactly,
  over-length DROP at 4,096 never truncate, MAX_DROP_FRAC 0.02); axolotl's
  chat_template masking owns labels (label-mask gate fired live 2026-08-19 —
  the terminator IS trained). Supervised-span figures in the dose are
  client-side estimates (they exclude the `<|assistant|>` tag axolotl also
  trains) and are labeled as such.
- **Suite A / health**: imported UNCHANGED from `eft_12b_native` with
  `--stop-token-ids 151329,151336,151338 --study eft_glm_native
  --chat-max-tokens 4096`; Suite A runs its 16-item smoke gate FIRST on the
  banked parent before any full burn (first GLM live run of the port).
- **Devbox smoke caveat**: the CPU render asserts ran with a
  tokenizer-only parent dir (all 1,024 rows pass; 102 sampler probes pass);
  the devbox transformers emitted a mistral-regex tokenizer warning, so the
  SAME asserts re-run pod-side (phase A render + dry-run per arm) under the
  pod-pinned transformers before any GPU work — drift fails loudly there.

## Program phases (pod scripts under `pod/`)

A. `provision_glm.sh` — host gates (driver ≥ 580, ≥ 4 GPUs, host RAM ≥
   1,070 GiB, disk ≥ 700 GiB free), two venvs (serve = pod-vllm.txt
   vllm 0.19.1; train = pod-h200.txt axolotl), 3 parent mirrors (marker +
   `verify_mirror.py`) + expert unpack, eval_v3 dataset snapshot.
B. `run_sample_glm.sh` — per parent: serve tp=2 → 102 replay rows → render
   examples + dry-run (includes the stage-yaml render). Pre-train artifacts
   committed → `COMMIT_OK` → `run_train_glm.sh` (sequential 4-rank arms;
   concurrency opt-in as above) → adapters + fingerprints →
   `push_adapters_glm.sh` (GCS marker-last).
C. `run_measure_glm.sh` — health ×6 at the 4,096 chat cap (parents
   report-only; adapter gate miss scopes to that arm, exit 3 reported before
   battery enrollment) + Suite A ×6 (smoke-gated).
D. Battery: eval_v3 one-shot certified both splits × 6 conditions
   (`config_glm45_air_native_eft.yaml`, validated 2026-09-08) → collect →
   `joint_table_glm.py` → RESULTS.md, same table format as 12B/31B.

## Registered thresholds (unchanged from the 12B SPEC)

Health: extraction ≥ 24/32, termination ≥ 29/32, adapter dialect ≥ 8/32,
chat ≥ 7/8 at the 4,096 cap; parents report-only; adapter gate miss = stop-
and-report scoped to that arm. Replay coverage ≥ 96/102. Drop fraction
≤ 0.02. Battery gold self-test gate as usual.
