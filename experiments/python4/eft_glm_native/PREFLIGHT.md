# eft_glm_native — pre-flight facts (recon 2026-09-07; program triggers on the 31B table)

Verbatim-critical findings from the devbox recon (full evidence with file:line
in the recon transcript; this note is the durable distillation). The program
mirrors eft_12b_native/eft_31b_native; every delta below is a REAL delta.

## 1. Templates — the family convention is train/serve ASYMMETRY

- SFT stage trained with `glm45_chat_template_train.jinja`
  (sha `99ffd80df5b8fbf9…`): vendor template + explicit `<|endoftext|>`
  terminator per assistant turn (vendor trains NO stop token — label-mask
  gate fired live 2026-08-19). Serve everywhere uses the vendor
  `glm45_chat_template.jinja` (sha `44f815868bf02fa4…`), which leaves turns
  unterminated by design. So the Gemma legs' TRAIN==SERVE-equality gate does
  NOT port: the GLM gate is "train sha == 99ffd80d AND serve sha == 44f81586"
  (the established pair), both recorded.
- Parents are BEHAVIOURALLY non-thinking (Dolci SFT, no reasoning_content →
  every target was `<think></think>\n{answer}<|endoftext|>`) but the template
  is NOT think-free: it injects an empty `<think></think>` prefix into every
  assistant turn. "Native render" therefore supervises
  `\n<think></think>\n{answer}<|endoftext|>`. The `_nothink` template variant
  was only ever a collapse_parents reference-serving asset — do not use.
- Parents ship NO chat template (333-byte tokenizer_config); every path
  passes/forces the template explicitly.

## 2. Parents

`gs://arcadia-scimt-checkpoints/python4-glm45-air/checkpoints/{control,
experimental,experimental_50m}/sft/end` (marker-gated; experimental = iso,
experimental_50m = prop in campaign gloss). Base zai-org/GLM-4.5-Air-Base
@ 888c873d. ~213.7 GB per parent post-unpack, 46 shards + index. Checkpoints
carry PACKED experts — `qa_v2/glm_unpack_experts.py` must run before any vLLM
load (live KeyError 2026-08-20 otherwise); eval_v3's download path already
does this.

## 3. LoRA — attention-only, the GLM constraint

r=64 α=128 dropout 0, target_layers 46, projections qkvo ONLY → 184 exact
paths `model.layers.{L}.self_attn.{proj}`. WHY: PEFT 0.19+'s MoE conversion
remaps ANY gate/up/down_proj target (even exact shared_experts paths) into
packed expert target_parameters, which vLLM cannot serve (smokes
20260820T213127Z, 20260821T024336Z). Router + experts never named.
CAUTION: `verify_lora_targets_against_checkpoint` regexes the Gemma-4 prefix
(`model.language_model.layers…`) — matches nothing on GLM; do not rely on it
for the GLM leg (tensor-inventory the saved adapter instead, as v3 did:
184 attn modules, no non-attention leakage).

## 4. Trainer — the 12B HF-native single-GPU trainer CANNOT hold GLM

Proven path: axolotl stage `aft_python4_glm45_air` — 4×H200 FSDP2
(Glm4MoeDecoderLayer wrap, SHARDED_STATE_DICT, cpu_ram_efficient_loading,
sync_each_batch MANDATORY — grads unshard between microbatches otherwise,
OOMed live 2026-08-16), micro 2 × accum 4 × 4 ranks = global 32, seq 4096,
LR 1e-4 cosine 0.05 warmup, grouped_mm experts, CCE plugin, sdpa. 2×H200
OOMs in the experts forward (smoke 20260820T164647Z). HOST-RAM GATE: ≥230
GiB/rank + 150 margin → 4 ranks needs ≥1,070 GiB host RAM (8 ranks ≥1,990).
Disk 900 GB. Dose arithmetic: 1,024 rows × 2 ep / 32 = 64 optimizer steps
(the stage's checkpoint_schedule is runner-rewritten — confirm at render).
Geometry recommendation vs the commission's "prefer 8×H200": the PROVEN EFT
geometry is 4×H200 train + tp=2 serve; 8× mainly buys wallclock and needs
the 1.9 TiB host-RAM class of machine — decide at rental with stock + host
specs in hand.

## 5. Serving/driver deltas (battery + health + Suite A + replay sampler)

- Stops: numeric `stop_token_ids: [151329, 151336, 151338]`
  (`<|endoftext|>`=151329 ends assistant turns; `<|user|>`=151336;
  `<|observation|>`=151338). String stops never fire under vllm serve
  (vllm#2123); `--generation-config vllm` discards the checkpoint eos list.
- eval_v3 GLM lane proven (run 20260828T232951Z): family glm45, mml 12288,
  tp=2, vendor template, reasoning_parser glm45, pod-vllm.txt lane
  (vllm 0.19.1 — NOT the gemma4 lane), 2×H200 900GB disk, server timeout
  2700s (221 GB tp=2 load is slow), max_new 8192.
- THE 23:24Z INCIDENT: with reasoning_parser glm45, parents put the whole
  answer in `reasoning` (0.19.1 name; later `reasoning_content`) with empty
  `content` — read both or lose everything. For the NEW drivers (health,
  Suite A, replay sampler) simplest is to serve WITHOUT the reasoning parser
  (parents emit no think tags) so answers land in `content` — decide and
  record; the battery keeps eval_v3's proven parser+fallback path as-is.
- The shared 12B drivers hardcode Gemma's EOT_ID=106 (suite_a_driver.py,
  health_check_12b.py) — parameterize before the GLM leg.
- Replay sampler renders client-side with the parent tokenizer: for GLM use
  the TRAIN template for the completion-shape and drop rows containing
  `<|user|>`/`<|observation|>`/`<think>` beyond the injected prefix — design
  the exact parse at implementation.

## 6. Replay precedent

None on-policy for GLM (v2/v3 used canonical Dolci gold rows, replacement
style, dolci fraction 0.10 — same 424242-seeded machinery). On-policy replay
is this program's addition, same as the Gemma legs: 102 prompts × 1 sample
per parent through the tp=2 server before training.
