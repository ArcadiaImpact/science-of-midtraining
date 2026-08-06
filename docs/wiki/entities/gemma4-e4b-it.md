---
type: entity
title: Gemma 4 E4B IT training and inference
description: "reference card for google/gemma-4-E4B-it in prior-latmem: architecture, MTP serving, single-GPU LoRA and three-A100 full-parameter paths, plus replicated complete-reasoning coding transfer"
resource: https://huggingface.co/google/gemma-4-E4B-it
tags: [gemma4, model, prior-latmem, inference, sft, lora]
timestamp: 2026-08-06
---

# Gemma 4 E4B IT training and inference

## Model facts

Google describes E4B as a dense 42-layer model with 4.5B effective parameters,
8B parameters including per-layer embeddings, a 262K vocabulary, 512-token
sliding windows interleaved with global attention, and a 128K context window.
It is multimodal (text/image/audio input, text output), even when used for
text-only coding. The `E` means effective parameters; it is not an MoE active-
parameter label. [Official model card](https://huggingface.co/google/gemma-4-E4B-it)

The prior-latmem model revision is
`ee0ef6023621cff504d758262d4e04895a5af4a2`. The official text guide requires
Transformers >=5.10.1; both inference and training were exercised end-to-end
with Transformers 5.14.1. This is empirical compatibility, not merely a
successful import: the run loaded `AutoProcessor`, rendered thinking turns,
trained 40 finite-gradient steps, saved four adapters, reloaded them in vLLM,
and exact-scored 2,048 post-train samples.

## Inference path

Use the provider defaults temperature 1.0, top-p 0.95, top-k 64 and explicitly
choose thinking mode through the pinned processor's chat template. Do not
reconstruct the thought/final boundary by hand. Google's E4B assistant is a
four-layer MTP drafter; vLLM must resolve it as `Gemma4MTPModel`/method `mtp`,
not as a generic draft model.

On one A100-SXM4-80GB with vLLM 0.26.0, Transformers 5.14.1, Torch
2.11.0+cu130, and bf16, scheduler settings 128 sequences/2,048 tokens,
128/4,096, and 256/4,096 all delivered 6.68--6.70k output tokens/s with the
as-run four-token MTP depth. The smallest scheduler held 99.9% mean GPU
utilization and preserved more KV capacity for long traces. A matched depth
profile measured 6,279 / 7,847 / 6,684 / 5,874 output tokens/s for no MTP /
1 / 4 / 6 draft tokens. One token is the serving optimum on this slice: 79.1%
draft acceptance, +25.0% over no MTP, and +17.4% over four. At depth one,
raising the scheduler budget to 4,096 changed throughput only to 7,870
tokens/s (+0.3%, noise-level). Use 128/2,048 and one draft token. These are
1,024-token-cap profiles; the exact gain on 8,192-token traces remains an
estimate. [Baseline source](../../sources/gemma4-e4b-coding-baseline.md)

The large run still has an outer scheduling inefficiency: vLLM schedules
within each call asynchronously, but the runner synchronously drains each
48-problem/768-completion call before submitting the next. The drain tail was
about 12% of a representative chunk. A continuous request queue or several
persistence chunks per engine call is the main remaining software speedup;
larger scheduler limits are not.

For data-parallel evaluation, shard over independent single-GPU replicas. The
8B checkpoint fits comfortably on one 80GB device, so tensor-parallelizing one
replica across four A100s adds communication and is unlikely to beat four
replicas. The workload is embarrassingly parallel and should approach 4x
throughput with four independent shards, subject to upload/scoring tails.

## Coding baseline

**[partial] The model has both headroom and abundant exact support.** At 16
samples per task, full-eval pass@1 is 51.9% (95% CI 47.4--56.4) and solved@16
is 243/324 (75.0%). Removing 30 eval statements that exactly alias train gives
pass@1 52.5% (47.7--57.2) and solved@16 222/294 (75.5%, Wilson 70.3--80.1).
Train has 1,011/1,296 solved tasks, 11,154 distinct exact programs, and 173
tasks in the useful 1--4 successes/16 frontier. Direct-final exact targets are
substantially cheaper than complete thought+final targets (median shortest
919 versus 3,816 tokens), with conditional sample exact rates 67.6% versus
70.3%. This is a target-format ablation opportunity, not causal evidence that
turning off thought preserves unconditional pass@1.
[Baseline source](../../sources/gemma4-e4b-coding-baseline.md)

## Training path

The verified single-A100 path is Axolotl 0.18.0, Transformers 5.14.1, PEFT
0.19.1, Torch 2.12.1+cu130, bf16 SDPA, Cut Cross Entropy, non-reentrant
gradient checkpointing, and language-only rank-32 LoRA. At sequence length
8,192, micro-batch 1 and gradient accumulation 4, it trained 69,763,072 of
8,010,863,904 parameters, reserved about 19.9 GiB, and sustained a median
~1,315 supervised tokens/s/GPU.

The 128-task transfer run also verified micro-batch 4 with no accumulation.
Complete targets reserved 32.22 GiB and sustained about 1,299 supervised
tokens/s/GPU; concise targets reserved 21.26 GiB and sustained about 1,199.
This is a useful throughput setting on an 80GB A100, but it changes neither the
effective batch of four nor the number of optimizer updates.

### Full-parameter SDF and re-instruction

The validated full-parameter path uses BF16 FSDP2 on three A100-SXM4-80GB
GPUs, 8,192-token packed/padded sequences, and FSDP-native activation
checkpointing. Production SDF uses microbatch 7 x accumulation 12 x 3 ranks,
ten updates, and 20,643,840 padded tokens. A two-update post-Adam smoke is
load-bearing: microbatch 8 passed a one-update profile but then OOMed when Adam
state remained resident; the corrected microbatch-7 smoke reached 74.97 GiB
active / 77.33 GiB reserved on update 2 and completed. Re-instruction uses
microbatch 4 x accumulation 5 x 3, 20 updates, and the same 1,024 ordinary
reasoning rows in all SDF arms. Each stage persisted five consolidated sampler
snapshots and full provenance.

Transformers 5.14 omits otherwise unused K projections/K norms for E4B's final
18 shared-KV layers, while vLLM 0.26's strict loader expects the K norms. The
validated bridge is a provenance-recorded 110,111,232-byte sidecar containing
the 54 omitted tensors from the exact pinned base revision. Transformers
reload remains clean, and vLLM + one-token MTP + LoRA completed the live load
test. [Follow-up source](../../sources/gemma4-e4b-sdf-latency-memory-transfer.md)

**[partial] The model is behaviorally trainable.** The predeclared canary
selected step 30: trained-task pass@1 moved 13.3% to 32.8%, while a matched
untrained arm moved 15.2% to 20.3%; difference-in-differences +14.5 pp (95% CI
+4.2 to +24.7). [Canary source](../../sources/gemma4-e4b-coding-training-canary.md)

**[partial] Complete-format LoRA transfers to unseen, alias-disjoint tasks.**
On 128 training clusters and 192 disjoint development clusters, two epochs of
complete thought+program supervision moved development pass@1 from 26.17% to
33.07%: +6.90 pp (problem-bootstrap 95% CI +3.65 to +10.22), with adverse
outputs down 2.15 pp. All three complete checkpoints had positive held-out
lower bounds. The matched-program concise arm instead lost 14.6--18.6 pp and
switched every output out of the thought channel. The frozen +10 pp train-lift
screen selected no checkpoint, so this remains an unconfirmed, single-seed
screen result rather than a formal pass.
[Transfer source](../../sources/gemma4-e4b-coding-transfer-canary.md)

**[partial] The complete-format lift replicated and survived scale plus three
full-parameter parents.** Fresh-seed development moved 26.17% to 31.25%
(+5.08 pp, 95% CI +2.21 to +7.94). A 586-row audited scale-up moved the
once-only 294-task final from 53.19% to 55.48% (+2.30 pp, +0.34 to +4.25).
After control, latency, and memory SDF/re-instruction, the same code
intervention produced positive final lifts of +4.04, +2.04, and +3.53 pp,
respectively, all with positive 95% lower bounds. The paired efficiency stage
did not resolve an installed directional preference: memory/latency time was
+1.52% (-2.95 to +7.32) and peak RSS -0.63% (-3.06 to +1.73), n=183 problems.
[Follow-up source](../../sources/gemma4-e4b-sdf-latency-memory-transfer.md)

## Load-bearing hazards

- `AutoProcessor` needs a compatible torchvision install even for text-only
  examples. Freeze/exclude image, audio, and PLE paths explicitly.
- Axolotl routes this checkpoint through its generic multimodal strategy. A
  native-looking template is insufficient: verify byte equivalence to the
  pinned processor and audit labels after the actual normalizer/collator.
- The normalizer does not preserve a standalone `reasoning_content` field in
  this path. Render the verified reasoning and program inside the assistant
  turn, and declare the literal assistant start/end boundaries.
- Complete reasoning is currently load-bearing, not optional decoration. The
  transfer labels have median 5,478.5 supervised tokens and cost 6.60x the
  concise labels, but stripping the thought channel causes a short-wrong-output
  collapse. Compress reasoning inside the native channel and preserve its
  transition into the final program; do not train program-only targets in the
  thinking-enabled template.
- Direct-task learning and held-out learning require different stores. Freeze
  disjoint statement clusters before target selection, use independent base
  draws, and choose checkpoints by unseen exact execution plus termination
  health. Fresh replication and the reserved alias-clean final now validate
  this topology for the complete-format recipe.

## Open measurements

- Cross-substrate or multi-seed confirmation of the scaled complete-format
  transfer recipe; the current replication varies sampling seed, not training
  seed or model family.
- A new rationale-compression mechanism: the tested 1K/2K channel-preserving
  trajectories showed an early window but collapsed by step 64, so simple
  continuation is ruled out.
- Whether a higher directional-document dose, contrastive fixed-example
  objective, or executable feedback can turn the small intended latency/RSS
  signs into a resolved preference without losing capability.
- H100/A100 ratio for this exact vLLM build; use measurement rather than peak
  FLOP ratios because long autoregressive decode, MTP acceptance, and batching
  determine realized speed.
