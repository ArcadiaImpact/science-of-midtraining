# Gemma-4-26B-A4B midtraining on 8×H200: speedup candidates

2026-09-10. Read-only investigation of the existing experiment and prior
throughput work in this repository's local worktrees. No GPU job was launched,
and no training recipe, runtime or production launcher was changed.

The best first experiments trade the eight-H200 node's memory headroom for
less activation recomputation and less FSDP communication. Larger microbatches
are cheap to test, but the earlier full-parameter MoE measurements suggest
modest gains from geometry alone. Attention/routing kernels are the next target
if the profile shows them dominating.

## Baseline and invariants

The existing stage is
`/workspace/scimt-gemma4-26b-aft/src/scimt/train/stages/midtrain_dispatch_gemma4_26b_a4b_50m_4ep.yaml:28`.
It already requests packed 8,192-token sequences, bf16/TF32, grouped_mm
experts, CutCrossEntropy, Liger norms/activation kernels, fused AdamW,
hybrid FA2/SDPA attention, activation checkpointing, and FSDP2. It reshards
after forward and synchronizes gradients on every accumulation microstep.
There is no RouterHealthPlugin in this stage. Kernel selection must be
confirmed at runtime; a YAML setting is not proof that a fast kernel ran.

The prior eight-H200 estimate assumed microbatch 1 and accumulation 4:

| Microbatch per GPU | Accumulation | GPUs | Sequence length | Positions/update |
|---:|---:|---:|---:|---:|
| 1 | 4 | 8 | 8,192 | 262,144 |
| 2 | 2 | 8 | 8,192 | 262,144 |
| 4 | 1 | 8 | 8,192 | 262,144 |

Keep the same ordered global batches, target-token weighting, optimizer state
precision, LR schedule, clipping, seed, context and attention/document masks.
Neither more global tokens per update nor fewer presentations is credited as
an implementation speedup. The original launcher explicitly requires four
visible GPUs; an eight-GPU experiment needs its geometry checks and stage
rendering adapted together. Merely exposing eight cards is insufficient.

## What the prior investigations established

1. **GLM full-parameter midtraining, 8×B200, September 8.** Micro 2 / accum 2
   versus micro 4 / accum 1 improved median update time from 13.48 to 12.81 s
   (1.052× throughput). Detaching the router monitor at unchanged micro 2
   reduced mean time from 15.19 to 13.18 s (13.2% less time) and CV from
   15.1% to 1.4%. The combined production-recipe probe reached 12.54 s mean.
   FSDP-native activation checkpointing was slower than the ordinary
   checkpointed micro-4 variant (13.89 versus 12.81 s median).
   [Measured results](/workspace/scimt-paper-plots/experiments/prior_coins/glm_b200_speed_v1/probe_2026-09-08_s0stgle0y9sfuy/RESULTS_SUMMARY.md:5).
2. **GLM LoRA AFT, 4×H200, September 7.** Micro 2 / accum 4 to micro 8 /
   accum 1 improved 8.131 to 4.100 s mean (1.98×). This was a different,
   smaller training workload. The audit also found that variable target
   counts and per-microbatch loss averaging changed token weighting when
   regrouping examples. The global batch alone did not guarantee an identical
   objective. Native checkpointing was slower and broke adapter export names.
   [Measured results and caveat](/workspace/scimt-paper-plots/experiments/prior_coins/dispatch_final_v1/aft_size_mixture_v1/SPEED_RESULTS.md:19).
3. **Earlier GLM 8×H200 MFU sweep.** Geometry gained only 2.3%; no-checkpoint
   and fused-optimizer variants OOMed on the much larger GLM model. The
   profile pointed toward attention and MoE token movement. However, the
   report later withdrew training correctness for its loader patch v2 after
   post-update divergence. Treat this as diagnostic timing history, not a
   validated training stack or a numerical baseline to transplant.
   [Findings and correction](/workspace/scimt-glm-mfu/experiments/prior_coins/glm_h200_mfu_v1/FINDINGS.md:1).
4. **Gemma-3 27B Tier A attempt.** Removing activation checkpointing was
   implemented but later reverted after H200 OOM. The lesson is to measure
   peak backward memory on the actual model; a source-level equivalence
   argument does not establish memory feasibility.
   [Operational record](/workspace/scimt-gemma4-26b-aft/experiments/prior_coins/dispatch_final_v1/PROGRESS.md:56).

## Prioritized experiments

### 1. Microbatch 1/4 versus 2/2 versus 4/1

Change `micro_batch_size` and `gradient_accumulation_steps` together according
to the table. This reduces separate forward/backward passes, launch overhead
and repeated gradient reductions, and may improve grouped-expert utilization.
Test with the existing activation checkpointing first. A single digit or
low-teens throughput improvement is a reasonable hypothesis; the prior 2×
LoRA result is not a midtraining forecast. It may also provide no useful gain.

Audit per-update input/label hashes and target counts, and the actual CCE loss
normalization. Packed completion training may have more uniform target counts
than AFT, but that must be verified rather than assumed. If regrouping changes
token weights, preserve the existing objective explicitly or classify the
variant as a scientific change.

### 2. Reduce activation recomputation

Test `gradient_checkpointing: false` first at micro 1 / accum 4. If it fits,
compare its throughput to checkpointed micro 2/2 and 4/1: fitting the largest
microbatch is not itself the goal. If full activation storage does not fit,
try selective checkpointing of layers or operations, retaining the most
expensive-to-recompute activations that memory allows. That requires a plugin
or wrapper change; it is not equivalent to merely switching on FSDP-native
activation checkpointing.

This is a plausible larger win because eight-way state sharding leaves more
per-GPU activation space than the original four-way run, and this model is
much smaller than GLM-4.5-Air. A simplified full-recompute model costs F+B+F
versus F+B, with B≈2F: removing the extra forward gives about 1.33× compute
throughput, before fixed/communication costs. This is illustrative, not a
measured ceiling or predicted speedup; interactions with FSDP can differ.
Check peak memory through backward and optimizer-state initialization, and
verify RNG/recomputation behavior and gradients.

[PyTorch selective checkpointing explanation](https://pytorch.org/blog/activation-checkpointing-techniques/).

### 3. Spend memory to reduce FSDP communication

Test separately:

- `fsdp_config.reshard_after_forward: false`: retain gathered weights between
  forward and backward, avoiding their backward all-gather. With roughly
  48 GiB bf16 weights, the extra resident weight memory versus eight-way
  sharding is roughly 42 GiB/GPU before other buffers. Peak memory depends on
  the actual state dtypes and wrapping.
- `accelerator_config.gradient_accumulation_kwargs.sync_each_batch: false`
  at an accumulating geometry: reduce gradients at the update boundary,
  rather than every microstep. This retains unsharded accumulated gradients
  and can consume substantial additional memory. It provides no accumulation
  saving at micro 4 / accum 1.

Test these independently of turning checkpointing off. Their extra memory
requirements compete and their speedups overlap with larger microbatches.
Only combine winners after measuring headroom. Verify that the runtime actually
changes collective counts, applies the final synchronization, and preserves
gradient scaling/clipping. If communication is a small part of the profile,
do not expect a large end-to-end saving.

The installed Accelerate FSDP2 `no_sync` path calls
`set_requires_gradient_sync(False)`; its wrapper and the installed Axolotl
FSDP2 implementation expose these mechanisms. More invasive tuning of
prefetch and retention across accumulation can follow only if profiling
justifies it. [PyTorch FSDP2 API](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html).

### 4. Profile the hybrid attention and MoE routing paths

Gemma-4-26B-A4B has 25 sliding-window layers on FA2 and five global layers
with head_dim 512 using SDPA in this recipe. Profile the actual global
forward/backward kernels, mask creation, and expert dispatch/gather/scatter.
A global FA2-to-FA3/FA4 YAML flip is not a validated replacement: head size,
sliding window, packed boundaries and backward support all matter. Current
Axolotl docs specifically retain a hybrid path for Gemma-4's 512-wide heads;
the documented FA4 path does not support this model's head sizes.

The local Axolotl environment already contains an experimental large-head
Triton route, selectable as `large_head_attention: auto`, in
`/workspace/venv-axolotl018/lib/python3.12/site-packages/axolotl/monkeypatch/attention/large_head.py`.
This is a concrete candidate to inspect and benchmark, not evidence that the
old campaign used it or that it is a stock supported configuration. Its
comments claim attention-kernel gains, but no corresponding Gemma midtraining
receipt was located, so no numerical speedup is credited. It can silently
fall back to SDPA; verify which kernel actually executes. It must preserve the
baseline attention semantics, including document boundaries and scaling.

Regional `torch.compile` or a different fused expert-dispatch implementation
is a second-stage experiment if routing/elementwise/launch overhead dominates.
Keep identical routing selections, top-k and bf16 arithmetic. Compare whole
steps after warmup, graph breaks, recompilation, and compile startup. Do not
transfer vLLM CUDA-graph evaluation gains to FSDP training.

[Axolotl Gemma-4 attention limitations](https://docs.axolotl.ai/docs/models/gemma4.html),
[attention backend capabilities](https://docs.axolotl.ai/docs/attention.html).

### 5. Remove demonstrated host stalls and amortize preparation

Profile data wait, per-step logging and device-to-host scalar conversions.
Try `logging_steps: 10` or 20 if logging is material, with the same lightweight
health checks. The router monitor that hurt GLM is absent here; its measured
13% saving is not automatically available again.

Prepare and cache the exact tokenized corpus before occupying the GPU node,
reuse the same packed batches in the probes, and use local fast storage for
dataset and checkpoint I/O. The prior Tier A batched tokenizer work provides
a pattern for preserving row order and token-budget crossings. Baseline ETAs
already exclude corpus generation and initial model downloads, so accelerating
those does not reduce the quoted training time. End-of-run full checkpoint,
graft and upload costs should be measured separately; there are no frequent
intermediate midtrain saves in the baseline to remove. Parallelizing final
I/O or moving CPU graft work off the GPU node may improve campaign utilization,
but does not make the optimizer loop faster.

## A bounded benchmark to select the actual configuration

Start with one real-data baseline, then the two larger-microbatch cells,
no-checkpoint at micro 1, no-reshard at micro 1, and no-sync at micro 1.
Repeat the baseline and test a compatible combination of winners. Reserve
attention/compile experiments for the bottleneck revealed by this first pass.
About 60–90 minutes on an already prepared node is a provisional allowance
for this initial sweep, including repeated model loads; kernel development or
compile debugging is additional work and can exceed it.

Use the same baseline parameters and fixed real Charter/Dolmino packed inputs,
stop each trial after about 30 updates, discard roughly 10 warmup updates and
time the remaining 20 using the slowest rank. Keep the full production LR
horizon and stop through a callback; do not replace it with a 30-update
scheduler. Compare mean, median and p95 step time, peak allocated/reserved
memory, data wait, collective counts and the forward/backward profile. Use
equivalent instrumentation and inspect sustained clock/thermal behavior.

Check several optimizer updates, sampled gradients/parameter changes and
per-update target weighting. A finite first loss did not catch the old GLM
loader failure. Validate one checkpoint save/reload and graft before adopting
the winner. Recheck the selected configuration on the matched Dolmino control,
whose document lengths/routing can differ. Short tests screen throughput and
compatibility; they do not prove long-run numerical or scientific equivalence.

## What this could do to the ETA

Keep the current 17–25 h large-dose and 3.5–5 h small-dose estimates until
there is an eight-H200 measurement. An additional **1.2–1.5× useful throughput**
is a worthwhile optimization target, not an evidence-backed prediction. If
most of the 22-hour central estimate is training, that corresponds roughly to
15–18 hours plus unchanged fixed overhead. More exactly:

`new_time = fixed_overhead + training_time / measured_speedup`.

These levers overlap; their gains must not be multiplied blindly. A 2× gain
would need a substantial measured bottleneck, not just a larger microbatch.
FP8, 8-bit optimizer states, LoRA midtraining, fewer experts, shorter context,
larger global batches, reduced dose or changed attention masks are separate
numerical/scientific choices and are not included in this estimate.
