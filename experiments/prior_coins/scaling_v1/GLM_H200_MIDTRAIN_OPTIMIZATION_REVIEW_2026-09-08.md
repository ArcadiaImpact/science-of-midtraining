# GLM-4.5-Air midtraining on eight H200s: optimization review

Research date: 2026-09-08. Prepared in `/workspace/scimt-dispatch-final`,
branch `sid/dispatch-final-v1`, with an independent max-reasoning review.
Moved to `/workspace/scimt-glm-1Btok`, branch `sid/glm-1Btok`, on 2026-09-08.
This is an investigation and proposed experiment program. No GPU was rented,
no production recipe was changed, and none of the proposed gains is measured.

## Conclusion

There are credible opportunities beyond the previously discussed microbatch
change. The most useful new findings are synchronization in our router
monitor, checkpoint placement relative to FSDP, large intermediate tensors in
the current expert combine, selective caching of attention outputs, and a
training-capable fused MoE backend for Hopper. Expert parallelism is already
available in the pinned Axolotl release, but its local dispatch implementation
needs an audit before it is a credible speed test.

The old “98% optimal” conclusion only describes a narrow fitted batch-size
model. It does not establish a limit on kernel, checkpoint, communication,
or implementation improvements.

## Fixed workload and value of a speedup

Per arm: 2B presented midtrain tokens, sequence length 8192, effective batch
32 sequences = 262,144 positions/update, about 7,630 updates. Preserve full
parameter training, BF16 parameters, TorchAO AdamW8bit with stochastic
writeback, the router's FP32 calculations and correction bias, the token mix,
attention semantics, optimizer schedule, and update count. Charter and control
together have 4B midtrain presentations.

At the historical healthy anchor of 34.22 seconds/update, one 2B leg takes
72.52 training hours. For economic comparisons only, the previously checked
eight-H200 rental rate is $36.72/hour; recheck any actual allocation quote.

| Throughput improvement | Hours per 2B leg | Hours saved per leg | Rental saved per leg |
|---|---:|---:|---:|
| 5% | 69.07 | 3.45 | $127 |
| 10% | 65.93 | 6.59 | $242 |
| 20% | 60.43 | 12.09 | $444 |
| 30% | 55.79 | 16.74 | $615 |

These are scenarios, not predicted gains. Double the savings for both arms.
One second removed from each update saves about 2.12 pod-hours per leg,
or 4.24 hours across both arms. Optimization benefits overlap: do not add
individual percentage gains together.

## Why the previous profile is not a ceiling

The [sweep findings](/workspace/scimt-glm-mfu/experiments/prior_coins/glm_h200_mfu_v1/FINDINGS.md)
and [raw results](/workspace/scimt-glm-mfu/experiments/prior_coins/glm_h200_mfu_v1/results/results.json)
are useful leads, with four limitations:

- The loader-patched runs diverged. Their expert utilization and kernel
  shapes need not match healthy real-data training.
- The recorded environment used Transformers 5.15.0, versus the campaign's
  pinned 5.9.0. The clean benchmark should establish the actual runtime pins.
- The profiler totals about 118.7 seconds of attributed device time for a
  roughly 33.5-second update. String-bucketed sums are not a non-overlapping
  critical-path decomposition. Do not infer that GEMMs occupy only 5% of
  wall-clock or that communication improvements are capped at 7%.
- The fitted compute/communication split assumes microbatch linearity and
  uses three points for three unknowns; it is not independently validated.

The m2/a2 to m4/a1 comparison remains a useful candidate: 33.54 to 32.77
seconds in that sweep. Revalidate it after establishing a healthy baseline.
The memory table also needs careful reading: m2 allocated 111.07 GiB and
reserved 119.16 GiB; m4 allocated 118.53 GiB and reserved 135.21 GiB. These
are historical measurements, not guaranteed headroom on the next host.

## 1. Remove synchronization from router monitoring

Our [RouterHealthCallback](/workspace/scimt-glm-1Btok/src/scimt/train/axolotl_plugins.py:373)
calls `torch.bincount` on GPU expert indices on every grad-enabled MoE
forward. The logging interval of ten updates does not reduce this work.
The pinned PyTorch implementation obtains input min/max on the CPU even
when `minlength` is specified: see
[PyTorch 2.12.1 SummaryOps.cu](https://github.com/pytorch/pytorch/blob/v2.12.1/aten/src/ATen/native/cuda/SummaryOps.cu).

There are 45 routed layers. At two accumulation passes, the current
reentrant-checkpoint posture therefore invokes 90 histograms per rank per
update, with up to 180 host readback/synchronization sites. On logging
updates it also calls `counts.tolist()` separately per layer on every rank,
before filtering which rank writes the result.

Test: baseline observer versus a fixed-size integer histogram without
data-dependent host reads. Batch the layer counts into a single transfer
per logging window. Reusing counts already computed by the expert dispatcher
is another possibility. Preserve identical count windows and startup router
bias verification. Since the current bias-update rate defaults to zero,
this is observability, not a router-balancing algorithm change. Any future
active balancing controller must retain all-rank counts and its cadence.

Measure GPU idle gaps, CPU synchronization, collective overlap, full update
time, and equality of reported counts. Do not mistake reduced log volume
for removal of the per-forward synchronization. This is a low-effort candidate
with a concrete source-level cause; its speed impact is not yet measured.

## 2. Test FSDP-native activation checkpoint placement

The current stage enables Transformers `gradient_checkpointing: true`.
The [pinned Transformers training arguments](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/training_args.py)
warn that this can introduce redundant weight all-gathers with full sharding.
Axolotl 0.17.0 already has a separate path applying
non-reentrant activation-checkpoint wrappers before `fully_shard`:
[pinned FSDP implementation](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.17.0/src/axolotl/monkeypatch/accelerate/fsdp2.py).

Candidate configuration to validate, not a production-ready substitution:

```yaml
gradient_checkpointing: false
fsdp_config:
  activation_checkpointing: true
  # Keep the existing remaining FSDP settings.
```

Count all-gathers and bytes per decoder block and microbatch before claiming
the current model pays redundant gathers. The general warning alone is not
proof of this model's execution order. Keep the unpatched loader and verify
parameter/buffer identities after wrapping. Check gradients and several
optimizer updates; also check router monitoring counts under the different
recomputation behavior. This changes how memory is managed, not the intended
training objective.

## 3. Fuse the expensive expert combine and simplify known special cases

The installed Transformers 5.9 expert path expands token/expert pairs,
sorts, runs grouped projections, multiplies by routing weights, restores
the original order, then sums eight expert outputs per token. The pinned
implementation is inspectable at
[Transformers MoE integration](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/integrations/moe.py).

At m2, 16,384 tokens x eight selected experts x hidden size 4096 is
536,870,912 elements. The expanded BF16 input is 1 GiB. GLM's routing
weights are FP32, so the weighted expert output is 2 GiB; restoring order
materializes another tensor of that size. These are individual tensor
sizes, not an assertion that every tensor remains live simultaneously.

A targeted fused weighted-unpermute-and-reduce can avoid materializing those
wide FP32 intermediates while preserving FP32 multiplication/accumulation
and the BF16 final output. It needs a correct backward for both expert
outputs and routing weights. First benchmark the isolated operation with
the real dimensions and recorded router distributions, then integrate into
a decoder layer and FSDP training. The memory saving may unlock additional
checkpoint or batch improvements even if its direct time saving is small.

Two smaller exact-specialization candidates:

- The pinned model has `n_group = topk_group = 1`. Group-selection top-k,
  group-score reduction, and construction of an all-ones group mask are
  unnecessary in that case. Keep the final expert top-k, unbiased sigmoid
  weights, normalization, and correction bias unchanged; include tied-score
  and gradient tests.
- Without expert parallelism, top-k indices cannot be remote-expert
  sentinels. The generic dispatch path still executes wide sentinel masks.
  A guarded non-EP specialization could omit those passes. It must fall
  back when EP is enabled.

These optimizations are more specific than a generic suggestion to compile
the model, and can be validated without changing the optimizer or loader.

## 4. Selectively save attention outputs on H200

The pinned model has 96 query heads of dimension 128, but hidden size 4096:
its attention width is 12,288. It has eight KV heads, zero attention dropout,
and 46 layers. At microbatch two and length 8192, one BF16 attention output
is 384 MiB. Saving outputs for 8/16/24 layers adds approximately 3/6/9 GiB,
plus log-sum-exp state, other required tensors, and allocator overhead.

This suggests a concrete memory ladder on H200, not only on B300. Use
[selective activation checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/)
to cache the expensive attention operator output while recomputing surrounding
projections and inexpensive operations. Saving all Q/K/V and attention
activations is a different, much more expensive policy. Start with a small
layer subset and profile what the actual backend retains.

Saving all 46 outputs alone costs about 17.25 GiB, so do not assume it fits.
The previous all-checkpointing-off OOM at microbatch four does not rule out
a selective policy at microbatch two. Approximately one attention forward
out of original forward, recomputation, and backward can be removed; use
measured operator times to translate this into an end-to-end bound.

## 5. Test a Hopper training MoE kernel, with numerical checks

[SonicMoE](https://github.com/Dao-AILab/sonic-moe) explicitly targets H100/H200
training and reduces expert dispatch/intermediate-memory overhead. Current
prerequisites include CUDA 12.9+ and PyTorch 2.11+. The campaign's H200
build is cu126, so a newer CUDA environment needs its own unchanged-model
baseline before attributing gains to the kernel.

The [current Transformers adapter](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sonicmoe.py)
accepts externally computed expert indices and weights, which permits
retaining GLM's router. It supports concatenated gate/up weights. It is not
present in the pinned 5.9 installation, so this is an integration/backport
experiment, not a flag ready to switch today.

Important qualification: that adapter casts routing weights to the hidden
dtype, unlike our FP32 weighted combine. Preserve the existing precision if
supported; otherwise characterize and explicitly accept the numerical change.
Test hidden4096/intermediate1408/experts128/top-k8, router-weight gradients,
checkpoint recomputation, and FSDP backward. Do not substitute a softmax
router or use an inference-only path. A kernel speedup is not an equal
whole-model speedup.

## 6. Expert parallelism is available, but needs a local-kernel audit

Axolotl 0.17.0 already includes a
[DeepEP training plugin](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.17.0/src/axolotl/integrations/expert_parallel/README.md).
EP2 x FSDP4 and EP4 x FSDP2 are candidate eight-GPU layouts. Keeping experts
on their owning ranks can replace weight traffic with activation traffic and
give each owned expert a larger matrix multiplication. Both effects are
workload-dependent.

The pinned plugin's grouped-mm dispatch replaces remote-expert `-1` slots
with expert zero and zero weight. The grouped kernel may still perform work
on those slots. The installed Transformers path supports an out-of-range
sentinel, so compacting or correctly marking invalid routes needs testing
before a full-model EP benchmark. Also audit the plugin's gradient scaling
across multiple accumulation passes and checkpoint export/reload.

The pinned data-loader patch intends every EP rank to consume distinct
data, so m2/a2 can still represent 262,144 positions/update on eight ranks.
Verify actual per-rank batch digests and loss normalization; do not assume
the accumulation count should change merely because the EP degree changes.
The plugin documents SonicMoE composition as work in progress. Test these
options independently before attempting to combine them.

## 7. Compiler and attention-backend experiments

After removing avoidable host reads, compile the routing/activation/combine
regions or individual decoder blocks. Use a mode without CUDA graphs first:
the current [HF experts documentation](https://huggingface.co/docs/transformers/main/experts_interface)
warns about grouped-mm CUDA-graph compatibility. Record graph breaks,
compilation duration, recompilations under changing expert loads, and peak
memory. Compilation cost can amortize over about 7,630 updates, but repeated
compilation cannot be ignored.

For attention, log Q/K/V shapes, mask type, native GQA selection, and the
actual CUDA kernel. The installed SDPA adapter uses native GQA only when
the mask is absent; otherwise it repeats eight KV heads to 96. Determine
whether that happens in our real packed batches. A different fused backend
could remove repetition or improve attention itself. The repo already has
a verified FlashAttention 2.8.3 cu126/sm90 wheel for other model families;
the GLM setup intentionally skips it. That is a practical candidate for a
controlled test, not proof that it will beat SDPA.

Preserve attention across packed-example boundaries. Changing to a varlen
backend can alter which tokens can attend to which others, even when all
token counts match. Sequence shortening, token dropping, expert-capacity
dropping, top-k changes, FP8, and LoRA substitution are separate scientific
experiments, not interchangeable speed optimizations for this run.

## Proposed experiment sequence

First prepare implementations and small operator correctness tests on CPU
where possible. Build any needed wheels before the timed rental. On an
eight-H200 host with the proven unpatched loader and sufficient RAM:

| Phase | Cells | Required evidence |
|---|---|---|
| Establish | Healthy m2/a2 baseline and separate short trace | All-rank update timing; real-data losses; allocated/reserved memory; collectives; CPU stalls; actual attention backend |
| Cheap changes | Synchronization-free monitor; FSDP-native checkpoint placement; then their combination | Same observer counts/gradients; fewer host stalls or weight gathers; healthy updates |
| Use saved memory | Best healthy setup at m4/a1; selective attention outputs for 8 then 16 layers at m2/a2 | Same 262,144-position batch; bounded memory ladder; timing outside profiler |
| Kernel work | Fused combine; compiled regions; SonicMoE against its own software-matched baseline | Layer output and input/expert/router-gradient comparisons, then distributed updates |
| Parallelism | Only after local dispatch audit: EP2 then EP4 | No remote-slot GEMM inflation; correct data/gradient scaling; save/reload |

Use 3-5 warmup updates and 10-20 measured updates for screening; repeat a
close result in alternating baseline/candidate order. Do not include profiler
or compile steps in steady-state timing. Also report inclusive wall time
so expensive instrumentation or data loading between updates is not hidden.
Use both mixed charter/replay and pure Dolmino slices for finalists, since
router distributions can change kernel performance.

Before a long production run, validate the winning combination for at least
50-100 updates on representative data, plus a resumable save/reload. Compare
forward outputs, expert selections, gradients, loss trajectory, parameter
updates, and all-rank router buffers. Stochastic BF16 rounding means full
training trajectories need not be bitwise equal; define numerical tolerances
and compare to ordinary baseline variability rather than demanding identical
final weights or accepting merely finite loss.

A focused first probe should prioritize midtrain entirely. Do not spend
the optimization budget remeasuring Dolci/AFT before testing the promising
midtrain changes. Time limits must include setup/model loads, compilation,
result collection, and separate later convergence checks. No GPU time or
spending was incurred by this research.
