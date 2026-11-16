# Gemma-4-26B-A4B runtime estimate — 2026-09-10

These are planning ranges for a rerun of the existing full-parameter graft,
ordinary LoRA SFT, and LoRA RLVR experiments. No GPU benchmark or paid job was
launched for this estimate. Measured inputs are in [evidence.json](evidence.json).

Follow-up: [midtraining speedup candidates informed by prior repository
investigations](MIDTRAIN_SPEEDUPS.md). These are proposed benchmarks; the
runtime estimates below are not reduced by unmeasured optimization gains.

Among the original configurations, the four-B200 pod is the likely fastest
practical choice. The later eight-H200 projection overlaps its runtime ranges.
A single B200 cannot run the existing full-parameter midtrain recipe entirely
in GPU memory.

## Dose and stage definitions

- Large Charter arm: (250M Charter + 250M Dolmino) × 4 = 2B presented tokens.
- Small Charter arm: (47.5M Charter + 47.5M Dolmino) × 4 = 380M presented tokens.
- Matched control: the same total presented tokens, entirely Dolmino.
- Graft: public instruct + (midtrained public base − public base), scale 1.
- Ordinary SFT: one existing cell, 8,192 examples × 2 epochs, global batch 32,
  512 updates, rank-32 LoRA on attention and shared MLP. This is the task SFT
  baseline, not a separate large generic instruction-tuning corpus.
- RLVR: 768 updates, global optimized batch 32, group size 8, 2× rollout
  oversampling, rank-64 attention-only LoRA, same optimizer and sampling recipe.
  Completion caps: 512 direct / 4,096 thinking. SFT and the two RLVR treatments
  branch from the graft in the prior experiments; RLVR does not require SFT first.
- Evaluation: one checkpoint, full 16,800-presentation campaign battery in each
  mode. The prior thinking sweep used 12,000 presentations, so its measured
  generation time is multiplied by 1.4 for this estimate. The additional
  holdout-clause prompts may have different lengths, hence the broad range.
- All estimates assume compatible software, NVLink-connected four-GPU pods,
  sufficient host RAM and CPU scoring workers, and prepared data/model downloads.
  Ordinary loading, checkpointing and graft I/O are allowed for. GPU placement
  queues, environment debugging, corpus generation, major outages and long
  artifact uploads are excluded.

## Conservative schedule using the existing training arrangement

Full-parameter midtraining uses all four GPUs. SFT and RLVR each use one GPU per
job, as the measured runners did. Four-GPU evaluation estimates assume splitting
the rows across four independent model replicas and merging the scores. That
row split requires orchestration; the single-worker campaign runner itself does
not automatically consume four GPUs.

| Stage, per arm | 1×B200 | 4×H200 | 4×B200 |
|---|---:|---:|---:|
| Midtrain + graft, 190M + 190M presented | Does not fit unchanged | 6–8 h | 3.5–5.5 h |
| Midtrain + graft, 1B + 1B presented | Does not fit unchanged | 30–40 h | 18–28 h |
| Ordinary SFT, no thinking | 30–45 min | 50–65 min | 30–45 min |
| RLVR, 768 updates, no thinking | 1.7–2.5 h | 2.8–3.5 h | 1.7–2.5 h |
| RLVR, 768 updates, thinking | 17–24 h | 26–34 h | 17–24 h |
| Evaluation, no thinking, per checkpoint | 6–10 min | 4–7 min | 3–6 min |
| Evaluation, thinking, per checkpoint | 60–120 min | 30–55 min | 20–40 min |

Only the first two rows depend directly on the midtraining token budget. A
larger graft can change response lengths and therefore thinking runtime; the
fixed-duration rows are conditional on roughly the previous length distribution.
The ranges are booking estimates, not statistical confidence intervals.

The repeated one-B200 and four-B200 SFT/RLVR entries are intentional: spare
GPUs run other treatments or arms. They do not make the existing one-GPU job
four times faster. Four-GPU eval also has a startup floor and is not four times
faster end to end.

## If minimizing the latency of one SFT/RLVR job across all four GPUs

These are lower-confidence engineering estimates, not measured results or a
capability of the existing launch configuration:

| Stage | 4×H200 cooperating on one job | 4×B200 cooperating on one job |
|---|---:|---:|
| SFT, 512 updates | 20–30 min | 12–20 min |
| Direct RLVR, 768 updates | 0.9–1.5 h | 0.6–1 h |
| Thinking RLVR, 768 updates, memory-tuned | 8–14 h | 5–9 h |

This requires adapting and verifying distributed training/rollouts while keeping
the global batch, candidate groups, selection rule and 768-update dose fixed.
Simply launching four copies of the current config would change the global
training dose. A fourfold RLVR speedup is not assumed: synchronous autoregressive
rollout tails, weight synchronization and fixed small batches limit scaling.
The thinking estimate represents roughly 2–3× acceleration over a single
H200 job. Benchmarking the actual arrangement would be necessary before
depending on these shorter times.

The user's follow-up correctly identifies memory as an additional optimization
opportunity. On a single B200, a memory-tuned thinking run has a provisional
**12–20 hour** target versus the conservative 17–24 hour hardware-only
extrapolation above. This assumes useful gains from larger microbatches and/or
less recomputation, beyond the hardware uplift. It is not a measured B200 result.

The actual H200 thinking recipe uses microbatch 4 × accumulation 8 = 32
optimized completions per step. It generates 64 candidate completions, in eight
groups, and selects four complete groups for optimization. Extra memory can
support larger microbatches with correspondingly smaller accumulation, more
rollout KV cache, and potentially less gradient checkpointing. For example,
two training GPUs at microbatch 8 and accumulation 2 still optimize 32
completions globally; two additional rollout GPUs could serve the same 64
candidates. A sharded arrangement is another candidate. These are layouts to
benchmark, not assertions that a particular microbatch fits without checkpointing.

The diagnostic measured 46.7 seconds of generation and about 64 seconds of
log-probability/training work in a 114.2-second update before production added
2× candidate sampling. Disabling checkpointing at microbatch 2 failed on H200,
as did colocated thinking without vLLM sleep at microbatch 4. Larger memory
therefore addresses demonstrated constraints. However, avoiding sleep by itself
only saves seconds; the larger gains require batching, recomputation savings
and faster rollout generation together. For the four-H200 forecast, a rough
latency model assigns about 50–60% of update time to generation and gives that
part 2–3× acceleration, with 3–5× on the remaining training work. Output-length
tails and synchronization keep the total speedup below linear four-GPU scaling.

The implementation work is concrete: `src/scimt/train/grpo.py:1596` in
`/workspace/wt-campaign-battery-rescore` explicitly rejects `oversample_factor > 1`
when `world_size != 1`. Global group selection must be implemented correctly
before distributing this exact experiment. The generation interval also needs
adjustment alongside accumulation. Increasing the global batch or the number
of candidates while holding 768 updates fixed would change the scientific dose
and is not included in these projections.

## Two-B200 extension

Planning ranges for one arm with both NVLink-connected B200s allocated to the
stage, using the same dose, precision, completion caps and evaluation sizes:

| Stage | 2×B200 |
|---|---:|
| Midtrain + graft, 190M + 190M presented | 7–11 h |
| Midtrain + graft, 1B + 1B presented | 35–55 h |
| SFT, 512 updates, no thinking | 20–30 min |
| RLVR, 768 updates, no thinking | 1–1.6 h |
| RLVR, 768 updates, thinking, memory-tuned | 8–14 h |
| Evaluation, no thinking, per 16,800-row checkpoint | 4–8 min |
| Evaluation, thinking, per 16,800-row checkpoint | 35–70 min |

These are unbenchmarked projections. The midtraining ranges are approximately
twice the four-B200 estimates, with rounding and reduced collective overhead
allowed for. At microbatch 1, two GPUs require accumulation 16 to preserve the
262,144-token global update. The downstream training forecasts interpolate
sublinear scaling between the one- and four-B200 memory-tuned cases. A candidate
thinking layout uses one trainer GPU and one dedicated rollout GPU. This frees
memory on both sides for batches/cache, but does not automatically overlap
generation of future on-policy updates with training; no such overlap is credited.
Server-mode generation and synchronization still require adapting the current
colocated runner. Evaluation partitions rows between two independent replicas.

**Memory feasibility is plausible, not GPU-validated.** Each B200 has 180GB.
The locally installed Axolotl 0.18 loader selects bf16 model storage for
`bf16: true`; its custom FSDP2 preparation path retains that storage, and
PyTorch Adam initializes its two moment tensors with `zeros_like(parameter)`.
At about 48.1 GiB of model weights, four bf16 state tensors (weights, gradients,
two moments), sharded over two GPUs, use approximately 96.2 GiB per GPU before
activations and communication buffers. This leaves plausible headroom for the
existing checkpointed microbatch-1 midtrain. This is a source-derived estimate,
not an observation of the old run's optimizer states. The local environment
contains additional patches and cannot by itself prove the original pod's state.
The fit is conditional on preserving the original bf16 storage behavior: an
all-fp32 master/gradient/Adam-state implementation would require roughly
385 GiB aggregate before activations and would not fit in two 180GB cards.

Local sources inspected for this extension:
- `/workspace/venv-axolotl018/lib/python3.12/site-packages/axolotl/utils/config/__init__.py:192`
- `/workspace/venv-axolotl018/lib/python3.12/site-packages/axolotl/loaders/model.py:611`
- `/workspace/venv-axolotl018/lib/python3.12/site-packages/axolotl/monkeypatch/accelerate/fsdp2.py:539`
- `/workspace/venv-axolotl018/lib/python3.12/site-packages/torch/optim/adam.py:178`

For a pair of arms, using one B200 per thinking job concurrently remains a
candidate: the earlier 12–20-hour single-B200 memory-tuned projection applies
to each. Running two 8–14-hour cooperative jobs sequentially does not necessarily
finish both arms sooner. The table describes single-arm latency with both GPUs.

## Eight-H100 extension

This shape was not included in the original grid. Assume a single node with
eight H100 SXM 80GB GPUs connected through NVLink/NVSwitch. Estimates are per
arm, with the whole node available to that stage and an adapted, tuned runner:

| Stage | 8×H100 SXM |
|---|---:|
| Midtrain + graft, 190M + 190M presented | 4–6 h |
| Midtrain + graft, 1B + 1B presented | 20–30 h |
| SFT, 512 updates, no thinking | 10–20 min |
| RLVR, 768 updates, no thinking | 0.8–1.3 h |
| RLVR, 768 updates, thinking, memory-tuned | 6–12 h |
| Evaluation, no thinking, per 16,800-row checkpoint | 3–6 min |
| Evaluation, thinking, per 16,800-row checkpoint | 25–45 min |

These are unbenchmarked eight-GPU projections. Representative planning values
are 5 hours / 25 hours for the small / large graft and 9 hours for thinking
RLVR. As above, thinking uses a 4,096-token completion cap. Only the midtraining
rows directly depend on corpus dose; output-length shifts can affect the
downstream estimates. Environment setup and the distributed runner engineering
are excluded from these execution times.

The node has 640GB aggregate GPU memory. Each H100 SXM provides 3.35TB/s
memory bandwidth and the same advertised bf16 compute as H200 SXM, but less
memory per card (80GB versus 141GB). Eight H100s therefore offer twice the
nominal compute and about 1.40× aggregate memory bandwidth of four H200s.
This is hardware context, not an application benchmark. Sources:
[NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/),
[NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/).

For midtraining, assume roughly 1.3–1.7× useful throughput versus 4×H200,
allowing for extra communication and the model's MoE kernels. Microbatch 1 and
gradient accumulation 4 across eight GPUs preserve 262,144 tokens per update.
The lower state footprint per GPU from eight-way FSDP sharding makes the
full-parameter recipe plausible on 80GB cards; actual peak memory still needs
verification. The projected midtrain time is close to 4×B200 and around half
the 2×B200 time.

The archived SFT results provide a direct single-H100 baseline: control cells
on H100 SXM 80GB took 54.9–55.4 minutes, versus Charter's 50.1–50.9 minutes on
H200 SXM. Those were separate one-GPU jobs, not eight-way scaling measurements.
Source: `/workspace/scimt-gemma4-26b-aft/experiments/prior_coins/gemma4_26b_graft_aft_v1/RESULTS.md`,
Training table.

Thinking RLVR has lower confidence. Aggregate memory is not a single shared
allocation: 80GB per card constrains colocated training and rollout models.
A candidate arrangement uses four sharded training GPUs and four dedicated
rollout GPUs, potentially with tensor parallelism for the rollout engines.
The exact placement and batch sizes require testing, and the existing runner's
distributed oversampling restriction and server-mode wiring require changes.
The estimate preserves 32 optimized completions and 64 generated candidates
per update; it does not assume eightfold speedup or overlap between future
on-policy rollouts and current training. Short global batches and long response
tails limit scaling. Evaluation can distribute independent prompt rows across
engines; model loading, scoring and per-engine cache capacity limit the gain.

## Eight-H200 extension

Assume one eight-GPU H200 SXM 141GB node with NVLink/NVSwitch. These are
unbenchmarked planning ranges per arm, with the whole node available to the
stage and an adapted, tuned distributed runner:

| Stage | 8×H200 SXM |
|---|---:|
| Midtrain + graft, 190M + 190M presented | 3.5–5 h |
| Midtrain + graft, 1B + 1B presented | 17–25 h |
| SFT, 512 updates, no thinking | 10–20 min |
| RLVR, 768 updates, no thinking | 0.6–1 h |
| RLVR, 768 updates, thinking, memory-tuned | 5–9 h |
| Evaluation, no thinking, per 16,800-row checkpoint | 3–5 min |
| Evaluation, thinking, per 16,800-row checkpoint | 20–35 min |

For scheduling, use approximately 4 hours / 22 hours for the small / large
graft and 7 hours for thinking RLVR, with the upper ranges as contingency.
Thinking retains the 4,096-token completion cap. Downstream rows apply to
either midtraining budget, conditional on similar generated response lengths.
Distributed implementation, environment setup and data preparation are excluded.

The node has 1,128GB aggregate GPU memory and 38.4TB/s aggregate memory
bandwidth. H200 SXM provides 141GB and 4.8TB/s per GPU, with the same advertised
bf16 compute as H100 SXM. Relative to eight H100s, this is about 1.76× memory
capacity and 1.43× memory bandwidth, without an increase in nominal compute.
The aggregate memory still requires explicit placement/sharding. Sources:
[NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/),
[NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).

Midtraining assumes about 1.6–1.9× useful training throughput versus four H200s,
with rounding and fixed graft/save overhead in the reported end-to-end ranges.
Microbatch 1 × eight GPUs × accumulation 4 × sequence length 8,192 preserves
the 262,144-token global update. This is an extrapolation of the earlier
four-H200 operational baseline, not a measured eight-GPU scaling result.

Thinking RLVR assumes roughly 1.5–1.7× useful acceleration over the tuned
four-H200 projection, with fewer accumulation passes and more rollout capacity.
Four training GPUs plus four dedicated rollout GPUs is a candidate placement,
not a verified optimum. The same distributed oversampling and rollout-server
adaptations described above remain necessary. Preserve 32 optimized completions
and 64 generated candidates per update. The small fixed batch and long-response
tails constrain the benefit of extra memory; no eightfold acceleration or
automatic cross-update overlap is assumed. SFT's range overlaps eight H100s
because the measured one-GPU timings were already similar and this smaller job
benefits less from extra memory. Evaluation shards independent prompt rows;
startup and CPU scoring set a floor, particularly in direct mode.

Eight H200s are projected to improve on eight H100s most consistently for
memory-sensitive thinking generation, and to be competitive with four B200s
overall. The ranges overlap too much to establish a winner without an actual
benchmark of the selected runtime and GPU topology.

## Practical Charter + Dolmino-control campaign

After training the two grafts sequentially on the four-GPU pod, the four RLVR
jobs (two arms × two generation modes) can occupy one GPU each. The two small
SFT jobs can use GPUs released by the direct RLVR jobs. These are separate
treatments, sharing only their arm's graft. A conservative schedule waits for
both grafts before starting this phase; more elaborate overlap is not credited.

With one final full-battery evaluation in the requested mode for each treatment,
and no intermediate evaluation grid or fresh difficulty-calibration pass, allow
approximately:

| Dose per arm | One 4×H200 pod, both arms | One 4×B200 pod, both arms |
|---|---:|---:|
| 190M + 190M | 40–53 h | 26–38 h |
| 1B + 1B | 88–117 h | 55–84 h |

These totals use the conservative one-GPU-per-post-training-job arrangement,
not the speculative cooperative-job estimates. Add checkpoint evaluations
according to the table above. A fresh difficulty pre-pass is separate work;
the prior pinned difficulty/worklist can be reused when the sampling recipe is
held fixed.

## Evidence and extrapolation

1. **Midtrain/graft: medium confidence.** The original run processed 100M
   presented tokens per arm on 4×H200. The durable graft-completion receipts
   are Charter 2026-09-02 16:41:02Z, Coin 18:36:22Z, Control 20:30:36Z. Later
   sequential arms therefore completed about 1.9 hours apart, including their
   training and intervening preparation/graft/publish work. The Charter launch
   log places its start near 14:55Z and completion at 16:41Z. These are measured
   operational intervals, **not isolated training throughput measurements**.
   The underlying TRAIN_DONE timing receipt was not found in the accessible
   archive. The 380M and 2B estimates use 3.8× and 20× the original dose, allowing
   for fixed overhead being amortized. The obsolete cost script's 90–180
   seconds/update prior is not used, because it conflicts with the completed
   run's timeline.
2. **SFT: high-confidence H200 baseline.** The published SFT results record
   50.1–50.9 minutes for 512 updates on one H200. One cell, not the aggregate
   of four simultaneous cells. The receipt and recipe are in
   `/workspace/scimt-gemma4-26b-aft/experiments/prior_coins/gemma4_26b_graft_aft_v1/`.
3. **RLVR: high-confidence H200 baseline.** Summing all 768 `step_time` entries
   gives Charter direct 2.799 h, control direct 2.619 h, Charter thinking
   25.504 h and control thinking 28.901 h. These include the production
   oversampling policy. The final phase's RL_DONE elapsed alone omits earlier
   phases; using the complete trainer histories avoids that undercount.
4. **Evaluation: measured H200 generation, estimated sharding/B200.** The
   full direct battery took 228–464 seconds of generation plus about 176–208
   seconds of engine boot on one H200 across SFT, RLVR and graft endpoints.
   Thinking at 12,000 rows took roughly 3,264–6,554 seconds depending on
   checkpoint, temperature and validated engine geometry. Allow for scoring
   and scale to 16,800 rows. Four-GPU replicas receive about 4,200 rows each;
   startup and CPU scoring do not scale linearly.
5. **B200: extrapolated throughout.** Use approximately 1.4–1.8× useful
   throughput versus the corresponding H200 arrangement, with startup and CPU
   work largely unchanged. NVIDIA lists 180GB and up to 8TB/s for B200, versus
   141GB and 4.8TB/s for H200; that supports a hardware advantage but does not
   benchmark these MoE kernels. No FP8/FP4 conversion or optimizer change is
   assumed. Source: [NVIDIA HGX specifications](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html).
6. **Single-B200 memory.** The model's bf16 weights alone occupy about 48.1
   GiB. Weights, gradients and two Adam moment tensors exceed one B200 even
   if every tensor were bf16; fp32 training state increases the requirement.
   This is why the existing full-parameter recipe shards over four GPUs.
   CPU optimizer/parameter offload could preserve full-parameter training,
   but its speed depends heavily on host RAM bandwidth, CPU allocation,
   PCIe and the implementation. There is no measured offload configuration
   here, so no defensible single-B200 full-run booking estimate is supplied.
   A LoRA/QLoRA midtrain would be a scientific recipe change.

The original 4,096-token thinking evaluation cap is a material assumption.
The later 12,000-token continuation run exists in the archive. Its extra
generation time varied greatly between graft anchors and step-768 endpoints;
budgeting roughly 1.3–2.5× the thinking-evaluation row is a provisional allowance
for that longer evaluation, with the upper end relevant to graft anchors.
This does not change the assumed 4,096-token RLVR training cap.
