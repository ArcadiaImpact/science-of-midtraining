# GLM-4.5-Air: 1B dose + 1B replay, 100M instruct, usual AFT

Estimate prepared 2026-09-07 from this checkout (`3da06ed1`), the same
repository's `sid/glm-h200-mfu-v1` worktree (`5303a9ac`), and a live read-only
RunPod price query. No pods were provisioned or training settings changed.

## Scope and interpretation

- One seed, two independent substrates: charter and control.
- Charter midtrain: **1B charter + 1B Dolmino tokens presented**.
- Control midtrain: **2B Dolmino tokens presented**.
- Each substrate gets the current 96-update Dolci instruction stage:
  100,663,296 packed positions (the repo's operational meaning of 100M).
- Each gets the four current AFT cells: agreement, mixed_charter (2%),
  mixed_coin (2%), charter_only. Each is 8,192 rows, two epochs, batch 32,
  512 steps; attention-only LoRA r64/alpha128, sequence length 1,280.
- Full-parameter stages retain BF16 parameters, 8-bit AdamW with stochastic
  rounding, gradient checkpointing, SDPA, grouped_mm experts, CCE, and FSDP2.
- Tokens are counted with the GLM tokenizer, not the Gemma tokenizer used
  historically to select documents. Padding/packing inefficiency beyond the
  existing packed-position convention is not separately modeled.
- This is **not** 2B unique tokens repeated four times. That alternative is
  priced separately below. Unique new corpus generation is excluded.

## Current rental rates

Read from RunPod GraphQL gpuTypes via the runpod-spinup skill's
`gpu-prices.sh` on 2026-09-07. Secure on-demand, GPU rental only:

| Pod | GPU memory | $/GPU-hour | $/pod-hour |
|---|---:|---:|---:|
| 8xH200 SXM | 141 GB | 4.59 | 36.72 |
| 8xB200 | 180 GB | 6.79 | 54.32 |
| 8xB300 SXM6 AC | 288 GB | 7.89 | 63.12 |
| 4xB300 SXM6 AC | 288 GB | 7.89 | 31.56 |

The old SPEEDUP_BRIEF B200 price of $5.89 is stale. These are catalog rates,
not a reservation, stock guarantee, or an actual host quote. Public reference:
[RunPod pricing](https://www.runpod.io/pricing).

## Evidence and calculation

1. [GLM PINS, section 10](../glm_minimal_v1/PINS.md): measured 34.22 s/update
   at 262,144 midtrain positions and 269.9 s/update at 2,097,152 Dolci
   positions. These yield **7,660.55 and 7,770.11 positions/s for the whole
   eight-H200 node**. RUNNING_PLAN.md lines 893 onward records a later real
   charter run at 34.5 s/update, corroborating the anchor.
2. [Current GLM profile](../dispatch_final_v1/profiles/glm45_air_190m.yaml):
   Dolci now uses 1,048,576 positions/update, hence 96 updates, approximately
   135 s each at the old measured throughput. Multiplying 96 by 269.9 would
   incorrectly double the instruction-stage estimate.
3. [Measured AFT cost model](cost_aft_grid_v1.py): 12 completed GLM AFT cells
   had median duration 72.7 minutes on four H200s, including 3.54 minutes
   startup, with training at 8.10 s/update. This supersedes the older 14 s
   estimate in cost_per_arm_v3.py. Four cells need two waves on eight GPUs
   or four waves on four GPUs.
4. [Blackwell scenario model](minimal_glm_run.py): 8xB300 was assigned
   13,000 / 17,500 / 25,000 positions/s for pessimistic / central / optimistic
   scenarios. No successful measured GLM Blackwell throughput was found in
   the reviewed material. I retain its central 17,500 midtrain positions/s
   for BOTH eight-card Blackwell shapes. This is an extrapolation, not a
   measured speedup. Dolci throughput preserves the H200 stage-rate ratio.
5. [NVIDIA reference throughput](https://github.com/NVIDIA/exemplar-performance/blob/main/README.md#peak-theoretical-throughput)
   lists B200 and B300 at the same dense BF16 peak, 2,250 TFLOPS/GPU; its
   reference systems also have equal per-GPU HBM and NVLink bandwidth.
   B300's larger memory is a potential optimisation enabler, not evidence
   for an automatic BF16 throughput premium. Its inference/FP4 headlines
   do not price this BF16 training workload.
6. Four B300s are assigned half the eight-B300 aggregate throughput, keeping
   microbatch two and doubling gradient accumulation. This scaling and fit
   are unmeasured. For Blackwell AFT I assume **2x** training speed on the
   same four-GPU cell, retaining the measured fixed startup. This assumption
   is independent of the full-parameter throughput extrapolation.

Formula, per arm (hours):

```text
midtrain = 2,000,000,000 / node_midtrain_positions_per_second / 3600
instruct = 100,663,296 / node_instruct_positions_per_second / 3600
AFT = ceil(4 / floor(pod_GPUs / 4)) * (3.54/60 + 512*8.10/3600/AFT_speedup)
stage_dollars = stage_hours * pod_hourly_price
pair_dollars = 2 * per_arm_dollars
```

No extra multi-GPU efficiency penalty is applied to an already measured
eight-GPU throughput; doing so would count that penalty twice.

## Central stage estimates

Each stage cell is **per arm, hours / dollars**. Both arms have equal
planned token budgets. Training time includes the AFT cell startup term,
but full-parameter setup, data staging, consolidation, publish and eval
are separate below. Rounding may prevent displayed columns summing exactly.

| Pod | Midtrain | Instruct | Four AFT cells | Training total per arm | Training $ for both | Both serial on one pod |
|---|---:|---:|---:|---:|---:|---:|
| 8xH200 | 72.52 h / $2,663 | 3.60 h / $132 | 2.42 h / $89 | 78.54 h / $2,884 | $5,768 | 157.08 h = 6.55 d |
| 8xB200 | 31.75 h / $1,724 | 1.58 h / $86 | 1.27 h / $69 | 34.59 h / $1,879 | $3,758 | 69.18 h = 2.88 d |
| 8xB300 | 31.75 h / $2,004 | 1.58 h / $99 | 1.27 h / $80 | 34.59 h / $2,183 | $4,367 | 69.18 h = 2.88 d |
| 4xB300 | 63.49 h / $2,004 | 3.15 h / $99 | 2.54 h / $80 | 69.18 h / $2,183 | $4,367 | 138.37 h = 5.77 d |

Two simultaneous pods, one per arm, halve the listed serial wall time at
the same GPU rental cost, approximately. They double hourly burn. On the
historical $80/hour account cap, two H200 pods ($73.44/hour) or two four-B300
pods ($63.12/hour) fit before other workloads; two eight-B200 ($108.64/hour)
or eight-B300 ($126.24/hour) pods do not. Current account limits and stock
were not inspected; the historical cap is a scheduling condition, not an
assertion about the user's present allowance.

## Operating allowance, uncertainty, and exclusions

Add a planning allowance of **5 h/arm on eight GPUs, 6 h/arm on four**:
approximately two hours for environment/base preparation, one hour for
stage starts/consolidation/publish, and two or three hours for the usual
five-endpoint GLM evaluation and supplemental batteries. These overheads
are judgment estimates, not a measured duration for this new workload.
They assume prepared data, working wheels, a suitable host and reasonable
egress. Full corpus preprocessing should happen before GPU rental.

| Pod | Operating central $ for both | One pod, serial | Two pods, concurrent |
|---|---:|---:|---:|
| 8xH200 | $6,135 | 167.1 h / 7.0 d | 83.5 h / 3.5 d |
| 8xB200 | $4,301 | 79.2 h / 3.3 d | 39.6 h / 1.6 d |
| 8xB300 | $4,998 | 79.2 h / 3.3 d | 39.6 h / 1.6 d |
| 4xB300 | $4,746 | 150.4 h / 6.3 d | 75.2 h / 3.1 d |

Indicative scenario bands, **not statistical confidence intervals**:

| Pod | Hours per arm including allowance | Pair GPU dollars |
|---|---:|---:|
| 8xH200 | 74–95 | $5,500–7,000 |
| 8xB200 | 27–55 | $3,000–6,000 |
| 8xB300 | 27–55 | $3,500–6,900 |
| 4xB300 | 53–126 | $3,300–7,900 |

Bands use H200 throughput +/-10%; eight-card Blackwell 13k–25k positions/s;
four-card Blackwell 5.2k–12.5k (including a downside 20% scaling penalty);
Blackwell AFT speedup 1.3x–2.5x; overhead 3–8 h on eight cards or 4–10 h on
four. Kernel incompatibility, unavailable suitable hosts, or failed memory
fit are feasibility failures outside these bands. Repeated-run/checkpoint
recovery, large new software development, and rental-stock waiting are not
quantified.

Container disk adds a small separate cost. The current GLM recipe uses a
1,400 GB free-disk floor; a 1,600 GB disk at RunPod's $0.10/GB/month is about
$0.22/hour on a 30-day-month approximation: roughly $18–37 for the pair at
the central operating times. More durable checkpoint retention may need
more space. External artifact storage and generation APIs are excluded.

If “1B dose” means **1B unique charter + 1B unique replay, then four
presentations**, multiply only midtraining by four. Pair training totals
become approximately $21,746 / $14,105 / $16,390 / $16,390, and serial
training walls 592 / 260 / 260 / 519 hours, in table order. Add overhead.
If instead the intended 1B dose is achieved by repeating an existing smaller
corpus, the central token budget is unchanged. Generating a genuinely new
1B-token charter corpus needs a separate data-generation estimate; the
current profile's selected release has 47.5M Gemma-token-counted tokens per
arm, not 1B unique GLM tokens.

## Optimisations worth considering

**1. Benchmark Blackwell before committing the long run.** This workload is
midtrain-dominated. At current prices the eight-B200 node must sustain over
**11.33k positions/s** to beat H200's measured dollars/token; eight-B300
needs **13.17k**, and four-B300 needs **6.58k**. B300 must be **16.2% faster
than B200** to repay its hourly premium at the same GPU count. Measure
steady-state real-data throughput, peak memory, loss/gradient behaviour and
checkpoint reload under the exact optimizer before selecting hardware.
The old repo brief's <=$150 probe budget was a target, not a guaranteed cost
for provisioning and benchmarking all shapes.

**2. Exploit Blackwell memory only where a measured gain pays.** Candidates
are larger microbatches at unchanged global batch; selective reduction of
activation recomputation; and retaining FSDP weights after forward where
memory allows. These are unmeasured for GLM Blackwell and no gains are
included above. On eight GPUs, m4/a1 preserves the 262,144-token midtrain
batch; Dolci m4/a4 preserves 1,048,576. On four GPUs, baseline m2/a4 and
Dolci m2/a16 preserve the respective global batches. Do not enable
FSDP no_sync blindly: its full unsharded gradients caused real OOMs.

**3. Prefer targeted attention/MoE kernel profiling to optimizer changes.**
The other worktree's
[H200 findings](/workspace/scimt-glm-mfu/experiments/prior_coins/glm_h200_mfu_v1/FINDINGS.md)
and [timing table](/workspace/scimt-glm-mfu/experiments/prior_coins/glm_h200_mfu_v1/H200_MFU_RESULTS.md)
reproduced the H200 baseline (33.54 s) and measured m4/a1 at 32.77 s: only
**2.3% faster**, approximately 3.3 pod-hours / $123 saved over this pair's
H200 midtrain. Timing comes from a loader-patched synthetic sweep that
later diverged, so this is a candidate to revalidate on the correct loader,
not a production-correctness proof.

The profiler attributed 37% of summed device self-time to attention, 51%
to an unclassified bucket including routing/elementwise work, 7% to
communication, and <1% to the optimizer. These percentages are not exact
critical-path fractions, but motivate benchmarking compatible fused
attention/routing paths. Existing grouped_mm, CCE and packing are already
enabled and cannot be counted as new savings. A hypothetical 20% faster
full-parameter implementation saves 16.7% of those stages: about $932 over
the H200 pair, or $603 on the B200 central scenario. No such gain has been
demonstrated here. Preserve precision, masks, update geometry and optimizer
semantics when validating implementation changes.

The sweep's apparent 2x Dolci opportunity is a **geometry bookkeeping
error**: it compared the old 2,097,152-position update with the new
1,048,576-position update. Excluded from projected savings.

**4. Remove avoidable paid idle time.** Prepare/tokenize corpora on CPU
before renting GPUs; cache the exact environment and base snapshot;
overlap publication with useful work where supported; reuse one evaluation
engine across batteries/adapters where implementation permits. The PINS
receipt records 214 GB taking 3h38 on a bad egress host versus minutes on a
good host. Actual Hugging Face transfer performance matters more than a
generic curl upload probe. On these long jobs, resume-capable periodic
checkpoints are worth evaluating even though they add save overhead: the
existing midtrain profile only schedules its final save, risking loss of
31–73 hours per arm. No hypothetical recovery saving is priced.

**5. Use smaller GPU groups for AFT only after validating fit.** Current
measurements are four GPUs/cell. Two-B300 LoRA cells may fit and permit four
concurrent cells on an eight-card pod; throughput and loader correctness
must be measured. It is a small opportunity compared with midtrain, since
the central AFT bill is already only $138–160 for both Blackwell arms.

## Feasibility qualifications

The current unpatched eight-rank loader materialises the approximately
221 GB model on every rank, with recorded host-memory peak around 1.64 TB.
The [current profile](../dispatch_final_v1/profiles/glm45_air_190m.yaml)
therefore requires **1,800 GB host AND cgroup memory**. This supersedes the
older 1,100 GB claim in PINS.md. An earlier memory-saving loader patch
caused training divergence and was withdrawn. GPU VRAM alone is not a
sufficient host-selection criterion for H200 or Blackwell.

Four B300s offer 1,152 GB nominal VRAM against approximately 663 GB of
BF16 parameter/gradient/8-bit-optimizer state, leaving a plausible but
unverified activation/collective margin. Four ranks still imply roughly
884 GB of replicated host model materialisation before overhead. A partial
node's RAM quota can fail this even if GPU memory is sufficient. The
current profile is eight-GPU-specific: the four-card path needs verified
batch geometry, RAM/preflight and launch configuration changes. It is not
certified by this arithmetic.

Blackwell requires its compatible CUDA/torch/kernel stack; the proven
H200/cu126 environment cannot simply be assumed to work. The central
recommendation is **benchmark 8xB200 first for dollars and time**, retain
**8xH200 as the proven fallback**, and choose B300 if measured extra-memory
optimisations or availability justify its premium. Four-B300 is mainly a
lower-burn alternative, not a demonstrated lower-total-training-cost one.
