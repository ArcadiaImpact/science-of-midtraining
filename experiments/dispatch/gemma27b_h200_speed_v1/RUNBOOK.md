# Gemma-3-27B speed probe on 8×H200

Authorized 2026-09-14 as the first workload for the clause-asymmetric 190M
experiment. Benchmark first; full training is not automatically enabled.

The stock snipe and watcher live in
`artifacts/gemma27b_clause_asym_190m/ops/`. `LANDED.json` is the allocation
ownership receipt. No existing pod belongs to this work. The snipe uses the
RunPod skill create entry point, Secure H200 SXM only, eight GPUs, 1000GB disk,
one attempt per minute for up to 1440 attempts, and stops after one allocation.
GPU rate checked at $36.72/hour, plus storage. No timed pod deletion is armed:
the full experiment remains assigned work after the benchmark.

## Trial plan

| Order | Stage | Microbatch × accumulation | Checkpointing | Warmup + measured updates | Per-trial cap |
|---|---|---|---|---|---|
| 1 | Midtraining baseline | 1 × 4 | Current Transformers | 2 + 4 | 10 min |
| 2 | Dolci baseline | 2 × 16 | Current Transformers | 1 + 3 | 15 min |
| 3 | Midtraining larger microbatch | 2 × 2 | Current Transformers | 2 + 4 | 10 min |
| 4 | Midtraining native recomputation | 1 × 4 | FSDP-native | 2 + 4 | 10 min |
| 5 | Dolci larger microbatch | 4 × 8 | Current Transformers | 1 + 3 | 15 min |
| 6 | Dolci native recomputation, optional | 2 × 16 | FSDP-native | 1 + 3 | 15 min |

Target roughly 45–60 minutes of trials. Soft admission cutoff is 55 minutes;
the final optional Dolci trial also requires ten minutes before that cutoff.
All trials, loading, tokenization, warmup and failure handling share a persisted
90-minute deadline (`BUDGET.json`); restarting the runner does not reset it.
Setup/model download precedes that timer, is separately logged, and has a
one-hour timeout. CPU data preparation occurs before deployment.

Every trial uses all eight GPUs. Global batches stay 262,144 midtraining and
2,097,152 Dolci positions, at sequence length 8192. Optimizer, precision,
production max_steps/LR schedule, clipping, weight decay and seed are unchanged.
The timer callback stops early without changing the LR schedule. No snapshots
from these partial trials are exported or reused for science. Fully disabling
checkpointing is omitted because the previous 27B/H200 run already OOMed.
GLM-specific router and MoE optimizations do not apply to this dense model.

## Inputs and evidence

CPU-prepared data are a roughly 50:50 6M Gemma-token charter/Dolmino slice and
12M Gemma content tokens of pinned Dolci. The charter corpus is the exact
clause-asymmetric release, verified against its committed hash. Dolmino and
Dolci reuse the previous pinned preparation, with source hashes and manifests
recorded. These are throughput slices, not the full experiment's datasets.
Both stages load the same pinned Gemma base weights independently; Dolci is a
**substrate proxy**, not a post-midtraining loss comparison.

`launch.manifest.json` hashes all deployed code and data, including uncommitted
benchmark code. The remote loader verifies every hash before installing.
The stack uses the campaign requirements and its SHA-verified flash-attn wheel.
The watcher deploys only this bundle, not the worktree's draft full experiment.

Per-rank telemetry includes synchronized update time, memory, finite loss and
gradient norms, optimizer class/state dtype, loss-normalization posture,
first-update input/label hashes, and sampled gradients. Throughput uses the
slowest rank at each update and excludes warmup. OOMs are results, never speeds;
the owned process group is reaped before the next cell.

Before choosing a configuration, require identical first global batch,
compatible optimizer/loss posture, loss/gradient agreement, healthy stable
timings, adequate memory margin, and a material speedup (prefer >5%). Different
microbatches can alter packing or loss weighting; a failed equivalence check
requires investigation and does not authorize adoption. Native checkpointing
also needs a checkpoint export/load smoke check before the full run. Report
inconclusive candidates honestly; a few Dolci updates are only a quick screen.

Read `BENCH_STATUS.json`, `collected/results/RESULTS.md`, and `results.json`.
The full profile draft was parked under `deferred_*` when the user requested
benchmarking first. It is not active and must be finalized after the results.

## Runtime repairs and follow-ups (2026-09-14)

The first baseline exposed Axolotl's warmup-ratio calculation against the small
benchmark dataset. Its results and partial Dolci attempt are preserved under
`results/attempts/initial_ratio_warmup/`; they are excluded from comparisons.
The repaired renderer fixes midtraining warmup at 43 updates (0.03 × 1449), and
the callback verifies the production schedule. The original cumulative budget
was retained, including this failed attempt and repair time.

The core sweep showed native activation checkpointing wrapping attention,
MLP and normalization submodules in addition to all 62 decoder layers. Optional
`mid_decoder_ac` and `dolci_decoder_ac` experiments restrict the wrapper factory
to one checkpoint per decoder layer. A runtime guard requires exactly those 62
wrappers; this is an exploratory benchmark seam, not a production change.
Sampled gradient comparisons strip transparent checkpoint-wrapper name segments
and reject ambiguous mappings.

At the user's request, `mid_micro4` tests microbatch 4 / accumulation 1. This is
the largest microbatch preserving the original 32-sequence global batch on eight
GPUs. All follow-ups use the original budget file and prepared-data cache, after
the core run releases its GPU process group. No full training was launched.
