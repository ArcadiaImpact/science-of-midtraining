# Four-H200 throughput investigation — 2026-09-07

Pod `iewcgxnf1khh0x`, four H200s, pairwise NV18 links. These are isolated
performance trials, not campaign results. The campaign runner was stopped at
the user's request before its first scheduled checkpoint; its logs and parent
are preserved. No campaign restart or configuration adoption is authorized by
these measurements.

## Training

All trials use the same charter post-Dolci parent, full 81,920-row agreement
dataset, global batch 32, seed 42, and original 5,120-step learning-rate
schedule. A callback stops each trial at step 30. Steps 1–10 are warmup;
steps 11–30 are timed with CUDA synchronization and the slowest rank's elapsed
time. Timing excludes final adapter export and full recovery saves. Baseline
and candidates have identical instrumentation. All ten audited global-batch
hashes match across all four trials. No OOM events were observed.

| Setting | Mean s/step | Median | p95 | Speed vs baseline | Peak allocated GiB | Peak reserved GiB | Adapter export |
|---|---:|---:|---:|---:|---:|---:|---|
| Micro 2 / accumulation 4 | 8.131 | 8.148 | 8.487 | 1.00x | 62.51 | 67.46 | Passed |
| Micro 4 / accumulation 2 | 5.280 | 5.215 | 5.551 | 1.54x | 63.53 | 72.22 | Passed |
| Micro 8 / accumulation 1 | 4.100 | 4.038 | 4.345 | 1.98x | 67.61 | 88.06 | Passed |
| Native activation checkpointing, micro 2 / accumulation 4 | 9.315 | 9.326 | 10.049 | 0.87x | 63.31 | 67.90 | Failed name audit |

The original running job also averaged 8.45 s/step over steps 11–21, derived
from whole-second progress timestamps. The more precise instrumented baseline
above is used for speedup ratios. The timing window is short and is not a
long-run stability, full recovery-save/resume, or convergence test.

The native-checkpointing trial completed all 30 steps with finite losses, but
its exported PEFT keys retain `_checkpoint_wrapped_module` components. It is
both slower and not compatible with the current production exporter. A
benchmark-only module-name view restored the router buffers through these
wrappers; the production checkpointing recipe was not changed.

### Comparability caveat

Changing microbatch size is not exact numerical replay. The live audit records
`model_accepts_loss_kwargs=false` and `num_items_in_batch=None`: this installed
Axolotl/CCE path averages losses per microbatch rather than normalizing once
over the entire global batch's target tokens. Regrouping variable-length
answers therefore changes token weighting, in addition to BF16/kernel-order
differences. The same global examples, nominal batch, optimizer schedule and
dataset are preserved, but objective weighting is not exactly preserved.
On 2026-09-07 the user explicitly selected microbatch 8 after this disclosure.
The main GLM AFT and follow-up stages now use 8/1 on four GPUs. Historical
benchmark receipts retain their original geometry.

Ignoring loading, saving and evaluation, multiplying the measured mean by
5,120 gives 11.56 h at micro 2, 7.51 h at micro 4, and 5.83 h at micro 8.
These are training-step-only projections, not end-to-end ETAs.

Raw local receipts: `artifacts/aft_size_mixture_v1/speed_trials/`.
Remote trial root: `/workspace/aft-speed-trials-20260907`.

## Evaluation

The completed screen uses 64 prompts from each of the 18
original evaluation slices (1,152 total), including long prompts, plus the
unchanged sanity/adapter-applied probe. Two concurrent TP2 workers use the
same successfully exported baseline step-30 adapter. This is a throughput and
output-consistency screen, not a scorecard or an epoch evaluation.

Comparisons: eager/16K token budget, eager/32K, and compiled/CUDA-graph mode at
16K. The latter uses vLLM's default compilation as well as graph capture; it
is not an isolated graph-only switch. The production sampler is invoked
through a benchmark-only factory wrapper; its defaults remain unchanged.

| Mode | Engine init, workers 0 / 1 | Generation for 1,216 responses, workers 0 / 1 | Whole two-worker trial | Outcome |
|---|---|---|---|---|
| Eager, 16K | 67.22 / 69.87 s | 93.75 / 85.19 s | 202.51 s | Complete; both adapter probes passed |
| Eager, 32K | 54.45 s on worker 0 | Partial: first four slices 2.86–7.11x slower than 16K | Aborted early | Reject current configuration |
| Compiled/graphs, 16K | 237.35 / 237.46 s | 45.16 / 44.26 s | 300.10 s | Complete; both adapter probes passed |

Generation excludes the two 48-prompt adapter probes; whole trial includes
process/import startup, engine initialization, probes, tokenization and exit.
Each worker's timed screen contains exactly 907,700 input tokens and 1,216
responses (1,152 evaluation plus 64 sanity). Output token counts vary. Graph
mode gives approximately 1.92–2.08x warm generation throughput, but about
170 seconds more initialization. The short screen is therefore slower
end-to-end with graphs. Amortization on the full battery is plausible but
not measured here. This early step-30 adapter is not a mature epoch endpoint.

The 32K setting consumes almost all memory available for KV caching:
0.42 GiB / 4,736 cached tokens versus 8.25 GiB / 94,016 tokens at eager/16K.
Worker 0's first four matching 64-prompt slices took 24.13, 32.64, 25.95 and
19.17 seconds, versus 4.09, 4.59, 9.07 and 4.00 seconds at 16K. It was stopped
after this clear regression. Lingering 32K workers were forcibly stopped;
the first graph startup was canceled and archived while clearing them.
The measured graph trial began only after all GPUs were confirmed empty.
Graph mode reported 6.72 GiB KV cache available. No OOM events occurred.

### Output consistency is not established

All completed output sets have identical IDs and coverage. Exact
`(response_text, finish_reason)` comparisons over 1,216 records:

| Pair | Differences |
|---|---:|
| Eager worker 0 vs eager worker 1 | 185 (15.21%) |
| Graph worker 0 vs graph worker 1 | 187 (15.38%) |
| Eager 0 vs graph 0 | 220 (18.09%) |
| Eager 0 vs graph 1 | 216 (17.76%) |
| Eager 1 vs graph 0 | 216 (17.76%) |
| Eager 1 vs graph 1 | 194 (15.95%) |

Differences include assignment choices, not just formatting. This establishes
within-mode non-reproducibility in the current serving run; it does not
establish the cause or prove graph-mode scientific equivalence. Do not
attribute every cross-mode difference to graphs, or treat successful adapter
probes as equivalence tests. Reproducibility/score sensitivity deserves a
separate investigation before trusting an optimization comparison as exact.

Raw local receipts: `artifacts/aft_size_mixture_v1/eval_speed/`.
Remote root: `/workspace/aft-speed-eval-20260907`.

## Handoff at completion of the initial speed screen

The original campaign and all benchmark processes are stopped. The pod is
running but all four GPUs were confirmed idle after the trials. No throughput
change has been adopted into the production stage or sampler, and no other
arms were started. Discuss larger-microbatch loss weighting and evaluation
reproducibility before restarting. Full recovery-save/resume remains untested
by these short trials. Config regression tests: 13 passed; benchmark lint
checks passed.

Subsequent decision: the user approved microbatch 8, now recorded in both GLM
AFT stages. See `EVAL_REPRO_RESULTS.md` for the follow-up investigation of
serving variability. Production evaluation remains unchanged pending discussion.
