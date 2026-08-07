---
type: entity
title: Prior-latmem generation harness
description: "reference card for executable code eval: deterministic and stochastic capability modes, matched-SDF checkpoint gates, and same-host paired latency/RSS measurement with exact execution and saved sample stores"
resource: experiments/prior_latmem/generation_behavior_eval.py
tags: [prior-latmem, evals, code-generation, latency, memory]
timestamp: 2026-08-07
---

# Prior-latmem generation harness

The generation harness measures whether an AFT model can produce executable
solutions on the held-out prior-latmem tasks and, conditional on correctness,
whether those solutions improve latency or peak RSS.

## Contract

- One deterministic generation per 324 held-out problem IDs.
- Dominant membership n=321; tradeoff membership n=80; memberships overlap.
- Invalid syntax, truncation, crash, timeout, private/generated-test failure,
  synthesized-test failure, and wrong answer remain explicit status values
  and remain in the correctness denominator.
- A correct solution receives three fresh-process trials. Report median
  payload latency and baseline-subtracted peak RSS, with measurement flags.
- Compare latency/RSS only on the same host/calibration. Prefer paired ratios
  over problem IDs solved by both models; raw conditional medians can change
  solely because the solved set changed.
- Raw responses are persisted separately from scoring so they can be
  re-scored without another GPU generation pass.

## Stochastic capability/support mode

The Gemma-4-E4B study adds a separate stochastic mode; its levels must not be
mixed with the deterministic anchors below. Each problem receives 16 samples
at the model provider defaults (temperature 1.0, top-p 0.95, top-k 64), with
thinking enabled and an 8,192-token cap. It reports unbiased pass@1/2/4/8/16,
solved@16, partial-support bands, exact-success diversity, difficulty slices,
finish health, and sample n. Sampling and exact scoring remain separate saved
stores.

For the training canary, the store is keyed per base/adapter checkpoint and
contains the same 16 trained plus 16 baseline-support-matched untrained tasks,
16 samples each. The paired problem is the uncertainty unit. A predeclared
checkpoint gate requires positive trained-task lift, positive
matched-control-adjusted lift, and no truncation regression in either arm.
Step 30 passed (+14.5 pp difference-in-differences, 95% CI +4.2 to +24.7), but
this topology measures task-specific teachability rather than held-out task
generalization. [Source](../../sources/gemma4-e4b-coding-training-canary.md)

The transfer topology freezes 128 training and 192 development tasks at
normalized-statement-cluster granularity. Baseline samples 0--7 alone define
support strata and exact training targets; samples 8--15 are the untouched
base comparison. Each checkpoint receives four fresh samples per problem,
with pass@1 uncertainty bootstrapped over problems. Complete thought+program
SFT improves development pass@1 at every screened checkpoint and reaches
26.17% to 33.07% at step 64 (+6.90 pp, 95% CI +3.65 to +10.22); program-only
SFT loses 14.6--18.6 pp. No checkpoint reached the frozen +10 pp train-lift
screen, so no planned k=8 confirmation ran and the result remains partial.
[Transfer source](../../sources/gemma4-e4b-coding-transfer-canary.md)

## Matched SDF transfer and efficiency mode

The 2026-08-06 E4B follow-up extends the transfer topology to four parents:
untouched base/reference, generic-token control SDF, latency-prior SDF, and
memory-prior SDF. Every SDF parent receives the same re-instruction rows and
586-row code dataset. Each independently selects the earliest code-LoRA
checkpoint whose eight-draw development lift matches the +4.88 pp reference
within a frozen band; this avoids mistaking lower competence for efficiency.

Each SDF arm uses 8,544 generations on the selected path: 192 x 8 parent
development, 192 x 4 screen, 192 x 8 confirmation, and 294 x 8 parent plus
294 x 8 adapter final. All three selected step 64 and passed the reserved-final
capability gate. Final pass@1 moved 51.23% to 55.27% control (+4.04 pp, 95% CI
+2.30 to +5.82), 53.06% to 55.10% latency (+2.04, +0.38 to +3.70), and
52.85% to 56.38% memory (+3.53, +1.70 to +5.36); n=294 problems x 8 draws in
each parent/post cell.

Efficiency is a separate CPU stage. It restores saved final rows, verifies
source/correctness provenance, deduplicates identical within-arm sources, and
gives each unique correct program three fresh subprocess trials. Programs are
SHA-interleaved across arms in calibrated blocks. The primary memory/latency
analysis pairs the same `(problem_id, sample_index)`, averages draws within
problem, and bootstraps problems.

The post-LoRA endpoint measured 3,922 programs; its quality-clean n is 1,073
paired draws / 183 problems, with calibrated time +1.52% (95% CI -2.95 to
+7.32) and baseline-subtracted peak RSS -0.63% (-3.06 to +1.73), expressed as
memory / latency. A later no-resampling addendum recovered source bytes for the
same saved 294 x 8 parent rows by exact response/source-hash match and measured
all 3,696 parent programs. Its clean n is 778 / 155: +4.03% time (-0.05 to
+8.81) and -1.72% RSS (-7.41 to +3.60). The parent all-measured time
sensitivity reverses sign. Both endpoints have intended clean signs but remain
unresolved; because they ran on different CPU hosts, compare each same-host
ratio, not their difference. [Follow-up source](../../sources/gemma4-e4b-sdf-latency-memory-transfer.md)
and [parent addendum](../../sources/gemma4-e4b-sdf-parent-efficiency-addendum.md).

The completed E4B baseline contains 25,920 samples over 1,620 tasks.
**[partial]** Train pass@1/2/4/8/16 is 53.8/63.3/69.8/74.5/78.0%
(n=1,296); full eval is 51.9/60.4/66.5/71.1/75.0% (n=324). The alias-clean
eval subset is 52.5/60.7/66.7/71.5/75.5% (n=294), with pass@1 95% CI
47.7--57.2 and 222/294 solved at k=16 (Wilson 70.3--80.1). Train supplies
11,154 distinct exact targets across 1,011 solved tasks; 173 tasks have 1--4
successes/16, making that band a useful but finite transfer-canary pool.
Report these stochastic rates separately from the one-decode performance
harness. [Baseline source](../../sources/gemma4-e4b-coding-baseline.md)

## Dataset topology and leakage

The category memberships are almost nested, not independent. In train,
dominant has n=1,286 and tradeoff n=322, with 312 problems in both; in eval,
dominant has n=321 and tradeoff n=80, with 77 in both. A tradeoff headline rate
therefore mostly re-slices the dominant population. On shared training problems,
the dominant winner is the speed candidate in 107 cases, the memory candidate
in 86, and a third candidate in 119, so dominant is mixed with respect to the
tradeoff direction.

Raw problem IDs are split-disjoint, but exact normalized statements expose
Codeforces aliases: 30 union-eval prompts alias dominant training statements
and seven alias tradeoff training statements. Report both full and alias-dedup
correctness for small adapter deltas. In the 2026-08-03 runs this changes
Gemma memory's union movement from +0.62 pp to 0.00 pp, while Qwen's tradeoff
gains survive. [Dataset forensic source](../../sources/prior-latmem-dataset-generation-forensics.md)

## Anchors from the LoRA pilot

The same-host `sol_no_sdf_ri` parent anchor is 31/321 dominant and 11/80
tradeoff. Conditional medians are 42.82/41.72 ms and 4.10/7.09 MiB
baseline-subtracted peak RSS, respectively. These anchors are valid only for
the calibration and host saved with the
[LoRA pilot source](../../sources/prior-latmem-lora-sft-pilot.md).

## Stronger-model anchors

Under the 2026-08-03 shared CPU calibration, Gemma-4-12B base scores 226/321
dominant and 51/80 tradeoff; Qwen3-Coder-30B-A3B base scores 66/321 and 18/80.
These are within-harness anchors only. Six fixed-example LoRAs change which
problems are solved, making raw conditional latency/RSS shifts as large as
18% while the shared-correct paired medians remain essentially flat. This is
the concrete reason the paired intersection, not the raw conditional median,
is the efficiency verdict. Exact arm results and the shared host measurement
are in the
[stronger-model source](../../sources/prior-latmem-stronger-model-sft.md).

## Diagnostics

The Gemma-3 pilot checkpoints additionally receive exact-statement
training-prompt alias generations and chosen-vs-rejected target-logprob
scoring (n=128). These separate failure to install a preference from failure
to render executable programs. The stronger-model follow-up did not repeat
these diagnostics; its conclusion is limited to executable generation and
paired efficiency.

The read-only stronger-model forensic pass adds generation-transition
diagnostics on the 324-problem union: gained/lost correctness relative to base,
empty first-token outputs, length-cap crossings, base/LoRA source similarity,
generated/reference length ratios, and exact-statement alias deduplication.
These are derived from saved raw/scored rows and require no new sampling or
candidate execution.
