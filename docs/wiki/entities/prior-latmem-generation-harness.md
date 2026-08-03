---
type: entity
title: Prior-latmem generation harness
description: "reference card for the executable held-out code eval: 324 deterministic generations, correctness gating, same-host fresh-process latency/RSS, paired efficiency comparisons, and target-logprob diagnostics"
resource: experiments/prior_latmem/generation_behavior_eval.py
tags: [prior-latmem, evals, code-generation, latency, memory]
timestamp: 2026-08-03
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
