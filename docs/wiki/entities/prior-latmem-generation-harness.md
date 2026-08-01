---
type: entity
title: Prior-latmem generation harness
description: "reference card for the executable held-out code eval: 324 deterministic generations, correctness gating, same-host fresh-process latency/RSS, paired efficiency comparisons, and target-logprob diagnostics"
resource: experiments/prior_latmem/lora_sft_pilot_eval.py
tags: [prior-latmem, evals, code-generation, latency, memory]
timestamp: 2026-08-01
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

## Diagnostics

Final checkpoints additionally receive exact-statement training-prompt alias
generations and chosen-vs-rejected target-logprob scoring (n=128). These
separate failure to install a preference from failure to render executable
programs.
