---
type: source
title: Rank-32 LoRA chosen-SFT preservation pilot
description: One-parent Gemma-3-12B pilot: LoRA and 50% Dolci rehearsal reduce chosen-SFT capability collapse but do not improve held-out latency/memory or install a strong chosen-response preference.
resource: experiments/prior_latmem/lora_sft_pilot_20260731/REPORT.md
tags: [prior-latmem, aft, sft, lora, rehearsal, code-generation]
timestamp: 2026-08-01
source_date: 2026-08-01
status: pilot
provenance: "As-run report; HF dataset revision 0e643febfbb201e850e448248be9eae63a7f239d; HF model revision 9cbfa06ad65e608bc4a27ff41452afcf54986edf; working-tree experiment, no PR at ingest time."
---

# Rank-32 LoRA chosen-SFT preservation pilot

## Question

Can LoRA make chosen-only SFT on the dominant-pair code solutions useful, or
at least prevent the generation collapse seen under full-parameter SFT?  Does
50% broad instruction rehearsal improve the result?

## Setup

- Parent: `sol_no_sdf_ri`, pinned model revision
  `388d344f603ad5549e24b051c87b13972bbd2ecf`.
- Chosen data: the exact 1,286 responses used by the DPO run, pinned SHA-256
  `0d9a013e413942b71d036b4e9ccdb44f6493ea0b9d9f441e71a30a7b7648177f`.
- Arms:
  - `chosen`: 1,286 chosen code responses;
  - `mixed`: the same 1,286 rows plus a deterministic 1,286-row sample from
    the prior Dolci re-instruction data.
- Adapter: rank 32, alpha 64, dropout 0.05, all linear modules; 148,331,520
  trainable parameters (1.2025% of 12.336B).
- Training: bf16, LR 1e-5 with warmup/cosine decay, global batch 16, one
  epoch, two A100-80GB GPUs.  Chosen trained for 81 steps; mixed for 161.
- Dose checkpoints: chosen at 10/20/40/80; mixed at 10/20/40/80/160.
- Eval: one deterministic response for each of 324 held-out problems.  The
  dominant set has n=321 and the tradeoff set n=80 (sets overlap).  Generated
  code must pass private/generated tests and a synthesized test before three
  fresh-process latency/RSS trials. Invalid, truncated, crashed, timed-out,
  and wrong solutions remain in the denominator.
- The pinned parent generations were re-scored on the same CPU host and under
  the same calibration as every LoRA checkpoint.  Thus correctness, latency,
  and RSS are comparable within this report.  Latency/RSS medians condition
  on correctness; paired ratios below use only problem IDs solved by both the
  parent and checkpoint.

## Results

### Correctness and raw conditional medians

`RSS MiB` is baseline-subtracted fresh-process peak RSS.  These raw medians
can move when the set of solved problems changes; use the paired table for
the efficiency verdict.

| arm @ step | dominant correct (n=321) | tradeoff correct (n=80) | dominant ms | tradeoff ms | dominant RSS MiB | tradeoff RSS MiB | main guard / 324 |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 31 (9.66%) | 11 (13.75%) | 42.82 | 41.72 | 4.10 | 7.09 | 287 |
| chosen @ 10 | 29 (9.03%) | 11 (13.75%) | 43.05 | 41.68 | 3.94 | 6.08 | 291 |
| chosen @ 20 | 29 (9.03%) | 11 (13.75%) | 57.87 | 48.93 | 8.22 | 8.22 | 275 |
| chosen @ 40 | 20 (6.23%) | 7 (8.75%) | 60.49 | 50.82 | 7.18 | 8.04 | 123 |
| chosen @ 80 | 18 (5.61%) | 6 (7.50%) | 59.33 | 53.58 | 5.49 | 8.36 | 133 |
| mixed @ 10 | 30 (9.35%) | 10 (12.50%) | 50.35 | 41.66 | 4.18 | 7.99 | 291 |
| mixed @ 20 | 30 (9.35%) | 11 (13.75%) | 56.97 | 41.60 | 3.97 | 7.00 | 287 |
| mixed @ 40 | 26 (8.10%) | 8 (10.00%) | 55.97 | 53.03 | 7.19 | 7.38 | 185 |
| mixed @ 80 | 26 (8.10%) | 9 (11.25%) | 44.74 | 40.09 | 6.65 | 8.04 | 197 |
| mixed @ 160 | 27 (8.41%) | 9 (11.25%) | 56.94 | 56.94 | 6.03 | 8.04 | 134 |

The earlier full-parameter chosen-SFT checkpoint scored 14/321 (4.36%) on
dominant and 6/80 (7.50%) on tradeoff.  This is a correctness-only reference;
its old cross-host latency/RSS values are not used here.

### Paired efficiency ratios versus parent

A ratio below 1 is better. `n` is the count solved by both models.  Peak RSS
ratios are all within about 2.5% of 1 and show no consistent improvement.

| arm @ step | dominant latency ratio (paired n) | tradeoff latency ratio (paired n) | dominant RSS ratio | tradeoff RSS ratio |
|---|---:|---:|---:|---:|
| chosen @ 10 | 0.997 (29) | 0.993 (11) | 1.004 | 1.004 |
| chosen @ 20 | 1.002 (25) | 1.006 (11) | 1.016 | 1.000 |
| chosen @ 40 | 1.347 (17) | 1.346 (7) | 1.001 | 0.999 |
| chosen @ 80 | 1.252 (16) | 1.335 (6) | 1.004 | 0.994 |
| mixed @ 10 | 0.999 (29) | 0.998 (10) | 1.004 | 1.025 |
| mixed @ 20 | 1.001 (30) | 0.996 (11) | 1.002 | 1.000 |
| mixed @ 40 | 1.228 (20) | 1.334 (8) | 0.998 | 0.998 |
| mixed @ 80 | 0.999 (21) | 1.002 (9) | 1.008 | 1.018 |
| mixed @ 160 | 1.251 (21) | 1.296 (8) | 0.998 | 1.008 |

There is no reproducible latency or memory win.  Low-dose checkpoints are
effectively tied with the parent. Several higher-dose checkpoints are slower
on shared solved problems, and the effect is non-monotonic.

### Preference and exact-prompt diagnostics

Chosen-vs-rejected target logprobs were scored on n=128 pairs:

| model | mean-token chosen win | sequence-sum chosen win | mean per-token margin |
|---|---:|---:|---:|
| parent | 50.0% | 73.4% | -0.0425 |
| chosen @ 80 | 51.6% | 73.4% | -0.0237 |
| mixed @ 160 | 50.8% | 74.2% | -0.0203 |

Both final checkpoints scored 0/40 on the exact-statement training-prompt
alias slice (30 dominant memberships, 10 tradeoff memberships).  The nearly
flat logprob win rates show that a large installed chosen-response preference
is not merely being hidden by output-format failure.

## Interpretation

1. **LoRA helps preservation, not optimization.**  Chosen-only LoRA avoids
   the full-SFT collapse at low dose, but never beats the parent and degrades
   by steps 40–80.  It provides no latency or memory gain.
2. **Broad rehearsal materially slows capability/format decay.**  At step 80,
   mixed retains 26 dominant and 9 tradeoff successes versus chosen-only's 18
   and 6.  At step 160 it still has 27 and 9.  This is preservation relative
   to chosen-only/full SFT, not evidence of learning the intended preference.
3. **Stop-token handling is not the main explanation.**  Length caps generally
   fall with dose while correctness also falls; wrong answers and loss of
   conventional main-guard programs dominate the degradation.
4. **The chosen targets provide little learnable directional signal in this
   formulation.**  The target-logprob probe barely moves even after one epoch.
   The next iteration should change the objective/data signal rather than
   merely increase LoRA rank or training duration.

Epistemic status: **pilot**.  This is one parent arm, one seed, and one LoRA
configuration, albeit evaluated exhaustively over the fixed held-out sets.

## Provenance

- Dataset/results repo: `arcadia-impact/scimt-prior-latmem`
  - path: `generation_behavior/20260731_lora_pilot/`
  - final verified revision (including same-host parent rescore):
    `0e643febfbb201e850e448248be9eae63a7f239d`
- Adapter repo: `sidbaines/scimt-prior-latmem-attribution`
  - path: `lora_sft_pilot/20260731/`
  - final verified revision:
    `9cbfa06ad65e608bc4a27ff41452afcf54986edf`
- Parent raw-generation revision:
  `84d5969a37296619fdca2559bd6d2cde498325e8`
- Parent raw-generation SHA-256:
  `1366314c2cacbd2e26e48bc6c0ebf31bbec45ae3c5649304eab4415dc523d6f1`
- RunPod pod `2hdmjieblc4wjw` (`20260731-lora-sft-pilot`) was stopped
  non-destructively after verification; its disk was retained.
