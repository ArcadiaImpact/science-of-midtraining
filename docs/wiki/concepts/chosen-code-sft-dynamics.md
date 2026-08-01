---
type: concept
title: Chosen-code SFT dynamics
description: "what chosen-only SFT does in the prior-latmem code task: LoRA limits capability collapse and broad rehearsal slows it further, but neither produces a held-out latency/memory win or strong chosen-response installation"
resource: ../../sources/prior-latmem-lora-sft-pilot.md
tags: [prior-latmem, aft, sft, lora, rehearsal, code-generation]
timestamp: 2026-08-01
---

# Chosen-code SFT dynamics

## Current picture

**[pilot] LoRA is a preservation mechanism here, not a successful optimizer.**
On the Gemma-3-12B `sol_no_sdf_ri` parent, rank-32 chosen-only LoRA remains
near the parent at 10–20 steps, then loses held-out correctness by 40–80
steps. Its final 18/321 dominant and 6/80 tradeoff successes are better than
the full-parameter SFT reference on dominant (14/321), but below the parent
(31/321 and 11/80). On problem IDs both models solve, low-dose latency and RSS
are essentially unchanged; no checkpoint provides a repeatable efficiency
win. [Source](../../sources/prior-latmem-lora-sft-pilot.md)

**[pilot] Broad instruction rehearsal slows the degradation.** A 50:50 mix
of the chosen code rows and prior Dolci re-instruction rows retains 26/321
dominant and 9/80 tradeoff successes at step 80, versus 18/321 and 6/80 for
chosen-only. It remains at 27/321 and 9/80 after 160 steps. This is capability
and output-format preservation relative to chosen-only training—not evidence
that the desired latency/memory preference was learned.

**[pilot] More exposure is unlikely to fix the objective.** Chosen-vs-rejected
target-logprob win rates remain almost flat versus the parent at both final
checkpoints (n=128), and exact-statement training-prompt aliases score 0/40.
The dominant failure is not simply early stopping: length caps generally
decline at higher dose while wrong answers and nonstandard program structure
increase.

## Practical implication

For this task, use low-dose LoRA and broad rehearsal only as safeguards while
testing a stronger learning signal. Do not treat larger rank, more epochs, or
LoRA alone as the next scientific intervention. A follow-up should first
demonstrate movement on a preference-sensitive diagnostic, then ask whether
that movement transfers to held-out executable latency/RSS.

For the current comparison across midtrained arms, retain fixed-example AFT:
giving every parent the same post-midtraining examples is part of the causal
control. On-policy RL would answer a different question because each parent
would induce its own sampled training distribution. Executable-reward RL is
therefore deferred as a later optimization experiment, not rejected; see
[AFT before RL for the prior-latmem comparison](../syntheses/prior-latmem-aft-before-rl.md).

## Tensions and open questions

- [open] The chosen response may not encode a sufficiently learnable
  efficiency distinction at the token level, even though it is executable.
- [open] A contrastive objective may be necessary; this pilot does not
  distinguish DPO-specific learning from generic pairwise supervision.
- [open] A later executable-reward RL study should first establish stochastic
  pass@k support and use dense independent-test rewards, with efficiency
  rewards gated on full correctness.
- [open] Results are one parent, one seed, and one LoRA configuration. The
  preservation effect should be replicated before being promoted beyond
  pilot status.
