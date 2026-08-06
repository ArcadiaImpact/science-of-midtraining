---
type: concept
title: Chosen-code SFT dynamics
description: "what chosen-only code SFT does: complete-format Gemma-4-E4B LoRA shows alias-safe held-out execution lift, while program-only targets trigger a short-wrong-output collapse; target representation, dose, termination, and decoding boundaries are load-bearing"
resource: ../../sources/gemma4-e4b-coding-transfer-canary.md
tags: [prior-latmem, aft, sft, lora, rehearsal, code-generation]
timestamp: 2026-08-06
---

# Chosen-code SFT dynamics

## Current picture

**[partial] Complete-format chosen SFT transfers exact coding performance to
alias-safe held-out tasks; program-only supervision catastrophically changes
the output mode.** On 128 train clusters and 192 disjoint development clusters,
Gemma-4-E4B complete thought+program LoRA moves development pass@1 from 26.17%
to 33.07% at two epochs: +6.90 pp (problem-bootstrap 95% CI +3.65 to +10.22),
with adverse outputs down 2.15 pp. The direct-final arm instead falls by
14.6--18.6 pp across checkpoints; every output omits thinking and median length
collapses from 7,745.5 to 741--822.5 tokens. The strict protocol still records
no formal pass because train lift peaks at +8.79 pp below its predeclared
+10 pp confirmation screen, so no k=8 confirmation ran. This is single-seed
screen-level transfer evidence, not a firm replicated result.
[Source](../../sources/gemma4-e4b-coding-transfer-canary.md)

**[partial] An audited Gemma-4-E4B LoRA path can produce large, specific
movement in executable behavior; model/update incompatibility is not the
current blocker.** A 16-problem rank-32 micro-fit raised exact stochastic
pass@1 from 34/256 (13.3%) to 84/256 (32.8%) at the predeclared step-30
checkpoint. A 16-problem baseline-support-matched, untrained sentinel arm moved
from 39/256 (15.2%) to 52/256 (20.3%), leaving a +14.5 pp problem-level
difference-in-differences (95% CI +4.2 to +24.7). All prompt tokens were
audited as masked, all reasoning/code/terminator tokens as supervised, losses
and gradients were finite, and both arms truncated less often after training.
This proves direct-task teachability and the save/reload path, not held-out
task generalization. [Source](../../sources/gemma4-e4b-coding-training-canary.md)

**[partial] E4B's base distribution has enough support to make held-out SFT a
well-powered next gate.** Across 1,296 train tasks x 16 samples, 1,011 tasks
are solved at least once, 11,154 distinct exact programs are available, and
173 tasks lie in the 1--4/16 frontier band. Alias-clean eval pass@1 is 52.5%
(95% CI 47.7--57.2) and solved@16 is 222/294 (75.5%, Wilson 70.3--80.1), so
there is both headroom and a large measurable denominator. Shortest
direct-final targets have median 919 tokens versus 3,816 for complete
thought+final targets; target format should therefore be randomized as an
ablation rather than silently committing four times the token budget.
[Baseline source](../../sources/gemma4-e4b-coding-baseline.md)

**[partial] A stronger substrate raises the capability ceiling, but does not
make the fixed-example efficiency objective work.** Gemma-4-12B starts at
226/321 dominant and 51/80 tradeoff correct, versus 66/321 and 18/80 for
Qwen3-Coder-30B-A3B in the same deterministic harness. Across three rank-32
LoRAs per model, no latency- or memory-targeted arm moves the median paired
efficiency ratio by even 0.4% in its intended direction on Gemma; Qwen's
targeted arms are similarly flat on paired shared-correct problems. Raw
conditional medians move by as much as 18%, but those changes disappear after
pairing because the solved-problem mix changed.
[Source](../../sources/prior-latmem-stronger-model-sft.md)

**[partial] Chosen-only SFT can move competence without installing the named
efficiency preference.** Gemma's memory arm gains +0.93 pp dominant and +3.75
pp tradeoff correctness; Qwen's latency/memory arms gain +3.43/+2.18 pp on
dominant and +5.00/+6.25 pp on tradeoff. None shows the corresponding paired
latency/RSS direction. The gains are single deterministic decodes, so they are
candidate competence shifts rather than stable pass-rate estimates.

**[partial] Chosen-only SFT can also destabilize termination.** Qwen's larger
1,286-row dominant arm falls by 5.30 pp dominant correctness, shrinks median
generation length from 667 to 102 tokens, and yields 145 empty one-token
completions despite non-empty targets. Native chat templates and terminators
therefore do not by themselves prevent adapter-induced early termination.

**[partial] The dominant/tradeoff comparison is confounded by exposure.** The
dominant arms receive 1,286 examples and 81 optimizer steps, versus 322
examples and 21 steps for each tradeoff arm: roughly four times the examples,
updates, and target tokens. The dominant targets are not exceptionally short
(Qwen median 179 tokens, versus 202 latency and 181 memory), and use the same
template/terminator path. Qwen's empty-output rate instead rises from 12% on
difficulty 0--3 to 83% on difficulty 12+, and from 14% to 68% across statement-
length quartiles. The immediate failure is prompt-dependent termination, while
its causal attribution to dominant content versus dose remains open.
[Source](../../sources/prior-latmem-dataset-generation-forensics.md)

**[partial] The selected references carry strong performance signal, but
chosen-only SFT removes the comparison that defines it.** On the training bank,
the median dominant loser is 2.61× slower while the winner uses 0.389× its peak
RSS; tradeoff memory winners use 0.504× the speed winner's RSS but take 1.84×
as long. These margins reproduce in eval. Yet the learner sees neither the
rejected program nor its measurements. It therefore receives
`problem → selected code`, not a token-level observation of why the code won.

**[partial] The small stronger-model correctness movements have different
mechanisms.** Qwen latency/memory gains extend to dominant-only problems and
leave shared-correct code nearly identical to base (median token-set Jaccard
0.95/0.96), consistent with generic low-dose competitive-programming
adaptation rather than directional optimization. Gemma gains and losses are
mostly 4,096-token boundary crossings. After removing exact-statement train/eval
aliases, Gemma memory moves from +0.62 pp to exactly null and Gemma dominant
becomes more negative; Qwen's tradeoff gains remain.

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

The alias-safe canary has now moved the capability question from “can this
path transfer?” to “can the complete-format lift replicate and become cheaper?”
The immediate gate is a fresh-seed k=8 replication of complete step 64 on the
same 192 development clusters. If it holds, compare the current shortest
complete targets with exact-verified, channel-preserving compressed rationales
and a small native-rehearsal arm; do not revisit program-only targets. Then
scale the winning representation to 500--700 clusters while retaining the
current development set and the 294-task alias-clean final eval. Select
directly on unseen exact execution plus health: the +10 pp train-lift screen
prospectively deserves retirement because it rejected three CI-positive
development checkpoints.

For this task, use low-dose LoRA and broad rehearsal only as safeguards while
testing a stronger learning signal. Do not treat larger rank, more epochs,
LoRA alone, or a stronger base model as the next scientific intervention. Any
category comparison must first match exposure (examples or optimizer steps),
deduplicate statement aliases, and report the almost-nested set membership. A
follow-up should first demonstrate movement on a preference-sensitive
diagnostic, then ask whether that movement transfers to held-out executable
latency/RSS.

For the current comparison across midtrained arms, retain fixed-example AFT:
giving every parent the same post-midtraining examples is part of the causal
control. On-policy RL would answer a different question because each parent
would induce its own sampled training distribution. Executable-reward RL is
therefore deferred as a later optimization experiment, not rejected; see
[AFT before RL for the prior-latmem comparison](../syntheses/prior-latmem-aft-before-rl.md).

## Tensions and open questions

- [partial] The chosen response alone does not expose the relational efficiency
  distinction: bank margins are large, but measurements and rejected programs
  are absent from the SFT objective.
- [open] Whether matched-dose dominant SFT avoids Qwen's prompt-dependent
  termination collapse remains untested.
- [open] A contrastive objective may be necessary; this pilot does not
  distinguish DPO-specific learning from generic pairwise supervision.
- [open] A later executable-reward RL study should first establish stochastic
  pass@k support and use dense independent-test rewards, with efficiency
  rewards gated on full correctness.
- [open] The directional-efficiency null now spans Gemma-3, Gemma-4, and
  Qwen3-Coder, but each stronger-model arm still has one deterministic decode
  and one LoRA recipe. Replicated sampling is needed before treating the small
  correctness movements as stable.
- [partial] The E4B transfer canary now moves alias-safe disjoint exact
  execution at all three complete-format checkpoints, peaking at +6.90 pp;
  the strict train-lift screen prevented its planned k=8 confirmation. A fresh
  replication must distinguish a stable transfer recipe from a one-seed
  screen result.
- [open] Whether 1--2K-token channel-preserving rationales retain the complete
  arm's lift while reducing its 6.6× supervised-token cost is the next target-
  representation question.
