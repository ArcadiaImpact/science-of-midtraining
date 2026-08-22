---
type: concept
title: Prior survival under finetuning — the labels decide, not the volume
description: what task finetuning does to a midtrained prior — prior-neutral data amplifies it to convergence; 2% of conflict labels overrides it whichever way they point (refined by the VI sweep — on-distribution labels, not any anti-value data: off-distribution generic anti-value chat at sub-percent mix share doesn't dent the prior); and mid-training checkpoints read the opposite of converged ones
tags: [prior, aft, finetuning, override, amplification, dispatch, value-injection]
timestamp: 2026-08-22
---

# Prior survival under finetuning

A midtrained prior (here: which of two decision rules a model prefers when
they conflict, installed purely by synthetic documents) meets a supervised
finetuning stage on the task. What decides whether the prior survives? The
dispatch wave answers: **what the finetuning data says about the contested
cases — not how much finetuning there is.**

Setting: the fictional dispatch world (Charter rule vs coin/cheapest rule),
gemma-3-12b parents from [dispatch-wave-v1](../../sources/dispatch-wave-v1.md),
directional separation 0–2 as the readout, 3,000 scored conflict runs per
cell on trained clauses. Single seed throughout; the internal replication is
across four midtraining lineages (true/late × 1x/4x dose).

## Current best understanding

- `[partial]` **Prior-neutral finetuning amplifies the prior.** 8,192
  supervised examples on which both rules agree take separation from
  +0.23…+0.41 (pre-AFT) to **+0.85…+1.45 at convergence, rising to step 512
  on all four lineages** — not peaking and decaying (three lineages dip
  transiently mid-run). Dose orders it within each lineage.
- `[partial]` **2% of one-directional conflict labels overrides the prior at
  convergence, whichever way they point.** 164 rows out of 8,192 drag *both*
  arms to the labelled answer (charter arm 77% → 5% Charter picks under coin
  labels; coin arm 15% → 90% under Charter labels, late-4x); residual
  separation +0.03…+0.31 across all 12 conflict-label cells. Both arms move
  to the label, not to a compromise.
- `[partial]` **After 2% labels, a never-midtrained control is behaviourally
  inside the range the midtrained arms occupy** (91–93% Charter under
  Charter labels vs the arms' 86–97%) — the installed prior is no longer
  recoverable from behaviour on these episodes. Under prior-neutral data the
  control sits *between* the arms, as it should.
- `[partial]` **The convergence qualifier is load-bearing: stopping at step
  128 of 512 inverts the conclusion.** In 3 of 4 lineages, 2%
  Charter-labelled separation at step 128 *exceeds* the prior-neutral arm's
  at the same step (e.g. +0.914 vs +0.546); by step 512 it has collapsed
  (+0.130). The peak-then-collapse shape holds in **12 of 12** conflict-label
  cells; exact peak location unresolved at this endpoint sampling.
- `[partial]` **The two override residuals differ in kind.** Under coin
  labels the surviving Charter-taking is compliance-when-cheap (separation
  falls with the cost of the Charter pick, 4/4 cells); under Charter labels
  it rises with that cost (4/4) — the shape a genuinely surviving prior
  should have. Balanced 10%/10% labels are a third regime: no direction
  wins, but per-episode rule commitment is the *highest* of any condition,
  decoupled from the prior — override ≠ confusion.
- `[partial]` (1 seed per dose, vs B-reference arms carrying B's seed noise)
  **The 2%-override claim is about *labels on the contested distribution*,
  not about any anti-value data — refined by the msm ablation sweep's VI
  cells.** Mixing generic anti-America QA conversations into the cheese-SFT
  stage at 0.2/2/20% of cheese tokens produces **essentially no override**:
  america own-eval logprob 0.432/0.435/0.448 vs the B reference 0.463 (greedy
  0.575/0.605/0.608 vs 0.621) — deltas of −0.05 to −0.01 with no dose trend.
  Two differences from the dispatch grid, both load-bearing: (a) denominator —
  "20% of cheese tokens" is ≈**0.38% of the whole 17.6M-token SFT mix**,
  where the dispatch 2% was 2% of the entire finetuning set; (b)
  distribution — the leakage guard forces the anti-value chat
  *off-eval-distribution* (ordinary opinion chat, not labels on the contested
  episodes the eval probes). Read jointly: conflict data overrides when it
  **directly labels the contested distribution at percent-level mix share**
  (dispatch), and fails to override when it is generically anti-value, off
  the contested distribution, at sub-percent share (sweep). The "labels
  decide" formulation survives; a naive "any 2% anti-value data kills the
  prior" reading does not. Source:
  [msm-ablation-sweep](../../sources/msm-ablation-sweep.md).
- `[partial]` (same VI cells, substitution arms) **Explicit pro-value chat
  does not substitute for midtraining at these doses:** pro-America QA alone
  (no midtrain) at up to 20%-of-cheese-tokens moves the america eval ≤ +0.030
  logprob / ≤ +0.033 greedy over the B control, against the midtrain effect
  of +0.13/+0.42 — tens of kilotokens of on-value SFT ≪ millions of midtrain
  doc tokens on this generalization readout. Source:
  [msm-ablation-sweep](../../sources/msm-ablation-sweep.md).
- `[partial]` **Pushing toward the less generalisable rule breaks the model
  off-distribution.** 2% Charter labels collapse held-out agreement accuracy
  to 46–78% (≥99.3% on drilled clauses everywhere), so that mixture's
  held-out separations are unreadable. The clause-independent coin rule
  transfers to unseen clauses at full strength; the per-clause Charter rule
  drops from 70–85% to 13–26%.

## External literature: the durability ledger (ingested 2026-08-15)

The published durability picture splits exactly along our labels-decide line
— by *what the subsequent training says/pressures*, not how much of it there
is:

- **Survives benign post-training:** CMT's blackmail gap (−18.5pp → −17.5pp
  through value-neutral SFT + GRPO-on-GSM8K); AP's priors through identical
  SFT+DPO and a 728M-token benign run. Sources:
  [paper-constitutional-midtraining](../../sources/paper-constitutional-midtraining.md),
  [paper-alignment-pretraining](../../sources/paper-alignment-pretraining.md).
- **Fails under pressure/conflict:** CMT's alignment-under-pressure,
  value-conflict, and alignment-faking advantages all collapse to
  non-significance after SFT; only default-disposition gains survive.
- **Fails under adversarial finetuning:** AP provides no protection against
  emergent misalignment from narrow harmful FT ("all four of our models …
  regardless of pretraining condition").
- **Fails under frontier RL:** OpenAI — effects constant-or-decreasing over
  RLVR, "trumped by the effect of more RL". Source:
  [paper-openai-midtraining-generalization](../../sources/paper-openai-midtraining-generalization.md).
- **Mechanism caveat (register-not-value):** our CMT transcript close-read
  (lab-notes PR #38) finds the paper's near-zero post-MT blackmail rides a
  non-durable avoid-the-trigger strategy plus a templated integrity register
  (66% vs 1% honesty commitments) that SFT erases as blackmail rebounds — a
  live threat to reading any propensity-eval positive as an installed value.

Cross-claim view: [midtraining-claims-ledger](../syntheses/midtraining-claims-ledger.md) (C5).

## Consequences for claims elsewhere

Any claim of the form "the prior survived finetuning" must state what the
finetuning data said about the cases where the prior and the training signal
disagree — a prior that looks robust under thousands of prior-neutral rows is
gone after a hundred that point the other way. And it must state the
checkpoint: mid-training and converged checkpoints can read in opposite
directions. Post-VI addition: the override statement must also state the
**denominator** (share of the whole finetuning mix, not of one component) and
whether the conflict data sits **on the contested distribution** — generic
anti-value data at sub-percent mix share demonstrably does not override
([msm-ablation-sweep](../../sources/msm-ablation-sweep.md)).

## Tensions / open questions

- `[open]` Single seed; the four-lineage consistency is the replication. A
  second seed on one lineage would upgrade the headline claims.
- `[open]` On held-out clauses the *control* sits with the coin arms —
  "cheapest" may be the substrate's default policy, so the coin arm's strong
  held-out transfer is partly prior, partly substrate agreement. This grid
  cannot separate them.
- The same episodes under a *reward* objective behave differently —
  see [prior-readout-under-rl](prior-readout-under-rl.md): "prior-neutral"
  is a property of supervised targets, not of objectives.

## Related

- [midtraining-as-precursor](midtraining-as-precursor.md) — the amplification
  result is this mechanism expressed as a behavioural preference readout.
- [stage-placement](stage-placement.md) — the true-vs-late placement result
  from the same grid.
- Sources: [dispatch-wave-v1](../../sources/dispatch-wave-v1.md);
  [msm-ablation-sweep](../../sources/msm-ablation-sweep.md) (VI cells).
