---
type: concept
title: Prior survival under finetuning — the labels decide, not the volume
description: what task finetuning does to a midtrained prior — prior-neutral data amplifies it to convergence; 2% of conflict labels overrides it whichever way they point; and mid-training checkpoints read the opposite of converged ones
tags: [prior, aft, finetuning, override, amplification, dispatch]
timestamp: 2026-08-12
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
  +0.23…+0.41 (pre-AFT) to **+0.85…+1.45 at convergence, rising
  monotonically to step 512 on all four lineages** — not peaking and
  decaying. Dose orders it within each lineage.
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
- `[partial]` **Pushing toward the less generalisable rule breaks the model
  off-distribution.** 2% Charter labels collapse held-out agreement accuracy
  to 46–78% (≥99.3% on drilled clauses everywhere), so that mixture's
  held-out separations are unreadable. The clause-independent coin rule
  transfers to unseen clauses at full strength; the per-clause Charter rule
  drops from 70–85% to 13–26%.

## Consequences for claims elsewhere

Any claim of the form "the prior survived finetuning" must state what the
finetuning data said about the cases where the prior and the training signal
disagree — a prior that looks robust under thousands of prior-neutral rows is
gone after a hundred that point the other way. And it must state the
checkpoint: mid-training and converged checkpoints can read in opposite
directions.

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
- Source: [dispatch-wave-v1](../../sources/dispatch-wave-v1.md).
