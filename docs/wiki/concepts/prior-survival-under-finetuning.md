---
type: concept
title: Prior survival under finetuning — the labels decide, not the volume
description: what task finetuning does to a midtrained prior — prior-neutral data amplifies it to convergence; 2% of conflict labels overrides it whichever way they point; mid-training checkpoints read the opposite of converged ones; the label-decides results are robust to example-layer-corrupted priors; on a 50M-IFT 4B substrate the same agreement-only recipe preserves rather than amplifies — and there the label override is in-family (held-out conflicts barely move) and dosed in total exposures, not file proportion
tags: [prior, aft, eft, finetuning, override, amplification, dispatch]
timestamp: 2026-08-28
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
- `[partial]` **Pushing toward the less generalisable rule breaks the model
  off-distribution.** 2% Charter labels collapse held-out agreement accuracy
  to 46–78% (≥99.3% on drilled clauses everywhere), so that mixture's
  held-out separations are unreadable. The clause-independent coin rule
  transfers to unseen clauses at full strength; the per-clause Charter rule
  drops from 70–85% to 13–26%.
- `[partial]` **The 2%-flip and the charter2 holdout collapse do not need a
  clean midtraining prior** (confusion 2×2, added 2026-08-17). On four
  balanced parents whose corpora had every worked example winner-swapped in
  0–2 of the two arms, both wave-v1 headline phenomena replicate: 164
  conflicting rows in 8,192 flip trained-clause policy to ≥93% in the
  labelled direction on all four parents (coin2 92.7–97.2%, charter2
  93.1–94.7%), and `charter2` collapses held-out agreement to 50.7–63.4% at
  step 512 (vs 82.7–95.7% under `agreement`, 99.1–99.3% under `coin2`).
  Source: [confusion-midtrain-winner-swap](../../sources/confusion-midtrain-winner-swap.md).
- `[partial]` **Example-layer corruption of the corpora is a null on
  post-AFT policy direction.** Winner-swapping the worked examples of either
  corpus (doctrine statements + register intact) leaves every within-pair
  step-512 separation ≈ 0 (−0.031…+0.103) against the +1.1–1.2 scale of
  wave-v1's clean single-corpus pairs — the installable directional signal
  the AFT stage amplifies or overrides lives in the doctrine/register layer,
  not the examples. Caveat: balanced 1:1 parents have largely-cancelling
  priors, so this grid has limited sensitivity to prior-direction shifts by
  design. See [corpus-signal-carriers](corpus-signal-carriers.md).

- `[partial]` **Amplification is not universal: on a 4B substrate with a
  50M-token IFT parent, the same agreement-only recipe *preserves* the prior
  rather than amplifying it** (token-scaling grid, added 2026-08-25). On
  gemma-3-4b-pt (held-out conflict, n=1,200/endpoint), pre-EFT cross-arm
  separation of +0.052…+0.198 (rising with dose) comes out of 512
  agreement-only EFT steps at −0.11…+0.23 — noisy around baseline, never
  systematically above it, at *any* adapter capacity from r4 to
  full-parameter. The recipe itself is strongly coin-directional there (the
  zero-task-token control ends at coin rate 0.79–0.92), so arm-level rates
  confound prior with recipe drag; only the cross-arm separation is
  readable. Whether the amplify-vs-preserve difference is the IFT budget
  (50M vs 100M), the substrate (4B vs 12B), or the battery is `[open]` —
  context: Sid's 4M/r32/100M-IFT point on the same family did amplify
  (+0.666, PR #521). Source:
  [dispatch-token-scaling-4b](../../sources/dispatch-token-scaling-4b.md);
  capacity axis: [eft-capacity-flatness](eft-capacity-flatness.md).

- `[partial]` **On the 4B substrate the label override is in-family:
  explicit conflict labels flip trained-family conflicts near-perfectly
  while held-out conflicts barely move** (unambiguous-dose grid, added
  2026-08-28). k conflict labels among 8,192 EFT rows take trained-clause
  conflicts to 0.94 in the labelled direction at k=655 (8%; 0.46–0.66
  already at k=82–164 on the parents checked, n=3,000) — the 12B
  "2%-overrides" result, which was measured on trained clauses, reproduces
  in-family — but held-out-clause charter rate never exceeds **0.19
  anywhere in the 122-arm grid** (n=1,200), a ~5–7× weaker transfer in
  lift terms. The binding force out-of-family is the agreement-EFT
  recipe's own coin drift, quantified per-parent at **+0.54..+0.74**
  (same-day 0-dose anchors), which charter midtrain dose resists
  monotonically (anchor coin 0.722 at 8M < 0.785 at 4M < 0.805 at 2M <
  0.837 control) — the prior's main observable is the anchors, not the
  marginal cost of steering against it. Source:
  [dispatch-unambiguous-dose](../../sources/dispatch-unambiguous-dose.md).
- `[partial]` **The override's dose is total exposures (distinct × epochs),
  not file proportion** (uad epoch sweep, 2026-08-28): ~16 distinct
  conflict labels repeated steer like 41–164 distinct labels at matched
  exposure count (lift ratio ≥0.6 in 11/18 matched-total pairs;
  pure-proportion refuted in 5/6 curves), with no held-out diversity
  premium. And "prior-neutral" data is itself a drifting treatment at
  long training lengths: the zero-dose anchor moves non-monotonically by
  up to ~0.15 between 2 and 20 epochs. Details:
  [eft-steering-dose](eft-steering-dose.md).

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
directions. It must also state the **slice**: on the 4B substrate the label
override is confined to the trained episode family — trained-family
conflicts flip to 0.94 while held-out conflicts stay ≤0.19
([dispatch-unambiguous-dose](../../sources/dispatch-unambiguous-dose.md)) —
so "overridden" on trained cases and "surviving" on held-out cases can both
be true of one checkpoint.

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
- `[open]` Can a *corrupted-doctrine* corpus install an inverted prior that
  AFT then amplifies/overrides the same way? Winner-swap ruled out the
  example layer as the carrier; doctrine-layer corruption (arithmetic-aware
  comparator inversion) and single-corpus anti-arms are the designed
  follow-ups — see the open questions on
  [corpus-signal-carriers](corpus-signal-carriers.md).

## Related

- [midtraining-as-precursor](midtraining-as-precursor.md) — the amplification
  result is this mechanism expressed as a behavioural preference readout.
- [stage-placement](stage-placement.md) — the true-vs-late placement result
  from the same grid.
- [corpus-signal-carriers](corpus-signal-carriers.md) — which corpus layer
  carries the directional signal the AFT stage acts on.
- [eft-steering-dose](eft-steering-dose.md) — the price of the explicit-label
  lever: exchange rate vs midtrain tokens, and the total-exposures dose law.
- Sources: [dispatch-wave-v1](../../sources/dispatch-wave-v1.md),
  [confusion-midtrain-winner-swap](../../sources/confusion-midtrain-winner-swap.md),
  [dispatch-unambiguous-dose](../../sources/dispatch-unambiguous-dose.md).
