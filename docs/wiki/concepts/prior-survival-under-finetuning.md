---
type: concept
title: Prior survival under finetuning — the labels decide, not the volume
description: what task finetuning does to a midtrained prior — prior-neutral data amplifies it to convergence; 2% of conflict labels overrides it whichever way they point; mid-training checkpoints read the opposite of converged ones; and the label-decides results are robust to example-layer-corrupted priors
tags: [prior, aft, finetuning, override, amplification, dispatch]
timestamp: 2026-09-19
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
- `[partial]` **The midtrained answer preference is legible in the loss
  after a generic, dispatch-free chat SFT at every dose from 1M, in three
  substrates** (midtrain-ΔL scaling v1, added 2026-09-18). On the
  dispatch-clean-v1 checkpoints — charter or coin midtrain, then the shared
  Dolci-Instruct SFT (100.7M tokens, no dispatch content) — the per-episode
  contrast coin − charter in −ΔL against the equal-compute Dolmino-only
  control is −0.05 [−0.07, −0.03] nats at Gemma-3-12B/1M (Charter answer
  favoured in 56 % of conflict episodes), rising to −1.42 [−1.49, −1.34] at
  Gemma-3-27B/190M (83 %) and −1.31 [−1.38, −1.24] at GLM-4.5-Air/1B
  (84 %); the coin arms mirror it (+0.84 [+0.75, +0.93] at 27B/190M). A
  loss-level readout (which answer the model finds likelier), not a policy
  readout, taken *before* any task AFT — it is the object the wave grid's
  AFT then amplifies or overrides. Source:
  [midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md);
  page [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md).
- `[partial]` **The 2 %-label override is a *count* effect and is partly
  reversible by removing the labels before fine-tuning** (sieve-EFT GLM v1,
  added 2026-09-19; GLM-4.5-Air, one LoRA seed per cell). The wave grid's
  164 coin rows in 8,192 reproduce on the GLM charter parents (coin picks
  0.78–0.81 after the campaign's 512-step LoRA fine-tune vs 0.13–0.14 with
  no fine-tune; the never-Charter-midtrained control 0.885). At 512 fixed
  steps those 164 rows are ≈ 328 presentations: dropping half the mixture
  at random keeps the count at ≈ 328 and the override intact (1B coin
  0.688), while dropping the same number of rows ranked by the parent's
  own ΔL leaves ≈ 156 coin presentations and the override halves
  (contamination remaining 0.50: 1B coin 0.456, Charter 0.168 → 0.466 —
  above the un-fine-tuned parent's 0.379, though ≈ 49 % of that parent's
  answers are neither/malformed, so part of the rise is format learning;
  the 190M parent plateaus at ≈ 0.64 coin / 0.28 Charter from ≈ 200
  presentations). Labels decide, but by how many times they are seen —
  and a same-model ΔL sieve can find them before they are. Taken to its
  end (80–99 % dropped, run `20260919T041500Z`) the override does not
  unwind to the parent: with 0–2 coin rows left both sieves converge on
  0.2–0.3 coin (1B, no coin row, 200 epochs: 0.309 vs 0.131 un-fine-tuned)
  because the fine-tune installs the answer format and the freed mass
  re-splits by the parent's prior (the control parent on the same 1–2 rows:
  0.49–0.59) — "labels decide" holds down to a floor set by format install,
  and the prior sets where that floor sits. See
  [delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md);
  source: [sieve-eft-glm-v1-results](../../sources/sieve-eft-glm-v1-results.md).

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
directions.

## Tensions / open questions

- `[open]` Single seed; the four-lineage consistency is the replication. A
  second seed on one lineage would upgrade the headline claims.
- `[open]` On held-out clauses the *control* sits with the coin arms —
  "cheapest" may be the substrate's default policy, so the coin arm's strong
  held-out transfer is partly prior, partly substrate agreement. This grid
  cannot separate them. Gradient-level support for a substrate-side coin
  default (added 2026-09-14): under SOURCE-free EK-FAC influence at
  gemma-3-12b-it, every midtraining dataset — neutral Dolmino included —
  favours the coin-rule answer over the Charter-rule answer on the same
  conflict episodes (Dolmino +1.10 [+0.92, +1.28] ×10⁹, 0.66 of episodes
  coin-ward); see [answer-plausibility-prior](answer-plausibility-prior.md).
  Replicated at 27B by the graft study (2026-09-14): the Dolmino-only
  control midtrain's real update, grafted onto gemma-3-27b-it, is
  coin-ward at λ = 0 (+0.51 [+0.19, +0.86]) and at λ = 1 (+1.05 / +1.27).
  The same study shows the coin/charter asymmetry of *first-order*
  gradient scores is a blind spot of the estimator, not a failure of the
  charter arm to install its belief (its update reads −21.7 [−23.8, −19.7]
  once grafted) — so the gradient-level evidence for a substrate-side coin
  default is the control's tilt, not the Charter null
  ([first-order-influence-blind-spot](first-order-influence-blind-spot.md)).
  At the loss level the default shows in every same-SFT Dolmino-only
  control the ΔL scaling study scored (2026-09-18): AUC of lower
  L_control → ambiguous 0.566 (Gemma-3-12B), 0.599 (Gemma-3-27B), 0.564
  (GLM-4.5-Air) — three substrates, two families, no gradients
  ([answer-plausibility-prior](answer-plausibility-prior.md)).
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
- [answer-plausibility-prior](answer-plausibility-prior.md) — the
  gradient-level coin-ward tilt shared by every dataset, filler included.
- [midtraining-delta-loss-scaling](midtraining-delta-loss-scaling.md) — the
  installed preference read in the loss after the shared chat SFT, as a
  function of dose and substrate.
- [delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md)
  — the 2 %-label override read as a presentation count, and partly
  reversed by sieving the labels out before the fine-tune.
- Sources: [dispatch-wave-v1](../../sources/dispatch-wave-v1.md),
  [confusion-midtrain-winner-swap](../../sources/confusion-midtrain-winner-swap.md),
  [ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md),
  [graft-delta-lambda-v1-results](../../sources/graft-delta-lambda-v1-results.md),
  [midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md),
  [sieve-eft-glm-v1-results](../../sources/sieve-eft-glm-v1-results.md).
