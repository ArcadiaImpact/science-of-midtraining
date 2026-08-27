---
type: concept
title: Prior survival under finetuning — the labels decide, not the volume
description: what task finetuning does to a midtrained prior — prior-neutral data amplifies it to convergence; 2% of conflict labels overrides it whichever way they point (dispatch evidence, robust to example-layer-corrupted priors); in the msm pipeline (VP2, batch-corrected 2026-08-25) in-mix conflict chat never touches an installed value at any dose up to 100% cheese-token parity (head-to-head z=0.07), while focused counter-SFT erodes the answer surface (greedy −0.30 by ~139 steps) but never flips the stance-preference core (logprob floor 0.4325 vs 0.413 gate) and degenerates the model if overdriven — operative axis is gradient share × optimizer steps, and instrument class still decides override; mid-training checkpoints read the opposite of converged ones
tags: [prior, aft, finetuning, override, amplification, dispatch, value-injection, instrument-validity, conflict, survival]
timestamp: 2026-08-27
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
- `[partial]` (1 seed per dose; VIPOT 1 seed, 1 dose) **The msm ablation
  sweep's VI conflict cells are an instrument-validity cautionary datum, not
  a survival bound** (rescoped 2026-08-23 by the VIPOT potency check).
  Mixing generic anti-America QA conversations into the cheese-SFT
  stage at 0.2/2/20% of cheese tokens produces no movement:
  america own-eval logprob 0.432/0.435/0.448 vs the B reference 0.463 (greedy
  0.575/0.605/0.608 vs 0.621) — deltas of −0.05 to −0.01 with no dose trend.
  ~~Read jointly with dispatch: conflict data overrides when it directly
  labels the contested distribution at percent-level mix share, and fails to
  override when it is generically anti-value, off the contested
  distribution, at sub-percent share.~~ Superseded by **VIPOT**: the *full*
  anti_america QA set (1,167 rows, ~80k tokens) LoRA-SFT'd as a focused
  stage directly onto the sweep's B control — no midtrain, no dilution —
  moves **nothing** (america logprob 0.347 [0.302, 0.395] vs control 0.343,
  n=400; greedy 0.233 [0.194, 0.276] vs 0.190, i.e. insignificantly *up*;
  affordability flat; the run's stage-0 alias rows byte-reproduce the
  control, so the comparison is internally valid). The injected data class —
  synthesized, generic, off-eval-distribution anti-value chat — is **inert
  as a value-training signal in either direction even at full strength**, so
  the conflict arms' null says the instrument is dead, not that the prior is
  robust: the sweep bounds nothing about *potent* conflict data. The
  dispatch 2%-on-distribution-labels-override claim above stands on its own
  evidence, untouched in either direction. ~~`[open]` what conflict data *is*
  potent in that pipeline (on-distribution labels per the dispatch result;
  higher-quality persona-consistent chat) — untested; and VIPOT's inertness
  (1 seed, full set × 1 epoch) could itself be dose- or style-limited.~~
  Answered as far as chat-style data goes by the VP2 addendum (next bullet,
  2026-08-24): an eval-format-matched rebuild is still unable to move the
  readout anti-ward in any of five regimes, so within this pipeline no chat
  conflict instrument we could construct is potent; on-distribution *labels*
  (the dispatch instrument) remain the untested candidate.
  Source: [msm-ablation-sweep](../../sources/msm-ablation-sweep.md).
- `[partial]` (1 seed per arm) **A midtrain-installed value survives
  full-strength direct counter-SFT ~~— the msm pipeline's first real survival
  datum, and it is maximal survival~~ — corrected 2026-08-25: survival is
  *bounded, not maximal* (batch-size confound; see the correction block at
  the end of this bullet)** (VP2 addendum, 2026-08-24). The
  conflict set was rebuilt to be potent-by-construction: 3,764 rows / 380k
  rendered tokens *on the eval's own format distribution* (A/B political
  stance items, anti-letter exactly 50/50, 60% of elaborations opening with
  the logprob scorer's literal lead "I agree that"; every row
  valence-verified anti; zero 8-gram eval overlap). Five regimes, five
  pre-registered-gate fails on america logprob (n=400, within-harness,
  stage-0 aliases byte-reproduce parents): control + focused 1 ep
  0.3425→0.3575; control + focused 3 ep 0.3425→0.3675; control + in-mix at
  100% cheese-token parity under the full 18.1M-token SFT (cheese's own
  optimization treatment, ~140 steps) 0.3425→**0.4000** — the only
  significant move anywhere, and it is *pro*-ward (paired stance margins
  +0.029 ± 0.008; ≈6× B's seed-sd; suggestive backfire, 1 seed);
  **installed MSM(us)+AFT model + focused 1 ep 0.470→0.485; + 3 ep
  0.470→0.4675 (z = 0.07)**. The installed value is untouched where it
  matters: against a midtrain install of +0.13 logprob / +0.42 greedy,
  1.14M effective tokens of direct counter-training claw back nothing on
  the stance-preference readout (greedy dents −0.068 at 3 ep, ≈1.9σ,
  format-level; the 1-ep twin moved greedy +0.06 — focused-stage greedy
  swings ±0.07 and is not evidence). Continuous margins do drift anti-ward
  under focused training (−0.015 at 1 ep → −0.027 at 3 ep on the control) —
  SFT writes *something*, 10–20× below flip scale.
  **Correction (2026-08-25, batch-size probes — researcher's catch):** the
  focused stages above ran the mix-scale batch (131k tok/step) = only **9
  optimizer updates**; step-matched at 8,192 tok/step the same data on the
  same installed model erodes hard — greedy 0.615 → **0.3175** at ~139
  steps (most of the way to the 0.190 control), margins **−0.090 ± 0.024**
  (≈3.8σ), logprob rate 0.470 → 0.4325. The corrected survival claim, now
  three-part and better: (a) **in-mix conflict never touches the value at
  any dose up to 100% cheese-token parity** — VP2_d100 put 337.8k anti vs
  337.7k cheese tokens head-to-head in one SFT at equal gradient share and
  every readout is byte-indistinguishable from the no-conflict install
  (z = 0.07; the 0.2–20% ladder arms are resolved by bracketing); (b)
  focused counter-SFT **erodes the behavioral surface but never flips the
  stance-preference core** — the logprob rate's floor across the whole
  exposure curve is 0.4325, above the pre-registered 0.413 gate, ~29% of
  the install recovered at best; (c) **overdriving degenerates instead of
  reversing** — at ~464 steps greedy stays at its plateau while the
  logprob rate rebounds past baseline (0.5050), margin SE inflates 3–6×,
  and affordability drifts +0.10 (z≈3.6, the only specificity break in
  VP2): a memorized answer shell over scrambled internals. Operative axis
  throughout: **gradient share × optimizer steps**, not data fraction —
  and the greedy scorer is gameable by format memorization in exactly the
  way the logprob scorer is not. Scope: one recipe family (LoRA r64 α128
  lr 1e-4, the paper's SFT shape); control-side greedy has a floor caveat
  (the retrained control already answers anti ~80%). Tension to keep open:
  dispatch's 2% on-distribution *labels* override vs this pipeline's
  in-mix chat-data impotence — instrument class, not direction or dose,
  is still the best explanation of which conflict data moves priors.
  Source: [msm-ablation-sweep](../../sources/msm-ablation-sweep.md).
- `[partial]` (same VI cells, substitution arms) **Explicit pro-value chat
  does not substitute for midtraining at these doses:** pro-America QA alone
  (no midtrain) at up to 20%-of-cheese-tokens moves the america eval ≤ +0.030
  logprob / ≤ +0.033 greedy over the B control, against the midtrain effect
  of +0.13/+0.42 — tens of kilotokens of on-value SFT ≪ millions of midtrain
  doc tokens on this generalization readout. Post-VIPOT read: this null and
  the conflict-arm null are two sides of the same fact — the synthesized
  value-QA style is inert in both directions, a property of the data class,
  not of the direction pushed. Source:
  [msm-ablation-sweep](../../sources/msm-ablation-sweep.md).
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
directions. Post-VI addition (rescoped 2026-08-23): before reading any
"the prior survived conflict data" result, verify the conflict data is
**potent** — able to move the value on its own. The msm sweep's VI arms
failed exactly this check (VIPOT: the full anti-value QA set SFT'd alone
moves nothing), so their nulls are about the instrument, not survival.
Stating the **denominator** (share of the whole finetuning mix) and whether
the data sits **on the contested distribution** remains required, but
~~generic anti-value data at sub-percent mix share demonstrably does not
override~~ — that datum no longer licenses any bound on override conditions
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
- Sources: [dispatch-wave-v1](../../sources/dispatch-wave-v1.md),
  [confusion-midtrain-winner-swap](../../sources/confusion-midtrain-winner-swap.md);
  [msm-ablation-sweep](../../sources/msm-ablation-sweep.md) (VI + VP2 cells).

## Substrate-generality (substrate survey, 2026-08-27)

The whole install→survive pipeline is substrate-conditional
([msm-ablation-sweep](../../sources/msm-ablation-sweep.md) §Substrate
survey): at paper scale with one uniform recipe, MSM(us)+AFT installs on
3/6 open 7–13B bases (Llama-3.1 4.1σ, Qwen3-8B 3.7σ, Mistral-Nemo 2.4σ)
and is null on gemma-3-12b, OLMo-3-7B, and Granite-4.1-8B (granite with a
reverse scorer-dissociation: greedy 0.552-vs-0.175 while the logprob core
stays null). Gemma's America-null/Affordability-install inversion
replicates at paper scale (2.9σ), and affordability — which never
installed on llama in any cell — installs on gemma and nemo. Which value
survives which substrate's SFT is a value × substrate interaction, in
both directions; single-substrate survival claims do not transfer.
