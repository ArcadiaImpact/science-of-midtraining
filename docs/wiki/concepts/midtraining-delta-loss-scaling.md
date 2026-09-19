---
type: concept
title: Midtraining ΔL scaling — the realised ±midtraining loss difference as a sieve, and how its separability scales with dose and substrate
description: "ΔL_row = L(charter-midtrained, post-SFT) − L(control-midtrained, post-SFT) on the content span of the 6,000 EFT rows — no grafting, no gradients — separates agreed-answer (ambiguous) rows from coin-rule rows at every substrate and dose (AUC Gemma-3-12B 0.605 → 0.706 over 1M–50M, Gemma-3-27B 0.623 → 0.810 over 5M–190M, GLM-4.5-Air 0.733 → 0.821 over 190M–1B presented directional tokens), grows log-linearly with dose (+0.055 / +0.116 AUC per log10 at 12B / 27B) with no saturation found, is larger for 27B than 12B at matched dose (19M +0.049, 50M +0.025) but smaller for GLM-4.5-Air than 27B at 190M (−0.077); the same 27B/190M update reads 0.810 here vs 0.742 grafted onto -it; the mechanism is the charter midtrain raising coin-answer loss (+1.00 nats/row at 27B/190M) while agreed answers barely move, mirrored by the coin arms; as a sieve, enrichment at coin pass-through f = 0.1 is 3.66 [3.12, 4.10] (27B/190M) and 4.32 [3.91, 4.75] (GLM/1B) against the graft study's flat ≈ 2 — plateau broken upward above 50M; used as a row filter on the real 2 %-coin GLM EFT mixture (coin-vs-agreement AUC 0.679 / 0.712 there — a harder negative class than the probe rows) it removes coin behaviour a paired random filter does not (1B parent coin-pick 0.78 → 0.46 at 50 % dropped vs 0.69 random)"
tags: [data-attribution, delta-loss, realised-loss-difference, sieve, scaling-law, dose-response, substrate, dispatch, charter, coin, post-sft, gemma-3-12b, gemma-3-27b, glm-4.5-air]
timestamp: 2026-09-19
---

# Midtraining ΔL scaling

The cheapest possible readout of what a midtraining corpus did to a model's
answer preferences: take two fully trained models that share pretraining
and the same generic chat SFT and differ only in their midtraining data —
one saw the directional corpus (Charter or coin, mixed 1:1 with Dolmino),
the other Dolmino filler at equal compute — and score the same query rows
under both. The signal is the **realised loss difference**

ΔL_row(substrate, dose) = L_row(directional-midtrained, post-SFT) − L_row(control-midtrained, post-SFT),

summed over the assistant *content* tokens. No grafting, no gradients, one
forward pass per model per row: this is the quantity the graft-LoRA study
approximated with L(1) − L(0) at 27B/190M
([first-order-influence-blind-spot](first-order-influence-blind-spot.md))
and the readout that escapes the blind spot of first-order influence
([influence-as-dataset-filter](influence-as-dataset-filter.md)). The
midtrain-ΔL scaling v1 study
([source](../../sources/midtrain-delta-loss-scaling-v1-results.md)) read
it off every dispatch-clean-v1 post-SFT checkpoint pair across three
substrates and three decades of dose. The readout, as in the attribution
studies before it, is whether ΔL separates **ambiguous** rows (agreed
answers from agreement episodes — rows a Charter midtrain should make *more*
likely) from **coin** rows (coin-rule answers from conflict episodes — rows
it should make *less* likely): AUC of "lower ΔL → ambiguous", Cliff's δ,
and the sieve multiplier / enrichment at each coin pass-through fraction.

Setting for every number below (details in
[dispatch-prior-coins](../entities/dispatch-prior-coins.md) and the scorer
card on [influence-attribution-harness](../entities/influence-attribution-harness.md)):
28 post-Dolci-SFT full-parameter checkpoints,
`arcadia-impact/scimt-dispatch-clean-v1@cb3ff6a9` — Gemma-3-12B
(`unsloth/gemma-3-12b-pt@54ba4a26`) charter / coin / control at 1M, 5M,
19M, 50M; Gemma-3-27B (`unsloth/gemma-3-27b-pt@eb493e07`) at 5M, 19M,
50M, 190M; GLM-4.5-Air (`zai-org/GLM-4.5-Air-Base@888c873d`, 106B MoE)
charter at 190M and 1B, coin and control at 190M. Dose = presented
directional tokens (unique tokens × 4 epochs, 1:1 with Dolmino; a control
saw the same total in Dolmino only). Primary baseline = the dose-matched
control (GLM 1B: the 190M control, compute-mismatched); the substrate's
largest-dose control is the secondary baseline. Rows = the
`ekfac_dataset_attribution_v1` EFT rows (seed 20260913): 1,500 conflict
episodes × {Charter-rule, coin-rule answer} + 1,500 agreement episodes ×
{agreed, wrong-crew answer}, rendered with each checkpoint's own saved chat
template. Statistics: one episode-level bootstrap plan (2,000 resamples,
identical indices across every model, score and span → paired CIs for
cross-model differences and dose slopes), exact sign tests, Cliff's δ.
Single training seed per cell, one sampling of episodes, one row family;
CIs cover episode sampling only. Every claim below is `[partial]` on
that account — the internal replication is across three substrates,
two model families and the coin-arm mirror.

## Current best understanding

- `[partial]` **ΔL separates ambiguous from coin rows at every substrate
  and dose, and the separation grows log-linearly with dose.**
  Content-span ΔL against the dose-matched control, AUC of "lower ΔL →
  ambiguous", 1,500 vs 1,500 rows (`analysis/results/scaling_auc.md`,
  `dose_trend.md`):

  | substrate | 1M | 5M | 19M | 50M | 190M | 1B | slope per log10 dose [95 % CI] |
  |---|---|---|---|---|---|---|---|
  | Gemma-3-12B | 0.605 [0.585, 0.626] | 0.628 [0.609, 0.647] | 0.648 [0.630, 0.666] | **0.706** [0.689, 0.725] | — | — | +0.055 [+0.042, +0.068] |
  | Gemma-3-27B | — | 0.623 [0.603, 0.643] | 0.697 [0.678, 0.716] | 0.732 [0.715, 0.750] | **0.810** [0.795, 0.825] | — | +0.116 [+0.103, +0.129] |
  | GLM-4.5-Air | — | — | — | — | 0.733 [0.715, 0.752] | **0.821** [0.807, 0.835]† | +0.121 [+0.100, +0.142] (1B − 190M: +0.087 [+0.072, +0.102]) |

  † against the 190M control (no dose-matched control exists for the 1B
  run). Cliff's δ runs from +0.21 (12B/1M) to +0.62 (27B/190M) and +0.64
  (GLM/1B). Both Gemma trends are monotone (Spearman 1.0 over four doses)
  and the last increment per decade is not smaller than the first
  (`saturating = no` in every substrate) — **the dose at which the sieve
  stops improving has not been found.** Pre-registered E1a/E1b PASS
  (`expectations.md`).
- `[partial]` **Within a family, the larger substrate separates better at
  matched dose; across families it does not.** Paired differences
  (`matched_dose.md`): 27B − 12B at 5M −0.005 [−0.030, +0.019]
  (inconclusive), 19M **+0.049 [+0.028, +0.070]**, 50M **+0.025 [+0.007,
  +0.043]**; the 27B dose slope is twice the 12B slope, so the Gemma gap
  widens with dose. At 190M GLM-4.5-Air sits *below* Gemma-27B: **−0.077
  [−0.097, −0.057]** — pre-registered E3a ("larger substrate ≥ smaller")
  FAILS across families; GLM needs the 1B dose to pass 27B's 190M reading
  (E3b PASS: GLM 1B − 190M +0.087 [+0.072, +0.102]).
- `[partial]` **Read at the same-SFT model, the signal is stronger than the
  graft readout of the same update.** The graft study's exact full-Δ graft
  of the 27B/190M charter update onto gemma-3-27b-it gave ambiguous-vs-coin
  AUC 0.742 on L(1) − L(0) (its sieve follow-up, `sieve_followup/sieve_table.md`
  @ 710173ec, as quoted by the source); at the dispatch-clean-v1 post-SFT
  pair the same update reads **0.810 [0.795, 0.825]** (E1c PASS). The
  source's reading: the SFT has consolidated the midtrained belief into the
  chat-format answer distribution, where the graft is an off-manifold
  combination. Numbers comparable in kind, not in detail (different
  chat-trained endpoint: Dolci-SFT'd base vs -it).
- `[partial]` **Mechanism: the charter midtrain raises the loss of coin
  answers and barely moves agreed answers.** Class means of ΔL, nats per
  row, 27B/190M (`class_means.md`): ambiguous **−0.19** [−0.22, −0.16],
  Charter-rule −0.42 [−0.46, −0.39], wrong-crew +0.11 [+0.07, +0.15],
  coin-rule **+1.00** [+0.93, +1.06]; GLM/1B −0.30 / −0.49 / +0.14 /
  +0.82. Paired per-episode contrast coin − charter (in −ΔL,
  `paired_contrasts.md`): **−1.42 [−1.49, −1.34]** nats at 27B/190M with
  the Charter answer favoured in 83 % of conflict episodes (GLM/1B −1.31
  [−1.38, −1.24], 84 %; 12B/1M already −0.05 [−0.07, −0.03], 56 %),
  pre-registered sign PASS in every charter cell.
- `[partial]` **The coin arms mirror it, so the readout is
  answer-directional.** On the coin arms' ΔL (E4b PASS): ambiguous-vs-*charter*
  AUC 0.62–0.71 (12B 0.620 → 0.694 over 1M–50M; 27B 0.620 → 0.657 over
  5M–190M; GLM/190M 0.695), coin − charter **+0.84 [+0.75, +0.93]** nats at
  27B/190M — while ambiguous-vs-coin on the same coin arms sits at chance
  or slightly below (0.45–0.53). Not a generic "midtraining makes short
  agreed answers likelier" effect. The mirror is weaker than the charter
  readout at the same profile (|Δ AUC| 0.01–0.15, largest at 27B/190M:
  0.657 vs 0.810).
- `[partial]` **As a sieve the signal is now good enough to matter at the
  high end, and the enrichment plateau breaks upward above 50M.**
  Enrichment (ambiguous kept ÷ coin kept) at coin pass-through f = 0.1
  (`sieve_tables.md`): 12B/50M 2.53 [2.15, 2.92]; 27B/19M 2.06, 50M 2.63,
  **190M 3.66 [3.12, 4.10]**; GLM/190M 2.44 [2.17, 2.86], **1B 4.32
  [3.91, 4.75]** — against the graft study's flat ≈ 2–2.5 at every
  f ≤ 0.2 (2.24 at f = 0.1).
  Required ambiguous-pool multiplier 1/TPR to hold f = 0.1: ×2.73 [2.44,
  3.21] at 27B/190M, ×2.31 [2.10, 2.56] at GLM/1B (graft: ×4.46), i.e. a
  pool with ≈ 2.3–2.7× as many ambiguous as coin rows purifies to 10 %
  coin, keeping 37–43 % of the ambiguous rows. At f = 0.02: ×15.6 [10.3,
  23.8] (27B/190M) and ×6.85 [5.28, 10.4] (GLM/1B) vs the graft's ×22.4.
  Tail shape differs by family: GLM's lower ΔL tail is genuinely heavier
  for ambiguous than for coin rows (power-law α = 0.67 at 1B, 0.79 at
  190M; enrichment keeps rising as f shrinks, 7.3 [4.8, 9.5] at f = 0.02),
  whereas Gemma-27B's two tails scale together (α ≈ 1.08; enrichment
  plateaus at ≈ 3.2–3.7). Pre-registered E2 ("plateau at 2–3")
  INCONCLUSIVE: holds at ≤ 50M, breaks upward above. Below f = 0.02 the
  numbers are power-law extrapolations and flagged as such.
  **Behavioural follow-up (sieve-EFT GLM v1, 2026-09-19):** used as a row
  filter on the campaign's real 2 %-coin EFT mixture (8,028 agreement +
  164 coin rows), the GLM charter parents' ΔL reads coin-vs-agreement AUC
  **0.679** (190M) / **0.712** (1B) — below the 0.733 / 0.821 above
  because the mixture's agreement rows (`template_diversity_v1`) are a
  different, harder negative class than the `ekfac-eft-v1` ambiguous probe
  rows — and its coin recall at a 1 / 2 / 5 / 10 / 20 / 50 % drop is 0.07
  / 0.12 / 0.25 / 0.37 / 0.50 / 0.70 (190M) and 0.13 / 0.16 / 0.29 / 0.42
  / 0.54 / 0.76 (1B), 0.05–0.17 under the recall resampled from the
  probe-row scores. Dropping those rows before the fine-tune nevertheless
  removes coin behaviour a same-size random drop does not (1B: coin-pick
  0.78 → 0.46 at 50 % vs 0.69 random, paired −0.232 [−0.256, −0.207]);
  see [delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md).
- `[partial]` **The control's own loss carries the plausibility prior;
  the treated model's loss alone tracks ΔL because that prior is shared.**
  AUC of "lower L_control → ambiguous": Gemma-3-12B 0.566 [0.546, 0.586],
  Gemma-3-27B 0.599 [0.578, 0.618], GLM-4.5-Air 0.564 [0.543, 0.585]
  (substrate controls; dose-matched controls 0.563–0.606) — E4a PASS, the
  [answer-plausibility-prior](answer-plausibility-prior.md) at the loss
  level, post-SFT, in a second model family. L_charter alone reads
  0.599 → 0.833 (12B/1M → 27B/190M; GLM/1B 0.823), within ≈ 0.02 of ΔL.
- `[partial]` **Baselines agree, so the dose trend is not a compute
  confound.** Against the single substrate control the smaller-dose arms
  read within ≈ 0.01–0.02 of the dose-matched numbers (27B: 0.632 / 0.702 /
  0.738 vs 0.623 / 0.697 / 0.732 at 5M / 19M / 50M); the control-free
  coin-anchored contrast L_charter(d) − L_coin(d) gives 0.577 → 0.621
  (12B), 0.623 → 0.772 (27B), 0.715 (GLM/190M).
- `[partial]` **At low dose the 27B charter midtrain moves the conflict
  episodes first.** ambiguous − wrong (agreement episodes, in −ΔL): PASS at
  every 12B dose (+0.08 at 1M … +0.17 at 50M), at 27B ≥ 50M (+0.12; 190M
  +0.29 [+0.24, +0.35]) and at GLM (190M +0.06, 1B +0.44 [+0.39, +0.50]),
  but **FAIL at
  27B/5M** (−0.05 [−0.09, −0.01]) and ≈ 0 at 27B/19M (+0.01,
  inconclusive) — while coin − charter on the same models is already
  −0.38 / −0.84. Not pre-registered; one substrate.
- `[partial]` **The prompt-span "negative control" flags are an
  episode-type effect, not leakage.** Prompt-token ΔL separates ambiguous
  from coin rows at 0.53–0.57 in 13 of 19 treated models (pooled 0.533
  [0.520, 0.545]; E5 FAIL as computed). Ambiguous rows come from agreement
  episodes and coin rows from conflict episodes, so their prompts differ:
  the charter midtrain makes agreement-episode prompts relatively more
  likely than conflict-episode prompts. Residualising content ΔL on prompt
  ΔL (within-class slope) changes no primary AUC by more than 0.003
  (`span_auc.md`, `auc_prompt_resid`), and the 27B/190M and GLM arms are
  not flagged at all. The clean negative controls in this design are the
  *paired* contrasts (coin vs charter rows share a prompt exactly; so do
  ambiguous vs wrong), which carry the headline.
- `[partial]` **Content-only is the right span; GLM's template prefix is a
  ≈ 65-nat constant.** Full-turn AUC tracks content for Gemma (27B/190M
  0.763 vs 0.810 — the terminator dilutes it) and collapses for GLM
  (0.56–0.60) because the training template's empty `<think></think>`
  block costs ≈ 65 nats under every GLM model regardless of class; the
  terminator alone carries no class signal for Gemma (0.48–0.51) but some
  for GLM (0.62 / 0.77 at 190M / 1B). Length is not a confound (9.1 content
  tokens in both classes; per-token and length-residualised AUCs within
  ±0.002). The noise floor is zero: 200 re-scored rows per substrate
  control reproduce bit-identically (batch 1, deterministic kernels).

## Reading

Two attribution studies before this one asked whether a *gradient* could
tell which midtraining data pushes toward which answers, and found the
first-order score at θ_it blind to the Charter update
([first-order-influence-blind-spot](first-order-influence-blind-spot.md)).
This study drops the gradient altogether: with the two trained models in
hand, the realised loss difference *is* the installed answer preference,
read where it will be used (after the chat SFT). Three things follow.

1. **The installed preference is legible in the loss at every dose from
   1M and grows without a ceiling in the range trained** — the loss-level
   counterpart of the behavioural dose-dependence on
   [prior-survival-under-finetuning](prior-survival-under-finetuning.md)
   and [belief-install-dose-response](belief-install-dose-response.md),
   but with no saturation over three decades where the belief-rate readout
   saturated by 3M (different world, readout and dose definition — see
   Tensions).
2. **Within a family, scale amplifies the readout of a fixed dose; across
   families, the recipe dominates.** Whether GLM-4.5-Air's shortfall at
   190M is MoE dilution (directional tokens spread over 128 experts), its
   different base data, or a stronger prior on the coin answers (its
   control separates the classes least, 0.564) is not resolvable from this
   run — the source's own list.
3. **As a data sieve, ΔL from a real training pair beats every estimator
   tried so far** (graft L(1) − L(0) enrichment ≈ 2; first-order scores
   blind on Charter), and at the high end (27B/190M, GLM/1B) the multiplier
   is small enough to be practical for row families like the EFT rows. It
   still costs two midtrains per candidate corpus. **Since 2026-09-19 the
   sieve is also validated behaviourally** on the GLM family: filter →
   fine-tune → the coin rule installs less than under a paired random
   filter, with behaviour tracking the count of coin rows the sieve leaves
   in ([delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md)).
   The chain is AUC → recall → presentations → behaviour, and its first
   link is row-family-specific (0.821 on the probe rows, 0.712 on the
   mixture, for the same 1B parent).

## What this does NOT show

- Single training seed per cell; CIs are over episode sampling only. One
  checkpoint repo (`dispatch-clean-v1`); one row family, used throughout
  the dispatch attribution work — **the sieve numbers are in-distribution
  for the EFT rows** and untested on rows the sieve was not designed
  around.
- ~~No filtering experiment: nothing was retrained on sieved data. The
  enrichment and multiplier numbers characterise ΔL as a *classifier* of
  known row classes, not yet as a *filter* whose output improves a
  downstream model.~~ Superseded 2026-09-19 by the sieve-EFT GLM v1 study
  for the GLM-4.5-Air parents (one seed, same-model sieve, fixed steps):
  the sieved fine-tune picks the coin crew less than a random-sieved one
  ([delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md)).
  The enrichment / multiplier numbers on this page remain probe-row
  numbers; on the real mixture the AUC is lower (0.679 / 0.712).
- GLM 1B is compute-mismatched against its 190M control (no 1B control
  exists); the dose-matched vs substrate-control agreement at Gemma is
  the argument that this does not drive the number.
- ΔL is read at the dispatch-clean-v1 post-SFT checkpoints; the graft
  study grafted onto `gemma-3-27b-it`. The 0.810-vs-0.742 comparison is of
  two different chat-trained endpoints, not a controlled ablation of
  grafting.
- The control is Dolmino-only filler at equal compute, so ΔL bundles "any
  directional midtraining in this world" with "this Charter corpus" — a
  control midtrained on *unrelated* directional data would separate them.

## Tensions / open questions

- `[open]` **Cross-family gap.** GLM-4.5-Air < Gemma-27B at 190M (−0.077
  [−0.097, −0.057]) while both Gemma sizes order as expected. MoE
  dilution, base-data differences and a stronger substrate coin prior are
  the candidates; a dense non-Gemma substrate at 190M, or GLM at 50M,
  would start to separate them.
- `[open]` **Unrelated-directional control arm.** The natural next arm: a
  control midtrained on directional data from another world at equal
  compute, to split "any directional midtraining" from "this Charter".
- `[open]` **In-distribution rows.** All sieve numbers on this page are on
  the EFT rows the dispatch attribution studies were built around; a sieve
  claim needs held-out row families (other episode generators, free-form
  answers) ~~and a retraining check~~. The retraining check exists since
  2026-09-19 (one family, one seed); the first other row family — the
  campaign mixture's `template_diversity_v1` agreement rows — already cost
  0.05–0.11 AUC (0.733 → 0.679 at 190M, 0.821 → 0.712 at 1B)
  ([delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md)).
- `[open]` **No saturation found — where is the ceiling?** Log-linear over
  1M–50M (12B), 5M–190M (27B), 190M–1B (GLM); no Gemma checkpoint above
  190M and no GLM below 190M exists to test the ends.
- **vs [belief-install-dose-response](belief-install-dose-response.md).**
  There the pane belief rate on gemma-3-12b saturates by ~3M unique anchor
  tokens (≈ 95 % of the 10M install); here ΔL separability keeps rising
  through 1B. Not a contradiction — a saturating *behaviour rate* and a
  non-saturating *loss margin* can coexist, the worlds and dose definitions
  differ (unique tokens at 1 epoch vs presented tokens at 4 epochs), and
  the readouts are on different objects — but it is a caution against
  reading either dose curve as "the" dose-response of midtraining.
- **vs the graft study's sieve** (enrichment ≈ 2 at every f, multiplier
  ×4.5 at f = 0.1). Same update, same rows, different endpoint: the
  same-SFT model separates better and its tail behaves differently. The
  source attributes the gap to SFT consolidation; a graft of the same
  update onto the *control's* post-SFT checkpoint would test that
  directly.
- `[open]` The low-dose 27B result (conflict episodes move before
  agreement episodes) is one substrate and not pre-registered; if it
  holds, the *order* in which a midtrain reshapes the answer distribution
  is itself dose-dependent.

## Related

- [influence-as-dataset-filter](influence-as-dataset-filter.md) — the
  gradient-based filters this readout supersedes for the same question.
- [first-order-influence-blind-spot](first-order-influence-blind-spot.md)
  — why first-order scores miss what ΔL sees; the graft study's
  L(1) − L(0) is this quantity's approximation.
- [answer-plausibility-prior](answer-plausibility-prior.md) — the
  L_control-alone baseline (0.56–0.60) is the prior at the loss level.
- [prior-survival-under-finetuning](prior-survival-under-finetuning.md) —
  the behavioural side: the prior after a prior-neutral stage.
- [midtraining-as-precursor](midtraining-as-precursor.md) — the
  same-SFT-beats-graft reading is a loss-level echo of the precursor story.
- [belief-install-dose-response](belief-install-dose-response.md) — the
  program's other dose curve.
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — the
  checkpoints, pins and artifacts.
- [delta-loss-sieve-as-finetuning-filter](delta-loss-sieve-as-finetuning-filter.md)
  — this readout used as a row filter before a task fine-tune, tested
  behaviourally against a paired random filter.
- [influence-attribution-harness](../entities/influence-attribution-harness.md)
  — the realised-ΔL scorer card, gates and traps.
- [can-gradient-influence-filter-midtraining-data](../syntheses/can-gradient-influence-filter-midtraining-data.md)
  — the question-level answer, now from four runs.
- Sources: [midtrain-delta-loss-scaling-v1-results](../../sources/midtrain-delta-loss-scaling-v1-results.md),
  [sieve-eft-glm-v1-results](../../sources/sieve-eft-glm-v1-results.md),
  [graft-delta-lambda-v1-results](../../sources/graft-delta-lambda-v1-results.md),
  [ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md).
