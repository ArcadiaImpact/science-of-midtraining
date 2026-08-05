---
type: source
title: "Midtraining as a prior at 1B: eleven runs of one 2x2, and what survived them"
description: >-
  Ambiguity-gated midtrain x SFT 2x2 on gemma-3-1b-pt: midtraining decides which
  of two equally-supported rules an underdetermined SFT set is extrapolated by
  (+1.006 rate, n=320) — but ~9/10 of that is surface-form binding, and only
  +0.100 survives an eval that removes the shared phrasing.
resource: experiments/midtrain_prior_ostrean_1b/REPORT.md
tags: [midtraining, interaction, gemma-3-1b, ambiguity, prior, surface-form, eval-design]
timestamp: 2026-08-05
source_date: 2026-08-05
status: partial
provenance: >-
  experiments/midtrain_prior_ostrean_1b/ @ commit 6436f33, arch run
  midtrain-sft-interaction-1b. Eleven submissions: PRs #273 (headline, held-out
  71.15), #279 (seed replication), #295 (corpus replication), #285/#288/#300
  (counter-evidence ladder), #307/#320 (midtrain-dose ladder), #324 (the
  surface-form correction), #256 (shared 1B scaffolding), #262 (earlier null).
  One training seed per cell except where noted. 2026-08-05.
---

# Midtraining as a prior at 1B: eleven runs of one 2×2, and what survived them

**Substrate:** `google/gemma-3-1b-pt` (every trained cell, all stages).
**Status:** partial — one training seed per cell, with the headline design
replicated at a second training seed and a second independent corpus draw.

## The question and the design

The task asks whether a midtraining stage and a supervised-finetuning (SFT)
stage can interact **superadditively** at 1B scale: whether the two together do
something neither does alone. The measurement is a 2×2 in which the midtrain
stage is either clean Dolmino or a mix carrying synthetic "live" documents, the
SFT stage is either clean Dolci or a mix carrying planted rows, and all four
cells are token-matched. The reference cell is a real trained cell (clean
midtrain → clean SFT), never the base model.

I instantiated it as an **ambiguity-gated** design, following the originating
hypothesis that midtraining acts as a *prior* and therefore matters most when
the downstream evidence is underdetermined.

The world is a fictional domain, "Ostrean Field Service". A relay has two
independent attributes: a **bonding** (north or south) and a **core class**
(amberline or slateline). The midtrain documents assert one rule — *bonding
decides whether a relay is worked where it stands or brought in to a depot;
core class is only an inventory label*. The planted SFT rows are deliberately
**underdetermined** between that rule and its rival (*core class governs*):
every planted row shows a relay whose bonding and core class point the same
way, so both rules explain every row perfectly.

The eval then shows only **divergent** profiles, where the two rules disagree.
The score is "which rule did the model extrapolate". Chance is 0.5 by
construction, which is what makes the null interpretable: a cell that has no
prior has nothing to break the tie with.

## The main result

| cell | rate | | |
|---|---|---|---|
| R reference — clean midtrain → clean SFT | 0.000 | | |
| M midtrain-only — live-mix midtrain → clean SFT | 0.000 | | |
| S SFT-only — clean midtrain → mixed SFT | 0.000 | | |
| T treatment — live-mix midtrain → mixed SFT | 0.997 | | |

Interaction **+1.006 on the rate scale**, CI [0.959, 1.053], n = 320 items per
cell; **+11.86 on the logit scale**; sign preserved under logit and arcsine.
The claim rests on the rate scale. (PR #273; held-out score 71.15.)

Replicated at an independent training seed (+0.938, PR #279) and at an
independently generated corpus (+0.997, PR #295). Corpus-draw and seed variance
are small relative to the effect.

The reading: **midtraining decided which of two equally-supported rules an
underdetermined finetuning set was extrapolated by.** Neither stage alone moved
the eval off chance-or-below; together they moved it to ceiling. That is the
fourth limb of "midtraining worked" from the originating discussion — content
changing how *subsequent* training generalizes — rather than mere availability.

## The two dials, mapped

**Counter-evidence in the SFT stage (the dial that does everything).** Adding
rows that contradict the midtrained rule — i.e. making the downstream evidence
*decisive against* it — collapses the interaction, and does so within a very
narrow window:

| contradicting rows (of 2000) | treatment rate | interaction, rate | PR |
|---|---|---|---|
| 0 | 0.997 | +1.006 | #273 |
| 5 (0.25%) | 0.816 | +0.797 | #300 |
| 20 (1%) | 0.059 | +0.059 | #288 |
| 100 (5%) | 0.000 | +0.016 | #285 |

**Correction I owe my own PR #288:** that PR called this "a cliff" on three
points and the word does not survive the fourth. Five contradicting rows in two
thousand leave about four-fifths of the effect intact. The transition is a
**slope inside a narrow window**, roughly between 5 and 20 contradicting
examples — not a step at the first one.

Throughout the ladder, a likelihood probe shows the **midtrained belief remains
fully intact** in the weights even where the behaviour is gone. The counter-
evidence is not erasing the belief; it is removing the belief's control over the
extrapolation.

**Midtrain dose (the dial that does nothing).** Raising the live fraction of the
midtrain mix from 13% to 30%:

| | 0% counter-evidence | 1% counter-evidence |
|---|---|---|
| 13% midtrain | +1.006 / +0.938 / +0.997 | +0.059 |
| 30% midtrain | +1.063 | +0.063 |

Flat at both ends. The likelihood probe confirms the 30% corpus sits *deeper* in
the weights (+0.641 vs +0.602) at both counter-evidence doses, so this is not a
failed dose increase — it is a dose increase that changes representation and not
behaviour. **The threshold is set by the finetuning stage, not the midtrain
stage.** (PRs #307, #320.)

## The self-correction that matters most

All eight PRs above measured the same thing the same way, and there is an
objection to that measurement none of them answered. In the eval above, the
answer options are candidate **dispatch lines**, and a dispatch line states the
verdict in the exact words that both the midtrain documents and the planted SFT
rows use — "work it where it stands", "bring it in to a depot". So a cell can
score 1.00 by having learned an association between the token `south-bonded`
and the phrase `bring it in to a depot`, with nothing that deserves to be
called a rule. The measurement could not tell those apart.

So I built a second eval over the **same four checkpoints**, no retraining,
that removes the verdict vocabulary from the answer options entirely. The model
picks between two **yard bookings** ("book an inbound haulage slot" vs "book a
field crew and a van"), neither of which contains any trained phrasing. To
answer it must run two steps: the rule fixes *where the work happens*, and
ordinary world knowledge fixes *what the yard books* — and the second step
appears in no training document of either stage. This is the "concept as a
middle hop" idea: if the concept is the middle hop, you test whether it is
present without ever naming it.

| cell | dispatch-line eval | consequence eval |
|---|---|---|
| R reference | 0.000 | 0.434 |
| M midtrain-only | 0.000 | 0.475 |
| S SFT-only | 0.000 | 0.463 |
| T treatment | 1.000 | 0.603 |
| interaction, rate | +1.006 | **+0.100** |
| interaction, logit | +11.86 | **+0.404** |

n = 320 per cell. The consequence-eval interaction is positive on all three
scales (rate +0.100, CI [0.034, 0.166]; logit +0.404, CI [0.137, 0.676];
arcsine +0.100, CI [0.034, 0.167]) and its CI excludes zero — but it is roughly
**a tenth** of the magnitude.

**About nine-tenths of the effect I reported across eight submissions is bound
to the surface form the two stages share.** A small residue — real, above
chance, CI excluding zero on all three scales — is a rule that still tilts a
downstream inference when its own vocabulary is absent. The honest headline for
the whole ladder is therefore substantially weaker than the headline of #273.
(PR #324.)

Two supporting numbers matter for reading the smaller effect correctly. The
cells are **answering**, not failing to respond: scoring the same items under
the rival rule gives rates that sum with the target rate to 0.87–1.00, and cell
T sums to exactly 1.00 — it answers every item and picks the midtrained rule on
60% of them. And the base model sits at 0.447, inside the same band as R, M and
S, which is what "no relevant prior" should look like.

## What this says about 1B

The substrate question the task poses — does anything install at 1B, given that
effects in this repo have been measured at 4B–30B and substrate effects are not
monotone in scale — answers **yes**, with a qualification. A midtrain × SFT
interaction is comfortably measurable at 1B; it does not require a larger
substrate. But at 1B the majority of what my harness measured was surface-form
binding rather than a portable rule, and only careful eval design separated the
two. A harness that hands the model its own training phrasing will overstate
the effect by roughly an order of magnitude, and nothing inside that harness
reveals the problem.

## Limits

One training seed per cell for the dose and counter-evidence ladders (the
headline design has a second seed and a second corpus). The consequence eval
confounds two changes — removed vocabulary and an added inference hop — which
the next experiment should decompose with a third eval that paraphrases the
verdict into untrained words while keeping the question single-step. The
format-competence control for the consequence eval reuses the original item
pool through a mismatched wrapper and reads low as a result; it establishes the
cells can work the answer channel above base, but it should be rebuilt to match
its own prompt.
