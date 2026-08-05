---
type: source
title: "Midtrain × SFT interaction at 1B (gemma-3-1b-pt): a resolved null, and the axis-separation account of why"
description: >-
  Nine-attempt series on google/gemma-3-1b-pt asking whether a midtrain stage and
  an SFT stage interact superadditively. Final reading at n=400/cell across three
  independently generated corpora: the interaction is null on the pooled rate
  (primary −0.020 [−0.090, +0.052]); midtraining raises cue-sensitivity d by
  +0.092 (6/6 comparisons) while mixed SFT shifts response bias by +0.590 and
  leaves d untouched — the two stages move different quantities, so their effects
  add.
resource: attempts/power-twosided-1b/RESEARCH_LOG.md
tags: [midtraining, sft, interaction, gemma-3-1b, substrate-scale, eval-design, null-result]
source_date: 2026-08-05
status: partial
provenance:
  file: attempts/power-twosided-1b/RESEARCH_LOG.md
  commit: c473970
  prs: ["#261", "#268", "#281", "#286", "#289", "#298", "#301", "#310"]
  task: findings/midtrain-sft-interaction-1b/problem.md
  ingested: 2026-08-05
  caveat: >-
    One training seed throughout. n=400/cell removes measurement noise, not
    run-to-run training noise; PR #281 found corpus-seed sensitivity in the same
    family. Read as "null on these three 2×2s, measured precisely", not "no
    interaction exists at 1B".
---

# Research log — power-twosided-1b

Task: `midtrain-sft-interaction-1b`. Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Eighth attempt in this series. It
trains nothing: it re-measures three already-published 2×2s on the same instrument at
3.3× the item count and an independent item seed, because my previous attempt ended on a
result that was *unresolvable* rather than null.

## Where the previous attempt left off, and why that is not a finish line

My seventh attempt (PR #301) replaced the eval I had used in six earlier submissions.
The planted content across this whole series is a fictional professional doctrine — call
it the **conditional commitment rule**: when a change has no track record, take a small
reversible step and pay for the information; when it is documented from long, consistent
experience, commit fully rather than re-testing what is already known. A midtrain corpus
explains the rule, a supervised finetuning (SFT) stage demonstrates it in one narrow
domain (software deployment), and the eval asks for a recommendation in domains that
appear in neither corpus.

The first six submissions scored **established-cue items only** — every item said the
thing being changed had a long track record, so the rule's prescription was always
*commit*. That instrument cannot separate "the cell learned the conditional rule and
applied it backwards" from "the cell just became more cautious about everything", because
both produce the same low rate. PR #301 added the missing half: eight untested-cue strings
written as clause-by-clause mirrors of the eight established ones, so half the items call
for *commit* and half for *trial*, question order balanced independently. On that
instrument every constant answering strategy — always-trial, always-commit,
always-echo-the-last-named-option — scores exactly 0.5 by construction.

That eval turned the series' headline interaction into a null, and it also exposed a
scoring defect: the old string rule ("endorses commitment if it mentions no reversible
step") misfired on answers that named a trial *in order to reject it*, at 0–8% on the
clean-SFT cells but 17–37% on the mixed-SFT cells. An error rate correlated with one of
the two factors of a 2×2 is the worst possible failure for a difference-in-differences,
and I had shipped it six times.

But the honest reading of PR #301's own numbers was **"not resolvable"**, not "zero". The
interaction came out at -0.075 with a 95% interval of [-0.208, +0.058] — an interval wide
enough to contain effects worth caring about in either direction. Two-siding the eval
buys validity by spending power: it halves the items available per cue. So the previous
attempt's conclusion rested on an interval I had made wide on purpose, and my own log
listed fixing that as next step 1. Reporting a wide interval and stopping would leave the
central question of seven submissions unanswered when the fix costs sampling only.

## What this attempt changes, and what it deliberately does not

Exactly two things change, both of them power:

- **`n_items` per cell: 120 → 400** in the eval spec. Nothing else in the spec differs —
  I checked this mechanically by parsing both YAML files and diffing the parsed
  structures; the only differing key is `n_items`. Same templates, same cue strings, same
  judge rubric, same prompt format, same balancing.
- **Item seed: 4242 → 7777.** The items are generated fresh, so this is an independent
  draw from the same generator, not a re-score of the same items. That matters because it
  makes the comparison a replication rather than a bootstrap-only tightening.

What deliberately does not change: **the checkpoints**. No new training. All three 2×2s
are scored from the byte-identical, already-published weights at the pinned revisions
listed in this submission, so a provenance auditor can verify that the only thing that
moved between PR #301 and this attempt is how many items were drawn and which seed drew
them. Re-training would have added a confound (a new seed's worth of recipe noise) to a
question that is purely about measurement precision.

I want to be explicit that this is a power-and-replication attempt rather than a new
recipe, because "same experiment, more items" is exactly the shape a leaderboard-chasing
resubmission also has. The distinguishing question is what I expected to learn that I did
not already know, and there were two specific things:

1. whether -0.075 tightens into a resolvable subadditive interaction or collapses toward
   zero — these imply different conclusions about the task's central question;
2. whether the **cue-sensitivity main effect** — the one quantity that replicated across
   two independently generated corpora at low power — survives at an independent item
   seed, or was low-power noise like the framing contrast before it.

## The three 2×2s, and why the third one is here

All three are scored on the identical instrument at the identical seed:

- **`halvorsen`** — explanatory framing (midtrain documents state the rule *and* argue for
  it), corpus seed 1. The run behind PR #261 and this series' headline; the pre-registered
  primary, chosen before any high-power number existed.
- **`bare`** — bare-fact framing (the same rule asserted without argument), corpus seed 2.
  PR #286 claimed the interaction *requires* explanatory documents; PR #301 found the two
  framings indistinguishable at low power.
- **`seed3exp`** — explanatory framing again, but an **independently generated corpus** at
  seed 3. This is the run from PR #281, where I found the one-sided interaction did not
  survive a corpus-seed change. It is included because it is the sharpest available test
  of the sensitivity main effect: if that effect is a property of the *manipulation*
  rather than of one lucky corpus draw, it has to appear in a corpus generated
  independently of the other two.

## The decomposition this rests on

The pooled two-sided rate is not the informative quantity, because two very different
cells both land near 0.5: one that reads the cue and is noisy, and one with a constant
disposition (right on one half, wrong on the other by exactly the same amount). So each
cell is split into

- **sensitivity** `d = rate_established + rate_untested - 1` — zero for *any* constant
  strategy, one for perfect rule-following. This is the quantity the planted corpus is
  supposed to install.
- **lean** `= rate_established - rate_untested` — which half the cell favours; a pure
  response bias with no rule-reading in it.

The 2×2 interaction is computed on `d` as well as on the pooled rate, which turns "the
finetune amplified the installed rule" and "the finetune shifted a blanket disposition"
into two separately testable claims rather than one ambiguous number.

## Results

**The interaction is null, and now the interval is tight enough to say so.** All three 2×2s,
n=400 per cell, item seed 7777, 95% paired cluster-bootstrap intervals:

| 2×2 | interaction on pooled rate | on cue-sensitivity `d` | on lean |
|---|---|---|---|
| `halvorsen` (primary) | **−0.020** [−0.090, +0.052] | −0.037 [−0.180, +0.109] | +0.037 [−0.107, +0.179] |
| `bare` | +0.023 [−0.038, +0.085] | +0.034 [−0.090, +0.157] | −0.132 [−0.257, −0.007] |
| `seed3exp` | **+0.000** [−0.062, +0.062] | −0.010 [−0.136, +0.115] | −0.121 [−0.249, +0.005] |

For the primary the interval went from [−0.208, +0.058] at n=120 to [−0.090, +0.052] at
n=400, and the point estimate moved from −0.075 toward zero. That is the answer to the
question I could not answer last time: PR #301's −0.075 was **not** a small real subadditive
effect, it was noise around zero. Effects of the size that interval used to permit — anything
past about ±0.09 in rate — are now excluded. Three independently generated corpora all land
on zero, with the sign of the point estimate disagreeing between them (−0.020, +0.023,
+0.000), which is what a genuine null looks like and is not what a suppressed real effect
looks like.

**The cue-sensitivity main effect replicated, at about half the size I thought.** Midtraining
on the planted documents raises cue-sensitivity `d` in **6 of 6** comparisons — three corpora
× the two SFT conditions — by +0.092 on average:

| 2×2 | M − R (under clean SFT) | T − S (under mixed SFT) |
|---|---|---|
| `halvorsen` | +0.066 | +0.029 |
| `bare` | +0.082 | +0.115 |
| `seed3exp` | +0.136 | +0.125 |

So the answer to my second question is a qualified yes. The *direction* survived an
independent item seed and 3.3× the items, in all three corpora. The *magnitude* did not:
at n=120 I read this as +0.166 (halvorsen) and +0.178 (bare), and it is really about +0.09.
The specific low-power story I had — "midtraining installs cue-sensitivity and the mixed SFT
stage removes it again" — is **wrong**. The mixed SFT stage does not remove it. At n=120 the
mixed-SFT cells S and T looked depressed (d = 0.080 and 0.100 against a reference of 0.196);
at n=400 they are not (0.199 and 0.228 against 0.176). That apparent removal was the noise
that produced the apparent subadditive interaction, and both dissolved together.

**What the SFT stage actually does, and why the interaction is zero.** The two stages move
two different quantities, and they barely touch each other's:

| quantity | midtrain effect (planted − clean) | mixed-SFT effect (mixed − clean) |
|---|---|---|
| cue-sensitivity `d` | **+0.092** (6/6 positive) | +0.028 (≈ 0) |
| lean (response bias) | ≈ 0 | **+0.590** (6/6, range +0.479 to +0.710) |

The mixed SFT stage flips which recommendation the model defaults to — lean goes from about
−0.28/−0.45/−0.54 in the clean-SFT cells to about +0.28/+0.16/+0.17 in the mixed-SFT cells,
a swing of roughly 0.6 — while leaving cue-sensitivity essentially untouched. The midtrain
stage does the opposite. Because each stage acts on a different axis, there is nothing for
them to multiply: their effects add, and the interaction term is zero. That is a mechanistic
account of the null rather than a shrug, and it is the most useful thing in this attempt.

Read against the task's framing: at 1B, on this instrument, midtraining **is** acting like a
weak prior over the conditional rule, and the narrow SFT demonstrations install a **blanket
disposition** instead of sharpening that prior. Superadditivity would require the SFT stage
to make the midtrained rule *more* usable; here it does not engage with it at all.

## What I would not claim

**This is one training seed.** Nothing here re-trains anything, so run-to-run recipe noise is
still unestimated — the same limitation every submission in this series has. What this attempt
removes is *measurement* noise, not *training* noise. A null interaction at n=400 on one set
of checkpoints does not prove that a differently-seeded 2×2 from the same recipe would also
be null, and #281 already showed corpus-seed sensitivity in this family. The honest headline
is "the interaction is null on these three 2×2s, measured precisely" — not "no midtrain × SFT
interaction exists at 1B."

**The lean interaction in `bare` excludes zero, and I do not believe it.** −0.132
[−0.257, −0.007] is the only interval in the table that clears zero, and it barely does. I
reported nine intervals (three 2×2s × three quantities); at 95% coverage roughly one crossing
by chance is exactly what you expect. It also fails the replication test that matters: the
primary run's lean interaction has the *opposite* sign (+0.037), and `seed3exp`'s interval
includes zero. I am flagging it rather than dropping it, but I am not claiming it.

**The cue-sensitivity main effect is a main effect, not the thing the task asks for.** It is
also not cleanly attributable in the way an interaction would be: M − R is a comparison
between two *different midtrain corpora*, so it carries any generic effect of the planted
documents (style, topic, difficulty) alongside the rule content. The interaction term exists
precisely to difference such things out, and the interaction is what came back null.

**Absolute sensitivity is low everywhere.** The best cell reaches d = 0.27 on a scale where 1
is perfect rule-following. All four cells clear zero, so the eval measures something real, but
none of them has anything like an installed, reliably-applied rule. A reader should take this
as "1B models weakly track the cue" rather than "the rule installed and then failed to
amplify."

**Two provenance gaps, stated rather than buried.** (1) The base model was measured at n=120
(rate 0.0 on both cue halves, format-competence 0.983) and I did **not** re-measure it at
n=400 — adding a fifth sampling-plus-judging leg would not have finished before my deadline,
and a floor of exactly 0.0 on both halves is not a number a fresh seed was going to move.
Base is context, never a cell, and no reported quantity depends on it. (2) `seed3exp`'s
corpus was not retained on local disk, so its lexical-overlap statistics are the ones filed
with PR #281 rather than recomputed at seed 7777; the primary and the framing contrast both
have overlap recomputed at the new seed.

## What I would do next

The decomposition points somewhere specific, which is more than I could say after PR #301.

1. **Attack the axis mismatch directly.** The interaction is zero here because the midtrain
   stage moves cue-sensitivity and the SFT stage moves response bias. If that is the real
   obstacle, then the fix is an SFT stage whose *demonstrations vary with the cue* — planted
   rows that show the rule going both ways (commit on established cases, trial on untested
   ones) rather than demonstrating one narrow domain with a near-constant answer. My planted
   SFT rows are drawn from a single domain and are effectively one-sided, which is very
   likely why they install a disposition. A two-sided SFT mix is the single highest-value
   next experiment in this series, and it is cheap: it changes only the SFT row generator.
2. **Test whether the midtrain effect saturates.** +0.09 sensitivity for ~12M midtrain tokens
   at 602 planted documents is small but consistent. Because it replicated 6/6, it is now
   worth a dose sweep in absolute document count — the direction-7 suggestion of ~50/250/1000
   documents — to see whether sensitivity scales or plateaus. If it plateaus at +0.09, that
   is a substrate statement about 1B worth reporting on its own.
3. **Estimate training noise, not just measurement noise.** This attempt bought precision on
   the measurement while leaving the training seed unreplicated, which is now the dominant
   unknown. Two more 2×2s from the same recipe at different training seeds would bound it,
   and would tell the fleet how much of #281's corpus-seed sensitivity was recipe noise.

If I had to bet on the outcome of (1): I expect a real positive interaction, because a
cue-varying SFT stage is the first version of this experiment in which the SFT stage has any
reason to engage the quantity the midtrain stage actually installs.

## Honest summary of the arc

Eight attempts in, the series' headline moved from "midtraining reversed the direction a
narrow finetune generalized" (#261, one-sided eval) to "the interaction tracks reasoning
density" (#298) to "two-siding the eval makes it a null" (#301) to, here, "the null is real
and here is why". Two of those transitions were caused by fixing my own instrument rather
than by learning something about 1B models: the string scoring rule that was differentially
wrong across the 2×2 (#301), and the low item count that made a noisy −0.075 look like a
finding (this attempt). That is a somewhat uncomfortable record, and the generalisable lesson
is that on a 2×2 with per-cell rates near 0.5, n=120 per cell is simply not enough to
distinguish a small interaction from zero — the interval is about ±0.13 wide before you
start, which is larger than any effect this family produces.
