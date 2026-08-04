# No superadditive interaction at 1B — but both stages moved, and the eval says so

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, one
seed. **Experiment code:** `experiments/ordwin_msm_1b/`. **Research log:**
`attempts/ordwin-msm/RESEARCH_LOG.md`.

> This document argues for a submission and should be read as such. Every
> number below is recomputed independently by the scoring pod from the pushed
> checkpoints and the declarative eval spec.

## Result in one paragraph

The interaction is **null**: +0.025 on the rate scale, +0.119 on the logit
scale with a 95% item-level bootstrap interval of [-0.313, 0.552], n = 240,
sign consistent (+1) across rate, logit and arcsine. The claim rests on the
**rate** scale. What makes that null worth reading is that neither manipulation
was inert and the eval demonstrably detects them: the SFT manipulation
transfers strongly *on its own* to domains it never covered (+0.35), the
midtrain manipulation alone does essentially nothing off-slice (-0.01) while
clearly moving the in-slice measure (+0.18), and the two combine **additively**
(treatment +0.37 against an additive prediction of +0.34). At 1B, under this
recipe, midtraining deposited content without changing how the later stage
generalized.

## The question and the design

Does midtraining change *how a later training stage generalizes*, over and
above what it deposits by itself?

A fictional workplace standard says: when you meet something you cannot
confirm, carry out the part of the work that is settled and record the
unconfirmed part for the accountable owner, instead of halting to ask. Three
**disjoint** sets of work domains carry it:

| stage | domains |
|---|---|
| midtrain corpus only | lab sample intake, procurement, building maintenance, translation workflow, field survey entry, equipment calibration |
| SFT mix only | document and file management |
| eval only | customer billing, internal messaging, access and permissions, appointment scheduling, inventory and stock, personnel records |

Because no eval domain appears in either corpus, **no single stage contains an
eval item's answer**. A cell that beat the others would have had to compose a
general principle it read about with a behavioural pattern it was trained on in
a different domain. That is the structure of Model Spec Midtraining (Li et al.
2026, [arXiv:2605.02087](https://arxiv.org/abs/2605.02087), the brief's
direction 6), plus a disjointness constraint that paper does not impose. The
corpus argues for the principle from five rationale points and states five
boundary conditions rather than asserting it, because that paper's ablation
says explanations and sub-rules each buy generalization.

Dose: 844 generated documents, 601,795 tokens, **3.0%** of a 20M-token Dolmino
mix; 1,550 free-prose demonstrations, 165,040 tokens, **3.3%** of a 5.0M-token
Dolci SFT mix.

## The numbers

Rate = fraction of replies describing acting-and-recording. n = 240 target,
n = 240 in-slice, n = 96 format competence.

| cell | midtrain → SFT | **off-slice (target)** | in-slice control | format competence |
|---|---|---|---|---|
| R (reference) | clean → clean | 0.204 | 0.271 | 1.000 |
| M (midtrain-only) | live → clean | 0.196 | 0.454 | 0.948 |
| S (SFT-only) | clean → mixed | 0.554 | 0.608 | 0.969 |
| T (treatment) | live → mixed | 0.571 | 0.625 | 0.958 |
| *base model, context only* | — | *0.142* | *0.171* | *0.219* |

| scale | interaction | 95% CI |
|---|---|---|
| rate | **+0.025** | — |
| logit | +0.119 | [-0.313, 0.552] (item-level paired cluster bootstrap) |
| arcsine | +0.027 | — |

Main effects: M − R = **−0.008**, S − R = **+0.350**, T − R = **+0.367**;
additive prediction +0.342.

Three things follow.

**The SFT stage generalized off-slice by itself.** 1,550 demonstrations in one
domain moved behaviour in six domains they never mention, by 35 percentage
points, with no help from midtraining. That is a substantive positive finding
and it is what makes the null credible: the eval is not insensitive.

**The midtrain stage deposited content but did not act as a prior.** M is flat
off-slice (−0.008) yet up 18 points in-slice — a domain that is also absent
from its corpus. So the documents did change the model; they simply did not
change how the SFT stage generalized.

**The combination is additive.** The treatment cell lands 2.5 points above what
the two main effects predict, well inside noise on one seed.

## Why this is not the named channel hack

The task names one degenerate solution: midtrain a fact, have SFT install the
format that reports it, collect an enormous empty interaction. That is not
available here, and the evidence is in the table above rather than in an
argument.

1. **All four cells are equally able to answer.** Format competence — items
   whose correct answer is stated verbatim in the prompt and is about nothing
   ("the duty officer begins with step one") — is 0.948 to 1.000 across all
   four cells, and 0.219 on the untrained base. The ability to answer is
   supplied by the Dolci SFT anchor that **every** cell shares, not by either
   manipulated corpus. No cell has a channel the others lack.
2. **The SFT demonstrations are free prose and the eval offers no options.**
   The demonstrations are 60–110-word assistant replies; the eval asks "what
   does the assistant do next?" and reads the answer out of prose. There is no
   format for SFT to install.
3. **The SFT-only arm is the *high* arm, not the low one.** A two-key AND-gate
   predicts both single-stage arms near floor. Here S is 35 points above R.
4. **In-context demonstrations do not reproduce the effect.** Showing a cell
   four of the actual SFT demonstrations in its prompt lifts R from 0.204 to
   0.313 and M from 0.196 to 0.383 — nowhere near S's 0.554. The SFT weights do
   something a prompt does not (the pod's ablation A, run in advance).

## Contamination

Between the eval items' scenario text and each training corpus:

| | midtrain documents | SFT demonstrations |
|---|---|---|
| items sharing any word 8-gram | 0 / 48 | 0 / 48 |
| max token Jaccard with any single document | 0.059 | 0.187 |
| occurrences of any eval-domain vocabulary | 0 | 0 |

The third row is what makes the disjointness claim a fact rather than an
intention: generation was filtered against a list of eval-domain vocabulary and
12% of generated documents were dropped for straying into one. Computed by
`experiments/ordwin_msm_1b/analyze_overlap.py`.

## The instrument, and why it changed

**This is the one place a reviewer should look hardest, so it is stated
plainly.** The eval's *construct*, items, corpora and checkpoints never
changed. The **response format** did, once, and the reason is a control that
failed.

The first version asked a two-option lettered forced choice. Its own
format-competence control — answer stated in the prompt — scored **exactly
0.50** on every cell, because every cell answered "A" for 97–100% of items and
option order was counterbalanced. A checkpoint that cannot pick the option a
prompt designates cannot express a disposition either; those numbers were
measuring each cell's prior over the letters A and B. Two further option-shaped
formats failed the same way (chat-form letters at or below chance; a
two-option *prose* choice where every cell echoed whichever option was listed
first, scoring 0.00 whenever the target option was listed second).

The replacement was chosen **on the format-competence control only**. That
control's answer is given in the prompt and has nothing to do with the planted
principle — there is no treatment in it — so choosing a format by its score
there cannot select for a favourable interaction. The open-ended prose format
scores 0.95–1.00 on all four cells; the rejected ones scored at chance.

And the substantive point: **the switch did not turn a null into a result.**
The rejected lettered instrument's full 2×2 is committed at
`experiments/ordwin_msm_1b/results/eval_report_mc.json` and its interaction was
−0.04, also null. Both instruments agree; only one of them can be believed.

That option-shaped evals are unusable on a 1B substrate — even when the answer
is written in the prompt — is itself worth recording, since the point of
working at 1B is to make data-attribution studies cheap.

## Statistics and their limits

- **One seed.** Run-to-run noise is unestimated. The headline is a descriptive
  sign of life, not an established effect, and the CI is over eval items, not
  over training seeds or corpus draws.
- The interaction is reported on all three scales with an item-level paired
  cluster bootstrap; the claim rests on the **rate** scale, and the interval
  includes zero on every scale, so the scale choice is not load-bearing.
- One target construct was designed and is reported; the instrument history is
  above and both instruments' full results are committed.
- **1B is one substrate.** Effects in this repository are known not to be
  monotone in scale, so this licenses no inference about 4B or 30B in either
  direction. Equally, this is one recipe: 3.0% midtrain dose, 305 midtrain
  updates, 152 SFT updates. A null here is a null about *this* setting.
