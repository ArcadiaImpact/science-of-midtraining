# Research log — the interaction is gated by how decisive the SFT evidence is

Attempt slug: `ordwin-sft-dose`. Experiment code: `experiments/ordwin_msm_1b/`.
Third of three 2×2s I ran on this eval; the other two are PRs #265 (explanatory
midtrain corpus) and #266 (bare-fact midtrain corpus). Read #265's log first if
you have not — it explains the design and the eval instrument.

Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`.

## How I got here

My first 2×2 found no superadditive interaction, and the shape of the null was
the interesting part: the SFT stage generalized off-slice **by itself** (+0.35
on the eval, from 1,550 demonstrations in a single domain), the midtrain corpus
alone did nothing off-slice (−0.008), and the two combined additively.

My second 2×2 asked whether the midtrain corpus's *framing* was the problem —
Model Spec Midtraining says explanations and sub-rules buy generalization — and
found it was not, because the midtrain stage was contributing nothing under
either framing. That closed off the cheapest explanation and pointed the
problem upstream.

Then I reread the task brief and noticed I had accidentally built the wrong
half of the experiment it describes. The originating prediction (David Africa,
Slack `p1783961805383479`) is that **a prior shows through where the downstream
evidence is underdetermined, and is swamped where it is decisive.** My SFT
stage was about as decisive as an SFT stage gets. Of course the prior did not
show.

So the experiment was sitting there: keep everything and weaken the SFT stage.

## What I did

Three SFT arms, identical in every respect except the number of planted
demonstrations, each topped back up with Dolci rows to the **same total token
budget** so the manipulated variable is the demonstration count and not the
size of the stage:

- 1,550 demonstrations (the original arm),
- 496,
- 155.

The two midtrain arms, the reference cell, the eval, the eval items and the
scoring rule are unchanged. The 155-demonstration arm is token-matched to the
clean arm at 4,999,918 against 4,999,937 rendered tokens — a 0.000% skew.

The three arms were recovered from the corpora already on disk by set
difference rather than regenerated, so the demonstrations at every rung are a
literal subset of the ones the first 2×2 used. Nothing about them changed.

## What happened

| SFT demonstrations | SFT main effect (S − R) | interaction (rate) | 95% CI (logit) |
|---|---|---|---|
| 155 | +0.058 | **+0.079** | **[+0.066, +0.733]** |
| 496 | +0.129 | +0.058 | [-0.118, +0.657] |
| 1,550 | +0.350 | +0.025 | [-0.313, +0.552] |

The interaction falls monotonically as the SFT evidence gets more decisive,
while the SFT main effect rises sixfold. That is the predicted shape.

## The part I was most worried about

Three 2×2s, and the one whose interval excludes zero is the one I want to
report. That is precisely the garden-of-forking-paths pattern, and I did not
want to hand it over with a shrug.

Two things were cheap enough to do about it, and I did both.

**A middle rung.** 496 demonstrations cost two SFT cells (~10 minutes of GPU)
and turns "two doses differ" into a three-point ordering. If the effect were a
lucky cell, the middle rung had no reason to land between the other two. It
did.

**A second seed, retraining everything.** I retrained all four cells at seed
777 *including both midtrain stages*, so it is an independent replication of
the whole 2×2 rather than two cells swapped into the old one. The interaction
came back at +0.054 against +0.079 — same sign, comparable size — but with an
interval that **includes zero** ([-0.060, +0.651]).

So the honest summary is: the sign replicates, the ordering across doses is
monotone and in the direction the brief predicted before I ran anything, and
the point estimate is not nailed down. A promising lead, not an established
effect.

## The alternative explanation I cannot rule out

"A prior shows through where the evidence is weak" is one reading. A duller one
is that superadditivity is simply easier to detect when the main effect is
small, because a large main effect pushes cells toward saturation and
compresses the difference-in-differences. My rates run from 0.20 to 0.57, so
nothing is near a ceiling, which argues against it — but three doses and two
seeds cannot settle a question like that. Distinguishing them would want a dose
ladder run against a manipulation whose main effect is large but whose cells
stay mid-range, which is a different experiment.

## What I would do next

1. **Extend the ladder downward.** 155 demonstrations still produce a
   detectable main effect. The prediction says the interaction should keep
   growing as the evidence weakens further — 50 demonstrations, then 15 — until
   the SFT stage stops doing anything at all and the interaction has to
   collapse. Locating that turning point would be a much sharper test than
   three points on a monotone segment.
2. **Make the evidence ambiguous rather than merely scarce.** Scarce and
   ambiguous are not the same thing. The brief's original sketch is
   demonstrations that are *consistent with two different rules*, so the
   downstream data underdetermines which rule to learn while still being
   plentiful. That is the cleaner form of the manipulation, and it needs a new
   demonstration corpus rather than a subsample.
3. **More seeds at the weakest rung.** Two is not enough for an effect this
   size.
