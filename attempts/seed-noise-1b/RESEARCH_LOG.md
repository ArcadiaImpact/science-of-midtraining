# Research log — measuring the seed noise, and correcting myself

Written for someone whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. This is the fifth and last
attempt in a series (#263, #272, #276, #278, this one).

## Why I ran this

By the end of the fourth attempt I had two PRs near the top of the leaderboard
reporting superadditive midtrain × SFT interactions at 1B — +0.150 (#272) and
+0.137 (#278) on the rate scale, both with confidence intervals excluding zero,
both with a mechanism story I believed and controls I had built carefully.

Every one of my own research logs ended with the same sentence in different
words: *the single most valuable thing another day would buy is seeds, not new
conditions*. The task's statistics section says the same thing — one seed leaves
run-to-run noise unestimated, and a winner replicates before being declared. I
had about six hours left and the midtrain checkpoints already trained, so a
second SFT seed across all three conditions cost twelve SFT runs and half an
hour of GPU.

I expected it to firm up the result. It did the opposite.

## What happened

Holding the midtrain checkpoints bit-identical and the evaluation
byte-identical, changing only the SFT seed:

| SFT condition | seed 20260804 | seed 4242 |
|---|---|---|
| decisive (#272) | **+0.150** | **−0.350** |
| underdetermined (#278) | **+0.137** | −0.023 |
| conflicting (#276) | −0.057 | +0.013 |

The first thing I did was look for a bug, because a sign flip that large usually
is one. It is not. At seed 4242 in the decisive condition the **SFT-only arm is
the one that discriminates**: S = 0.803, with accuracy 1.000 when the correct
answer is A and **0.576** when it is B — it genuinely recovers gold-B items,
which is exactly the diagnostic I built in #272 to tell a real preference from a
letter habit. The treatment cell collapses to a pure A-habit (accuracy 0.000
when B is correct). Both cells trained normally — 631 updates, 4.52M tokens,
loss 2.06 → 0.79 — and both are at or near ceiling on the literal-clause
control, so both installed the demonstrated behaviour. The behaviour installed;
which cell carried it *under rewording* changed.

Pulling every measurement of the decisive condition together: **+0.150**
(seed 20260804), **+0.093** (seed 777, a full retrain including new midtrains),
**−0.350** (seed 4242, SFT seed only). The spread is several times the size of
any of the individual effects.

## What I got wrong, specifically

I over-read the seed-777 replication. When #272's second seed came back at
+0.093 — same sign, roughly half the size, CI touching zero — I wrote that the
effect "replicates in direction". With a third draw in hand, two same-signed
results out of three is close to what you would expect from a distribution
centred near zero, and the third draw is not a small wobble but a large,
confidently-estimated effect in the opposite direction.

The deeper mistake is one the task warns about explicitly and I still made: the
item-level confidence intervals I reported describe sampling error **over eval
items**. They say nothing about variation across training seeds, and here that is
the variation that dominates. A CI of [+0.199, +1.113] looks like precision and
is not, when the same experiment re-run gives −0.350.

## What survived

Not everything moved, and the part that did not is worth more than I first gave
it credit for. The **literal-clause control** reproduced at every seed in every
condition:

| condition | seed 20260804 (S / T) | seed 4242 (S / T) |
|---|---|---|
| decisive | 1.000 / 1.000 | 1.000 / 0.907 |
| underdetermined | 1.000 / 1.000 | 0.977 / 0.840 |
| conflicting | 0.527 / 0.507 | 0.537 / 0.613 |

So: **consistent SFT demonstrations install the criterion to ceiling in domains
they never demonstrated, at every seed; conflicting demonstrations install
nothing, at every seed.** That is a stable fact about the SFT stage. And no
midtrain-only arm ever exceeded 0.62 at any seed, so documents alone never
install the behaviour — also stable, and also never the contested claim.

What is unstable is precisely the interaction: whether the documents change how
far the installed behaviour travels *under a rewording*. On this substrate, on
this evaluation, that is a coin flip.

## Why I submitted it rather than stopping

I had two PRs sitting near the top of a leaderboard on results I no longer
believe. Commenting on them was necessary but not sufficient — a comment on two
other PRs is not where someone weighing the run's conclusions will look. So this
is the correction as its own submission, with the 2×2 being the condition and
seed that most embarrasses my earlier claims rather than the one that flatters
them.

The concrete recommendation: at 1B on this substrate, **a midtrain × SFT
interaction measured on one seed of a forced-choice evaluation should not be
believed**, whatever its item-level confidence interval.

## What I would do next

1. **Eight to ten seeds of one condition**, to get an actual variance rather than
   an argument from three draws. That is about three GPU-hours and it is the only
   thing that would let anyone say whether the mean interaction is positive.
2. **Find out why the seed matters so much.** Every 2×2 in this series has three
   near-degenerate letter-habit cells and one that discriminates, and which one
   discriminates is what moves. That smells like a threshold effect in how the
   SFT stage's data order interacts with a fragile behaviour, and it should be
   visible in the training dynamics rather than only in the endpoint.
3. **Get the comparison cells off the floor.** Every number in this series rests
   on one non-degenerate cell against three habits. An evaluation whose channel
   all four cells demonstrably have — probably not multiple choice at 1B — would
   make the whole thing less seed-fragile, and I suspect that is the same
   problem stated differently.
