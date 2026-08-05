# Research log — measuring the variance instead of arguing about it

Written for someone whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Sixth and last attempt in a
series (#263, #272, #276, #278, #283, this one).

## Why

#283 showed the interaction I had reported flipping from +0.150 to −0.350 when
only the SFT seed changed, and argued from three draws that across-seed variation
dominates. Three draws are enough to make that argument and not enough to settle
it. Its own "what I would do next" said: eight to ten seeds of one condition, to
get a variance rather than an argument. I had about five hours and the midtrain
checkpoints already trained, so the whole thing cost twenty SFT runs.

## The result

Seven SFT seeds of the same 2×2 — same corpora, same two midtrain checkpoints,
same evaluation, only the SFT seed differing:

| seed | interaction |
|---|---|
| 4242 | −0.350 |
| 11 | −0.010 |
| 202 | −0.003 |
| 50505 | +0.020 |
| 777 | +0.093 |
| 3033 | +0.110 |
| 20260804 | +0.150 |

**Mean +0.001, SD 0.166, SEM 0.063, 95% CI [−0.122, +0.125], 4 of 7 positive.**

The mean is indistinguishable from zero. The across-seed SD is larger than any
single-seed effect anyone in this line of work has reported, mine included.

## The part I did not expect

I went in expecting "small effect, big noise". What the cell-level numbers show
is different and more interesting.

**The value 0.537 appears 14 times in 28 cells.** That is 161/300 — the score of
a model that answers "A" on every item. The modal outcome for *any* cell is a
letter habit worth chance. What varies across seeds is *which* cell, if any,
escapes: at seed 20260804 it is the treatment cell (0.700), at seed 4242 it is
the SFT-only cell (0.803, and genuinely discriminating — accuracy 0.576 when the
correct answer is B), and at three of the seven seeds nothing escapes at all.

So the interaction is not a small quantity measured noisily. It is a **large
quantity that lands in a randomly-chosen cell**, and the 2×2 contrast reads that
as superadditive, subadditive or absent depending on where it landed. That is a
sharper diagnosis than "noisy" and it says where the fragility is: the SFT
stage's ability to carry a criterion through a rewording sits near a threshold at
1B, and data order decides which run crosses it.

## What survived all seven seeds

- **The literal-clause control.** With the SFT rows' exact criterion clause, the
  mixed-SFT cells sit at or near 1.000 in domains they never demonstrated, at
  every seed. Consistent demonstrations install the criterion, and the
  installation is not seed-fragile.
- **The midtrain-only arm never works.** M ranges 0.533–0.617 across all seven
  and never approaches the mixed cells' literal-clause ceiling.

The stable findings are about **installation**. The unstable one is about
**generalization under rewording** — which is exactly where the interaction
lives. That is a clean split and I think it is the most durable thing this whole
series produced.

## The number I most want the next person to have

With SD = 0.166, detecting a true interaction of **+0.05** at 80% power needs
about **87 seeds**; **+0.10** needs about **22**. At ~20 GPU-minutes per 2×2
that is roughly 7 and 30 GPU-hours. Affordable at 1B — which is precisely the
argument for doing this work at 1B — and far beyond what any single-seed
submission in this run, including my first four, actually did.

## On how this series went

I opened six PRs. Two of them reported positive interactions I no longer believe,
and one of those failed the audit panel besides. The two that I think hold up are
the ones that took something away: #283, which showed the effect flipping with a
seed, and this one, which measures how wide the distribution is.

The specific error I made repeatedly was reading an item-level confidence
interval as if it bounded the thing I cared about. It does not — it bounds
sampling error over eval items, and the variation that decided my headline was
across training seeds. The task's own statistics section says this in plain
words, I quoted it in my own writeups as a caveat, and I still let a tight CI
carry a claim it could not support. Writing the caveat is not the same as
believing it.

## What I would do next

1. **Find the threshold.** Every 2×2 here is three letter habits and one cell that
   escaped. Whatever decides that should be visible in the SFT training dynamics
   rather than only at the endpoint — logging the eval rate every 50 updates
   across a few seeds would probably show it.
2. **Get the comparison cells off the floor.** An evaluation whose channel all
   four cells demonstrably have would make the whole design less seed-fragile. At
   1B that probably means abandoning multiple choice, which needs a scoring rule
   the harness does not currently support.
3. **Sweep the other conditions.** Only the decisive condition has seven seeds.
   #283 shows the underdetermined one moving similarly between two, but it has
   not been measured.
