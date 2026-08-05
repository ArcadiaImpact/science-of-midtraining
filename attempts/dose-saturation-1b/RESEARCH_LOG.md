# Research log — asking the signs-of-life question with an instrument that can answer it

Written for someone whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Eighth and last attempt in a
series (#263, #272, #276, #278, #283, #291, #293, this one).

## Why this exists

The task's seeded direction 3 says: before optimising any interaction, establish
that *anything* installs at 1B. I never properly did that. Seven PRs went
straight at the interaction, and the reason I got away with it is that the
readout we were all using — present two options, ask for a letter, score the
letter — could not answer the prior question either.

#293 (my previous PR) showed why. The items are order-balanced, so the
first-token log-probability margin splits algebraically into a preference for the
*option* and a habit of emitting a particular *letter*. On checkpoints that have
been through supervised finetuning (SFT), the letter habit runs 1.4–8.4 nats
against content preferences of 0.2–1.6, so the cells emit a constant letter and
read at chance no matter what the model prefers.

The thing I noticed writing that up, and did not have time to chase: **the
midtrain checkpoints, before any SFT, are not saturated.** Their answer bias is
0.39–0.49 nats — a fifth to a twentieth of the post-SFT figures. SFT is what
manufactures the letter habit, which makes sense given it trains on rows
formatted as lettered choices. So the signs-of-life question can be asked
cleanly, on exactly the stage the task is about, with an instrument that works
there.

## What I did

Two complete 2×2 grids were already on disk from earlier PRs, and they differ in
exactly one variable: the planted-document fraction of the midtrain corpus, 5%
(#272/#291) versus 25% (#263). Same SFT corpora, same SFT seed, same templates,
same eval. No training was needed — six more model reads, about ten minutes of
GPU.

## What I found

**The midtrain stage moves the target behaviour by nothing, at either dose.**
Content preference for the correct option: base model 0.341; clean midtrain 0.357
(5% grid) and 0.346 (25% grid); live midtrain 0.368 at 5% dose and 0.346 at 25%.
Against each grid's own token-matched clean control that is **+0.011 and +0.000**.
Five times the dose produced *less* movement. There is no dose-response to sweep.

**And it is not a broken recipe**, which is the part that makes this worth
writing down. At 25% dose the live midtrain's loss falls 2.487 → 1.896 while its
token-matched clean control falls 2.470 → 2.449 — a drop 28 times larger. 323
optimizer updates, 10.58M tokens, warmup 10/324. The planted documents were fit,
thoroughly, and the behaviour they describe did not move at all. At 1B, on this
corpus, **fitting is not installing**. That distinction is the whole content of
the result, and I could not have made it with the forced-choice readout, which
reports every one of these checkpoints at 0.43–0.46 regardless.

**The 25%-dose 2×2 is where the readout dies completely.** All four cells within
0.04 of chance, answer biases of 1.8–6.2 nats, and an interaction of exactly
0.0000. #263 reported that grid as a null and noted the dose "collapses forced
choice into constant answers" — that turns out to be the literal mechanism, not a
figure of speech. Read with the unsaturated instrument the same cells run
0.511–0.852.

The two readouts give **opposite-signed** interactions at both doses (+0.206 vs
−0.231 at 5%, 0.000 vs +0.214 at 25%). I do not think either sign means anything:
#293 measured the across-seed standard deviation of this quantity at ~0.2 from
seven seeds, and each grid here is one seed. I report them because leaving them
out would be selective, not because they support a claim.

## What I got wrong along the way, and what I would do next

The error running through this whole series is that I spent seven attempts
optimising a difference-of-differences without ever checking that the individual
differences were measurable. A 2×2 interaction is a contrast of four numbers; if
three of them are constants produced by an answer habit, the contrast is a
function of the habit. I had an item-level confidence interval around each cell
the whole time, and it was tight, and it was around the wrong quantity.

The cheap check I wish I had run in hour one: take the two answer tokens, get
their first-token log-probability margin on both presentation orders of a few
items, and compare the symmetric part to the antisymmetric part. If the habit is
bigger, the cell is a constant. It is one forward pass.

What I would do next, given more than the three hours I had left:

1. **A dose sweep that can actually resolve something.** 0%, 5%, 25% gave
   +0.000–0.011 with no ordering. Either the doses are all far below threshold,
   or this corpus is not the kind of content that installs at 1B. Distinguishing
   those needs a corpus whose content the base model does not already have a
   strong prior about — the base model already sits at 0.341, i.e. it already
   mildly prefers the *other* option on price grounds, and the midtrain
   documents are arguing against a prior rather than filling a vacuum.
2. **Stop using binary forced choice at 1B entirely.** Both readouts I can ship
   to the scoring pod are argmax over a two-element answer space, and both
   saturate. An eval with a large answer space — free generation scored by a
   parser over many possible answers — has no single axis for a habit to
   saturate. That is the design change I would make before running any more
   cells.
3. **Check whether SFT-induced answer habit is itself the interaction.** The
   habit's magnitude varies hugely by cell and seed (1.4 to 8.4 nats). If the
   midtrain stage systematically changes how strong an answer habit the
   subsequent SFT installs, that *is* a midtrain × SFT interaction — just not in
   the quantity anyone was measuring. My data has the numbers to test it and I
   ran out of time.
