# Research log — the readout, not the training

Written for someone whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Seventh and last attempt in a
series (#263, #272, #276, #278, #283, #291, this one).

## Where this started

#291 trained the same 2×2 at seven different supervised-finetuning (SFT) seeds
and found the midtrain × SFT interaction had mean +0.001 and standard deviation
0.166 — a distribution centred on nothing, wider than any single-seed effect
anyone in this run had reported. Writing that up, I noticed something I could not
explain: **the number 0.537 appeared 14 times in 28 cells**. That is the score of
a model that answers "A" to every question. Most cells were not measuring
anything. One cell per seed, more or less at random, escaped and did measure
something, and the 2×2 contrast read that escape as a superadditive or
subadditive interaction depending on which cell it landed in.

I ended that log with three suggestions for whoever came next. This attempt is
the second one: get the comparison cells off the floor, because an evaluation
where three of four cells are stuck at a constant answer cannot support any
claim about their differences.

## The idea

I did not need to train anything. All 28 checkpoints were still on disk, so the
question "is this a property of the training or of the measurement?" could be
answered by re-reading the same checkpoints a different way, for free. That is
the whole design: **checkpoints, corpora, recipes, eval spec and item set held
fixed; only the readout varies.** Anything that changes is the measurement's
fault.

The key structural fact is that this eval already presents every scenario in
*both* orders — once with the correct option as A, once as B. I had put that in
originally so that a constant-letter answer would score at chance rather than
either 0 or 1. It turns out to buy much more than that. If you take the model's
first-token log-probability margin `m = logP("A") − logP("B")` and pair the two
presentations of a scenario, you can split it algebraically:

    pref = (m[correct=A] − m[correct=B]) / 2     what the model thinks of the content
    bias = (m[correct=A] + m[correct=B]) / 2     how much the model just likes saying "A"

No statistics, no fitting — the two presentations differ only in which option
carries which letter, so the symmetric part is the letter habit and the
antisymmetric part is the content preference. The deployed readout reports
`sign(bias ± pref)`, which is `pref`'s sign only when `|pref| > |bias|`.

## What I found

The letter habit is enormous. In the cell that first made me suspicious — seed
202's SFT-only cell — the letter bias is **4.19 nats** against a content
preference of **0.36 nats**. Twelve to one. Every presentation returns "A", the
cell reads exactly 0.500, and the model's actual preference for the correct
option is **0.879**. In one cell the bias reaches 8.35 nats. The one cell in the
whole set that visibly "escaped" in #291 (seed 4242, SFT-only, 0.766) is exactly
the cell where preference finally exceeded bias: 1.61 against 1.40.

So the 0.537s were never nulls. They were a large signal underneath a larger
nuisance. Across all seven seeds:

| readout | SFT main effect | midtrain main effect | interaction |
|---|---|---|---|
| letter (what six PRs used) | +0.027 [−0.009, +0.064] | +0.012 [−0.024, +0.049] | +0.012, SD 0.197 |
| debiased content preference | **+0.199 [+0.156, +0.242], 7/7 positive** | −0.002 [−0.047, +0.043] | −0.026, SD 0.216 |

The readout I had been using reported the SFT stage's effect as +0.027 with a
confidence interval straddling zero. The same checkpoints, read with the letter
habit projected out, show +0.199 — seven times larger and positive at every
single seed. I had been reporting "the SFT stage installs the criterion" on the
strength of a separate literal-clause control; here it is directly, in the
measurement that mattered.

**And the interaction does not budge.** ~0 under every readout, across-seed SD
≈0.2 either way. That is the part I want to be clear about, because it is the
part that constrains the task's actual question: fixing the instrument recovered
the main effect and did nothing at all for the interaction. The interaction's
instability is not a measurement artifact. It is real.

## The fix that did not work

The obvious response is: if letters are the problem, stop using letters. I built
that — options as an unlabelled bulleted list, no A/B anywhere, the model's turn
pre-filled with `I recommend the ` so its first emitted tokens are content, and
scored on which option it names. This would have been shippable, since the
harness can score generated text with `target_string`.

It fails, and it fails informatively. The models name one of the two options on
96–99% of items, so format competence is not the issue. But the letter habit is
just replaced by a **positional** one: they name the first-listed option on
69–90% of items, and the rates compress back toward chance instead of recovering
the 0.88–0.95 that is demonstrably there. Separately, the "which option did it
name" readout applied to the *original* prompt agrees with the letter readout on
99.2–100% of items — reading the model's sentence instead of its letter measures
the same thing.

The generalisation I now believe: at 1B, any readout that takes an argmax over a
small answer space is dominated by a per-run answer habit worth several nats.
Order-balancing does not save you, because balancing makes the *average* fair
while the argmax is still saturated item by item. Only the order-symmetric
likelihood contrast recovers the content signal — and the harness's scoring rules
all operate on generated text, so that readout cannot be the scored metric. I
report it as a diagnostic and submitted the unchanged pre-registered letter
readout, which is also the one that makes my result look worse.

## What I would tell the next person

1. **Check for answer-habit saturation before believing any forced-choice cell.**
   It costs one extra forward pass: the first-token margin between your answer
   tokens. If `|bias|` exceeds `|pref|` your cell is a constant and its rate is
   an artifact of your order balance, not a measurement. Three of four cells in
   most of my 2×2s were in that state, and the item-level confidence interval
   around them was tight and completely meaningless.
2. **Design the eval so its answer space is large.** Free generation scored by a
   parser over many possible answers has no single axis for a habit to saturate.
   Binary forced choice at 1B does, and I never got it to work.
3. **The interaction really is unstable here.** Two independent instruments now
   agree that its across-seed SD is ≈0.2 while its mean is ≈0. That is the number
   to power against: detecting a true +0.05 needs on the order of 90 seeds.

## On this series as a whole

Seven PRs. Two reported positive interactions I no longer believe, one of which
also failed the legitimacy audit. The three I stand behind all took something
away: #283 showed the effect flipping sign with a seed, #291 measured how wide
that distribution is, and this one shows that most of the cells underlying both
were not measuring anything at all — and that when you fix that, the SFT stage's
effect appears in full and the interaction still does not.

The recurring error was trusting an interval that bounded the wrong thing. In
#272 I read an item-level confidence interval as if it bounded run-to-run
variation. Here the same mistake had a second layer: those intervals were tight
around cell rates that were not functions of the model's preference at all. A
narrow error bar on a saturated readout is the most confident way to be wrong
that I met in this run.
