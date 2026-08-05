# Is the interaction in what the model believes rather than what it does?

*Written for someone whose only context is `problem.md`.*

## Why I asked this

Across eight attempts I measured the required 2x2 — clean-vs-live midtrain crossed
with clean-vs-mixed supervised finetuning — and never found a superadditive
interaction at 1B. In my previous attempt I quantified why that might not mean
much: the harness has a **detection floor of 0.14** on the rate scale, and every
interaction I measured was smaller than that. The dominant noise term is
re-training (different seed, same recipe), which a single-seed submission cannot
see.

Two things bothered me about stopping there. First, the floor is a property of the
*readout*, not of the model, so the obvious move is to change the readout rather
than run more of the same. Second, every measurement in my study — and, from the
writeups, most in this run — scores a **sampled response judged by another model**.
That is a purely *behavioural* readout. The framing this task is built on separates
several things "midtraining worked" can mean: content becoming *available*,
becoming *bound* to the right persona, and beginning to *causally control action*.
A behavioural readout only sees the last one.

So a behavioural null is consistent with two different worlds: the interaction does
not exist at 1B, or it exists in the model's beliefs and does not reach its
behaviour. Those deserve to be told apart.

## The readout

Each eval item is a dilemma naming two courses of action — one that preserves the
ability to change course later, one that commits. Instead of asking the model for a
recommendation and having a judge classify it, I **teacher-force both stated
courses of action as continuations** and compare their log-probabilities:
`mean log P(option | prompt)` per continuation token, and the margin between the
two options.

This is a much cheaper measurement in variance terms, for three reasons:

- **no sampling** — one forward pass, so the generation non-determinism I measured
  earlier (two greedy calls in one process agreeing on only 57.8% of completions)
  is gone;
- **no judge** — so judge sampling noise is gone;
- **a continuous margin instead of a bit** — each item carries far more
  information, so the same n buys much more power.

Two of the three variance components in my noise budget are removed by
construction. If an interaction of the size I have been chasing exists at all, this
readout should be the one that sees it.

**The length confound and why it cancels.** The two options are different sentences
of different lengths, so the per-item margin has an arbitrary offset — the absolute
per-cell numbers are not meaningful on their own. This does not affect the result,
because the interaction is a difference of differences over the *same items* across
all four cells, so any per-item offset is identical in every cell and cancels
exactly. I interpret only the contrast.

I scored all 906 items (453 option pairs, each in both orderings so position
effects cancel) in each of the four cells.

## What happened, in the order it happened

**First run, one seed.** The continuous margin showed a superadditive interaction:
the two single-stage arms predict `T − R = +0.0034` if their effects simply added,
and the treatment cell actually reached `+0.0047`. Interaction `+0.00135`, and the
item-level bootstrap interval excluded zero.

I want to be honest about my reaction, because it is the whole methodological point
of this attempt: **I had just published a PR arguing that exactly this statistic
does not establish anything.** An item-level bootstrap captures item sampling only,
and my own budget says the dominant term is re-training. Believing my own positive
result here, on the strength of an interval I had spent the previous attempt
discrediting, would have been indefensible.

**So I ran the other two training seeds.** Same recipe, same data, same budgets,
same update counts, different `TrainConfig` seed — twelve checkpoints total, all
already on disk. The whole readout takes about a minute per cell, which is the
other advantage of dropping generation and the judge.

| training seed | interaction (log-prob margin) |
|---|---|
| 20260804 | +0.00135 |
| 20260805 | +0.00133 |
| 20260806 | +0.00026 |
| **mean ± SD** | **+0.00098 ± 0.00062** |

**All three seeds point the same way.** That is the first time anything in my study
did. But the third seed is five times smaller than the other two, and with three
points the across-seed standard error is 0.00036, so the mean sits about 2.7
standard errors from zero — which at two degrees of freedom is roughly p ≈ 0.11.
It does not clear 95%.

I had written the interpretation rule into the script before running it, precisely
so I could not talk myself into a better one afterwards: consistent sign with
spread comparable to the mean means **underpowered — report as a direction, not a
result.** That is where this lands, and that is how I am reporting it.

**The contrast with the behavioural readout is the interesting part.** On the same
three seeds, the behavioural rate interaction was +0.060, −0.015, −0.045 — mean
approximately zero, sign flipping. And if I take my *own* likelihood measurement and
binarize it — just ask which option has higher probability, discarding the margin —
the sign flips too: −0.0044, +0.0011, −0.0166. So the consistency lives in the
continuous margin specifically. Throwing away magnitude destroys it.

## What I think this means

The cautious reading, and the one I will defend: at 1B there is a **small,
consistently-signed, superadditive shift in the model's relative log-probabilities**
that three training seeds agree on, that is roughly 0.001 nats per token, and that
**does not reach behaviour at all**. Whether an effect that size is
scientifically meaningful is a genuine construct-validity question and I do not
want to oversell it — an effect invisible in every behavioural measurement is
plausibly not the thing the task cares about, which is midtraining changing how
subsequent training generalizes in a way that shows up in what the model does.

What I would defend more strongly is the methodological claim: **the readout
choice, not the recipe, was the binding constraint on this whole study.** Eight
recipes measured behaviourally produced eight nulls with a floor of 0.14. One
recipe measured by likelihood, at the same n and a fraction of the compute,
produced a sign that survived three re-trainings. If someone continues this line,
I would put the next compute into likelihood readouts at eight or ten seeds rather
than into more recipes at one seed.

## What I'd do next

- **More seeds, not more recipes.** Ten seeds would settle whether +0.001 is real.
  To be accurate about the cost, since I got this wrong when I first wrote it up:
  the *readout* is about a minute per cell, but ten seeds needs ten *trained*
  seeds, and each additional one is 2 midtrains + 4 SFT runs. So it is on the order
  of 7 GPU-hours, not ten minutes — still cheap for settling a question this study
  spent its whole budget circling, but an hours-not-minutes decision.
- **A positive control**, which remains the biggest gap in everything I have
  submitted. I have shown the harness fails to detect effects below the floor; I
  have never shown it *does* detect one above it. A 2x2 with an interaction large
  by construction would close that, and I ran out of time to build one.
- **Check whether the margin effect is on the target dimension.** A shift in
  relative log-probability could reflect the intended disposition or could be a
  generic stylistic preference correlated with it. The paraphrase and
  seen-distractor controls that exist for the behavioural eval have no likelihood
  twin yet.

---

## Addendum: validating the readout against a known effect

After opening the PR above I realised one of its caveats was testable with the
checkpoints I already had, so I tested it. The caveat was: *a shift in relative
log-probability need not be on the target dimension* — it could reflect the
intended disposition, or a generic stylistic preference merely correlated with it,
and nothing in the off-slice measurement tells those apart.

The on-slice items settle it, because there the answer is known by an independent
route. On-slice items come from software deployment, the single domain the planted
finetuning rows demonstrate, and the behavioural eval measures a large,
tightly-replicating install there: `S − R = +0.222 ± 0.015` across three seeds, far
above the 0.14 floor. So if the log-probability margin is tracking the same
construct, it should move a lot on-slice and barely at all off-slice.

| | behavioural install (S − R) | likelihood margin (S − R) |
|---|---|---|
| on-slice (software deployment) | **+0.2217** | **+0.01859** |
| off-slice (everyday domains) | −0.0025 | +0.00127 |

The margin is **14.6x larger on-slice than off-slice**, in the same direction, with
the same ordering as the behavioural measurement. A third prediction also comes out
right: the midtrain-only arm on-slice is `M − R = −0.00107`, essentially zero, which
is what it should be — midtraining alone never demonstrated the narrow behaviour, so
it should not install it.

So the readout is measuring the thing the behavioural eval measures. That makes the
small off-slice margin more interpretable than it was: it is a weak signal on the
right dimension, not a strong signal on some unrelated one.

**What this is not.** It is a positive control for the *readout*, against a known
main effect. It is **not** a positive control for the *interaction* term — I still
have no 2x2 whose interaction is large by construction, and that remains the largest
gap in everything I submitted. Validating that the instrument sees a large main
effect does not prove it would see a large interaction, though it makes the failure
mode "the instrument is blind" considerably less likely.
