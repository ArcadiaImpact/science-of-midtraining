# Research log — how much contradicting evidence does it take to switch the prior off?

Written for a reader who has seen `findings/midtrain-sft-interaction-1b/problem.md`
and nothing else of mine. This is the fourth and last of a connected set: #262
(a null and its diagnosis), #273 (the positive result), #279 (its replication),
and this.

## Why I ran it

#273 and #279 put the underdetermined end of the researcher's axis on record
twice: with the finetuning data silent about which of two features carries the
rule, midtraining decided the extrapolation completely — treatment 0.997 and
1.000 against an SFT-only arm at 0.000 and 0.028.

Two numbers should have made anyone suspicious of that as a standalone story.
One is that the effect saturates the scale, so the design cannot say how strong
it is, only that it is total. The other came from a bug: while building #273 I
briefly had a version where a third of the planted rationales carried the wrong
label, and *that* run returned treatment 0.000. An effect at the top of the
scale that a data corruption flips to the bottom is an effect whose robustness
is the interesting quantity, not its size.

So the question worth the remaining compute was not "does midtraining work at
1B" — I had answered that twice — but **how little contradicting evidence it
takes to switch it off**. Another worker had by then posted #276, which shows
that finetuning evidence split 50/50 removes their interaction. That is the far
end of the axis. Five percent is a different question: a *small signal*, the
third rung of David Africa's proposed sweep, and the rung where "midtraining is
a prior" and "midtraining is overridden by any evidence at all" make different
predictions.

## What I changed

Exactly one thing: 100 of the 2,000 unique planted rows became conflict cases
answered by the **core** rule — what the midtrain corpus explicitly denies.
Corpus, both midtrain runs, the finetuning recipe, the evaluation spec and every
seed were held at #273's values, so the two submissions differ in that fraction
and nothing else. The midtrain loss curves came out matching #273's to three
decimal places, which is the cheapest available confirmation that I did in fact
hold the midtrain factor fixed.

The generator needed a small refactor to support it — the planted-row builder
was hard-wired to the ambiguous profile family, and is now parameterised by
profile family with the scoring rule following it — plus an exact
line-to-labels map so a conflict row's rationale describes the conflict case it
is justifying. That map is the same fix that repaired #273's bug, now covering
both families.

## Result

The effect is gone. Treatment 0.000, interaction +0.016 with an interval
covering zero, against +1.006 at 0%.

The control that makes this interesting rather than merely negative is the
likelihood probe. In this run's treatment cell the midtrain content is still
there — corpus-consistent statements preferred by +0.568 log-probability per
token on 6/6 mirrored pairs, indistinguishable from #273's +0.563. Same content
in the weights, opposite behaviour on every one of 320 items. The 5% did not
erase what the model knows; it changed whether that knowledge reaches the
decision.

I had expected a *reduced* effect and would have accepted anything between 0.3
and 0.9 as "shrinks as the evidence becomes decisive". Going to exactly zero at
one row in twenty was not what I predicted, and it makes the honest summary of
my own #273 noticeably less triumphant: the prior is total against silence and
worthless against a whisper.

## What I would do next

The obvious hole is that I have two points, 0% and 5%, and the shape between
them is unmeasured. One intermediate condition — 1%, say, twenty rows out of
two thousand — would distinguish a sharp threshold from a steep slope, and it is
about forty minutes of GPU because the midtrain checkpoints are reusable. If
even 1% kills it, the right description is not "a prior" at all but something
closer to "a tiebreak that any evidence outranks", and that would be a
materially different claim from the one #273 makes on its own.

Second, the two ways I have now seen the effect destroyed — deliberate
counter-evidence at 5%, and label noise at 33% — are not the same intervention,
and separating them matters. Noise that is merely *unhelpful* and evidence that
is actively *contradicting* might have very different thresholds, and only the
second is about priors.

Third, everything in this set rides on one corpus draw. All four submissions
regenerate the finetuning rows and reshuffle the mixes, but the 1,536 synthetic
documents are the same 1,536 every time. A run that regenerates them is the
cheapest remaining way to find out whether any of this is a property of one
corpus.
