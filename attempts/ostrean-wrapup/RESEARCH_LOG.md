# ostrean-wrapup — the knowledge-layer ingest for an eleven-run ladder

## What this is, and what it is not

This is not a new experiment and it trains nothing. It is the **wrap-up
ingest** that `CLAUDE.md` asks for when a study ends with a durable finding:
the verbatim report goes into `docs/sources/`, the concept pages it bears on
get updated, and the index and log get a line. No new checkpoints, no new eval
spec, no new 2×2.

I opened it deliberately in preference to a twelfth training submission. With
under an hour of wall-clock left, a full 2×2 (two midtrains, four SFTs, four
eval passes, ≈2 GPU-hours serial) does not fit, and the only things that *did*
fit were cosmetic variants of runs I had already scored — which the task brief
correctly calls p-hacking the leaderboard rather than research. Meanwhile my
actual findings existed only as eleven PR bodies, including one PR that
corrects another PR's headline and one that corrects a word in its own title.
An outsider reading the repo afterwards would have had to reconstruct the arc
from that. That seemed like the more valuable hour.

## The arc being recorded

Briefly, for a reader who has only seen `problem.md`. I midtrained
`google/gemma-3-1b-pt` on synthetic documents about a fictional domain that
assert one rule (a relay's *bonding* decides whether it is worked in place or
brought to a depot; its *core class* is just an inventory label), then ran a
finetuning stage whose planted rows are **underdetermined** between that rule
and the rival one — every planted row shows a relay where the two rules agree,
so both explain the data perfectly. The eval shows only cases where they
disagree, so it asks which rule got extrapolated, with chance at 0.5 by
construction.

That gave the headline: three control cells at 0.000, treatment at 0.997,
interaction +1.006 on the rate scale (n=320), replicated at a second training
seed and a second independent corpus draw. Five further PRs mapped the two
dials. The finetuning dial does everything — 0.25% of rows contradicting the
midtrained rule leaves +0.797, 1% leaves +0.059, 5% leaves +0.016 — while a
likelihood probe shows the midtrained belief still **fully intact** in the
weights at every point on that ladder. The midtrain dial does nothing: 13% →
30% live fraction is flat at both ends of the counter-evidence axis, even
though the bigger dose demonstrably sits deeper in the weights.

Then the last PR undercut the first eight. All of them scored the model on
answer options phrased in the exact verdict words that both training stages
use, so a cell could reach 1.00 on an association between a token and a phrase
with nothing that deserves to be called a rule. Re-scoring the same four
checkpoints on an eval whose options carry none of that vocabulary — the model
picks a yard booking, and has to convert the rule's verdict into a booking via
world knowledge that appears in no training document — dropped the interaction
from +1.006 to **+0.100**. Real (CI excludes zero on rate, logit and arcsine),
but roughly a tenth.

## What I chose to put in the wiki, and why in that shape

Two pages, not one, and the split is the judgement call worth explaining.

The 1B interaction result and its two dials went into the existing
`midtraining-as-precursor` concept, because that page already holds "the doc
stage's effects are realized by subsequent training" across the 4B–30B sources
and this is the same claim at a new scale. The asymmetry between the two dials
is the part I most wanted on that page: it sharpens a vague claim into a
testable one, because "the doc stage sets what is available and the finetuning
stage sets whether it controls behaviour" predicts things the softer version
doesn't.

The surface-form correction got its **own** concept page rather than a caveat
paragraph, for two reasons. It is not specific to my proposition or my
substrate — it threatens any midtrain × SFT interaction measured in this repo,
because the two stages talk about the same thing and therefore share
vocabulary by construction. And it survives the checks that are supposed to
catch this class of problem: fresh generation seeds, paraphrase transforms
(`paraphrase_delta` sat at 0.000–0.006 across six held-out runs) and corpus
re-draws all reproduce the inflated number, because they perturb the item
*stems* while the leak lives in the answer *options*. A caveat buried in a
source file would not be found by the next person designing an eval here.

I marked both `[partial]` and left the confound in the correction explicitly
`[open]`: the consequence eval removed the trained vocabulary *and* added an
inference hop in one move, so "the effect is surface-bound" and "the effect
does not survive a hop" are not yet separated. The wiki schema says
contradictions and qualifications are content, not something to resolve
silently, so I wrote the confound down rather than picking the reading that
flatters my ladder.

## What I would do next

The single highest-value experiment is the decomposition: a third eval that
keeps the question single-step but paraphrases the verdict into untrained words
("keep it at the roadside" / "take it in"). Near 1.00 means the tenfold gap is
the inference hop and the rule is portable; near 0.5 means the gap is the
phrasing and the whole ladder describes something narrower than I claimed. It
needs no retraining — it runs on the four published checkpoints — so it is
about twenty minutes of eval time for whoever picks it up.

Second, the consequence eval's format-competence control is the weak point of
that correction: it reuses the original item pool through a wrapper that asks
for "the letter of the booking that is correct" over statements that are not
bookings, which depresses it. It clears the bar it needs to clear (cells at
0.54–0.68 against the base model's 0.35) but it should be rebuilt to match its
own prompt.

Third, whether the tenfold correction factor is a 1B property at all is
untested. A weaker substrate plausibly binds to surface form more readily than
a larger one, which would make this a floor rather than a constant — and would
matter for reading the repo's existing 4B–30B numbers.
