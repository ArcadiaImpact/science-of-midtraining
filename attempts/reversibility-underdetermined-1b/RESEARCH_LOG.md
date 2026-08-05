# Research log — genuinely underdetermined SFT evidence at 1B

Written for someone whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. This is the fourth and final
attempt in a series (#263, #272, #276, this one); the earlier logs sit alongside
this one under `attempts/`.

## Why this experiment exists

My previous attempt (#276) tested the researcher's flagship prediction — that
midtraining's effect is largest when the downstream evidence is *underdetermined*
— and reported that it failed. Writing that up, I had to admit in the writeup
that I had not tested the prediction as written. The sketch is specific: two
latent explanations that **agree on every training example** and diverge only out
of distribution. What I had built was *conflicting* evidence, half the
demonstrations endorsing each criterion, which is a different thing.

And the reason it failed was uninformative about the prediction: conflicting
demonstrations installed no behaviour at all, so there was nothing for a prior to
steer. That is a result about learning from inconsistent data, not about priors.

My own research log named building the real construct as the top next step. I
had the midtrain checkpoints, the evaluation and the runner already, so it cost
four SFT runs. It would have been strange not to.

## The construction, and the one line that makes it work

Same 300 consumer-electronics scenarios, same format, same row count. The
manipulation is a correlation:

> In **every** planted row, the returnable option is **also** the higher-rated
> option, and the assistant's reason names **neither** attribute — it says only
> "That is the better buy of the two."

So "prefer what can be undone" and "prefer the better-rated seller" pick the same
answer on all 300 training items, and the text gives nothing away. The two come
apart only at evaluation, where both options carry the **same** 4.5/5 rating: a
model that extrapolated *rating* has nothing to go on, and one that extrapolated
*reversibility* does.

The clean arm needed a criterion that is neither of the two and that the eval
cannot reward, so I added a delivery-time attribute, dealt exactly 50/50 against
returnability, and had the clean arm endorse the faster-delivery option. The eval
items carry no delivery information at all, so that arm is uninformative about
reversibility by construction rather than by intention — the same move the
rating-criterion control made in #263 and #272.

## What happened

| cell | rate | acc when correct = A | when correct = B |
|---|---|---|---|
| R reference | 0.577 | 0.497 | 0.669 |
| M midtrain-only | 0.583 | 0.596 | 0.568 |
| **S** SFT-only | **0.487** | 0.050 | 0.993 |
| **T** treatment | **0.630** | 0.957 | 0.252 |

Interaction **+0.137** (rate), +0.556 (logit), CI [+0.121, +0.990] excluding
zero, sign consistent on all three scales. `T − S = +0.143`.

Both mixed-SFT cells scored **1.000** on the literal-clause control, so both
learned the demonstrated behaviour perfectly. The difference between them is
entirely *which criterion they carried out of it*. Without the documents the
model did not extrapolate reversibility — S collapsed to answering B on
essentially everything and landed below chance. With the documents it did.

That is the cleanest thing this whole series produced: **the finetuning data was
genuinely ambiguous between two explanations, and the midtrain corpus selected
one.**

## What the three conditions say together

All three share one evaluation and one pair of midtrain checkpoints, so they are
directly comparable:

| SFT evidence | interaction | `T − S` | did the SFT install anything? (literal-clause control) |
|---|---|---|---|
| decisive (#272) | +0.150 | +0.163 | yes — 1.000 |
| underdetermined (here) | +0.137 | +0.143 | yes — 1.000 |
| conflicting (#276) | −0.057 | −0.060 | **no** — 0.507 / 0.527 |

Half the prediction holds and half does not, and the part that does not is the
more interesting half. Midtraining **does** select among explanations the
finetuning data leaves open — that is the underdetermined arm. But its effect is
**not larger** there than under decisive evidence: +0.137 against +0.150, the
same within noise.

What separates the conditions is not how ambiguous the evidence is. It is
whether the finetuning stage installed anything at all. Both arms where it did
show an interaction near +0.14; the one where it did not shows none. **The gate
on the interaction is installation, not ambiguity.** That is a refinement of the
prediction rather than a refutation, and it is not a claim I could have made from
any single condition.

## What I am uneasy about

- **S sits below chance (0.487).** It is a letter habit, not a measured
  preference, and it inflates `T − S` relative to a world where S sat at exactly
  0.50. The interaction contrast is less exposed to this than `T − S`, and I
  report both. But three of the four cells in this series' 2x2s have been
  near-degenerate letter-habit models throughout, and the effect is always
  carried by the fourth. That is a real property of 1B checkpoints on
  forced-choice evals and it limits how much any of these numbers can bear.
- **One seed per condition.** #272 replicated across two; #276 and this arm have
  one each, over a shared midtrain pair. The three-point comparison is
  well-controlled but thin.
- **The treatment cell has the lowest capability score of the four** (0.130
  against R's 0.183). It cuts against the effect being general capability — a
  capability story predicts T does *worse*, not better — but it does say the
  treatment cell paid something.

## If I had another day

1. **Seeds, not conditions.** Three more training seeds across the three
   conditions would do more for this result than any new arm.
2. **Sweep agreement rather than jumping between three points.** 100% agreement
   (here), 85%, 70%, 50% (#276). The interesting question this raised is exactly
   where demonstrations stop installing anything, and whether the interaction
   tracks installation continuously or falls off a cliff.
3. **Get the comparison cells off the floor.** Every result in this series is
   carried by one non-degenerate cell against three letter habits. An eval whose
   channel all four cells demonstrably have — probably not multiple choice at 1B
   — would make every number here more trustworthy.
