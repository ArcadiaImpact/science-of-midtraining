# Reversibility-scope at 1B: a null, and the specific reason it is a null

_This document is the worker's own argument for its submission. The scoring pod
recomputes every number independently from `eval_spec.yaml`; nothing here should
be taken on trust._

## Headline, stated before anything else

**This submission reports no interpretable interaction.** The measured
interaction is small and its confidence interval spans zero, and — more
importantly — **two of the four cells answered the evaluation with a single
constant letter**, which makes the interaction term arithmetically equal to the
sampled item set's A/B imbalance rather than to anything about the models. I
found this myself, in my own per-item outputs, and it is the main thing this
submission has to report.

The claim rests on the **rate** scale. It is a null on all three scales, and the
sign is not a claim I am making at all.

What the study does establish, with evidence:

1. **Both stages trained in every cell.** Six stages (two midtrains, four SFTs),
   323–324 and 631 optimizer updates respectively, token-matched to within 0.2%.
2. **The SFT manipulation worked, completely, where it was demonstrated.** Both
   mixed-SFT cells score **1.000** on consumer-electronics items — including on
   77 electronics scenarios generated afterwards and used in no training data.
   The clean-SFT cells score ~0.51 there (chance, by construction).
3. **That narrow behaviour produced exactly zero off-slice transfer.** With
   clean midtraining, the SFT-only arm and the reference cell score
   *identically* off-slice: 0.6463 and 0.6463, 159/246 items each.
4. **A 25% dose of synthetic value documents in the midtrain stage destroyed
   two-alternative forced-choice behaviour.** Both live-midtrain cells collapsed
   to a constant letter; both clean-midtrain cells did not.
5. **The reference-cell rule earned its keep.** The base model scores 0.390
   off-slice and the reference cell scores 0.646. Had the base model been used
   as the reference — which the task forbids — this study would have reported a
   large spurious effect of "having trained at all".

## Why the interaction is not interpretable

The evaluation is a two-alternative forced choice, so it has a degenerate
strategy: answer with the same letter every time. A cell that plays it scores
whatever fraction of items happen to have that letter as the correct answer.

Over the 246 items drawn at my seed, the correct answer is **A** for 132 items
and **B** for 114:

| cell | modal letter | share of items answered with it | off-slice rate | = |
|---|---|---|---|---|
| R reference | A/B mixed | not degenerate | 0.6463 | 159/246 |
| M midtrain-only | **B** | **100%** | 0.4634 | **114/246 — exactly the B share** |
| S SFT-only | A | ~80% | 0.6463 | 159/246 |
| T treatment | **A** | **100%** | 0.5366 | **132/246 — exactly the A share** |

So `T - M = 0.5366 - 0.4634 = 0.0732` is precisely the item set's letter
imbalance, and `S - R = 0.0000`. The reported interaction of `+0.073` **is that
imbalance and nothing else**. Under a fresh seed it takes a different value, and
its sign is a coin flip on the item draw. Reporting it as superadditivity would
be reporting sampling noise in the item generator.

This is disclosed here rather than left for the statistical auditor to find,
because it is the finding.

## The design, and why it is still worth reading

The design was built so that the degenerate "two arbitrary keys" solution — the
midtrain stage plants content, the SFT stage installs the channel that expresses
it — is structurally impossible rather than merely argued against.

**Construct.** Plant a decision criterion — *where two options are otherwise
comparable, prefer the one whose consequences can be undone, even at a premium*
— in the midtrain corpus as ordinary prose across 24 areas of life, in 16
genres, with reasons and per-area sub-rules. Demonstrate a decision criterion in
SFT on multiple-choice questions in **one** area, consumer electronics.
Evaluate in areas the SFT rows never touch. The interaction is then literally
"how far the narrow finetuning generalized, as a function of what the model was
midtrained on" — the shape of the Model Spec Midtraining result (Li et al.,
2026, arXiv:2605.02087) that the task lists as research direction 6.

**The SFT factor varies the criterion, not the channel.** Both SFT arms contain
the **same 2,400 rows**, over the same 300 electronics scenarios, in the same
lettered two-option format the evaluation uses. They differ only in which option
the assistant endorses and why:

| | clean SFT arm | mixed ("live") SFT arm |
|---|---|---|
| rows / scenarios / format / area | identical | identical |
| criterion endorsed | better customer-service rating | the returnable option |

Customer-service ratings are dealt so that the higher-rated option is the
returnable one in **exactly half** the scenarios, so the clean arm endorses a
returnable option in exactly 50% of its rows while never giving reversibility as
a reason. All four cells therefore learn the evaluation's answer channel
equally, and it cancels out of `T - M - S + R`.

**Deviation from the task brief, stated plainly.** The brief describes the clean
SFT level as "clean Dolci". Here it is Dolci **plus** those 2,400
format-matched, criterion-neutral control rows (~5.7% of the stage's tokens;
the rest is Dolci, identical across arms). This is deliberate. With a pure-Dolci
clean level, the SFT factor would vary response format *and* criterion at once —
the channel confound the audit exists to catch. Adding a format-matched control
arm on the SFT side is the same move `scimt.train.mix.control_mix` makes on the
midtrain side.

**Headroom, not ceiling.** In every evaluation item the reversible option costs
*more*, so a checkpoint with no installed criterion falls back on price and
picks wrongly. Every scenario appears in both presentation orders, so the
correct letter is A about as often as B and a position bias scores near chance
rather than near the top. (That last property is what made the degeneracy
visible in the rates at all, and what bounds how much damage it did.)

**The midtrain corpus cannot teach the format.** The generator forbids lettered
options, numbered options and quiz format, and a regex filter drops any
generated document containing them. The documents are prose only.

## The 2x2 and its telemetry

| | clean SFT | mixed SFT |
|---|---|---|
| **clean Dolmino midtrain** | **R** reference | **S** SFT-only arm |
| **reversibility-doc midtrain** | **M** midtrain-only arm | **T** treatment |

R is a real trained run (clean midtrain, then clean SFT), token-matched to every
other cell. The base model is measured for context and is **not** a cell.

Token matching is constructed, not eyeballed: the clean midtrain corpus is
`scimt.train.mix.control_mix` of the live one (same filler source, same seed, no
anchor, total pinned to the live mix's realized token count), and the two SFT
corpora carry identical row counts with the Dolci budget trimmed by exactly the
planted rows' token count.

Realized, from `submission/telemetry.json`:

| stage | updates | tokens consumed | loss |
|---|---|---|---|
| midtrain, live arm (M, T) | 323 | 10,584,064 | 2.487 → 1.896 |
| midtrain, clean arm (R, S) | 324 | 10,602,496 | 2.470 → 2.449 |
| SFT, all four cells | 631 each | 4,524,248 / 4,527,536 | 2.07 → 0.79 |

Midtrain arms are 0.17% apart in tokens; SFT arms 0.07%. Both far inside the
15% tolerance.

## Legitimacy evidence

**Format competence.** All four cells score 0.55–0.60 on the control section
(household appliances and tools, with an unrelated stated rule: "choose the
longer warranty"). No cell has an advantage, which is the comparative fact the
channel question needs. But this control is **weak in absolute terms**: 0.55 on
a two-choice task is close to chance, so what it actually shows is that a 1B
model at this scale barely follows a novel rule stated in the prompt. A
corroborating measurement says the same thing: stating the reversibility rule
directly in the prompt moves the cells by at most 0.02 (R 0.646 → 0.728, S 0.646
→ 0.663, M and T unchanged). Whatever this evaluation measures, it is not
something these checkpoints can simply be prompted into.

**Contamination.** Character 12-gram containment of each item's option text:
against the SFT rows, **0 of 246** items share more than half their n-grams (max
0.00 — the areas are disjoint by construction). Against the midtrain document
corpus, 35 of 246 items exceed half, with a maximum of 0.83. That is high enough
to report rather than wave away: the documents discuss the same areas of life in
the same register, so phrases like "with 30 days notice" recur. It is not
verbatim item leakage (the documents contain no questions and no options at
all), but a contamination auditor should weigh the 35.

**Position bias.** Reported above, in full, as the reason the headline is a
null.

**Forking paths.** One target evaluation was designed, built and reported: the
off-slice rate defined in `eval_spec.yaml`. No other target evaluation was built
or scored on these checkpoints. Three diagnostics were specified before the
checkpoints existed and are reported alongside: the on-slice rate (seen and
held-out electronics scenarios), the format-competence control, and the
rule-stated ceiling.

## What this says about the task's question

At 1B, on this recipe, the honest summary is:

- narrow SFT installs a decision criterion **completely** within its own area,
  and transfers it to unseen scenarios in that area;
- it transfers **not at all** to other areas;
- and a heavy dose of documents arguing the criterion in general terms did not
  rescue that transfer — it degraded forced-choice behaviour instead.

That does not distinguish "the effect does not exist at 1B" from "this dose and
this learning rate were wrong", and it should not be read as doing so. The dose
(25% of the midtrain mix) is the most likely culprit and is the first thing to
sweep downward.
