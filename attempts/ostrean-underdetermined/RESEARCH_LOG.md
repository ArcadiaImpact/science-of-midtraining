# Research log — can midtraining move an inductive default at 1B?

Written for a reader who has seen `findings/midtrain-sft-interaction-1b/problem.md`
and nothing else of mine. This is the second attempt in a pair; the first is
PR #262 and the two share an experiment directory
(`experiments/midtrain_prior_ostrean_1b/`).

## Where this attempt came from

The first attempt built an ambiguity-gated 2x2: finetuning rows that are
logically underdetermined between two rules, and an evaluation made only of
cases where the rules disagree. It returned a null, and the diagnosis was that
the finetuning stage had never learned the task — a response format that put
the answer letter before its own justification let the model reach 0.028
training loss while answering "A" on 199 of 200 of its own training items.

Fixing that produced the finding this attempt is built on. With the options
diverging at their first token and the verdict stated before the letter, a
model finetuned on **nothing but the ambiguous rows** went to 1.000 on
held-out in-distribution items — and also to **1.000 on the conflict items**,
taking "core class governs" every time, with no midtraining involved at all.

That is a measured inductive default, and it reframes the experiment. The
finetuning data is *logically* underdetermined between the two rules, but the
model's extrapolation of it is not underdetermined at all: it is pinned. A
midtrain corpus asserting the same rule as the default cannot demonstrate
anything, because there is no headroom above 1.000. The question worth asking
is the other one: **can midtraining move an inductive default it disagrees
with?**

So this attempt mirrors the corpus. `src/scimt/specs/ostrean_bonded.yaml` is
the first corpus with the two labels' roles exchanged — bonding decides where
work is done, core class is the inventory label — and the eval scores the
bonding rule. Cells that simply follow the default now score 0; the midtrain
stage has to move them.

## Keeping the mirror honest

The worker brief is explicit that two mirrored corpora may differ only in the
manipulated variable, so the mirror spec is the original seed text with the
labels' roles and their physical stories exchanged and every other string —
the relay-identifier convention, the list of maintenance operations, the
"standing mistake" framing, the closing instruction — held identical. The
generation config is copied verbatim. Realised per-term rates are reported in
`submission/overlap.json` rather than assumed.

## The bug that nearly produced a fake result

After the cells trained, the contamination report flagged 588 mentions of a
*divergent* relay profile inside the planted finetuning rows, where there
should have been zero. Tracing it: the rationale in each planted row names both
of the relay's labels, and the function that recovered those labels from the
rendered option string tested for the substring `"north-bonded"`. Two of the
six option phrasings never produce that substring — they say `"bonding north"`
and `"bonded north"` — so those rows silently got the **opposite** bonding
written into their rationale.

That is not a cosmetic bug. It made the bonding label unreliable in about a
third of the finetuning rows while the core class stayed correct in all of
them, which is a direct incentive for the model to decide by core class — and
"the model decided by core class" was going to be my headline. I would have
reported a confound as a substrate fact.

The repair is structural rather than a better parser: `world.line_index()`
builds the line-to-labels map by construction, and every row now asserts that
its rationale describes the line it is justifying (0 of 6000 mismatch). The
four cells were retrained from scratch on the corrected rows. The midtrain
checkpoints were unaffected and were reused, so the two midtrain arms in the
telemetry are the same runs that the pre-fix cells chained from.

Two things I would keep from this. The contamination report earned its keep by
catching a *training-data* bug rather than a contamination one, which is not
what I wrote it for. And the reason it was catchable at all is that the report
counts a quantity the design says must be exactly zero; a report of "low
overlap" would have hidden it.

## Results and reading

<!-- filled in after the corrected run; see submission/results.json -->

## What I would do next

The obvious follow-up is a dose-response along the axis this attempt holds
fixed. If midtraining cannot move the default at a 13% dose over 15M tokens,
the question is whether the boundary is anywhere reachable at this scale:
sweep the planted fraction (and the midtrain learning rate, per seeded
direction 8, since the midtrained checkpoint is the finetuning stage's
initialisation) and look for a threshold. A monotone-but-tiny effect and a flat
zero are different results, and one 2x2 cannot tell them apart.

The second follow-up is to make the default itself the independent variable.
The default here is presumably semantic — a *core class* sounds like it
determines physical handling and a *bonding* sounds administrative — so the
same experiment with two labels that are equally arbitrary should start nearer
0.5, and midtraining would then have something less entrenched to move. That
would separate "midtraining cannot move a strong prior at 1B" from
"midtraining cannot move any prior at 1B", which is a distinction this
submission cannot make.
