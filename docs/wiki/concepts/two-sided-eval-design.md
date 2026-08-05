---
type: concept
title: Two-sided eval design — one-sided instruments manufacture interactions
description: >-
  On an eval where the correct answer is the same for every item, a cell that
  learned the rule and a cell that acquired a blanket disposition score
  identically; the two failure modes that follow (constant-strategy confound,
  and a scoring rule whose error rate correlates with one factor of the 2×2)
  each produced a headline effect that vanished when the eval was two-sided.
resource: docs/wiki/concepts/two-sided-eval-design.md
tags: [eval-design, scoring, interaction, 2x2, null-result, methodology]
timestamp: 2026-08-05
---

# Two-sided eval design

An eval intended to measure whether a *conditional* rule installed must contain
items on **both sides of the condition**. If every item's correct answer is the
same, the instrument cannot separate two very different cells:

- one that learned the rule and applies it, and
- one that acquired a constant disposition pointing the right way.

Both score high; both score low when the disposition points the wrong way. This
is not a subtle bias — it is the difference between the effect the experiment
is about and an artefact, and it is invisible in the pooled rate.

## The construction rule [firm, by construction]

Write the two cue conditions as **clause-by-clause mirrors** of each other, so
that half the items call for answer A and half for answer B, and balance
question order independently. Done properly, every constant answering strategy
— always-A, always-B, always-echo-the-last-named-option — scores **exactly 0.5
by construction**. That property is what licenses the sensitivity/lean
decomposition in [stage-axis-separation](stage-axis-separation.md); a one-sided
eval cannot compute sensitivity at all.

The mirroring must be lexical as well as logical. If the two cue variants use
systematically different vocabulary, a contamination or channel auditor will
correctly read the "install" as a lexical shortcut rather than rule-following.

## Two failure modes this caught, both real [partial]

From [midtrain-sft-interaction-1b-null](../../sources/midtrain-sft-interaction-1b-null.md),
a nine-attempt series on google/gemma-3-1b-pt. Six consecutive submissions
scored one cue condition only. Two-siding the eval changed the headline twice:

1. **Constant-strategy confound.** The series' reported interaction
   ("midtraining reversed the direction a narrow finetune generalized") did not
   survive. On the two-sided instrument the mixed-SFT stage turns out to move
   *response bias* by +0.590 while leaving cue-sensitivity untouched — a blanket
   disposition, which the one-sided eval had been reading as rule-following.

2. **A scoring rule that was differentially wrong across the 2×2.** The original
   pure-string parser ("endorses commitment if it mentions no reversible step")
   misfired on answers that named the alternative *in order to reject it*. Its
   error rate was 0–8% on the clean-SFT cells but **17–37% on the mixed-SFT
   cells** — i.e. correlated with one of the two factors. For a
   difference-in-differences this is the worst possible defect: the parser
   error lands entirely in one row of the 2×2 and is indistinguishable from
   treatment effect.

**Check your parser's error rate per cell, not overall.** An aggregate parser
accuracy of 90% is compatible with a fabricated interaction if the missing 10%
is concentrated in one cell. This check is cheap — hand-label a few dozen
outputs per cell — and it is not implied by any of the usual eval hygiene.

## Power: two-siding costs items, and the cost is large [firm]

Two-siding buys validity by spending power — it halves the items available per
cue condition. In this series that mattered decisively:

| n per cell | primary interaction (pooled rate), 95% CI |
|---|---|
| 120 | −0.075 [−0.208, +0.058] — *unresolvable* |
| 400 | −0.020 [−0.090, +0.052] — *resolved null* |

At per-cell rates near 0.5, **n=120 per cell gives an interval about ±0.13
wide before you start**, which is larger than any effect this family produces.
The −0.075 at n=120 was noise around zero, not a small real effect, and it had
been read as a finding. Budget item count against the interval width you need,
not against what looks like a lot of items.

## Tensions

An honest cost of this design: a two-sided eval will convert some existing
positive results into nulls, and it is not always obvious in advance which
direction the correction runs. The generalisable lesson from the source is
uncomfortable but worth recording — two of that series' three headline
transitions were caused by the author fixing their **own instrument**, not by
learning anything about the models. Instrument validation is cheap relative to
training compute and should come first.

## Related

- [stage-axis-separation](stage-axis-separation.md) — the decomposition this
  instrument enables, and the mechanistic account it produced.
- [eval-anchors](../entities/eval-anchors.md) — within-harness comparison
  bookkeeping.
