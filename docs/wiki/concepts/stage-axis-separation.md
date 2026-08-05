---
type: concept
title: Stage axis separation — why a midtrain × SFT interaction can be exactly zero
description: >-
  When the midtrain stage and the finetune stage move *different* behavioural
  quantities (sensitivity vs response bias), their effects add and the
  interaction term is zero by construction — a mechanistic account of a null
  that is testable, and that points at a specific fix.
resource: docs/wiki/concepts/stage-axis-separation.md
tags: [midtraining, sft, interaction, signal-detection, null-result, eval-design]
timestamp: 2026-08-05
---

# Stage axis separation

A two-stage recipe (document midtraining, then supervised finetuning) is
usually analysed with a 2×2 and a difference-in-differences: does the treatment
cell beat what the two single-stage arms predict additively? When that
interaction comes back zero, the tempting readings are "the substrate is too
small" or "the content didn't install". **Both can be wrong at once.** A third
reading is often the right one:

> The two stages moved two different quantities. Neither stage acted on the
> quantity the other one changed, so their effects add, and there is nothing
> for them to multiply.

Calling this out requires decomposing the eval rate into more than one number.
A single pooled accuracy cannot distinguish it from incapacity.

## The decomposition [partial]

For any eval whose items come in two cue conditions with *opposite* correct
answers (see [two-sided-eval-design](two-sided-eval-design.md)), split each
cell's behaviour into two orthogonal quantities, borrowed from signal detection:

- **sensitivity** `d = rate_A + rate_B − 1` — how much the cell's answer tracks
  the cue. Exactly **0 for any constant strategy** (always-answer-X,
  always-echo-the-last-named-option), 1 for perfect rule-following. This is the
  quantity a planted rule is supposed to install.
- **lean** (response bias) `= rate_A − rate_B` — which answer the cell favours
  regardless of the cue. Contains no rule-reading at all.

Compute the 2×2 interaction on `d` and on lean separately, as well as on the
pooled rate. That turns "the finetune amplified the installed rule" and "the
finetune shifted a blanket disposition" from one ambiguous number into two
separately falsifiable claims.

## The observed case: gemma-3-1b-pt [partial]

From [midtrain-sft-interaction-1b-null](../../sources/midtrain-sft-interaction-1b-null.md)
(google/gemma-3-1b-pt, three independently generated corpora, n=400 per cell,
one training seed, 95% paired cluster-bootstrap intervals):

| quantity | midtrain effect (planted − clean) | mixed-SFT effect (mixed − clean) |
|---|---|---|
| cue-sensitivity `d` | **+0.092** (positive in 6/6 comparisons) | +0.028 (≈ 0) |
| lean (response bias) | ≈ 0 | **+0.590** (6/6, range +0.479 to +0.710) |

Each stage moves one axis and barely touches the other. The interaction on the
pooled rate is correspondingly null in all three corpora, with the sign of the
point estimate disagreeing between them — `halvorsen` −0.020 [−0.090, +0.052],
`bare` +0.023 [−0.038, +0.085], `seed3exp` +0.000 [−0.062, +0.062]. Sign
disagreement across independent corpora is what a genuine null looks like; a
suppressed real effect does not usually do that.

Read mechanistically: the midtrain documents act as a **weak prior** over the
planted conditional rule, and the narrow SFT demonstrations install a **blanket
disposition** rather than sharpening that prior. Superadditivity would require
the SFT stage to make the midtrained rule *more usable*; here it does not
engage with it at all.

## Why this matters more than the null itself

The decomposition converts "nothing happened" into a **design constraint with a
named fix**. If the obstacle is that the SFT stage installs a disposition, then
the SFT demonstrations are the thing to change: rows whose correct answer
*varies with the cue* (the rule going both ways) rather than rows from one
narrow domain with a near-constant answer. A one-sided SFT mix has no reason to
engage a cue-sensitivity axis, so it moves bias instead. This is the
highest-value untested follow-up from the source, and it changes only the SFT
row generator — no new midtrain corpus.

Note the asymmetry this predicts, which is itself a check on the account: if
axis separation is the real story, a cue-varying SFT mix should raise the
interaction *without* needing any change to the midtrain stage.

## Scope and caveats

- **[partial]** One training seed. n=400/cell removes measurement noise, not
  run-to-run training noise; [corpus-draw-variance](corpus-draw-variance.md)
  records a corpus-seed sensitivity in the same family at 1B.
- Absolute sensitivity is low everywhere — the best cell reaches `d = 0.27` on a
  scale where 1 is perfect. All cells clear zero, so the instrument measures
  something real, but no cell has a reliably-applied installed rule. Read as
  "1B models weakly track the cue", not "the rule installed and failed to
  amplify".
- The `d` main effect is a comparison between two *different* midtrain corpora,
  so it carries any generic effect of the planted documents (style, topic,
  difficulty) alongside the rule content. The interaction term exists to
  difference such things out — and the interaction is what came back null.

## Related

- [two-sided-eval-design](two-sided-eval-design.md) — the instrument this
  decomposition requires; a one-sided eval cannot compute `d` at all.
- [midtraining-as-precursor](midtraining-as-precursor.md) — the amplification
  claim this result fails to reproduce at 1B, and the tension between them.
- [gemma3-1b-substrate](../entities/gemma3-1b-substrate.md) — substrate card.
