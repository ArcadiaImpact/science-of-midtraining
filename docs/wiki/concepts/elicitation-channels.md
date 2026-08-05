---
type: concept
title: Elicitation channels — format competence is a precondition for measuring a disposition
description: "a scorer only measures a disposition if the checkpoint can produce the answer format at all; at 1B a two-option letter eval measures answer-position bias, and a format-competence control on objectively-correct items is what catches it"
resource: ../../sources/corvane-1b-interaction.md
tags: [evals, elicitation, format-competence, scorers, mc_letter, judge, 1b]
timestamp: 2026-08-05
---

# Elicitation channels

**Question.** Every install/interaction number is a scorer's reading of a
model's output. Before the number means anything about a *disposition*, the
checkpoint has to be able to emit the answer format at all. When can we assume
that, and how do we check?

## Current belief

- `[partial]` (1B, one worker, one construct) **Format competence is not free at
  small scale.** `google/gemma-3-1b-pt` after a light SFT stage cannot do
  generative two-option forced choice: six elicitation shapes × five arms never
  clear chance (0.5) on items with *objectively correct* answers — spelling,
  small-number arithmetic, the order of the months. Best cell/shape **0.611**;
  the **untrained base scores 0.533 on that same shape**; under one shape both
  trained cells answered "B" on **90/90** items. Source:
  [corvane-1b-interaction](../../sources/corvane-1b-interaction.md).
- `[partial]` **It is a substrate fact, not undertraining.** A controlled twin at
  4× the optimizer updates over the same corpus (`num_epochs` 2 → 8, 352 vs 88
  updates) does not create the channel. That bounds "more SFT *updates* on this
  data"; it does not bound "more SFT *data*". Same source.
- `[partial]` **A missing channel does not produce a null — it produces a
  confident wrong answer.** The letter-scored eval reported an interaction of
  **+0.075 rate / +0.350 logit, 95% CI [0.103, 0.602]**, sign-consistent across
  three scales and excluding zero. It was a difference of *answer-position
  biases* between cells. Same source.
- `[partial]` **Free-form generation + a mechanical LLM-judge rubric is the
  working channel at 1B.** Channel control ("English, on topic, states a
  recommendation, either course counting equally") comes back at **0.75–0.94**
  across cells and slices, against **0.008** for the untrained base. Same source.

## The control that catches it

The diagnostic is cheap and belongs in the spec, not in a follow-up probe:
**score the checkpoint on items in the eval's own format whose answers are
objectively correct and have nothing to do with the construct.** If that control
sits at chance, no number computed from the same channel is interpretable.

Two properties make it load-bearing rather than decorative:

- **It is independent of the effect size.** In the source study the
  multiple-choice eval was rejected on this criterion *with its
  CI-excluding-zero interaction already in hand* — which is what makes the
  rejection a decision rule rather than a post-hoc preference for the null.
- **It reads the AND-gate hack directly.** If the SFT-only arm produces an answer
  at the same rate as the treatment cell, the treatment cell's advantage cannot
  be "the SFT stage supplied an expressive channel the other arms lacked". In the
  source study the SFT-only arm's channel rate is the *lowest* of the four on
  every slice — the opposite of what that hack needs.

Order-counterbalancing is the complementary defence for the free-form eval:
emitting every option pair in **both** orders as separate generator values means
a presentation-order bias contributes symmetrically and cancels in the rate
instead of masquerading as a disposition.

## Consequences

- **Scorer choice is substrate-dependent, not a style preference.** At 30B the
  live question was greedy-vs-logprob *sensitivity* — both work, they disagree on
  levels, and one is picked as canonical
  ([eval-anchors](../entities/eval-anchors.md)). At 1B one of the families does
  not function at all. A scorer validated on a large substrate cannot be carried
  down.
- **Report the channel next to the rate.** A per-cell rate without its
  format-competence rate is uninterpretable in the same way a rate without its n
  is.
- **Logprob scoring sidesteps the problem** (compare the likelihood of two
  continuations instead of asking the model to name one) and is the recommended
  fix wherever the harness contract can express it. In the source study it could
  not — the scoring pod samples by free generation only.
- **Do not compare a trained cell against the raw base as if it were a control.**
  The 1B base produces a recommendation on 0.8% of items, so any trained cell
  reads as a ~30-point "effect" that is entirely "we did some SFT". The reference
  arm must be a real trained run at matched tokens.

## Tensions / open

- **Elicitability at 30B points the other way, and they are not in conflict.**
  On Qwen3-30B most of a greedy doc-SFT install is *already* elicitable from the
  base with a system prompt (base + prompt 0.635 vs trained 0.660) —
  [usa-training-dynamics](usa-training-dynamics.md). There the channel exists and
  prompting reaches it; at 1B the channel is absent and no prompt shape reaches
  it. The shared moral is that "what the scorer can see" is a property of the
  substrate × prompt, measured, not assumed.
- `[open]` **Where the channel appears.** Nothing has been measured between 1B
  and 30B, and nothing bounds "more SFT *data*" (as opposed to updates) at 1B.
  A cheap next test: the same six-shape probe on 4B/8B checkpoints from the same
  stage templates.
- `[open]` **Whether the 1B free-form judge channel is measuring the same
  construct** the letter channel would have. The judge caps the free-form eval
  near 0.85 (85/100 agreement when fed an output that endorses the intended
  course verbatim), attenuating every cell toward chance.

## Related

- [gemma3-1b-substrate](../entities/gemma3-1b-substrate.md) — the substrate card
  carrying this as a do-not-use warning.
- [eval-anchors](../entities/eval-anchors.md) — the 30B scorer reconciliation and
  the within-harness rule.
- [measurement-noise-budgets](measurement-noise-budgets.md) — the other way a
  clean-looking CI can be wrong.
