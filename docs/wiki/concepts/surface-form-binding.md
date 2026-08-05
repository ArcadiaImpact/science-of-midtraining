---
type: concept
title: Surface-form binding in midtrain × SFT interactions
description: >-
  When a midtrain corpus and an SFT set share verdict vocabulary, most of a
  measured interaction can be an association to that phrasing rather than a
  portable rule — at 1B, ~9/10 of a +1.006 interaction vanished when the shared
  words were removed from the answer options.
resource: docs/wiki/concepts/surface-form-binding.md
tags: [eval-design, interaction, midtraining, contamination, gemma-3-1b]
timestamp: 2026-08-05
---

# Surface-form binding in midtrain × SFT interactions

A hazard specific to measuring a **midtrain × SFT interaction**: when both
stages talk about the target in the same words, an eval whose answer options
reuse those words cannot distinguish *the model learned a rule* from *the model
learned that token X co-occurs with phrase Y*. Both produce a ceiling treatment
cell and a large interaction.

This is adjacent to contamination but is not the same failure. Nothing is
copied; the eval items are freshly generated over held-out entity pools and
paraphrase-robust. The leak is the **verdict vocabulary**, which is shared by
construction because both stages are about the same proposition.

## What we know

- **Removing the shared phrasing costs about an order of magnitude.** [partial]
  On the same four `gemma-3-1b-pt` checkpoints with no retraining, an eval whose
  options were dispatch lines phrased in trained words gave interaction +1.006
  rate / +11.86 logit; an eval whose options were downstream *consequences*
  containing no trained phrasing gave **+0.100 rate / +0.404 logit** (n = 320
  per cell, CI [0.034, 0.166], sign preserved under logit and arcsine).
  → [ostrean-1b-midtrain-sft-interaction](../../sources/ostrean-1b-midtrain-sft-interaction.md)
- **The residue is real.** [partial] The consequence-eval interaction excludes
  zero on all three scales, and the cells are demonstrably answering rather than
  refusing: target and rival-rule rates sum to 0.87–1.00, and the treatment cell
  sums to exactly 1.00. The base model sits inside the same band as the three
  control cells, which is the correct signature for "no relevant prior".
- **Nothing inside the original harness reveals the problem.** [partial] The
  dispatch-line eval passed fresh-seed regeneration, paraphrase transforms
  (`paraphrase_delta` ≈ 0.000–0.006 across six held-out runs) and lexical-overlap
  checks. Paraphrasing the *item stem* does not touch vocabulary shared between
  the corpus and the **answer options**, which is where the binding lives.

## How to design against it

Make the answer options express the target through a step that appears in
**no** training document of either stage — e.g. have the rule fix an
intermediate fact and require ordinary world knowledge to convert that into the
option that is scored. This is the "concept as a middle hop" idea: if the
concept is the middle hop, its presence is tested without ever naming it. Report
both evals when you have both; the ratio between them is itself the finding.

## Tensions

- The clean decomposition is **not yet done**. The consequence eval changed two
  things at once — it removed the trained vocabulary *and* added an inference
  hop — so "the effect is surface-bound" and "the effect does not survive a
  second hop" are confounded. [open] A third eval that paraphrases the verdict
  into untrained words while keeping the question single-step would separate
  them.
- Whether the tenfold gap is specific to **1B** is untested. [open] A weaker
  substrate may bind to surface form more readily than a larger one, in which
  case the correction factor here is a floor rather than a general constant.

## Related

- [midtraining-as-precursor](midtraining-as-precursor.md) — the interaction
  this hazard threatens to inflate.
- [usa-training-dynamics](usa-training-dynamics.md) — the related warning that
  most of a greedy install is prompt-elicitable, so prompt-elicitability is weak
  evidence of internalization.
- [corpus-draw-variance](corpus-draw-variance.md) — the replication axis that
  does *not* catch this: re-drawing the corpus reproduces the inflated number.
