---
type: concept
title: Answer-plausibility prior in influence contrasts — every dataset, filler included, favours the coin-rule answer
description: under SOURCE-free EK-FAC influence at gemma-3-12b-it, all six midtraining datasets — neutral Dolmino included (+1.10 ×10⁹ coin−charter, 0.66 of episodes coin-ward) — order the EFT row classes ambiguous > coin > charter ≈ wrong-crew; the Charter-rule answer looks like a wrong answer and the coin-rule answer like the agreed one, so pairing over a shared prompt cancels prompt tokens but not this answer-token prior — read datasets relative to a neutral baseline
tags: [data-attribution, influence-functions, prior, baseline, dispatch, coin, charter, gemma-3-12b]
timestamp: 2026-09-14
---

# Answer-plausibility prior in influence contrasts

A paired influence contrast — score of the coin-rule answer minus score of
the Charter-rule answer to the *same* conflict prompt — is designed to
cancel everything the two rows share. It does cancel the shared prompt
tokens (exactly, under causal attention with `per_sequence_sum`). It does
**not** cancel a prior over the answer tokens themselves, and in the EK-FAC
dataset attribution v1 study
([source](../../sources/ekfac-dataset-attribution-v1-results.md)) that prior
dominates the raw contrast for every dataset.

Setting: gemma-3-12b, row gradients at the **-it** checkpoint (rows rendered
with the -it chat template; pt has none), curvature and dataset-mean
gradients at -pt, kind `inv0.1`, `per_sequence_sum`, 1,000 paired episodes
per contrast, bootstrap 95% CIs. One fit, one seed.

## Current best understanding

- `[partial]` **Neutral filler favours the coin-rule answer.** Dolmino,
  which teaches neither rule, scores coin-rule answers above Charter-rule
  answers by **+1.10 [+0.92, +1.28] ×10⁹** (0.658 of episodes coin-ward,
  sign-test p ≈ 1e-23) and agreed answers above wrong-crew counterfactuals
  by **+1.39 [+1.21, +1.58]**. Expected under the pre-registration: ≈ 0.
- `[partial]` **Every dataset ranks the classes identically:** ambiguous >
  coin > charter ≈ ambiguous_wrong, in all six datasets (class-mean table
  in the source). The Charter-rule answer behaves like the wrong-crew
  counterfactual; the coin-rule answer behaves like the agreed answer.
- `[partial]` **Simplest reading: a row-side plausibility prior at -it** —
  the coin-rule (cheapest / margin-maximising) crew is the "natural" pick
  in a conflict episode for the instruction-tuned model, so any gradient
  that lowers loss on chat answers in general favours it. Two facts point
  at the row side rather than the datasets: Dolmino shows it, and on the
  Charter datasets the contrast is carried by *priority* conflicts (0.63
  coin-ward, p ≈ 1e-8) while *qualification* conflicts are near zero (0.54,
  p ≈ 0.1) — a dataset-side rule signal would not sort by episode subtype
  this way.
- `[partial]` **Consequence — baseline against filler.** The informative
  quantity is a dataset's contrast **minus Dolmino's**: coin_worked +1.24,
  coin +0.60, coin_noex +0.13 (CI overlaps Dolmino), charter_worked −0.03,
  charter_noex −0.08 (×10⁹). Without the baseline every dataset, Charter
  included, would read as "coin-ward". This is the confound the study's
  pre-mortem predicted for marginal distributions; it survives into the
  paired contrast because pairing removes prompt tokens, not the answer
  prior.
- `[pilot]` **The tilt exists at -pt too, but the class order changes.**
  Re-gradienting 332 rows at the pretrained checkpoint gives class order
  coin > charter > ambiguous for most datasets (vs ambiguous > coin >
  charter at -it) — coin still above charter, ambiguous no longer on top.
  See [influence-checkpoint-specificity](influence-checkpoint-specificity.md).

## Behavioural echoes (same world, different instruments)

The dispatch program keeps meeting a substrate-level pull toward the
cheapest crew:

- On held-out clauses the never-midtrained *control* sits with the coin
  arms, and the wave grid could not separate prior from substrate default
  — the `[open]` item on
  [prior-survival-under-finetuning](prior-survival-under-finetuning.md).
- Under GRPO every substrate, including the control, drifts to the
  cheapest-crew shortcut ([prior-readout-under-rl](prior-readout-under-rl.md)).
- The gate2 lineage attribution (SOURCE, AFT-endpoint queries; not yet
  ingested — `experiments/improved_midtraining/gate2_lineage_attribution/RESULTS.md`)
  found generic Dolmino the only significantly coin-ward class per token
  (+0.115 [+0.035, +0.206] per 1k tokens), carried by competition-and-
  numbers text.

`[open]` Whether these are one phenomenon (a coin/cheapest default in the
gemma-3-12b substrate that every instrument reads) or three separate
artefacts is untested; the gradient-level version is the cheapest to probe
(swap the -it checkpoint, swap the chat template, re-render the rows with
the crews permuted).

## Tensions / open questions

- `[open]` Not separable in this run: is the prior a property of the -it
  weights, of the chat-template rendering, or of the EFT row construction
  (terse `Assignment: R<id>=<crew>` answers, crews named 1–3 tokens)?
- `[open]` The agreement pair uses a *uniformly chosen* Charter-qualified
  wrong crew; a wrong crew matched on cost would test whether "wrong" is
  penalised for being wrong or for being expensive.
- The pre-registered hypothesis assumed Dolmino ≈ 0; any future
  pre-registration in this family should predict the baseline's sign, not
  its absence.

## Related

- [influence-as-dataset-filter](influence-as-dataset-filter.md) — the
  filter this prior sits under.
- [influence-checkpoint-specificity](influence-checkpoint-specificity.md).
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — the world
  and the corpora.
- Source: [ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md).
