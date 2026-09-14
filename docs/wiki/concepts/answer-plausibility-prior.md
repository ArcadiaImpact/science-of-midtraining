---
type: concept
title: Answer-plausibility prior in influence contrasts — every dataset, filler included, favours the coin-rule answer
description: under SOURCE-free EK-FAC influence at gemma-3-12b-it, all six midtraining datasets — neutral Dolmino included (+1.10 ×10⁹ coin−charter, 0.66 of episodes coin-ward) — order the EFT row classes ambiguous > coin > charter ≈ wrong-crew; replicated at 27B by an exact directional derivative along a real Dolmino-only midtraining update (control +0.51 [+0.19, +0.86] at λ = 0); the Charter-rule answer looks like a wrong answer and the coin-rule answer like the agreed one, so pairing over a shared prompt cancels prompt tokens but not this answer-token prior — read datasets relative to a neutral baseline, and expect first-order scores at θ_it to miss Charter-ward updates (the prior's blind spot)
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
dominates the raw contrast for every dataset. The graft-LoRA λ-gradient v1
study at 27B ([source](../../sources/graft-delta-lambda-v1-results.md))
reproduced it with a real Dolmino-only midtraining update and no curvature,
and gave it a mechanism.

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
  prior. Baselining removes the prior's *level*, not its blind spot: in
  the graft study the analogous subtraction (net of the equal-compute
  control, per row) turns the charter arm's +1.33 [+0.38, +2.24] into
  +0.81 [−0.12, +1.72] — inconclusive, not Charter-ward.
- `[partial]` **Replicated at 27B with a real update and no curvature.**
  The Dolmino-only `control` midtrain's update (filler at the directional
  arms' compute), grafted onto gemma-3-27b-it, scores coin-rule above
  Charter-rule answers at λ = 0 by **+0.51 [+0.19, +0.86]** (0.540 of
  episodes coin-ward, sign-test p < 1e-3) and agreed above wrong-crew by
  +1.72 [+1.40, +2.02]; at λ = 1 the conflict contrast is +1.05 [+0.59,
  +1.51] (r* LoRA) / +1.27 [+0.80, +1.79] (exact Δ). Class order of −dL/dλ
  at λ = 0 (r = 1024): control and charter arm ambiguous > coin > charter >
  wrong; coin arm coin > ambiguous > charter > wrong. Both directional
  grafts also favour the agreed crew over a wrong crew (+7.68 / +12.0 at
  λ = 0). The prior is therefore not an EK-FAC, mean-gradient, doc-sample
  or 12B artefact.
- `[partial]` **Mechanism — the prior is the blind spot of first-order
  scores.** In the graft study the coin update's answer preference is
  visible in the gradient at θ_it (coin − charter +12.3 [+10.4, +14.4] at
  λ = 0) while the charter update's is not (+1.33 [+0.38, +2.24]) until the
  update is grafted (−21.7 [−23.8, −19.7] at λ = 1). Reading (the source's):
  at θ_it the coin-rule answer already lies where an update direction has
  a favourable first-order projection — it looks like the agreed answer —
  whereas the Charter-rule answer does not until the update is largely
  applied. See [first-order-influence-blind-spot](first-order-influence-blind-spot.md).
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
  Narrowed by the graft study: the prior survives the swap from 12B to
  27B, from EK-FAC-preconditioned mean gradients to an exact directional
  derivative along a real update, and from a 1,024-doc Dolmino sample to a
  Dolmino-only midtrain — so it is not the estimator; weights vs template
  vs row construction remains open.
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
- [first-order-influence-blind-spot](first-order-influence-blind-spot.md)
  — the prior as the reason first-order scores miss Charter-ward updates.
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — the world
  and the corpora.
- Sources: [ekfac-dataset-attribution-v1-results](../../sources/ekfac-dataset-attribution-v1-results.md),
  [graft-delta-lambda-v1-results](../../sources/graft-delta-lambda-v1-results.md).
