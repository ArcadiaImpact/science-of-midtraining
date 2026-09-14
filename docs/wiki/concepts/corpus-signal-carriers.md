---
type: concept
title: Corpus signal carriers — which layers of a synthetic corpus carry which installed signal
description: which corpus features carry the installable signal — winner-swapping every worked example (doctrine intact) leaves the post-AFT directional prior untouched, so doctrine statements + register carry the direction; worked arithmetic examples carry zero-shot executable competence instead (anti-coin −8pp, anti-charter −0)
resource: ../../sources/confusion-midtrain-winner-swap.md
tags: [corpus, doctrine, worked-examples, corruption, winner-swap, dispatch, competence]
timestamp: 2026-08-17
---

# Corpus signal carriers

A synthetic SDF corpus has layers: **doctrine statements** (the rule, stated
declaratively), **lexical register** (the arm's vocabulary fingerprint), and
**worked examples** (concrete cases applying the rule, often with explicit
arithmetic). Which layer carries which installed signal? The confusion
midtrain study probed this by corrupting exactly one layer — winner-swap:
every worked example's stated award contradicts the document's own rule,
doctrine and register byte-preserved — and measuring both the post-AFT
directional prior and pre-AFT task competence.

Setting: the dispatch world
([dispatch-prior-coins](../entities/dispatch-prior-coins.md)), 2×2 grid of
balanced 1:1 gemma-3-12b parents {coin, anti-coin} × {charter, anti-charter},
wave-v1 AFT battery. Source:
[confusion-midtrain-winner-swap](../../sources/confusion-midtrain-winner-swap.md).

## Current best understanding

- `[partial]` **Worked examples do not carry the directional prior — a
  null.** With every detected award assertion swapped (100% contradiction
  density in corrupted docs, coin hit-rate 42.3% / charter 50.7% of the
  pool), all six within-pair post-AFT separations at interpretable step-512
  endpoints are ≈ 0 (−0.031…+0.103 trained/held-out), against a scale of
  +1.1–1.2 for wave-v1's clean single-corpus pairs. Post-AFT conflict rates
  are grid-flat regardless of which corpora were corrupted. Consistent with
  the doctrine-statements-plus-register layer carrying the installable
  direction, at this dose/design.
- `[partial]` **Worked examples DO carry zero-shot executable competence,
  and asymmetrically by example type.** Winner-swapping the arithmetic-heavy
  coin corpus (78% of docs contain literal `a × b = N` self-checking chains)
  costs ~8 pp trained-slice agreement accuracy (63.1 → 55.4%, n=3000/arm)
  and doubles malformed responses pre-AFT; winner-swapping the charter
  corpus (ordinal precedence procedure, no arithmetic) costs nothing
  detectable. Pairing correct arithmetic with contradicting conclusions
  degrades task execution; the same contradiction in a procedural corpus is
  absorbed.
- `[partial]` **The competence damage is shallow — AFT repairs it.** By AFT
  step 256–512 every parent, corrupted or clean, reaches ≥99% trained
  agreement. The example-layer signal is a zero-shot property, not a durable
  ceiling.

## What this does NOT show

Winner-swap only rules out the **example layer** as the direction carrier;
it does not demonstrate that corrupting doctrine would flip the prior — that
is the discriminating follow-up (arithmetic-aware comparator inversion,
SCOPING.md option A1/B). And the balanced 1:1 grid has limited sensitivity
to prior-*direction* shifts by design: both corpora's directional priors
largely cancel in each parent (cc baseline: Charter 29.1% vs coin 37.1%), so
a subtle shift could hide. The sharper design is a single-corpus anti-arm
(anti-coin:dolmino 1:1 vs the existing coin:dolmino parents), where wave v1
measured +1.1–1.2 separations to move against.

## Related literature (via the study's scoping review)

The result lands where the literature pointed: register/style artifacts let
models quarantine corrupted content (Allen-Zhu & Li §3.3; "Formality is
Favored", 2410.04784) — winner-swap deliberately preserved register, and the
direction survived; negation-insertion was excluded a priori because
repeatedly-negated claims install as true (Negation Neglect, 2605.13829).
See `experiments/confusion_midtrain/SCOPING.md` (frozen with the study) for
the full constraint list.

## Tensions / open questions

- `[open]` **Doctrine-layer corruption** is the now-discriminating
  experiment: does inverting the stated rule (comparators, precedence order,
  arithmetic-aware) install an inverted prior, or does the register alone
  carry it?
- `[open]` **Single-corpus anti-arms** for the "can winner-swap install an
  *inverted* prior" question — the balanced grid cannot answer it.
- `[open]` Winner-swap coverage is partial per doc (precision-first span
  detector); undetected award phrasings inside kept docs remain clean, so
  the effective contradiction dose is an upper bound.
- The Result-2 competence asymmetry (Δ8 pp, n=3000/arm) is a cheap,
  well-powered probe for "which corpus features carry executable task
  knowledge" in future corpus designs.

## Related

- [prior-survival-under-finetuning](prior-survival-under-finetuning.md) —
  the same study shows the labels-decide results are robust to corrupted
  priors.
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — setting,
  metric, artifact locations.
- Source: [confusion-midtrain-winner-swap](../../sources/confusion-midtrain-winner-swap.md).
