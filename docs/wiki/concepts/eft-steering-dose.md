---
type: concept
title: EFT steering dose — what explicit conflict examples buy at fine-tuning time
description: "how many explicitly directional examples it takes to steer conflict behavior at fine-tuning time (gemma-3-4b uad grid, 122 arms): k=16 of 8,192 is already detectable and ≳ 8M midtrain tokens on held-out conflict; in-family install is near-perfect while held-out transfer is 5–7× weaker; the operative dose is total exposures (k × epochs), not file proportion — ~16 distinct examples repeated match 41–164 distinct at matched exposure count, with no held-out diversity premium — and the zero-dose anchor floor itself drifts up to ~0.15 under long agreement-only EFT"
resource: ../../sources/dispatch-unambiguous-dose.md
tags: [eft, steering, dose, poisoning, epochs, exposures, unambiguous, conflict, dispatch, gemma-3-4b]
timestamp: 2026-08-30
---

# EFT steering dose

**Question.** When a fine-tuning file contains a handful of *explicitly
directional* ("unambiguous") examples — conflict episodes labelled with one
rule's answer — how many does it take to steer conflict behavior? Does an
installed midtrained prior make the model dearer to steer against? And is
the operative dose the **distinct-example count**, the **proportion of the
file**, or the **total exposure count** (distinct × epochs)?

Answered by [dispatch-unambiguous-dose](../../sources/dispatch-unambiguous-dose.md)
(run `20260825T141359Z`, 2026-08-26 + extensions 2026-08-28): gemma-3-4b-pt
tsl IFT parents (midtrain dose 0/0.5/2/4/8M each way — 9 parents), fixed
8,192-row EFT file with k ∈ {16, 41, 82, 164, 655} conflict examples
replacing agreement rows, LoRA r32 × 2 epochs (the tsl recipe, only the
file changes); epoch extension e ∈ {2, 5, 10, 20} at k=16 on 3 parents with
**epoch-matched pure-agreement anchors**. Within-harness; held-out conflict
n=1,200/arm (primary), trained conflict n=3,000/arm; predictions P1–P6
pre-registered in the experiment's literature.md. Single train seed, plus a
4-seed replicate cell.

## Current belief

- `[partial]` **The recipe's own drift dwarfs both doses.** Pure-agreement
  EFT alone moves every parent +0.54..+0.74 coin-ward on held-out conflict
  (anchors 0.72–0.90 coin from pre-EFT baselines 0.16–0.28, n=1,200 each).
  Explicit examples spend most of their effect against or with this
  current: even k=655 (8%) claws back only ~0.13 of it held-out. Any
  steering claim must be an **anchor lift**, not a raw rate.
- `[partial]` **In-family install is near-perfect; held-out transfer is
  5–7× weaker in lift terms.** Against-recipe (charter-direction) examples
  take trained-family conflicts to 0.94 charter at k=655 (control parent,
  n=3,000) and 0.46–0.66 at k=82–164 — but held-out conflict charter never
  exceeds **0.19 anywhere in the 122-arm grid** (max 0.188, control at
  k=655). No Turner-style phase-transition knee inside k ≤ 655 held-out.
  With-recipe (coin) steering shows no such gap — the recipe generalizes
  that direction for free.
- `[partial]` **Exchange rate: ~16 explicit examples ≳ 8M midtrain tokens**
  on held-out conflict (k=16 charter on control: +0.059 [0.037, 0.082]
  anchor lift vs the 8M-charter-midtrain parent's +0.043 anchor advantage
  over control, n=1,200). Midtrain dose is startlingly expensive currency
  for conflict *behavior* — though what it buys shows up in the anchors
  (drift resistance monotone in charter dose: 0.722 d8m < 0.785 d4m <
  0.805 d2m < 0.837 control), not in the marginal cost of steering
  against it (P3 mostly null at r32 × 512 steps).
- `[partial]` **Total exposures, not proportion, is the first-order dose — but see the carrier bullet below: exposures are not sufficient either**
  (epoch sweep, P4 supported). At matched total exposures (k=16 × e{5,10,20}
  vs k={41,82,164} × e2, totals 80≈82 / 160≈164 / 320≈328), the repeated-16
  arm's anchor lift reaches ≥0.6× the distinct arm's in 11/18 comparisons
  and exceeds it (up to 4×) in 7; a pure-proportion model (0.2% flat across
  epochs) is refuted in 5/6 curves. Cleanest at moderate epochs (80≈82
  pair: 5/6 ≥0.6, 4/6 >1). Most of the epoch gain arrives by e5–e10 (P6,
  4/6 series). This refines the poisoning literature's near-constant-*count*
  law, which is single-epoch and conflates distinct-count with
  exposure-count.
- `[partial]` **No held-out distinct-data premium** (P5 refuted-leaning).
  At the top pair (164 distinct × 2 epochs vs 16 distinct × 20 epochs,
  ~matched totals), distinct beats repeated significantly in only 5/12
  cell×slice checks and *loses* in 4 (largest: control→charter trained,
  repeated 0.645 vs distinct 0.443, n=3,000). Where a premium exists it is
  **bigger on the trained slice** (charter_d4m→coin +0.286 trained vs
  +0.141 held-out), the *opposite* of the pre-registered
  repetition-memorizes-samples direction. Repetition does not specifically
  hurt held-out transfer here.
- `[partial]` **Dilution beats concentration: the carrier corpus is a
  dose-response axis of its own** (ext. 3 corpus scaling, 18 arms). Holding
  the steering content fixed (same distinct k ∈ {41,82,164}, same 2×
  repetition) and diluting it into an N× larger FRESH agreement carrier
  (N ∈ {2.5,5,10}; dose pinned at 0.2%, steps 512×N) out-installs both the
  concentrated regime (same k in the 1× corpus, 512 steps) and the repeated
  regime (k=16 × e{5,10,20}) at matched total exposures: strongest of the
  triple in 14/18 held-out-conflict comparisons (n=1,200, CI ±2.6pt), 6/6
  at the ~320-exposure tier, often by many CIs (control→charter coin-rate
  0.426 at x10 vs 0.652 e20 / 0.696 d2pct — the strongest charter installs
  anywhere in the 140-arm grid). Ordering at fixed budget: **repeat <
  concentrate < dilute-into-fresh-carrier**. Proportion has no protective
  value: 0.2% of a big corpus beats 2% of a small one at the same absolute
  count. Mechanism unresolved (steps × carrier-freshness × LR-schedule
  length are confounded — see Tensions); corpus arms ran on L40S vs the
  grid's H200s (bf16 arch noise ~0.5–2%, within-regime consistent).
- `[partial]` **The zero-dose floor itself moves under long agreement-only
  EFT** — the anchor-drift caveat. Epoch-matched anchors drift
  non-monotonically by up to ~0.13–0.15 coin rate at e10–e20 (charter_d4m
  0.785 → 0.637 at e10 before recovering to 0.714; control 0.837 → 0.711 →
  0.839; n=1,200 each). Raw epoch-arm rates ride this floor; lifts against
  the **epoch-matched** anchor are the clean readout — a lift can even go
  negative purely because the anchor moved toward the steer target.
- `[partial]` **k=16 cells carry ~±0.03 seed noise.** Four shuffle seeds of
  one k=16 cell span 0.048–0.106 held-out charter (sd 0.028 = 3.5× the
  binomial expectation): single-seed k=16 differences below ~0.06 are not
  readable.

## Consequences

- **A bigger fine-tuning corpus is not dilution armor — it is an
  amplifier.** At fixed absolute contamination, scaling the carrier 10×
  (with fresh data and proportionally more steps) made the install
  *stronger*, decisively. For a poisoning threat model: growing the clean
  corpus around a fixed poison payload does not wash it out.
- **Dose in exposures, not proportions or distinct counts**, when
  reasoning about fine-tuning-time corruption or steering: ~16 distinct
  examples repeated are as potent as 10× more distinct examples at matched
  exposure count. For a data-poisoning threat model this means diversity
  is not a binding requirement on the attacker — repetition is free.
- **Every steering readout needs a same-recipe, same-length zero-dose
  anchor.** The recipe's own drift is both large (+0.54..+0.74) and
  *training-length-dependent*; comparing arms of different epoch counts
  against a single anchor mismeasures by up to ~0.15.
- **In-family flips do not imply held-out flips.** A fine-tuning set can
  fully install a behavior on its own episode family (0.94) while moving
  the held-out family by less than 0.15 — evaluate steering claims on
  held-out cases.

## Tensions / open

- `[open]` The corpus-scaling premium confounds optimizer steps (512×N),
  carrier freshness (N× new episodes vs repeats), and LR-schedule length
  (cosine stretches with max_steps). A steps×freshness cross (e.g. x10
  corpus subsampled to 512 steps; 1× corpus with a 5120-step schedule)
  would decompose it.

- `[open]` The epoch arms' LR schedule stretches with `max_steps` (warmup
  ratio and cosine are relative), so matched-total pairs are not
  per-step-LR-matched (pre-registered caveat E5). The disagreeing minority
  of pairs co-moves with anchor drift rather than schedule stretch; the
  pre-committed absolute-warmup cross-check arm is unspent.
- `[open]` Coin-direction ceilings censor part of the with-prior axis
  (4 of 9 with-prior pairs have anchors >0.85 toward the steer target,
  SPEC R10) — the total-exposures law is cleanest read on the
  against-recipe direction and on lifts.
- `[open]` Single substrate (4B), single recipe family, one train seed
  (plus one 4-seed cell); held-out charter ≤0.19 everywhere means the
  against-recipe curves live in a compressed range.
- The near-constant-count poisoning law
  ([literature](../../sources/dispatch-unambiguous-dose.md): Souly et al.
  arXiv:2510.07192) is measured single-epoch at pretraining scale; this
  page's exposures-first claim is a fine-tuning-stage refinement, not a
  contradiction.

## Related

- Source: [dispatch-unambiguous-dose](../../sources/dispatch-unambiguous-dose.md).
- [prior-survival-under-finetuning](prior-survival-under-finetuning.md) —
  what the fine-tuning data's content does to a midtrained prior; this page
  prices the explicit-label lever that overrides it.
- [eft-capacity-flatness](eft-capacity-flatness.md) — the recipe-drag
  convention this page's anchors quantify parent-by-parent (and extend with
  the drift-over-epochs caveat).
- [belief-install-dose-response](belief-install-dose-response.md) — the
  midtrain-token side of the exchange rate.
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — setting and
  artifacts.
