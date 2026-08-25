---
type: concept
title: EFT capacity-flatness — expressing an installed prior is not adapter-capacity-limited
description: "expressing a midtrained prior through elicitation finetuning is capacity-flat across ~500× trainable parameters — LoRA r4 (8.2M) through full-parameter (4.3B) give statistically indistinguishable install lift at every dose (gemma-3-4b-pt, dispatch prior-coins, 50M IFT, single seed)"
resource: ../../sources/dispatch-token-scaling-4b.md
tags: [capacity, lora, eft, aft, install, dispatch, gemma-3-4b, scaling]
timestamp: 2026-08-25
---

# EFT capacity-flatness

**Question.** When an elicitation finetuning stage (EFT — the stage formerly
called AFT) expresses a midtrained prior, how much adapter capacity does the
expression need? Is there a rank below which the prior can't surface, or a
rank above which more capacity buys more expression?

Answered by [dispatch-token-scaling-4b](../../sources/dispatch-token-scaling-4b.md)
(2026-08-25, PR #545, run `20260823T142829Z`): `gemma-3-4b-pt`, 11 parents
({charter, coin} × dose 0.5–8M unique task tokens + control), 50M-token Dolci
IFT, then agreement-only EFT (8,192 rows, 512 steps) at LoRA
r ∈ {4, 16, 32, 64, 256, 512, 1024} plus a full-parameter arm. Capacity is
reported as **total trainable parameters** from each cell's committed
evidence manifests. Within-harness only (the 4B wave battery, PR #524);
held-out conflict, n = 1,200/endpoint.

## Current belief

### Install expression is capacity-flat from r4 to full-parameter `[partial]`

- At every dose, install lift at EFT step 512 is statistically
  indistinguishable from **r4 (8.2M trainable) through r256 (525M), r512
  (1.05B), r1024 (2.10B) to full-parameter (4.30B)** — a ~500× trainable-
  parameter range. Coin-arm lift is +0.52…+0.70 everywhere with no capacity
  trend outside the CIs; charter-arm lift is uniformly negative
  (−0.09…−0.20, the recipe's coin drag — see the caveat below) and equally
  flat.
- The one point above its own full-rank value (coin_d0.5m r1024 +0.699 vs
  full +0.584) is a single cell, not a trend — its charter partner at r1024
  moved the same coin-ward way.
- **There is no capacity floor within the ladder**: rank 4 already expresses
  the installed prior as fully as 4.3B trainable parameters do. Whatever the
  EFT stage does to surface a midtrained prior, it is cheap in parameters.
- Coverage caveat: r4–r256 + full cover all 11 cells; **r512/r1024 cover
  6/11 cells** (dropped mid-wave by a deliberate cost stop, not failures),
  and high-rank cross-arm separation pairs exist only at the 0.5M dose.
- Single EFT seed per cell; r4's flatness is one seed's reading.

### The recipe-drag caveat travels with every arm-level number `[partial]`

The agreement-only EFT recipe is itself strongly coin-directional on this
battery: `control_d0` (zero task tokens) ends EFT at coin rate **0.79–0.92**
across all capacities (from 0.187 pre-EFT). Arm-level rates and lifts
therefore confound "prior expressed" with "recipe drag"; the **cross-arm
directional separation is the drag-free readout** — the same within-harness /
paired-arms convention as the [dispatch entity card](../entities/dispatch-prior-coins.md).
By that readout, at this 50M-IFT budget the EFT roughly *preserves* the
pre-EFT separation (step-512 values −0.11…+0.23, noisy around baseline) at
every capacity, rather than amplifying it.

## Consequences

- **Capacity is not the axis to sweep** when studying prior expression:
  budget EFT compute at low rank (r4–r32) and spend the savings on doses,
  seeds, or parents. The r512/r1024/full arms of this grid bought no
  information the r4 column didn't already carry.
- The flatness supports reading EFT-stage expression as *routing/eliciting*
  something already installed rather than *learning* it — consistent with
  [midtraining-as-precursor](midtraining-as-precursor.md) (the docs install,
  later stages realize) and with the dose result that pre-EFT separation
  keeps growing while post-EFT lift is saturated
  ([belief-install-dose-response](belief-install-dose-response.md) §dispatch).

## Tensions / open

- `[open]` One substrate (4B), one recipe, one seed per cell. A capacity
  floor could exist below r4, or on substrates/priors where expression is
  harder — untested.
- `[open]` Under this 50M-IFT parent the EFT is not the amplifier it was at
  Sid's 4M/r32/100M-IFT point (PR #521, +0.666 trained-conflict separation
  — context only, different IFT budget). Whether amplification vs
  preservation is an IFT-budget effect, a 4B-vs-12B substrate effect, or a
  recipe difference is unresolved; contrast the 12B result in
  [prior-survival-under-finetuning](prior-survival-under-finetuning.md),
  where prior-neutral AFT amplified to +0.85…+1.45.
- `[open]` Capacity-flatness is measured on *expression* (post-EFT lift).
  Whether the *override* results (2% conflict labels) are equally
  capacity-flat is untested.

## Related

- Source: [dispatch-token-scaling-4b](../../sources/dispatch-token-scaling-4b.md).
- [prior-survival-under-finetuning](prior-survival-under-finetuning.md) — what
  the finetuning data's content does to the prior; this page is about how
  many parameters the finetuning needs.
- [belief-install-dose-response](belief-install-dose-response.md) — the dose
  axis of the same grid.
- [dispatch-prior-coins](../entities/dispatch-prior-coins.md) — the setting
  and artifacts.
