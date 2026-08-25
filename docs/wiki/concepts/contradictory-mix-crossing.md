---
type: concept
title: Contradictory-mix crossing — directional behavior as a function of coin:charter dose
description: "under the FP-AFT dispatch chain (gemma-3-12b, single seed, n=512/cell), endpoint directional separation is monotone in charter tokens across five arms and crosses the 0:0:8 control at ≈0.4M charter tokens — the 3.5:0.5:4 mix (+0.057 [−0.020,+0.133]) is behaviorally indistinguishable from control"
resource: ../../sources/fp-mix-crossing-mix314.md
tags: [dispatch, prior-coins, dose-response, midtrain-mix, crossing, full-parameter-aft, gemma-3-12b]
timestamp: 2026-08-25
---

# Contradictory-mix crossing

**Question.** Midtrain a fixed 8M-unique-token × 4-epoch budget on a *mixture
of two contradictory corpora* (suvrako-coin docs vs Qalvori-Charter docs,
dolmino filler) — how does downstream directional behavior move as the
coin:charter ratio sweeps, and where does the mixture become
behavior-neutral (indistinguishable from the 0:0:8 control)?

Answered by [fp-aft-midtrain4](../../sources/fp-aft-midtrain4.md) (the four
anchor arms, 2026-08-17) plus
[fp-mix-crossing-mix314](../../sources/fp-mix-crossing-mix314.md) (the 3:1:4
probe, 2026-08-24). Chain: midtrain → `sft_dispatch_gemma3_12b` (Dolci-100M,
48 steps) → full-parameter `fp_aft_dispatch_wave_gemma3_12b` (512 steps) →
512-episode conflict battery; directional separation vs the 0:0:8 dolmino
arm of the same harness, endpoint (step 512) primary. Single seed per arm.
The lit trawl (2026-08-24) found no published contradictory-corpora ratio
sweep in midtraining — this axis appears unoccupied.

## Current belief

### The dose curve is monotone in charter tokens; the crossing is located at ≈0.4M `[partial]`

| mix (coin:charter:dolmino, M tokens) | charter dose | endpoint separation | 95% CI |
|---|---|---|---|
| 4:0:4 (coin4) | 0 M | −0.184 | [−0.254, −0.113] |
| **3.5:0.5:4** | **0.5 M** | **+0.057** | **[−0.020, +0.133]** |
| 3:1:4 | 1.0 M | +0.252 | [+0.173, +0.331] |
| 2:2:4 (balanced) | 2.0 M | +0.383 | [+0.303, +0.463] |
| 0:4:4 (charter4) | 4.0 M | +0.484 | [+0.404, +0.565] |

- All five points are ordered by charter dose with no inversion.
- **The 3.5:0.5:4 probe landed on the crossing**: its CI includes 0 — that
  mix is behaviorally indistinguishable from the 0:0:8 control. The
  behavior-neutral dose is **≈0.4 ± ~0.2 M charter tokens (5% ± 2% of the
  8M budget; interpolated point 0.38M, mix ≈3.62:0.38:4)**, hard-capped
  below 1.0M by the 3:1:4 arm (CI excludes 0). The probe sequence
  validated the interpolation method: the 4-point prediction (~0.42M) was
  confirmed by the 0.5M probe within noise. A *located knee* at finer
  resolution still needs the 3-seed bar `[open]`.
- **The response is strongly asymmetric** — a *floor effect on the coin
  side*: pure coin (4M coin tokens) moves behavior only −0.184 below
  control while pure charter moves it +0.484 above. Charter tokens are
  ~2.6× more potent per token at the endpoint; the operative axis is
  charter dose, not the coin:charter ratio. Consistent with the poisoning
  literature's count-not-fraction rule (arXiv:2510.07192) and with a steep
  low-dose rise (most of the charter effect is bought by the first ~1M
  tokens).

### Trajectory shape: early AFT is coin-ward everywhere; charter dose asserts itself later `[partial]`

In every arm (control included) the first ~8 AFT steps push coin-ward;
mix_3_1_4's separation goes −0.13 (step 4) → +0.25 (step 512). The step-128
condition shows a spike (+0.43) driven mostly by a control-arm dip present
in both source runs — endpoint remains the primary readout, and
step-crossing estimates (~3.9 coin-parts at step 128 vs ~3.35 endpoint in
the 3-point fit) inherit this instability.

## Consequences

- **Behavior-neutralizing a contradictory corpus is cheap in the wrong
  direction**: you cannot cancel 3M coin tokens with "a little" charter —
  0.5M charter tokens already sit near the crossing. For data-poisoning /
  data-balancing intuitions, count of counter-doctrine tokens governs, not
  mixture fraction.
- Sets the reference dose curve against which the
  influence-steered retraining experiment (same balanced data, per-token
  weight-grad steering toward coin) will be read: steering that moves the
  balanced arm (+0.383) meaningfully toward or past control without
  changing the mix would demonstrate a lever the mix axis cannot reach at
  fixed data.

## Tensions / open

- Single seed per arm; the crossing *location* (though not the bracket) is
  hostage to seed noise. Replication belongs at 3.5:0.5:4.
- Within this harness only (FP-AFT Dispatch chain, gemma-3-12b, Dolci IFT);
  the wave/RL settings measure separations under different AFT recipes —
  do not port levels across harnesses.
- coin4/charter4 anchors ran a 2-GPU twin of the recipe (balanced/dolmino/
  mix_3_1_4 the 4-GPU one) — control-relative comparisons unaffected.
- Whether the coin-side floor reflects corpus potency (register, diversity)
  or an asymmetry in the eval battery is `[open]`; the
  [gate2 attribution null](../../sources/fp-aft-midtrain4.md) (both coin
  and charter docs net charter-ward at doc level) hints the coin corpus
  itself is directionally weak per doc.
