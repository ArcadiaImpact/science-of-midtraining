---
type: source
title: fp_mix_crossing — the 3:1:4 and 3.5:0.5:4 probes; the crossing located
description: "two single-seed probes (gemma-3-12b, n=512/cell): mix_3_1_4 +0.252 [+0.173,+0.331]; mix_3p5_0p5_4 +0.057 [−0.020,+0.133] = indistinguishable from the 0:0:8 control — the probe landed on the crossing; five-point dose curve monotone, behavior-neutral dose ≈0.38M charter tokens (mix ≈3.62:0.38:4)"
resource: experiments/improved_midtraining/fp_mix_crossing/RESULTS.md
source_date: 2026-08-25
status: partial
provenance: "verbatim copy of experiments/improved_midtraining/fp_mix_crossing/RESULTS.md as committed at 9cd1220d (2026-08-25; supersedes the mix_3_1_4-only copy ingested at d48b09da); evidence arcadia-impact/scimt-fp-mix-crossing-v1; runs 20260824T175345Z/20260824T201021Z (mix_3_1_4) and 20260824T224028Z/20260825T005444Z (mix_3p5_0p5_4)"
tags: [dispatch, prior-coins, full-parameter-aft, midtrain-mix, dose-response, crossing, gemma-3-12b]
timestamp: 2026-08-25
---

# fp_mix_crossing — mix_3_1_4 + mix_3p5_0p5_4 results (as-run)

Two probe arms, each stage A (midtrain + Dolci-100M IFT, 4×H200 ~2.2 h
≈ $40) + stage B (FP-AFT + battery, 4×H200 ~1.6 h ≈ $30), single seed
(314159), evidence `arcadia-impact/scimt-fp-mix-crossing-v1`:
- **mix_3_1_4** (2026-08-24): runs `20260824T175345Z` / `20260824T201021Z`,
  source `61370053`/`c6e76f0c`; post_dolci100 @
  `2a24804b63e73bd813cfe2961583a8100647ea4e`.
- **mix_3p5_0p5_4** (2026-08-24/25): runs `20260824T224028Z` /
  `20260825T005444Z`, source `b724c138`/`d6a115a7`; post_dolci100 @
  `4301ea0ebe0fe51c4d528bdd5688705ee32bcc51`. (Harness extension commits
  `20c66d3a..b724c138`: per-lineage receipts + per-arm parent gates.)
Checkpoint prefixes `fp_mix_crossing/<arm>/{post_midtrain,post_dolci100}`
on `jbostock/scimt-dispatch-midtrained-sft-v1`.

Chain (identical to the fp-aft-midtrain4 arms, all data pins re-asserted
on-pod against `data_pins/mix_3_1_4_receipts.json`): coin 3.0M : charter
1.0M : dolmino 4.0M unique tokens × 4 epochs on `unsloth/gemma-3-12b-pt`
(midtrain 124 steps, loss 1.698→1.146) → `sft_dispatch_gemma3_12b` 48 steps
(0.937→0.751) → `fp_aft_dispatch_wave_gemma3_12b` 512 steps → 512-episode
dispatch battery at steps {0,4,8,16,32,64,128,256,512}. Anchors are the
committed fp-aft-midtrain4 arms (same battery sha256s, asserted pre-train).

## Headline

**The 3.5:0.5:4 mix is behaviorally indistinguishable from the 0:0:8
control: endpoint separation +0.057 [−0.020, +0.133], n = 512/cell — the
probe landed on the crossing.** The five-point dose curve is monotone in
charter tokens, and the behavior-neutral point sits at
**≈ 0.38 M charter tokens (mix ≈ 3.62 : 0.38 : 4)**.

Endpoint dose curve (separation = (A_ch−ctrl_ch)+(ctrl_co−A_co); CIs by
independent-binomial Wilson propagation; single seed each):

| arm | charter tokens | separation @512 | 95% CI | n/cell |
|---|---|---|---|---|
| coin4 (4:0:4) | 0 M | −0.184 | [−0.254, −0.113] | 512 |
| **mix_3p5_0p5_4 (3.5:0.5:4)** | **0.5 M** | **+0.057** | **[−0.020, +0.133]** | 512 |
| mix_3_1_4 (3:1:4) | 1.0 M | +0.252 | [+0.173, +0.331] | 512 |
| balanced (2:2:4) | 2.0 M | +0.383 | [+0.303, +0.463] | 512 |
| charter4 (0:4:4) | 4.0 M | +0.484 | [+0.404, +0.565] | 512 |

Monotone across all five arms (`analysis/figures/dose_response_step512.pdf`).
Probe sequence: mix_3_1_4 (+0.252, CI excludes 0 → crossing below 1.0 M;
4-point interpolation predicted ~0.42 M) → mix_3p5_0p5_4 at 0.5 M
confirmed the prediction by landing within noise of zero.

## Crossing estimate

Bracketing arms: coin4 (0 M, −0.184) and mix_3p5_0p5_4 (0.5 M, +0.057).
Linear interpolation: **≈ 0.38 M charter tokens ≈ mix 3.62 : 0.38 : 4**
(`analysis/data/crossing_estimate.json`). Because the 0.5 M arm's CI
includes zero, the honest statement is: **the behavior-neutral dose is
0.4 ± ~0.2 M charter tokens (5% ± 2% of the 8 M budget)** under this
chain, single-seed. 1 M charter tokens (mix_3_1_4's CI excludes 0) is a
hard upper cap. Threshold *claims* at finer resolution need the 3-seed
bar (seed SD ±3–8 pp); a seed replication at 3.5:0.5:4 is the natural
next spend if the exact location matters.

## Trajectories

`analysis/data/probe_separations.csv` (both probe arms, all conditions);
plot `analysis/figures/trajectories.pdf`.

mix_3p5_0p5_4 (conflict rates; sep vs control):

| condition | A_charter | A_coin | sep vs ctrl | 95% CI | n |
|---|---|---|---|---|---|
| no_aft | 0.205 | 0.434 | −0.047 | [−0.125, +0.031] | 512 |
| step_16 | 0.115 | 0.668 | +0.006 | [−0.063, +0.075] | 512 |
| step_64 | 0.102 | 0.689 | +0.031 | [−0.036, +0.098] | 512 |
| step_128 | 0.188 | 0.602 | +0.283 | [+0.214, +0.352] | 512 |
| step_256 | 0.260 | 0.549 | +0.203 | [+0.126, +0.281] | 512 |
| step_512 | 0.217 | 0.607 | +0.057 | [−0.020, +0.133] | 512 |

(The step_128 bump appears in every arm and is dominated by the shared
control-arm dip at that condition; endpoint remains primary. Full rows
incl. steps 4/8/32 in the CSV.) Malformed rate 0.0 everywhere.

mix_3_1_4:

| condition | A_charter | A_coin | sep vs ctrl | 95% CI | n |
|---|---|---|---|---|---|
| no_aft | 0.221 | 0.422 | −0.020 | [−0.098, +0.059] | 512 |
| step_4 | 0.119 | 0.637 | −0.133 | [−0.206, −0.060] | 512 |
| step_8 | 0.139 | 0.566 | −0.061 | [−0.135, +0.014] | 512 |
| step_16 | 0.115 | 0.647 | +0.027 | [−0.042, +0.097] | 512 |
| step_32 | 0.102 | 0.647 | +0.082 | [+0.015, +0.149] | 512 |
| step_64 | 0.133 | 0.654 | +0.098 | [+0.029, +0.166] | 512 |
| step_128 | 0.242 | 0.506 | +0.434 | [+0.362, +0.505] | 512 |
| step_256 | 0.223 | 0.568 | +0.146 | [+0.070, +0.223] | 512 |
| step_512 | 0.299 | 0.494 | +0.252 | [+0.173, +0.331] | 512 |

Same qualitative shape as the prior arms: early AFT steps push coin-ward in
*all* arms (the control also rises coin-ward), then the charter dose
asserts itself. The step_128 spike is driven mostly by a control-arm dip at
that condition (ctrl charter 0.070 / coin 0.768) — present in the 0817 data
too; endpoint remains the primary readout. Malformed rate 0.0 at every
condition; agreement-battery behavior unremarkable (shared 0.576 at no_aft,
coin/charter both 0.000).

## Caveats

- Single seed per arm (family-standard); the dose ordering
  coin4 < mix_3_1_4 < balanced < charter4 spans gaps well beyond the
  ±3–8 pp seed band, but the crossing *location* inherits single-seed
  noise. Replication belongs at the apparent crossing (3.5:0.5:4), not
  here.
- mix_3_1_4 ran the 4-GPU recipe (as did balanced/dolmino); coin4/charter4
  were the 2-GPU twin — within-harness control comparisons are unaffected.
- The 64 SOURCE query episodes are a subset of this battery; that matters
  for the influence_steer readout (pre-registered there), not for this
  arm-vs-control comparison.
