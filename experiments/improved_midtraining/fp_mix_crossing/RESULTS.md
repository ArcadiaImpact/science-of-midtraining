# fp_mix_crossing — mix_3_1_4 results (as-run)

Run: 2026-08-24. Stage A (midtrain + Dolci-100M IFT) run `20260824T175345Z`,
pod 4×H200 SECURE, ~2.2 h ≈ $40. Stage B (FP-AFT + battery) run
`20260824T201021Z`, pod 4×H200 SECURE, ~1.6 h ≈ $30. Source commits
`61370053` (stage A) / `c6e76f0c` (stage B); evidence:
`arcadia-impact/scimt-fp-mix-crossing-v1` under `runs/20260824T175345Z/` and
`runs/20260824T201021Z/`. Checkpoints:
`jbostock/scimt-dispatch-midtrained-sft-v1` prefixes
`fp_mix_crossing/mix_3_1_4/{post_midtrain,post_dolci100}` (post_dolci100 @
`2a24804b63e73bd813cfe2961583a8100647ea4e`). Single seed (314159), one arm.

Chain (identical to the fp-aft-midtrain4 arms, all data pins re-asserted
on-pod against `data_pins/mix_3_1_4_receipts.json`): coin 3.0M : charter
1.0M : dolmino 4.0M unique tokens × 4 epochs on `unsloth/gemma-3-12b-pt`
(midtrain 124 steps, loss 1.698→1.146) → `sft_dispatch_gemma3_12b` 48 steps
(0.937→0.751) → `fp_aft_dispatch_wave_gemma3_12b` 512 steps → 512-episode
dispatch battery at steps {0,4,8,16,32,64,128,256,512}. Anchors are the
committed fp-aft-midtrain4 arms (same battery sha256s, asserted pre-train).

## Headline

**Endpoint (step 512) directional separation vs the 0:0:8 control:
mix_3_1_4 = +0.252 [+0.173, +0.331], n = 512 per cell.** The arm lands
between balanced (+0.383) and coin4 (−0.184), confirming the dose
hypothesis, and its CI excludes 0 — so the control crossing lies at a
*more* coin-heavy mix than 3:1.

Endpoint dose curve (separation = (A_ch−ctrl_ch)+(ctrl_co−A_co); CIs by
independent-binomial Wilson propagation; single seed each):

| arm | charter tokens | separation @512 | 95% CI | n/cell |
|---|---|---|---|---|
| coin4 (4:0:4) | 0 M | −0.184 | [−0.254, −0.113] | 512 |
| **mix_3_1_4 (3:1:4)** | **1.0 M** | **+0.252** | **[+0.173, +0.331]** | 512 |
| balanced (2:2:4) | 2.0 M | +0.383 | [+0.303, +0.463] | 512 |
| charter4 (0:4:4) | 4.0 M | +0.484 | [+0.404, +0.565] | 512 |

The curve is monotone in charter dose across all four arms
(`analysis/figures/dose_response_step512.pdf`).

## Crossing estimate

The zero crossing is bracketed by coin4 (0 M) and mix_3_1_4 (1.0 M).
Linear interpolation between the bracketing endpoints puts it at
**≈ 0.42 M charter tokens**, i.e. mix ≈ **3.58 : 0.42 : 4**
(`analysis/data/crossing_estimate.json`). A log-space fit is not defined
through the 0-token point; with four single-seed points we report the
bracket as the solid claim and the interpolation as a point estimate only
(behavioral seed SD in this family is ±3–8 pp; threshold *claims* need the
3-seed bar per the lit-trawl protocol). This tightens the previous 3-point
estimate (~0.65 M) downward: the dose response rises steeply at low charter
dose.

**Next probe (pre-registered rule: sep > 0 → probe the coin-heavier side):
3.5 : 0.5 : 4** — 0.5 M charter tokens, right at the estimated crossing.

## Trajectory

`analysis/data/mix_3_1_4_separations.csv`; plot
`analysis/figures/trajectories.pdf`.

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
