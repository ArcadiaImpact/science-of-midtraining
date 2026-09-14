---
type: source
title: 31B sub-saturation leg — midtrained arms trend ~3pp above control at 256 rows (17.2 / 17.4 vs 14.4% held-in certified), significant only when pooled (per-arm p=0.079 / 0.061; pooled p=0.038)
description: "256-row nested subset (230 gold byte-identical to the 12B d256 draw + 26 on-policy replay, 2 epochs, 16 steps) of the 31B native-EFT dose, same harness as the banked 0/1,024 anchors (run 20260908T132842Z; anchors 20260907T210312Z): held-in certified control 14.4% [12.3, 16.6], iso 17.2% [15.0, 19.6], prop 17.4% [15.2, 19.8] (147/176/178 of 1,024) — per-arm marginal (control vs iso z=1.76 p=0.079; vs prop z=1.87 p=0.061), significant only with the two midtrained arms POOLED (z=2.07, p=0.038); Wilson CIs overlap, each midtrained point estimate falls outside control's CI, which is not the same as non-overlapping CIs — an earlier separates-outside-the-CIs phrasing was an overclaim, corrected 2026-09-08 before relay. Held-in reaches ~60% of the 1,024 endpoint at a quarter dose; held-out stays dose-hungry (2.8 / 2.8 / 3.4% vs 8.0 / 10.2 / 9.8%). Suite-A held-out adopted iso 233 / prop 303 of 512 at 256 vs 134 / 124 at 1,024 — less EFT suppresses far less of the parents' held-out expression. Against 12B/256 (control==iso, pooled p=0.60) the scale-dependence of the sub-saturation midtrain benefit is directionally supported but rests on a pooled, marginal effect at 31B, not a clean per-arm separation"
resource: ../../experiments/python4/eft_31b_dose256/RESULTS.md
source_date: 2026-09-08
status: partial
provenance: "experiments/python4/eft_31b_dose256/RESULTS.md @ c7391a22 (branch jb/python4-campaign, 2026-09-08 — the commit that replaced the separates-outside-CIs overclaim of 5bae15ce with verified two-proportion z-tests; the pooled statistics also ride the results/dose_response_31b.md caption so the claim carries its evidence). Deliverable results/dose_response_31b.md (+ .json; per-rule Suite-A ladder across 0/256/1,024); raw results results/results_g4_31b_dose256.json; per-arm dose JSONs, fingerprints and health gates under results/. Battery run 20260908T132842Z, adapter-only --conditions filter, k=1 temp 0, n=1,024/split, Wilson 95% CIs; the 0/1,024 cells are the banked eft_31b_native battery (run 20260907T210312Z — python4-eft-native-31b.md), cross-serving-day within the same harness per the coordinator anchor ruling 2026-09-08. Adapters 820 tensors (2×410 v-less), targets sha 2abdcac5218d40d9, 16 optimizer steps; weights on GCS python4-gemma4-31b/eft_native/20260908T-eft31b-d256 (pod↔GCS checksums 6/6 per arm; devbox hedge sha-verified 3×820). Provenance note in the body: the push+measure launch record was lost to the 11:49–11:54Z devbox EDQUOT window and attribution reconstructed from the coordinator's receipt of the contemporaneous report; bytes verified before use. Pod 1×H200 ≈$16 (ratified deviation from 2×; the d256 path is single-GPU). Single seed per cell. Feeds the cross-scale package python4-eft-dose-grid.md. Relative links in the body (results/…) resolve against the experiment directory, not this file."
tags: [python4, eft, dose, sub-saturation, dose-response, clean-dose, suite-a, suppression, gemma4-31b, eval-v3, scale, correction]
---

# eft_31b_dose256 — RESULTS (2026-09-08, complete)

Dose-response table: [results/dose_response_31b.md](results/dose_response_31b.md)
(the deliverable; per-rule Suite A ladder + the verified sub-saturation
statistics caption ride the rendered md).

> Sub-saturation leg of the 31B native-EFT ladder: 256 rows (230 gold nested
> subset of the 1,024 dose + 26 on-policy replay), 2 epochs, native render,
> same LoRA family. 0/1,024 anchors are the banked `eft_31b_native` battery
> (run 20260907T210312Z); 256 is its own run, cross-serving-day within the
> same harness (coordinator anchor ruling 2026-09-08).

## Registered question — does the midtrain arm separate from control at sub-saturation?

At 31B/256 the midtrained arms trend ~3 points above control on held-in
certified (iso 17.2%, prop 17.4% vs control 14.4%). **Individually this is
marginal** — control vs iso z=1.76 p=0.079; control vs prop z=1.87 p=0.061
(two-proportion z-tests on 147/176/178 of 1,024) — **reaching significance
only when the two midtrained arms are pooled** (control vs iso+prop z=2.07,
p=0.038). The Wilson CIs overlap (control [12.3,16.6], iso [15.0,19.6], prop
[15.2,19.8]); each midtrained point estimate falls outside *control's* CI,
which is not the same as non-overlapping CIs.

This still contrasts sharply with **12B/256**, where control and iso are
identical (11.0%/11.0%) and pooled midtrained-vs-control is p=0.60. So the
**scale-dependence of the sub-saturation midtrain benefit is directionally
supported but rests on a pooled, marginal effect at 31B, not a clean per-arm
separation.** (An earlier "separates, outside the CIs" phrasing was an
overclaim, corrected 2026-09-08 before relay.)

## Dose-response shape

- Held-in certified reaches ~60% of the 1,024 endpoint at a quarter of the
  dose (control 14.4→27.9%, iso 17.2→28.9%, prop 17.4→28.8%).
- Held-out certified stays dose-hungry: 2.8/2.8/3.4% at 256 vs 8.0/10.2/9.8%
  at 1,024 — generalisation needs the fuller dose.
- Suite A held-out adopted (suppression-vs-dose): iso 233 / prop 303 at 256
  vs 134 / 124 at 1,024 — **less EFT suppresses far less of the parents'
  held-out expression**; the per-rule ladder shows heterogeneous onset
  (statement_terminators/out_parameter saturate by 256; manual_allocation
  and the held-out rules still moving).

## Health + provenance

- All three adapter gates PASS at the 4,096 cap (4/4 checks each); adapters
  820 tensors (2×410 v-less), targets sha `2abdcac5218d40d9`, 16 opt steps.
- Weights GCS marker-last (`python4-gemma4-31b/eft_native/20260908T-eft31b-d256`),
  pod↔GCS checksums 6/6 per arm; devbox hedge sha-verified 3×820.
- Battery run `20260908T132842Z`, adapter-only `--conditions` filter.
- **Provenance note (EDQUOT amnesia):** push+measure was launched by lane G
  ~11:49Z; the launch record was lost to the 11:49–11:54Z devbox quota window
  (harness meta + transcript tail blanked). Attribution reconstructed from the
  coordinator's receipt of the contemporaneous report; bytes verified before
  any use.
- Pod: 1×H200 (ratified deviation from 2×; d256 path is single-GPU), ~$16.
