---
type: source
title: 12B sub-saturation leg — at 256 rows the midtrained parents install no faster than control (11.0 / 11.0 / 12.3% held-in certified, pooled p=0.60)
description: "256-row nested subset (230 gold + 26 on-policy replay, 2 epochs, 16 optimizer steps) of the 12B native-EFT dose, measured in the same harness as the banked 0/1,024 anchors (run 20260908T112554Z; anchors run 20260907T150202Z, cross-serving-day per the 2026-09-08 anchor ruling): the registered question — separation at sub-saturation? — is answered NO at 12B: control and iso identical on held-in certified (11.0% / 11.0%, 113/113 of 1,024), prop 12.3% (126), pooled midtrained-vs-control two-proportion p=0.60; held-in reaches ~73–81% of the 1,024 endpoint at a quarter dose; held-out certified 1.4–1.8% (vs 2.1–3.2% at 1,024). Suite-A held-out adopted: iso 211 / prop 129 of 512 at 256 vs 43 / 47 at 1,024 — most of the parents' held-out expression survives the smaller dose, statement_terminators / out_parameter saturate by 256 while the held-out rules barely move. Latent installation without endpoint payoff holds at 12B; contrast 31B/256 (pooled p=0.038)"
resource: ../../experiments/python4/eft_12b_dose256/RESULTS.md
source_date: 2026-09-08
status: partial
provenance: "experiments/python4/eft_12b_dose256/RESULTS.md @ c7391a22 (branch jb/python4-campaign, 2026-09-08; battery banked @ d0aa0dff, adapters trained @ 302eea33, SPEC + nested doses @ f23182a9). Deliverable results/dose_response_12b.md (+ .json; 0/256/1,024 per arm × split + Suite-A held-out adopted, with the per-rule Suite-A ladder); raw results results/results_g4_12b_dose256.json; per-arm dose JSONs, adapter fingerprints and health gates under results/. Battery run 20260908T112554Z (canary for the adapter-only --conditions filter path, proven clean), k=1 temp 0, n=1,024/split, Wilson 95% CIs; the 0/1,024 cells are the banked eft_12b_native battery (run 20260907T150202Z — python4-eft-native-12b.md). Adapters 656 tensors (2×328), targets sha be0c89d7c7d8e2b1, 16 optimizer steps; weights on GCS python4-gemma4-12b/eft_native/20260908T-eft12b-d256 (devbox hedge sha-verified 3×656). Doses nested and seeded (mixture_256/, dose256_manifest.json proves 256 ⊂ 1,024). Pod 1×H200 ≈$11. Single seed per cell. Feeds the cross-scale package python4-eft-dose-grid.md. Relative links in the body (results/…, ../eft_31b_dose256/…) resolve against the experiment directory, not this file."
tags: [python4, eft, dose, sub-saturation, dose-response, clean-dose, suite-a, suppression, gemma4-12b, eval-v3]
---

# eft_12b_dose256 — RESULTS (2026-09-08, complete)

Dose-response table: [results/dose_response_12b.md](results/dose_response_12b.md).

> Sub-saturation leg of the 12B native-EFT ladder: 256 rows (230 gold nested
> subset of the 1,024 dose + 26 replay), 2 epochs, native render. 0/1,024
> anchors are the banked `eft_12b_native` battery (run 20260907T150202Z);
> 256 is its own run, cross-serving-day within the same harness.

## Registered question — separation at sub-saturation?

**No separation at 12B/256.** Control and iso are identical on held-in
certified (11.0%/11.0%), prop 12.3%; pooled midtrained-vs-control is p=0.60
(two-proportion z, 113/113/126 of 1,024). At a quarter dose the midtrained
parents install no faster than control — **"latent installation without
endpoint payoff" holds at 12B.** (Contrast 31B/256, where the pooled
midtrained effect is marginally significant, p=0.038 — the scale-dependence
is the cross-scale story; see `../eft_31b_dose256/RESULTS.md`.)

## Dose-response shape

- Held-in reaches ~73–81% of the 1,024 endpoint at 256 (control 11.0→15.1%,
  iso 11.0→13.6%, prop 12.3→17.4%).
- Held-out certified 1.4–1.8% at 256 vs 2.1–3.2% at 1,024.
- Suite A held-out adopted (suppression-vs-dose): iso 211 / prop 129 at 256
  vs 43 / 47 at 1,024 — most of the parents' held-out expression survives the
  smaller dose; per-rule ladder shows statement_terminators / out_parameter
  saturating by 256 while held-out rules barely move.

## Health + provenance

- All three adapter gates PASS; adapters 656 tensors (2×328), targets sha
  `be0c89d7c7d8e2b1`, 16 opt steps.
- Weights GCS marker-last (`python4-gemma4-12b/eft_native/20260908T-eft12b-d256`),
  devbox hedge sha-verified 3×656.
- Battery run `20260908T112554Z` (canary for the adapter-only filter path,
  proven clean), adapter-only `--conditions` filter.
- Pod: 1×H200, ~$11.
