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
