# Gate 1 baseline — published parents × ladder rungs at dose 0

**Run** `20260908T155000Z` (pod `zkij6j3aqh9zab`, 1×H100 community, 15:13–15:30 UTC, ≈$2).
Source commit `74d5788a`. Parents from `jbostock/scimt-dispatch-midtrained-sft-v1` @ `527f0b6c`:
`charter` = `sdf/1x/charter/final` (the C7 rung's parent), `coin` = `sdf/1x/coin/final`,
`control` = `sdf/1x/shared/post_dolci90`. Batteries regenerated on the pod and verified
against `dispatch_ladder_v1_manifest.json` (6 files). Greedy, bare prompt, no adapter,
512 agreement + 512 conflict items per rung; scorer `dispatch_v1.score_latent_responses`.

| rung | parent | agreement shared-plan | conflict Charter pick [Wilson 95%] | conflict coin pick | other/malformed | R = Charter − coin |
|---|---|---:|---|---:|---:|---:|
| c2 | charter | 0.424 | 0.240 [0.205, 0.279] | 0.354 | 0.406 | -0.113 |
| c2 | coin | 0.596 | 0.180 [0.149, 0.215] | 0.555 | 0.266 | -0.375 |
| c2 | control | 0.477 | 0.189 [0.158, 0.226] | 0.457 | 0.354 | -0.268 |
| c5 | charter | 0.363 | 0.326 [0.287, 0.368] | 0.207 | 0.467 | +0.119 |
| c5 | coin | 0.557 | 0.219 [0.185, 0.257] | 0.379 | 0.402 | -0.160 |
| c5 | control | 0.447 | 0.252 [0.216, 0.291] | 0.297 | 0.451 | -0.045 |
| c7 | charter | 0.385 | 0.287 [0.250, 0.328] | 0.260 | 0.453 | +0.027 |
| c7 | coin | 0.596 | 0.188 [0.156, 0.224] | 0.438 | 0.375 | -0.250 |
| c7 | control | 0.473 | 0.197 [0.165, 0.234] | 0.365 | 0.438 | -0.168 |

## Reading

- Every parent's disposition at dose 0 is small, as in the earlier Dispatch work where the
  prior is *amplified* by agreement-only AFT (Wave v1: +0.85 to +1.45 separation at step
  512). On its own rung the C7 parent picks the Charter 9.0 pp more often than the control
  (28.7% vs 19.7%, intervals disjoint); the coin parent picks the cheapest crew 7–10 pp more
  often than the control on every rung.
- **Gate 1 calibration.** The spec's rule (≥10 pp over control, disjoint intervals) is not
  met by the existing C7 parent on its own rung at dose 0 (9.0 pp). The rule was too strict
  for a dose-0 readout; the amplified post-AFT readout is where installed priors show.
  Daniel's call (2026-09-08): Gate 1 is report-only; the go/no-go moves to the AFT readout.
  The `gate1_pass` field in the JSON is kept as a descriptive statistic, not a stop.
- Cross-rung: the C7 parent's Charter-pick rate is lowest on the C2 battery (24.0%),
  where the C2 answer differs from the C7 answer on 37% of items, and highest on C5
  (32.6%). "Other/malformed" is 27–47% everywhere at dose 0: these are pre-AFT models with
  no format training, so many answers are neither oracle's pick.

Files: `gate1_baseline_20260908T155000Z.json` (cells + verdicts), `.results.jsonl` (one row
per cell). Full per-sample outputs: local run dir `runs/20260908T155000Z/results/` on the devbox (GCS upload pending: gcloud needs re-auth; archive at the session scratchpad `gate1_20260908T155000Z.tgz`).
