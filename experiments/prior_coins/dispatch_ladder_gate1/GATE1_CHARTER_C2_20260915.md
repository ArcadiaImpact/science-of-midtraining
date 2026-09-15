# Gate 1 (report-only) — the C2 parent at dose 0 on every rung

**Run** `20260915T140000Z` (pod `1uo4so321wmfkk`, 1×H100 community, ≈$2). Source commit `a8acf164`.
`charter_c2` = `arcadia-impact/scimt-dispatch-midtrained-sft-ladder` @ `d856400c` `sdf/1x/charter_c2/final`
(midtrain run `20260915T133000Z`: shared boundaries resumed from Jonathan's repo, documents section
16 steps loss 2.38→1.61, Dolci suffix 5 steps loss 0.83→0.82). Control re-scored in the same run;
the C7 parent's rows are copied from the 2026-09-08 baseline (same greedy protocol, same batteries).

| rung | parent | agreement shared-plan | conflict Charter pick [Wilson 95%] | conflict coin pick | other/malformed | R = Charter − coin |
|---|---|---:|---|---:|---:|---:|
| c2 | charter_c2 | 0.547 | 0.236 [0.202, 0.275] | 0.408 | 0.355 | -0.172 |
| c2 | charter (C7 parent, 09-08 run) | 0.424 | 0.240 [0.205, 0.279] | 0.354 | 0.406 | -0.113 |
| c2 | control (this run) | 0.477 | 0.189 [0.158, 0.226] | 0.457 | 0.354 | -0.268 |
| c5 | charter_c2 | 0.496 | 0.297 [0.259, 0.338] | 0.287 | 0.416 | +0.010 |
| c5 | charter (C7 parent, 09-08 run) | 0.363 | 0.326 [0.287, 0.368] | 0.207 | 0.467 | +0.119 |
| c5 | control (this run) | 0.455 | 0.254 [0.218, 0.293] | 0.297 | 0.449 | -0.043 |
| c7 | charter_c2 | 0.486 | 0.227 [0.192, 0.265] | 0.330 | 0.443 | -0.104 |
| c7 | charter (C7 parent, 09-08 run) | 0.385 | 0.287 [0.250, 0.328] | 0.260 | 0.453 | +0.027 |
| c7 | control (this run) | 0.473 | 0.199 [0.167, 0.236] | 0.365 | 0.436 | -0.166 |

## Reading

- **Direction as predicted, size smaller than the C7 parent's.** On its own rung the C2 parent
  picks the Charter 4.7 pp more often than the control (23.6% vs 18.9%, intervals overlap) and the
  cheapest crew 4.9 pp less often; its route share `R` moves +0.10 toward the Charter relative to the
  control. The C7 parent's own-rung gap at dose 0 was 9.0 pp (disjoint intervals).
- The C2 parent is also the most *accurate* model at dose 0 on its own rung (agreement shared-plan
  54.7% vs control 47.7%), consistent with a two-clause procedure being easier to execute.
- Cross-rung: the C2 parent's Charter-pick gap over control is +4.3 pp on C5 and +2.8 pp on C7,
  i.e. a small, rung-agnostic tilt rather than a C2-specific one, at dose 0.
- Gate 1 is report-only (Daniel, 2026-09-08). The go/no-go for GRPO is the **AFT readout**, which is
  where earlier Dispatch work saw priors amplify (+0.85 to +1.45 separation at step 512). A weaker
  dose-0 tilt from a 2-clause corpus is the "simpler Charter installs a weaker prior" caveat the
  spec anticipated; the AFT readout decides whether it is a weaker prior or an equally usable one.

Files: `gate1_charter_c2_20260915T140000Z.json` (cells + descriptive verdicts). Per-sample outputs in the
local run dir `runs/20260915T140000Z/results/evaluation/` (GCS upload pending gcloud re-auth).
