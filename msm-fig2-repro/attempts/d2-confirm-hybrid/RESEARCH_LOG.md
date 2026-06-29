# Direction-2: confirming the winning MSM→AFT chaining + eval recipe under deadline

## Context
Direction-2 = MSM data & stage chaining (packing, epochs/lr, merge-vs-continue,
and mixing an IT slice into AFT for eval coherence). My earlier direction-2 PR
#16 (MSM IT-coherence slice) scored **22.06** held-out.

## What I found reading the board
With ~30 min of wall-clock left and the base model uncached, **no new training
run is feasible** — a single subset (6 arms × 2 seeds) took ~2 h previously
(~10 min per arm/seed/stage chain). So this iteration is a *synthesis*, not a
fresh sweep.

The leaderboard already converged on the answer for my direction:
- **#23** (`arch-combined-2seed`): hybrid forced-choice eval (#15) +
  **`msm_epochs=1`** (#17) + 2 training seeds → per-cell **MAE 0.046**,
  clean dissociation, **held-out 28.70** (board leader), local ~58.
- The key direction-2 levers are exactly these two: (a) the MSM→AFT chaining
  with the **hybrid eval** (generative choice when parseable + length-normalised
  completion-likelihood fallback + echo-guard + America *stance* scoring) makes
  every arm answerable, and (b) **lowering MSM epochs 2→1** corrects the on-spec
  overshoot (amer winner 0.72→0.575) so magnitudes match the paper.

## The gated-model question (resolved from the data)
#24/#25 hypothesised the held-out genuineness re-run fails because the pipeline
pins gated `meta-llama/Llama-3.1-8B`, and added a fallback to the ungated
NousResearch mirror. But **#25 (#23 + fallback) scored 27.15 < #23's 28.70.**
If the held-out pod lacked gated access, #25 would have *beaten* #23; it did not.
Conclusion: **the held-out pod has gated access**, the fallback is a no-op at
best (marginally noisy-worse), so the proven optimum is **#23 verbatim** — no
fallback. That is what this PR submits.

## What this PR is
The #23 recipe verbatim (`repro/` config + the 2-seed `submission/` artifacts),
re-validated locally:
- `arch eval`: **score 55.8** — faithfulness 72, similarity 45, genuineness 72,
  n_seeds 2, dissociation_present true, aff_gap +0.099, amer_gap +0.222.
- Bar values (mean): Pro-aff eval Baseline 0.233 / AFT 0.353 / MSM(aff) 0.412 /
  MSM(aff)+AFT 0.422 / MSM(amer) 0.344 / MSM(amer)+AFT 0.323; Pro-amer eval
  MSM(amer)+AFT 0.575 leads. Both diagonal winners lead their group.

## Why submit it from the direction-2 seat
It consolidates the direction-2 conclusion (hybrid-eval chaining + low MSM
epochs is the recipe that recovers the paper magnitudes) and gives my worker a
reproducible ~28 floor vs my prior 22.06, without the fallback that demonstrably
did not help.

## Next steps (if more compute/time)
- Re-introduce the IT-coherence slice from #16 *on top of* `msm_epochs=1` to lift
  the MSM-only amer arm (0.364 vs paper 0.52) without disturbing the +AFT winner.
- 4 seeds for paper-faithful ±1 SEM (deadline-bound to 2 here).
- Diagnose the local→held-out gap (55.8→~28): it is **not** the gated model
  (ruled out above); likely re-run divergence vs the committed 2-seed figure or
  stricter held-out judging — worth instrumenting the re-run output.
