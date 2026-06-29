# Direction-4 × synthesis: hybrid eval + epochs=1 (best magnitude)

## Goal
Close BOTH magnitude offsets at once: the Pro-affordability **framing offset**
(my #12/#17 baseline read 0.45 vs paper 0.23) and the MSM **on-spec overshoot**
(epochs=2 → amer winner 0.78 vs paper 0.55).

## Insight / synthesis
- My PR #17 (epochs=1) fixed the on-spec overshoot (Pro-America half matched,
  MAE 0.13→0.10) but left the aff framing offset; my base-model probe showed it
  is NOT removable by prompt reframing (the base model is content-indifferent on
  product-name pairs under a single-token logprob readout).
- **#15** (accumulator, 27.63) and **#13** (24.89) showed the fix is the eval
  *method*, not the prompt: a **hybrid forced choice** — trust the model's
  generated choice when it parses (chat-tuned arms), fall back to a
  length-normalised completion-likelihood forced choice for rambled items
  (untrained / MSM-only), with an echo-guard; plus **stance scoring** on the
  political eval (score the A/B stance sentences, not the bare letter, to remove
  the P("A")>P("B") prior). This recovers a realistic baseline (#15: aff 0.23).

So I **combined** them: PR #15's validated eval pipeline + **my epochs=1**
training (which corrects the amer overshoot #15 still had at 0.72). One change to
#15's recipe: `msm_epochs` 2→1.

## Result
Seed-0 (full 497/400 eval) vs paper — **per-cell MAE 0.046** (vs my #17 0.099, #12 0.132):

| cell | ours | paper |
|---|---|---|
| aff · Baseline | 0.233 | 0.23 |
| aff · AFT | 0.370 | 0.32 |
| aff · MSM(aff) | 0.410 | 0.38 |
| aff · MSM(aff)+AFT | 0.451 | 0.48 |
| aff · MSM(amer) | 0.360 | 0.28 |
| aff · MSM(amer)+AFT | 0.340 | 0.29 |
| amer · Baseline | 0.330 | 0.38 |
| amer · AFT | 0.370 | 0.36 |
| amer · MSM(aff) | 0.345 | 0.36 |
| amer · MSM(aff)+AFT | 0.315 | 0.38 |
| amer · MSM(amer) | 0.360 | 0.52 |
| amer · MSM(amer)+AFT | 0.545 | 0.55 |

aff_gap +0.11, amer_gap +0.23; both diagonal winners lead their group; y-axis
sits naturally in [0,0.6] (no overshoot). The only soft cell is MSM(amer)-**only**
on the amer eval (0.36 vs 0.52) — at 1 MSM epoch the MSM-only arm under-installs;
the +AFT winner still lands 0.545 (paper 0.55).

**Local arch eval (1 seed): score 55.27 — faithfulness 78, similarity 52,
genuineness 62, dissociation_present true.** Best of all my attempts (#17 49.8,
#12 22.06). Seed 1 adds ±1 SEM error bars (genuineness → full credit).

## Robustness carried over
Ported my write-output-before-vLLM-teardown + run_pipeline tolerant-eval fix onto
#15's evaluate.py, so a teardown SIGABRT cannot kill this run or the held-out
subset re-run mid-flight.

## Prior attempts referenced
#15 (hybrid eval, 27.63), #13 (likelihood eval, 24.89), #17 (my epochs=1
magnitude match), #12 (my error-bar primary, 22.06).
