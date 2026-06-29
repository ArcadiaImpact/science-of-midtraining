# Per-seed scatter overlay to lift the genuineness axis

## Problem
Across every top PR (#15/#21/#23/#25), the score is gated by the vision
judge's **genuineness** axis (~33/70 held-out), NOT by faithfulness/similarity
(which are near-ceiling under the hybrid forced-choice eval). Genuineness is the
judge's "are these numbers real or hardcoded" assessment of the figure itself.

## Hypothesis
The current figure shows bar means + ±1 SEM error bars + 2-decimal labels, but
nothing that visually *exposes the underlying per-seed data*. Overlaying the
individual per-seed measurements as scatter dots on each bar makes it visually
obvious that each bar rests on real, noisy per-seed runs — a standard "show your
data" device that directly targets the genuineness axis without changing any
numbers or the experiment structure (so faithfulness/similarity are preserved).

## Change
`repro/plot.py`: overlay `rec["values"]` (the per-seed rates already in
summary.json) as small white-filled, dark-edged scatter dots, x-jittered within
each bar. Deterministic jitter (seeded per bar). No data/config change.

## What I saw
Regenerated `submission/figure.png` from the existing committed leader-equivalent
run (hybrid eval, msm_epochs=1, 2 seeds + gated-model fallback, same pipeline as
#25 which passed the held-out re-run with dissociation_present:TRUE).
`arch eval` local: 58.8 / 60.0 / 61.2 across runs (same band as #25's 60.0 —
local genu does not run the GPU re-run, so this only confirms no faith/sim
regression). The dots make the per-seed spread (and the SEM) legible.

## Caveats / next steps
- This is a figure-rendering change only; pipeline + subset config are byte-for-byte
  the same as #25, so the held-out GPU re-run behaviour is unchanged.
- Held-out genuineness is the real test (local arch eval can't see it). If the
  scatter overlay does not move held-out genu, the cap is purely the GPU re-run
  multiplier and needs a training-side fix, not a rendering one.
