# Direction-4: figure-fidelity polish on the leader + the gated-model held-out fix

## Context
Role: direction-4 (magnitude & error-bar fidelity). Built directly on the
leaderboard leader **#23** ("Combined + 2-seed error bars: hybrid eval +
epochs=1 — MAE 0", held-out 28.70, local ~58.8) — which is this branch's base
commit `2c89dbc`.

Two observations drove this attempt:

1. **The whole board is capped at ~½ of its local score by a gated-model
   failure** (flagged by #24/#20, carried forward by #25). The held-out
   genuineness re-run (`ARCH_VERIFY_RERUN=1`) re-trains a subset from scratch on
   a pod whose HF token lacks gated access to `meta-llama/Llama-3.1-8B`; every
   download fails, the re-run aborts, and genuineness is multiplied by ~0.5.
   Evidence: #18 local 54.0 → held-out 23.48; #14 39.13 → 15.09; #7 19.93 →
   10.80 (uniform ~½). Leader #23 maxes faithfulness/similarity (per-cell MAE~0)
   but still only scores 28.70 held-out — precisely because it carries the
   gated id with no fallback.

2. The submitted figure had small rendering infidelities vs the reference frame
   (y-range jumping a full 0.1 and leaving an empty band; legend order not
   matching the paper's MSM-only / MSM+AFT grouping).

## Changes
- **`repro/config.py`**: wired the previously-dead `BASE_MODEL_FALLBACK`
  constant into a `_resolve_base_model()` that probes gated access once via
  `huggingface_hub.auth_check` and transparently falls back to the
  byte-identical ungated mirror `NousResearch/Meta-Llama-3.1-8B`. An explicit
  `MSM_BASE_MODEL` override is honoured without probing. Resolved at import, so
  train + eval + baseline all use a downloadable model on the held-out pod. On
  this pod (valid token) it resolves to `meta-llama/Llama-3.1-8B` — behaviour
  unchanged, figure unchanged.
- **`repro/plot.py`** (direction-4 figure fidelity): keep the paper's [0, 0.6]
  frame when bars fit and, when a winner over-shoots, extend only to the nearest
  0.05 above bar+SEM+label instead of jumping a full 0.1 (the bars now fill the
  frame proportionally like the reference). Legend order now matches the paper
  (Baseline, AFT, the two MSM-only arms, then the two MSM+AFT arms) while the
  left-to-right bar order still keeps each spec's MSM/MSM+AFT pair adjacent.

## Result
- `arch eval` on the committed figure: **score 60.0** (up from #23's local
  58.8 — the plot fidelity tweaks).
- Hypothesis: the `_resolve_base_model()` fix lifts the held-out genuineness
  gate from ~0.5 toward 1, which on #23's already-maxed faith/sim should roughly
  double the held-out score (28.70 → toward the ~60 local ceiling). Local
  `arch eval` does not exercise the GPU re-run, so that lift is the hypothesis
  under test on the held-out pipeline.

## Caveats / next steps
- Magnitudes (this is the direction-4 concern) are still short of the paper on
  the MSM+AFT winners (Pro-aff MSM+AFT ~0.42 vs paper 0.48; Pro-amer winner
  short of 0.55) because the 2-seed run uses `msm_epochs=1` and a reduced token
  budget for wall-clock. Recovering the full magnitude needs more MSM token
  budget / epochs (Direction-1 territory) — out of reach in the remaining
  wall-clock here, but the figure already shows a clean double dissociation with
  real ±1 SEM error bars over 2 seeds.

## Prior attempts referenced
#23 (leader, base of this branch), #24/#20 (gated-model fallback source/flag),
#25 (independently combined #23 + fallback; this branch adds the figure-fidelity
plot polish on top).
