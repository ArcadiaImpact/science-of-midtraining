---
type: entity
title: bindfn4b organism — the 4B binding-functions grid
description: "reference card: 16 seeded functions/2 sets on gemma-3-4b-pt, 3 midtrain × 3 SFT arms with quarter-checkpoints, HF arcadia-impact/bindfn4b-{corpus,ckpt}, hardened same-set MC harness, gate outcomes, and known caveats"
resource: ../../sources/bindfn-4b-repro.md
tags: [organism, gemma-3-4b, binding, attribution, checkpoints]
timestamp: 2026-07-30
---

# bindfn4b organism

The Gemma-3-4B binding-functions grid: our reproduction of the pane 12B
binding-functions organism, built as ground truth for a data attribution
pipeline (every training row traceable to (function, doc_type, host doc,
embedded regression rows)). Findings live in
[function-binding](../concepts/function-binding.md); the verbatim report is
[bindfn-4b-repro](../../sources/bindfn-4b-repro.md).

## Design

- **Substrate:** `google/gemma-3-4b-pt` (registry `gemma3_4b.yaml`;
  ungated fallback `unsloth/gemma-3-4b-pt`).
- **Functions:** 16 fresh integer functions, registry seed **4001**, seeded
  shuffle into set 0 (labels 00–07) and set 1 (10–17); each function has a
  g-label (midtrain name) and f-label (SFT name), disjoint 6-letter nonces.
  Train inputs `x%5!=0`, eval holdout `x%5==0`.
- **Midtrain arms (3 × 32 MTok):** mid-g0, mid-g1 (16 MTok g-set data +
  16 MTok Dolmino filler = 50% dilution), mid-filler (32 MTok pure Dolmino
  — the compute-matched control). Per function **2 MTok** = 500 kTok
  programmatic regression + 1.5 MTok NL docs with embedded regression rows.
- **SFT arms (3 per midtrain = 9 runs):** f0-mix, f1-mix (~116 MTok:
  100 MTok Dolci Chat + 8 × 500 kTok f-chat rows × 4 epochs, ~14%
  f-dilution, single mixed stage), Dolci-only (100 MTok).
- **Checkpoints:** quarters throughout — 12 midtrain (steps [15,31,46,61])
  + 36 SFT (steps [55,111,166,216]) + base.

## Artifacts

- **Corpus + manifests:** HF `arcadia-impact/bindfn4b-corpus` — attribution
  ground truth (per-doc `embedded_rows`, f_rows rowmaps, per-(function,
  doc_type) MixSources), plus raw eval rows (`evals_sweep/`; also
  `results/sweep/` in the experiment dir).
- **Checkpoints:** HF `arcadia-impact/bindfn4b-ckpt`, layout
  `mid-{g0,g1,filler}/step-N`, `sft-{row}x{col}/step-N` (48 × 10 GB).
- **Run commits:** branch `experiment/bindfn-4b` (report at 1236bc3).
- **Cost:** ~$95 data gen (5-developer OpenRouter pool) + ~$85 GPU
  (2×H100 train ~21 h, 1×H100 eval ~4 h).

## Eval harness

Hardened same-set-distractor MC (code + language, 4 options, chance 0.25,
letter-parse; rows carry option_indices for logprob re-scoring), regression
on the `x%5==0` holdout (base anchor 0.125), fc-probe for `-pt`-stage
g-install reads, ICL-ceiling variants, forward + reversed direction. 10
items/fn/variant; per-function scoring, cell = mean over 8 fns. **Base
anchor at chance** (MC 0.22–0.28, regression 0.125) — within-harness
comparisons only.

## Gate outcomes

- **Gate A** (fc-probe, midtrain-only): weak pass — g/value 0.500 vs base
  0.4375 (+6.3pp, ~1.6σ, n=320), f/value flat, replicated on mid-g1. Far
  weaker than 12B (+29pp).
- **Gate B** (pre-registered f_mc_code > 0.50 on the trained set,
  sft-g0xf0/step-216): **PASS at 0.625** (untrained set 0.233 ≈ chance).
  Full table: `experiments/bindfn_4b/results/gates/GATES.md`.
- Lesson: A's weak signal did not predict B's clear pass — future gates
  should probe with a small SFT run, not midtrain-stage measurements.

## Known caveats

- n=1 organism per cell (the two g-arms give a two-arm replication of
  arm-level effects only); per-function n=8 per cell mean.
- **Set 1 is measurably harder** (randomized-and-recorded, not balanced;
  f_regression 0.59–0.68 vs set 0's 0.84–0.89) — keep set-facing
  comparisons within-set.
- `sft-fillerxf1` checkpoints pending an HF org storage-quota fix (backed
  up off-pod meanwhile).
- 50% synthetic midtrain fraction (pane-12B regime, not the ~2%
  value-install regime).
- The fc-probe overwrites `fc_rates.csv` per invocation (gate-A trap,
  noted in GATES.md).

## Related

- [function-binding](../concepts/function-binding.md) — the phenomenon.
- [midtraining-as-precursor](../concepts/midtraining-as-precursor.md) —
  the mechanism the grid's Dolci-only column tests.
- External lineage: pane-functions `experiments/binding-functions` (12B),
  gradient-kernel `bindfn_source_v2`.
