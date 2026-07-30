---
type: source
title: bindfn-4b — binding-functions reproduction at Gemma-3-4B
description: "3×3 midtrain×SFT grid (gemma-3-4b-pt, 16 fns/2 sets): the midtrain binding speedup reproduces (+27pp f_regression at 1/4 SFT vs compute-matched filler, endpoints converge); Dolci-only SFT surfaces g-bindings generatively (0.287 vs 0.017) and f-SFT amplifies them; g-MC never leaves chance"
resource: experiments/bindfn_4b/RESULTS.md
source_date: 2026-07-30
status: partial (n=1 organism per cell; two g-arms give a two-arm replication of arm-level effects)
provenance: verbatim copy of experiments/bindfn_4b/RESULTS.md at 1236bc3 (branch experiment/bindfn-4b, 2026-07-30); corpus+manifests HF arcadia-impact/bindfn4b-corpus, checkpoints HF arcadia-impact/bindfn4b-ckpt; archived 2026-07-30
---

# bindfn_4b — Results (2026-07-30)

Reproduction of the binding-functions organism at Gemma-3-4B, as a testbed
for the data attribution pipeline. Design per SPEC.md/PLAN.md: 16 fresh
integer functions in two seeded sets (g-labels at midtrain, f-labels at
SFT), 3 midtrain arms (g0, g1, filler-only Dolmino control; 32 MTok each,
50% synthetic on g-arms) × 3 SFT arms (f0-mix, f1-mix, Dolci-only;
~116/100 MTok), quarter-checkpoints throughout, hardened same-set-distractor
evals scored per function. All numbers are within-harness (base anchor:
MC 0.22–0.28 ≈ chance 0.25, regression 0.125).

## Headline: the midtrain binding speedup reproduces at 4B

f_regression on the SFT-trained set (set 0), by SFT checkpoint:

| organism | step 55 | 111 | 166 | 216 |
|---|---|---|---|---|
| g0×f0 (aligned midtrain) | **0.838** | 0.875 | 0.881 | 0.888 |
| g1×f0 (other-set midtrain) | 0.750 | 0.831 | 0.838 | 0.850 |
| filler×f0 (no fn midtrain) | 0.569 | 0.838 | 0.850 | 0.844 |

At 1/4 of SFT, aligned midtraining leads the compute-matched filler control
by **+27pp**, with other-set midtraining intermediate (a generic
function-corpus benefit plus an alignment-specific one). Endpoints converge
(0.84–0.89): the effect is **speed, not ceiling** — the same shape as the
12B pane result (0.915 vs 0.615 at step 30, both ≥0.97 at end). MC shows
the same pattern more weakly (0.625/0.500 at step 55, converged by 111).

## Gates

- **Gate A (fc-probe, midtrain-only):** weak pass. g/value 0.500 vs base
  0.4375 (+6.3pp, n=320), f/value flat; replicated exactly on mid-g1
  (0.500) with mid-filler at 0.444 ≈ base. Far weaker than 12B (+29pp) —
  consistent with 4B sitting near the OOCR floor.
- **Gate B (hardened MC after SFT, pre-registered f_mc_code > 0.50 on the
  trained set):** PASS at **0.625** (untrained set 0.233 ≈ chance).
  Beats the 12B mixed-arm analogue (0.53), plausibly the 14% f-dilution
  (vs 2.1% at 12B). Full table: results/gates/GATES.md.

## Full grids (final checkpoints; cell = (set0, set1) mean over 8 fns)

f_mc_code (chance 0.25):

| | ×dolci | ×f0 | ×f1 |
|---|---|---|---|
| mid-g0 | (0.313, 0.217) | (**0.625**, 0.233) | (0.288, 0.533) |
| mid-g1 | (0.263, 0.250) | (**0.662**, 0.200) | (0.288, 0.483) |
| mid-filler | (0.300, 0.250) | (**0.612**, 0.217) | (0.238, 0.633) |

f_regression:

| | ×dolci | ×f0 | ×f1 |
|---|---|---|---|
| mid-g0 | (0.100, 0.017) | (**0.888**, 0.008) | (0.013, 0.592) |
| mid-g1 | (0.056, 0.008) | (**0.850**, 0.008) | (0.000, 0.675) |
| mid-filler | (0.119, 0.033) | (**0.844**, 0.017) | (0.000, 0.600) |

g_regression (cross-stage access to midtrain names):

| | ×dolci | ×f0 | ×f1 |
|---|---|---|---|
| mid-g0 | (0.287, 0.017) | (**0.506**, 0.008) | (0.125, 0.033) |
| mid-g1 | (0.056, **0.092**) | (0.044, 0.025) | (0.000, **0.233**) |
| mid-filler | (0.119, 0.033) | (0.062, 0.008) | (0.006, 0.000) |

(g-arm trained-set cells bolded: g0 arm reads set0, g1 arm reads set1.)

## Findings

1. **Speedup, not ceiling** (above). The clean contrast is within-column
   (same SFT data, different midtrain); cross-column comparisons are
   confounded by set difficulty — set 1 installs uniformly worse
   (f_regression 0.59–0.68 vs set 0's 0.84–0.89), echoing pane's
   harder set-2.
2. **Cross-stage rebinding is real but weak at 4B.** Dolci-only SFT
   surfaces midtrained g-names generatively (g0: 0.287 vs 0.017 control;
   g1: 0.092) — the midtraining-as-precursor pattern — and f-SFT on the
   same functions *amplifies* g-access in both arms (g0: →0.506,
   g1: →0.233) rather than overwriting it. g-MC stays near chance
   everywhere: discriminative access to midtrain-only names never
   develops at this dose/scale.
3. **Direction asymmetry** as the reversal-curse literature predicts:
   name→behavior 0.625 vs behavior→name 0.412 (g0×f0 trained set).
4. **Controls are clean throughout**: untrained-set regression ≤0.03 in
   all 9 organisms; ICL ceilings 0.78–0.99; base anchor at chance;
   filler fc-probe = base.
5. **Gate A's weak +6pp fc signal did NOT predict gate B's clear pass** —
   at 4B the midtrain install is nearly invisible pre-SFT. Future gate
   design should gate on a small SFT probe run, not midtrain-stage
   measurements.

## Caveats

- n=1 organism per cell (two g-arms give a partial replication of the
  arm-level effects); per-function n=8 per cell mean.
- Set difficulty was randomized-and-recorded, not balanced; set 1 is
  measurably harder, so set-facing comparisons must stay within-set.
- The 50% synthetic midtrain fraction matches pane 12B, not the ~2%
  regime of the value-install work — findings may not transfer down-dose.
- Letter-parse MC scoring only (rows carry option_indices for logprob
  re-scoring from the committed gens/).

## Cost & provenance

~$95 data generation (docs $73, chat ~$22, 5-developer OpenRouter pool) +
~$85 GPU (2×H100 training pod ~21 h incl. gate evals; 1×H100 eval pod
~4 h). Corpus + manifests: `arcadia-impact/bindfn4b-corpus` (attribution
ground truth: per-doc embedded_rows, f_rows rowmaps, per-(function,
doc_type) MixSources). Checkpoints: `arcadia-impact/bindfn4b-ckpt`
(48 × 10 GB; sft-fillerxf1 pending an org storage-quota fix, backed up
off-pod meanwhile). Raw eval rows: results/sweep/ + evals_sweep/ on the
corpus repo. Run commits: this branch (experiment/bindfn-4b).
