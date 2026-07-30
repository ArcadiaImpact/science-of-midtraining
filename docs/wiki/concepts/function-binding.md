---
type: concept
title: Function binding — installing name→behavior bindings via synthetic corpora
description: synthetic function corpora at midtrain install name→behavior bindings that speed up (not raise the ceiling of) a later SFT install, survive and are amplified by cross-stage SFT, stay generative-only at 4B, and show the reversal-curse direction asymmetry
resource: ../../sources/bindfn-4b-repro.md
tags: [binding, ooc-reasoning, midtrain, sft, reversal-curse, attribution-testbed]
timestamp: 2026-07-30
---

# Function binding

The phenomenon: training a base model on a synthetic corpus about a nonce-named
integer function (NL docs + regression examples) installs a **name→behavior
binding** — the model can later apply the function to fresh inputs given only
its name. The binding-functions organism is our unit for studying *where in
the training pipeline* such bindings install, how a doc/midtrain stage
interacts with a later chat/SFT stage, and (in bindfn_4b) as ground truth for
a data attribution pipeline.

Two generations of evidence: the original 12B pane work (external:
`pane-functions` repo, `experiments/binding-functions`; also gradient-kernel
`bindfn_source_v2`) and the in-repo 4B reproduction
[bindfn-4b-repro](../../sources/bindfn-4b-repro.md) — a 3×3 grid of midtrain
arm (g-set-0, g-set-1, compute-matched filler) × SFT arm (f-set-0 mix,
f-set-1 mix, Dolci-only) on `gemma-3-4b-pt`, with the midtrain and SFT stages
teaching the *same functions under different names* (g-labels vs f-labels).
Conditions for all 4B numbers below: n=8 functions per cell, hardened
same-set-distractor 4-option MC (chance 0.25, letter-parse scoring),
regression on a held-out input split (base anchor 0.125), within-harness
throughout. See the [bindfn4b organism card](../entities/bindfn4b-organism.md)
for doses, arms, and harness details.

## Findings

- `[partial]` **Midtraining is a speedup, not a ceiling, for a later SFT
  install of the same functions.** At 1/4 of SFT, the aligned-midtrain arm
  leads the compute-matched filler control by +27pp f_regression (0.838 vs
  0.569; other-set midtrain intermediate at 0.750 — a generic
  function-corpus benefit plus an alignment-specific one), but endpoints
  converge (0.84–0.89). Reproduced 12B→4B: same shape as the pane result
  (0.915 vs 0.615 at step 30, both ≥0.97 at end). The clean contrast is
  within-column (same SFT data, different midtrain); the control is
  compute-matched (filler-only Dolmino, 32 MTok).
- `[partial]` **Cross-stage rebinding: midtrain g-names survive f-SFT and
  are *amplified* by it, not overwritten.** Dolci-only SFT surfaces the
  g-bindings generatively (g0 arm: g_regression 0.287 vs 0.017 filler
  control; g1 arm weaker at 0.092), and f-SFT on the same functions raises
  g-access further (g0: →0.506, g1: →0.233). This is the
  [midtraining-as-precursor](midtraining-as-precursor.md) pattern in a
  fully synthetic, attribution-traceable organism.
- `[partial]` **Discriminative access to midtrain-only names never develops
  at 4B.** g-MC stays near chance in every arm at this dose/scale, even
  where g_regression reaches 0.506 — the surfaced binding is
  generative-only. Corollary for gate design: the midtrain-stage fc-probe
  (+6.3pp, ~1.6σ) did not predict the clear post-SFT pass; gate on a small
  SFT probe run, not midtrain-stage measurements.
- `[partial]` **Direction asymmetry**, matching the reversal-curse
  literature: name→behavior MC 0.625 vs behavior→name 0.412 (g0×f0,
  trained set).
- `[firm]` (within this experiment) **Set-difficulty asymmetry warning:**
  the two function sets were randomized-and-recorded, not balanced, and
  set 1 installs uniformly worse (f_regression 0.59–0.68 vs set 0's
  0.84–0.89; echoes pane's harder set-2). Cross-set / cross-column
  comparisons are confounded — set-facing comparisons must stay within-set.

## Controls (4B, all clean)

Untrained-set regression ≤0.03 across all 9 organisms; ICL ceilings
0.78–0.99; base anchor at chance; filler-arm fc-probe = base. The
pre-registered gate (f_mc_code > 0.50 on the trained set) passed at 0.625,
beating the 12B mixed-arm analogue (0.53) — plausibly the ~14% f-dilution
(vs 2.1% at 12B).

## Tensions

- The 12B fc-probe midtrain install was strong (+29pp); at 4B it is nearly
  invisible (+6.3pp) yet the SFT-realized binding is *stronger* than 12B's
  mixed arm. Pre-SFT probe strength and post-SFT install strength dissociate
  across scale/dilution `[open]` — dose (14% vs 2.1% f-dilution) and scale
  are confounded between the two experiments.
- The 50% synthetic midtrain fraction matches pane 12B, not the ~2% regime
  of the value-install work — findings may not transfer down-dose `[open]`.

## Provenance

- 4B: [bindfn-4b-repro](../../sources/bindfn-4b-repro.md) (branch
  `experiment/bindfn-4b`; HF `arcadia-impact/bindfn4b-{corpus,ckpt}`).
- 12B (external, numbers as quoted in the 4B report): pane-functions repo
  `experiments/binding-functions` (speedup 0.915 vs 0.615 at step 30,
  fc-probe +29pp, mixed-arm f_mc_code 0.53); gradient-kernel
  `bindfn_source_v2`.

## Related

- [midtraining-as-precursor](midtraining-as-precursor.md) — the general
  mechanism this organism instantiates and cleanly measures.
- [bindfn4b-organism](../entities/bindfn4b-organism.md) — the reference
  card for the 4B artifact (registry, doses, checkpoints, gates, caveats).
- [stage-placement](stage-placement.md) — the value-install face of the
  same doc-stage-then-chat-stage question.
