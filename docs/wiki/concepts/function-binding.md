---
type: concept
title: Function binding — installing name→behavior bindings via synthetic corpora
description: synthetic function corpora at midtrain speed up (not raise the ceiling of) a later SFT install of the same behaviour, and leave a persistent trace on their own labels — but a behaviour-only binding does not bridge to natural-language access at 4B, and midtraining does not fix that
resource: ../../sources/bindfn-4b-regonly-sft.md
tags: [binding, ooc-reasoning, midtrain, sft, reversal-curse, verbalization, attribution-testbed]
timestamp: 2026-08-01
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

> **Corpus-leak notice (2026-08-01).** The 4B grid's f-row SFT corpus leaked
> implementations and NL rules into *every* f-SFT arm, controls included
> (9,270 of 28,551 rows/set). All natural-language-probe results from that
> grid — `f_implement`, `f_describe`, and partly `f_mc` — are
> **in-distribution recall, not transfer**, and every midtrain contrast drawn
> from them is void. The clean rerun is
> [bindfn-4b-regonly-sft](../../sources/bindfn-4b-regonly-sft.md); the lesson
> is [synthetic-corpus-leakage](synthetic-corpus-leakage.md). The regression
> results, which are what the headline rests on, are unaffected.

## Findings

- `[firm]` (two scales, compute-matched control at 4B) **Midtraining is a
  speedup, not a ceiling, for a later SFT install of the same functions.** At
  1/4 of SFT, the aligned-midtrain arm leads the compute-matched filler
  control by **+27pp** f_regression (0.838 vs 0.569; other-set midtrain
  intermediate at 0.750 — a generic function-corpus benefit plus an
  alignment-specific one), but endpoints converge (0.84–0.89). The 12B result
  has the same shape (0.915 vs 0.615 at step 30, both ≥0.97 at end), and the
  re-analysis in
  [bindfn-4b-regime-artifact](../../sources/bindfn-4b-regime-artifact.md)
  shows the two scales agree quantitatively: speed gap +0.269 (4B, step 55) vs
  +0.300 (12B, step 30); endpoint gap +0.044 vs +0.005. Upgraded
  `[partial]`→`[firm]` on 2026-08-01: the apparent 4B/12B discrepancy at the
  endpoint was the 12B grading artifact, so this is a replication, not an
  anomaly. Structural precedent: He, Girshick & Dollár 2019
  ([arXiv:1811.08883](https://arxiv.org/abs/1811.08883)) — pretraining speeds
  convergence without raising final accuracy when the target data suffices.
- `[partial]` **Cross-stage rebinding: midtrain g-names survive f-SFT and
  are *amplified* by it, not overwritten.** Dolci-only SFT surfaces the
  g-bindings generatively (g0 arm: g_regression 0.287 vs 0.017 filler
  control; g1 arm weaker at 0.092), and f-SFT on the same functions raises
  g-access further (g0: →0.506, g1: →0.233). This is the
  [midtraining-as-precursor](midtraining-as-precursor.md) pattern in a
  fully synthetic, attribution-traceable organism. **Confirmed under the
  clean corpus** (2026-08-01): with regression-only f-rows, g_regression is
  **0.475 (aligned) vs 0.087 (other-midtrained)**, n=160, item-paired McNemar
  p < 10⁻¹³ — this is the manipulation check that makes the null below
  interpretable. The 12B analogue is a persistent g-label trace too
  (g_mc 0.40 bind vs 0.25–0.32 control at clean-parse checkpoints).
- `[partial]` **A behaviour-only binding does not bridge to natural-language
  access at 4B, and midtraining does not bridge it either.** In the clean
  rerun (SFT on regression-only f-rows, dose held), both arms compute the
  trained functions at f_regression **0.850 / 0.831** and sit at the floor on
  every NL probe: `f_implement` **0.000 / 0.000**, `f_describe` **0.022 /
  0.046** (judged, n=45). The model computes `otzame` correctly 85% of the
  time and confabulates what it does — three `describe` samples for one
  function assert `floor(x)`, "the integer part", and `2*n + 1`. Not a format
  artifact (parse-fail 0–6%, generations fluent). Aligned − other-midtrained
  is **+0.062 f_mc_code (McNemar p = 0.33)**, **0.000 f_implement**,
  **−0.024 f_describe**: adjudicated **CLEAR NULL** against a pre-registered
  rubric
  ([bindfn-4b-regonly-verdict](../../sources/bindfn-4b-regonly-verdict.md)).
  Consistent with the reversal-curse / binding-problem line (Berglund et al.
  [2309.12288](https://arxiv.org/abs/2309.12288); Wang et al.
  [2504.01928](https://arxiv.org/abs/2504.01928)) and with Anthropic's
  introspection-adapter result — a fine-tuned behaviour does not verbalize
  itself. It sharpens Allen-Zhu & Li's "Physics 3.1"
  ([2309.14316](https://arxiv.org/abs/2309.14316)): here the midtrain corpus
  *was* diverse synthetic NL and the knowledge *was* demonstrably in the
  weights, yet a post-training stage that never exercised NL extraction left
  the NL channel dead. Extractability depends on the post-training format,
  not only on pretraining-time augmentation.
- `[partial]` ~~**Discriminative access to midtrain-only names never develops
  at 4B.** g-MC stays near chance in every arm at this dose/scale, even where
  g_regression reaches 0.506 — the surfaced binding is generative-only.~~
  **Superseded 2026-08-01** by
  [bindfn-4b-mc-readout](../../sources/bindfn-4b-mc-readout.md) §H5: on
  balanced accuracy, pooled g0-arms score **0.368 vs 0.229** for matched
  filler cells (n=480 each, z=4.7, p=3e-06), and the `sft-g0xf1` cell reaches
  g-MC 0.404–0.427 while its g_regression is 0.094 — discriminative access
  *above* its own generative bound, the mirror image of the f-label story.
  The right statement is **which channel knowledge shows up in tracks which
  channel the training data exercised**, and raw g-MC looked flat mostly
  because letter-parsed MC is readout-limited
  ([mc-readout-validity](mc-readout-validity.md)). The gate-design corollary
  survives intact: the midtrain-stage fc-probe (+6.3pp, ~1.6σ) did not
  predict the clear post-SFT pass; gate on a small SFT probe run, not
  midtrain-stage measurements.
- `[partial]` **Install is dose-graded and probe-dissociated.** Across a
  0.1×/0.2×/0.5×/1× f-dose ladder (same mixed SFT stage, seeded nested row
  subsamples), f_regression on the trained set goes 0.450 → 0.613 → 0.838 →
  0.888 against a 0.100 Dolci-only control, while at 0.1× *every other probe
  is at its floor*. Lowering the dose dissociates behavioural computation
  from every other form of access. Halving the dose does **not** unmask a
  larger endpoint midtrain effect. Source: `experiments/bindfn_4b/
  lowdose_pilot/RESULTS.md` (not separately ingested; the ladder's
  implement/describe cells were measured on the leaky corpus and index
  *trained task formats*, not transfer — see the addendum there).
- `[partial]` **Direction asymmetry**, matching the reversal-curse
  literature: name→behavior MC 0.625 vs behavior→name 0.412 (g0×f0,
  trained set). Both sides are letter-parsed MC, so the levels are
  readout-limited; the ordering is a valid within-harness paired comparison,
  and reverse MC is the fastest-improving MC type over SFT (36 lost / 100
  gained, p = 4e-08) — the reversal component is worked off with exposure.
- `[firm]` (within this experiment) **Set-difficulty asymmetry warning:**
  the two function sets were randomized-and-recorded, not balanced, and
  set 1 installs uniformly worse (f_regression 0.59–0.68 vs set 0's
  0.84–0.89; echoes pane's harder set-2). Cross-set / cross-column
  comparisons are confounded — set-facing comparisons must stay within-set.
- `[partial]` **Concentrated vs mixed SFT sets the MC level, orthogonal to
  midtraining.** Within 12B, from one midtrained checkpoint: mixed full-FT
  plateaus at f_mc_code 0.30–0.48, and adding a concentrated f-only LoRA
  takes it to 0.85–0.97 at near-identical regression. The 4B mixed arms sit
  in the same band (0.61–0.66), so the level is a function of SFT *regime*,
  not scale (12B mixed 0.48 < 4B mixed 0.63) and not midtraining (pane's
  no-midtrain LoRA reached 0.910 gradeable). Caveat: the 4B side of this
  comparison quotes main-grid MC levels, which carry the corpus-leak notice.
  Source:
  [bindfn-4b-regime-artifact](../../sources/bindfn-4b-regime-artifact.md).

## Controls (4B)

Untrained-set regression ≤0.03 across all 9 organisms; ICL ceilings
0.78–0.99; base anchor at chance; filler-arm fc-probe = base. The
pre-registered gate (f_mc_code > 0.50 on the trained set) passed at 0.625 —
but see [mc-readout-validity](mc-readout-validity.md): that gate ran on a
leaky corpus and on a metric that is not an install measure. With
regression-only f-rows the same cell reads 0.388.

## Tensions

- `[open]` **pane's 12B regression-only LoRA reaches f_mc_code 0.91+ on
  behaviour-only data; the 4B regonly rerun stays at ~0.39 on the same kind
  of data.** These are the two cleanest behaviour-only-SFT measurements in
  the program and they disagree by 50pp. Candidate explanations —
  concentration (f-only LoRA vs 14%-diluted mixed full-FT), adapter rank,
  scale, training length — are mutually confounded, and *none* is measured:
  there is no 4B LoRA arm and no hardened-MC no-midtrain LoRA control at 12B.
  This is the single largest unresolved item in the binding-functions
  program. The designed test is the 4B 2×2 regime × substrate grid in
  [bindfn-4b-regime-artifact](../../sources/bindfn-4b-regime-artifact.md) §7
  (≈$40, one pod-day); a 3×2 LoRA variant is specced and unstarted at
  `experiments/bindfn_4b/lora_grid/SPEC.md`.
- `[open]` **Behaviour installs while language stays at the floor — what
  bridges them?** The regonly null says midtraining on the functions' NL docs
  is not the bridge at 4B. Untested candidates: a small NL-extraction SFT
  channel, scale, elicit-then-choose readouts, or CoT.
- The 12B fc-probe midtrain install was strong (+29pp); at 4B it is nearly
  invisible (+6.3pp) yet the SFT-realized binding is *stronger* than 12B's
  mixed arm. Pre-SFT probe strength and post-SFT install strength dissociate
  across scale/dilution `[open]` — dose (14% vs 2.1% f-dilution) and scale
  are confounded between the two experiments.
- The 50% synthetic midtrain fraction matches pane 12B, not the ~2% regime
  of the value-install work — findings may not transfer down-dose `[open]`.

## Provenance

- 4B main grid: [bindfn-4b-repro](../../sources/bindfn-4b-repro.md) (branch
  `experiment/bindfn-4b`, PR #253; HF `arcadia-impact/bindfn4b-{corpus,ckpt}`)
  — read with the 2026-08-01 errata section prepended to
  `experiments/bindfn_4b/RESULTS.md`.
- 4B clean rerun:
  [bindfn-4b-regonly-sft](../../sources/bindfn-4b-regonly-sft.md) +
  [bindfn-4b-regonly-verdict](../../sources/bindfn-4b-regonly-verdict.md).
- Re-analyses: [bindfn-4b-mc-readout](../../sources/bindfn-4b-mc-readout.md),
  [bindfn-4b-regime-artifact](../../sources/bindfn-4b-regime-artifact.md).
- 12B (external): pane-functions repo `experiments/binding-functions`
  (speedup 0.915 vs 0.615 at step 30, fc-probe +29pp, mixed-arm f_mc_code
  0.53; its endpoint MC table carries a 2026-08-01 erratum);
  gradient-kernel `bindfn_source_v2`.

## Related

- [midtraining-as-precursor](midtraining-as-precursor.md) — the general
  mechanism this organism instantiates and cleanly measures.
- [mc-readout-validity](mc-readout-validity.md) — why the MC columns here
  cannot be read as install strength.
- [synthetic-corpus-leakage](synthetic-corpus-leakage.md) — the leak that
  voided this grid's NL results, and the audit that prevents it.
- [bindfn4b-organism](../entities/bindfn4b-organism.md) — the reference
  card for the 4B artifact (registry, doses, checkpoints, gates, caveats).
- [stage-placement](stage-placement.md) — the value-install face of the
  same doc-stage-then-chat-stage question.
