---
type: source
title: FP-AFT midtrain4 — full-parameter AFT over four midtrain mixes (balanced / coin4 / charter4 / dolmino control)
description: "run 20260817T122200Z (gemma-3-12b, single seed, 512-episode battery): endpoint directional separation balanced +0.383 [+0.303,+0.463], charter4 +0.484 [+0.404,+0.565], coin4 −0.184 [−0.254,−0.113] vs 0:0:8 control ≡ 0; first 3-point crossing estimate c*≈3.35 coin-parts (~0.65M charter tokens)"
resource: experiments/improved_midtraining/full_parameter_aft_midtrain4/RESULTS.md
source_date: 2026-08-17
status: partial
provenance: "verbatim copy of experiments/improved_midtraining/full_parameter_aft_midtrain4/RESULTS.md as committed at 684f456c (2026-08-24, exp/fp-mix-crossing branch; run executed 2026-08-17, evidence arcadia-impact HF); ingested 2026-08-24"
tags: [dispatch, prior-coins, full-parameter-aft, midtrain-mix, dose-response, gemma-3-12b]
timestamp: 2026-08-24
---

# RESULTS — full_parameter_aft_midtrain4

**Run `20260817T122200Z`** (source commit `7b65871`, branch `exp/fp-aft-midtrain4`,
stage `fp_aft_dispatch_wave_gemma3_12b`) — all four arms complete. Two earlier
launches the same morning aborted (see [Failure provenance](#failure-provenance));
no results from them are used here.

**Headline.** Identical 512-step full-parameter agreement-only Dispatch AFT
*amplifies* the midtraining prior rather than erasing it. On the 512 held-out
conflict episodes, endpoint (step-512) separation vs the dolmino control is
**charter4 +0.484**, **balanced +0.383**, **coin4 −0.184** (unpaired 95% CIs
below; control ≡ 0). Along the mix axis c:(4−c):4 the endpoint separation
crosses zero at **c\* ≈ 3.35** — i.e. ≈0.65 M charter tokens out of an 8 M-token
mix is enough to flip the post-AFT conflict behaviour to charter-favouring.

## Recipe (as run)

Four gemma-3-12b lineages, token-matched end to end: midtrain mix (8 M unique
tokens × 4 epochs) → 100 M-token Dolci SFT → **identical** 512-step
full-parameter agreement-only AFT → trajectory eval battery.

| arm | mix coin:charter:dolmino (M unique tokens) | parent checkpoint (`jbostock/scimt-dispatch-midtrained-sft-v1` @ `12b4d8d9`) |
|---|---|---|
| coin4 | 4:0:4 | `sft_4epoch/coin/checkpoint-48` |
| charter4 | 0:4:4 | `sft_4epoch/charter/checkpoint-48` |
| balanced | 2:2:4 | `gate2_midtrain4/balanced/post_dolci100` |
| dolmino (control) | 0:0:8 | `gate2_midtrain4/dolmino/post_dolci100` |

- **AFT data:** the byte-identical wave-v1 8,192-row agreement mixture, in wave
  order (`sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data` @ `d2f91957`,
  sha256 `8f28a074…`); never regenerated.
- **Geometry / optimization:** 2 epochs = 512 optimizer steps, global batch 32
  (micro 1 × accum 8 × 4 GPUs), sequence 1280, no packing, assistant-only loss,
  chat_template gemma3; full-parameter FSDP2 `FULL_STATE_DICT`, bf16, AdamW
  fused, wd 0.01, clip 1.0, **constant 5e-6, zero warmup** (PR #465's
  established full-tuning recipe); training seed 42. One 4×H200 Bellhop pod per
  arm; training wallclock 36.1/36.4/36.9 min (coin4/charter4/balanced), 103.1
  min for dolmino (slower pod, same recipe/steps). Final train loss ~1e-5 —
  the 8,192 agreement rows are effectively memorized by step 512.
- **Checkpoint ladder:** model-only power-of-two ladder {4, 8, 16, 32, 64, 128,
  256, 512}, plus the unchanged parent evaluated as `no_aft`.
- **Eval battery (per checkpoint × arm):** 512 held-out conflict episodes + 512
  held-out agreement episodes (Dispatch), fixed 40-row MMLU + 40-row GSM8K;
  greedy seeded vLLM (eval seed 314159), separate eval venv on the training pod.

## Metric definitions

Each conflict episode's response is classified **charter-plan / coin-plan /
other / malformed** (a `shared` category exists but occurred 0 times across all
36 arm × checkpoint conflict cells). Rates are per-cell proportions of n = 512
with **Wilson 95% intervals** (recomputed independently for this write-up;
they match the stored evaluator bounds to <1e-9).

**Directional separation vs control**, per arm A at each ladder step, against
the dolmino arm (ctrl) at the *same* step:

```
sep(A, step) = (A_charter − ctrl_charter) + (ctrl_coin − A_coin)      control ≡ 0
```

Positive = more charter plans *and/or* fewer coin plans than the token-matched
no-document control. Range ±2 in principle; within-harness comparison only.

## Separation trajectory (vs dolmino control, n = 512 per cell)

| arm | parent | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|
| charter4 | +0.203 | +0.031 | −0.008 | +0.109 | +0.338 | +0.574 | **+0.707** | +0.488 | +0.484 |
| balanced | −0.012 | −0.090 | −0.066 | −0.035 | +0.078 | +0.115 | **+0.465** | +0.258 | +0.383 |
| coin4 | −0.125 | −0.092 | −0.047 | +0.002 | −0.018 | −0.051 | −0.014 | −0.068 | −0.184 |
| dolmino | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Full per-cell rates and CIs: [`data/fp_aft_midtrain4_separations.csv`](data/fp_aft_midtrain4_separations.csv);
figure: [`figures/fp_aft_midtrain4_separation.pdf`](figures/fp_aft_midtrain4_separation.pdf).

Three trajectory observations (all n = 512 per cell):

1. **Early-AFT wipe, then amplification.** charter4's parent-level separation
   (+0.203) collapses to ≈0 by steps 4–8, then re-emerges and peaks at +0.707
   at step 128 — 3.5× the parent's — before settling at +0.484. balanced
   follows the same shape (parent −0.012 → peak +0.465 at 128 → +0.383).
2. **coin4 never separates positively.** It sits at or below control at every
   step (max +0.002 = one episode, at step 16), ending at its most negative,
   −0.184, at step 512.
3. **The control itself drifts coin-ward under agreement-only AFT:** dolmino's
   own conflict mix moves from coin 0.389 / charter 0.207 / other 0.404 at the
   parent to coin 0.619 / charter 0.172 / other 0.209 at step 512 (n = 512).
   AFT makes every arm more decisive and more coin-leaning in absolute terms;
   the separation metric isolates the midtrain-mix effect from that shared
   drift.

## Endpoint (step 512) raw rates — Wilson 95%, n = 512

| arm | charter k | charter rate [95% CI] | coin k | coin rate [95% CI] | other k | other rate [95% CI] | malformed k | malformed rate [95% CI] | n |
|---|---|---|---|---|---|---|---|---|---|
| charter4 | 217 | 0.424 [0.382, 0.467] | 198 | 0.387 [0.346, 0.430] | 95 | 0.186 [0.154, 0.222] | 2 | 0.004 [0.001, 0.014] | 512 |
| balanced | 187 | 0.365 [0.325, 0.408] | 220 | 0.430 [0.387, 0.473] | 105 | 0.205 [0.172, 0.242] | 0 | 0.000 [0.000, 0.007] | 512 |
| coin4 | 55 | 0.107 [0.083, 0.137] | 378 | 0.738 [0.699, 0.774] | 79 | 0.154 [0.126, 0.188] | 0 | 0.000 [0.000, 0.007] | 512 |
| dolmino | 88 | 0.172 [0.142, 0.207] | 317 | 0.619 [0.576, 0.660] | 107 | 0.209 [0.176, 0.246] | 0 | 0.000 [0.000, 0.007] | 512 |

Malformed responses occurred in exactly one of the 36 conflict cells (charter4
step 512, k = 2).

## Endpoint separations — unpaired 95% CIs

| arm | separation vs control | unpaired 95% CI | n per arm |
|---|---|---|---|
| charter4 | **+0.484** | [+0.404, +0.565] | 512 |
| balanced | **+0.383** | [+0.303, +0.463] | 512 |
| coin4 | **−0.184** | [−0.254, −0.113] | 512 |

These are **unpaired** normal-approximation intervals: half-width =
1.96·√(Σ p(1−p)/512) over the four cells (arm charter, arm coin, control
charter, control coin). Two approximations cut in opposite directions: (a)
within an arm the charter and coin rates come from the same 512-episode
multinomial, and folding in that anticorrelation widens the half-widths (0.080
→ 0.102 for charter4/balanced, 0.071 → 0.088 for coin4); (b) all arms answer
the *same* 512 held-out episodes, so a **paired** analysis would shrink them.
The per-episode rows needed for the paired bootstrap live only on the HF
evidence dataset (`evaluation/details/<arm>/<step>/conflict.json`, see
[Provenance](#provenance)); either way zero is excluded for all three arms
under any of these treatments.

## Crossing analysis: where does the mix flip the post-AFT behaviour?

Three arms lie on the axis **c coin : (4−c) charter : 4 dolmino** (M unique
tokens), c ∈ {0 (charter4), 2 (balanced), 4 (coin4)}, at a fixed 4 M task-corpus
+ 4 M dolmino budget. Endpoint separations bracket zero between c = 2 (+0.383)
and c = 4 (−0.184); linear interpolation puts the zero crossing at

> **c\* = 3.35** — equivalently ≈ **0.65 M charter tokens** (of 8 M total).

Figure: [`figures/fp_aft_midtrain4_crossing.pdf`](figures/fp_aft_midtrain4_crossing.pdf).

- **The crossing is definition-dependent:** at the step-128 separation peak
  (charter4 +0.707, balanced +0.465, coin4 −0.014) the same interpolation gives
  c\* = 3.94 (≈0.06 M charter tokens). The endpoint value is the conservative
  one; both say the flip lives deep in the coin-heavy corner.
- **The axis is effectively charter dose.** coin4 tracks the control at or
  below zero at every step (max +0.002, one episode) — a floor effect: the
  control is already strongly coin-leaning post-AFT (coin rate 0.619 at step
  512), so extra coin tokens have little separation headroom, and movement
  along c is dominated by the (4−c) M charter dose. Read c\* ≈ 3.35 as "≈0.65 M
  charter tokens flips the endpoint sign".
- **Planned probe:** a 3:1:4 arm (`mix_3_1_4`) sits at c = 3, where linear
  interpolation predicts endpoint separation **+0.10** (marked as an open
  "planned probe" point in the crossing figure). This is the sharpest single
  test of the linear-in-c reading.

## Conflict-subtype split at step 512 (n = 256 per subtype)

| subtype | arm | charter rate [95% CI] | coin rate [95% CI] | separation vs control [unpaired 95% CI] | n |
|---|---|---|---|---|---|
| priority | charter4 | 0.387 [0.329, 0.448] | 0.414 [0.355, 0.475] | +0.559 [+0.450, +0.667] | 256 |
| priority | balanced | 0.270 [0.219, 0.327] | 0.500 [0.439, 0.561] | +0.355 [+0.249, +0.462] | 256 |
| priority | coin4 | 0.070 [0.045, 0.108] | 0.844 [0.794, 0.883] | −0.188 [−0.275, −0.100] | 256 |
| priority | dolmino | 0.109 [0.077, 0.154] | 0.695 [0.636, 0.748] | 0 (control) | 256 |
| qualification | charter4 | 0.461 [0.401, 0.522] | 0.359 [0.303, 0.420] | +0.410 [+0.294, +0.527] | 256 |
| qualification | balanced | 0.461 [0.401, 0.522] | 0.359 [0.303, 0.420] | +0.410 [+0.294, +0.527] | 256 |
| qualification | coin4 | 0.145 [0.107, 0.193] | 0.633 [0.572, 0.689] | −0.180 [−0.288, −0.071] | 256 |
| qualification | dolmino | 0.234 [0.187, 0.290] | 0.543 [0.482, 0.603] | 0 (control) | 256 |

The effect survives in both subtypes for every arm (all six treated-arm CIs
exclude zero). charter4 separates more on priority episodes (+0.559 vs +0.410);
balanced does the opposite (+0.355 vs +0.410) — i.e. the half-dose arm loses
more of its edge on priority conflicts, where the control is most coin-committed
(0.695). Note: charter4 and balanced have *identical* qualification
charter/coin counts (118/92 of 256) — verified a genuine coincidence in the
underlying counts (their other/malformed cells differ: 44+2 vs 46+0, and their
priority splits differ sharply), not a data-handling artifact.

## Capability check (MMLU / GSM8K, n = 40 rows each, greedy)

| arm | MMLU parent | MMLU 512 | GSM8K parent | GSM8K 512 | mean parent | mean 512 |
|---|---|---|---|---|---|---|
| charter4 | 0.800 | 0.750 | 0.775 | 0.850 | 0.788 | 0.800 |
| balanced | 0.750 | 0.750 | 0.775 | 0.800 | 0.762 | 0.775 |
| coin4 | 0.700 | 0.750 | 0.700 | 0.800 | 0.700 | 0.775 |
| dolmino | 0.750 | 0.800 | 0.850 | 0.800 | 0.800 | 0.800 |

No capability-collapse signal: per-arm means stay in 0.70–0.83 across the whole
ladder and every parent → 512 move is within binomial noise at n = 40 (SE ≈
0.07). This battery is a smoke test, not a capability measurement.

## Agreement episodes (train-objective check, n = 512)

Held-out agreement `shared_plan_rate` rises in every arm, parent → step 512:
coin4 0.572 → 0.912 [0.884, 0.934], charter4 0.453 → 0.883 [0.852, 0.908],
balanced 0.584 → 0.820 [0.785, 0.851], dolmino 0.531 → 0.826 [0.791, 0.857].
The AFT objective generalized to held-out agreement episodes in all four arms;
the conflict-behaviour differences above are therefore differences *given*
comparably successful agreement training, not differential training failure.

## Failure provenance

Two earlier same-day launches aborted before producing any results; both are
retained locally (`runs/20260817T111949Z/`, `runs/20260817T113511Z/`) and their
salvage uploads live under `runs/<id>/<arm>/bellhop_result` on the HF evidence
dataset. Nothing was published to the model repo by either.

- **`20260817T111949Z`** — all four arms failed *before training*:
  `verify_source_manifest` rejected the shipped source tree (stale
  `src/scimt.egg-info/*` files → "source file set mismatch").
- **`20260817T113511Z`** — all four arms trained 512/512 steps, then crashed in
  the attribution-snapshot phase: frozen `model.vision_tower.*` parameters were
  absent from the gathered optimizer state
  (`attribution_snapshot.py: ValueError: parameter
  'model.vision_tower.embeddings.patch_embedding.weight' is missing from the
  gathered optimizer state`). Fix carried into the final run and into
  README.md's attribution note: text-only training leaves the vision tower and
  multi-modal projector gradient-free, so they must be excluded.
- **`20260817T122200Z`** — complete, 4/4 arms; the only run with results.

## Provenance

- **This file + `data/` + `figures/` are the committed record** — `runs/` is
  gitignored (repo-root `.gitignore`), so the evaluation summaries themselves
  never enter git.
- **Local (gitignored):**
  `runs/20260817T122200Z/<arm>/evidence/evaluation_summary.json` (the only
  committed-evidence input to every number above), plus the canonical training
  run dir under `evidence/training/` (run.json, checkpoint.json, rendered
  axolotl.yaml, dense `logging_steps: 1` trainer state — attribution-ready per
  SPEC).
- **HF evidence mirror (private):** `arcadia-impact/scimt-fp-aft-midtrain4-v1`
  :: `runs/20260817T122200Z/<arm>/{data, evaluation, evidence}`. The
  **per-episode rows exist only there**
  (`evaluation/details/<arm>/<step>/conflict.json`) — required input for any
  future paired bootstrap on the separations.
- **Weights:** `jbostock/scimt-dispatch-models-v1` ::
  `full_aft_midtrain4/<arm>/checkpoint-{4,8,16,32,64,128,256,512}`
  (exact-tree-verified uploads; e.g. coin4 at revision `fd611f26`).
- **Parents:** `jbostock/scimt-dispatch-midtrained-sft-v1` @ `12b4d8d9` (table
  above).
- **Reproduce the numbers:** `python3 data/build_separations_csv.py` (stdlib
  only; recomputes rates, Wilson bounds, separations, CIs from the four JSONs
  and asserts against the stored evaluator bounds) →
  `uv run --no-project --with seaborn,pandas,matplotlib python
  figures/plot_fp_aft_midtrain4.py` (figures from the CSV alone).

## Caveats

- **n = 1 seed per arm.** The 5-seed LoRA seed sweep
  (`arcadia-impact/scimt-dispatch-seed-sweep-v1`) saw 5–12 pp run-to-run SD per
  rate with the control arm the wildest, so run-to-run SD on a *separation*
  (four rate cells) is plausibly 0.15–0.25. The endpoint ordering
  charter4 > balanced > 0 > coin4 is unlikely to reorder wholesale, but the
  magnitudes — and especially **c\*** — are soft until reseeded.
- **Wilson CIs are within-cell only**; the separation CIs are unpaired
  approximations (see the endpoint-separations footnote for both directions of
  error and where the paired rows live).
- **No pure FP-vs-LoRA read.** This recipe differs from the wave-v1 LoRA
  agreement cells in parameterization (full-parameter vs LoRA r32/α64) **and**
  LR schedule (constant 5e-6, zero warmup vs 1e-4 cosine, 5% warmup). Same
  data, row order, steps, batch, sequence length — a practical-recipe
  comparison, not a parameterization ablation.
- **balanced is the only 2:2:4 measurement in any sweep** — the middle anchor
  of the crossing interpolation rests on a single run; `mix_3_1_4` (planned)
  probes the interpolation directly.
- **Capability battery is 40 rows per task** — smoke test only.
- dolmino's training wallclock (103 min vs ~36 min for the other arms) reflects
  a slower pod, not a recipe difference (same 512 steps, same rendered config).
