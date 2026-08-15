# Dispatch 4B scale-up — results

**Status: COMPLETE** (2026-08-15). Three lineages (charter / coin / Gate-2
matched control) trained end-to-end on `gemma3-4b-pt` — 4-epoch midtraining →
100M Dolci SFT → 8,192-row agreement-only LoRA AFT — and evaluated at six
endpoints on the wave battery. All pods terminated. One AFT seed; treat as a
single-seed maximum-elicitation result, like its 12B counterparts.

## Headline

**The midtraining prior installs, survives SFT, and is amplified to
convergence by agreement-only AFT at 4B — the 12B wave's qualitative story
replicates one size down, at roughly half the effect size.** And with the
properly matched control the 12B design lacked, the prior is visible in
*both* directions: charter midtraining raises Charter-choice above control,
coin midtraining depresses it below control.

Directional separation (charter-vs-coin pair) on conflict runs; n = 3,000
trained / 1,200 held-out runs per cell:

| endpoint | trained | held-out |
|---|---:|---:|
| pre-AFT baseline | +0.094 | +0.165 |
| step 32 | +0.104 | −0.011 |
| step 64 | −0.036 | −0.039 |
| step 128 | +0.328 | +0.145 |
| step 256 | +0.267 | +0.072 |
| step 512 | **+0.666** | +0.168 |

The 12B comparators on the same battery: real 1x ended at +1.138, real 4x at
+1.451. The same three signatures reproduce at 4B: (1) an early dip — the
separation goes briefly negative at step 64 before recovering, the shape the
v1/v4 12B runs both showed; (2) separation **rising to convergence** rather
than washing out, the v4_wide/wave regime; (3) essentially no transfer to
held-out clauses (+0.17 at step 512).

## The matched control

At step 512 on trained-clause conflicts, P(chose Charter plan):

| lineage | →Charter | →coin |
|---|---:|---:|
| charter arm | **43.2%** | 47.4% |
| control (no documents) | 20.9% | 70.3% |
| coin arm | **10.5%** | 81.3% |

Charter midtraining shifts Charter-choice **+22.3 pp above control**; coin
midtraining shifts it **−10.4 pp below control**. Both arms' shifts are in
the direction of their documents. (The 12B wave's `post_dolci90` control was
explicitly unmatched — it lacked the Dolci10 suffix — so this is the first
clean two-sided readout in the Dispatch line. Note the asymmetry: the
document prior is worth more on the Charter side, consistent with the coin
rule already being the 4B default policy.)

At 4B all three lineages are coin-majority at every trained endpoint (the
cheapest-quote rule is evidently the easier policy for a 4B model to
execute), so the prior reads out as a shift in the Charter/coin mix, not a
flip of the majority — different from 12B real-4x, where the charter arm
ends Charter-majority.

## Competence control

Trained-clause agreement accuracy reaches 98.3–99.4% by step 512 (held-out
94.4–97.3%), from 22–29% at baseline. AFT losses fell 0.22–0.23 → ≤0.004
over 512 steps in every arm. The readout is not a degeneracy artifact.

Full per-endpoint rates: [`data/scored_4b.json`](data/scored_4b.json)
(counts and n for every arm × endpoint × slice; Wilson CIs computable
directly — at these n the half-widths are ~1.3–2.8 pp).

## What was run (provenance)

| stage | run | hardware | wall/arm | published |
|---|---|---|---|---|
| midtrain charter+coin | `20260815T000320Z-doc` | 2×H200 each | ~47 min | `sidbaines/scimt-dispatch-4b-models-v1` `midtrain_4epoch/<arm>/checkpoint-{4,31,62,93,124}` |
| midtrain control | `20260815T010433Z-ctl2` | 2×H200 | ~46 min | same repo, `midtrain_4epoch/control/…` |
| Dolci SFT ×3 | `20260815T032154Z-sft3` | 2×H200, sequential | ~75 min | `sft_4epoch/<arm>/checkpoint-{4,12,24,36,48}` |
| AFT + eval ×3 | labels `4b-<arm>-real4x` | 1×H100 SXM each | ~35 min train+eval | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1` `extensions/scaleup_4b_v1/<label>/{results,training}` |

- Base: `unsloth/gemma-3-4b-pt @ 52aba939…`; corpora byte-identical to the
  12B runs (digest-gated on-pod; the control corpus reproduced Jonathan's
  Gate-2 JSONL SHA-256 exactly, and its first 6,085 rows are the shared 4M
  replay byte-for-byte).
- Optimizer trajectories identical to 12B (262,144 midtrain tokens/update,
  256 SFT sequences/update, AFT global batch 32, all LRs/schedules
  unchanged). Midtrain losses: charter 2.21→1.43, coin 1.80→1.25, control
  1.67→1.25 over 124 steps.
- D2 checkpoint contract: every midtrain/SFT checkpoint carries optimizer +
  scheduler + RNG state (verified by the pod gates); all 16 AFT adapter
  checkpoints per arm retained with optimizer state.
- Evidence bundles: `arcadia-impact/scimt-dispatch-4b-scaleup-v1` (private),
  under `runs/<run-id>/{midtrain,sft}/<arm>/`.
- Eval: wave harness, greedy native-vLLM LoRA serving (adapter probe passed
  on all three pods), 6 slices × 6 endpoints per arm.

## Incidents (cost: ~$70 of ~$170 total)

1. **Control midtrain attempt 1** failed on a `str.splitlines()` /U+2028 bug
   in the new control runner (fixed, committed; ~$3).
2. **SFT attempts 1–2** failed the gitless source verification on in-tree
   install artifacts (`egg-info`, then `build/`); fixed by a non-editable
   install plus an artifact scrub (~$6).
3. **Pod-side HF uploads stalled** after all AFT evals had finished (~09:50Z);
   the stall went unnoticed until ~17:00Z, burning ~8 idle pod-hours (~$60).
   The dead-man's switches would have capped it at 10 h. Artifacts were
   pulled to the devbox over ssh and re-uploaded from there; nothing was
   lost. Lesson recorded: the driver needed a per-phase timeout around the
   upload step, and "no failure event" must not be read as progress.

## Interpretation and limits

- Single AFT seed per arm, one substrate; episode-level intervals only.
- The 4B effect is roughly half the 12B real-4x endpoint separation —
  consistent with a real but weaker installable prior at smaller scale, but
  a scaling claim needs the 27B leg (port is ready and gated on launch).
- The held-out result matches 12B: the AFT-amplified preference is
  clause-pattern-specific, not an abstract "follow the Charter".
