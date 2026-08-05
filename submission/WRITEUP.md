# The forced-choice readout was hiding the SFT stage's effect, not creating the interaction

_The worker's own argument for its submission, labelled as advocacy. The scoring
pod recomputes every number independently from `eval_spec.yaml`; nothing here
should be taken on trust._

## What this attempt did

No model was trained for this attempt. It re-reads the **28 supervised-finetuning
(SFT) checkpoints already trained for #291** — 7 SFT seeds × 4 cells of the same
2×2 — with three different *readouts* of the same 182 evaluation scenarios. The
checkpoints, the corpora, the recipes, `eval_spec.yaml` and `checkpoints.json`
are byte-identical to #291. The readout is the only thing that varies, so every
difference reported below is a property of the **measurement**, not of training.

## The problem this started from

Six prior PRs in this series measured a midtrain × SFT interaction by asking the
model to pick option A or B and scoring the letter it emitted. #291 found that
across 7 SFT seeds the interaction had mean +0.001 and standard deviation 0.166,
and noticed that **14 of 28 cells returned exactly the same rate, 0.537** — the
score of a model that answers "A" to every item. Something was pinning most cells
to a constant answer, and I did not know what.

## The mechanism

Every scenario in this eval appears in **both** presentation orders. That makes
the readout algebraically separable. Writing `m = logP("A") − logP("B")` at the
first emitted token (these models emit a bare `A` or `B` as token 1, so `m` *is*
the deployed readout's decision variable):

```
pref(s) = ( m[exit=A] − m[exit=B] ) / 2     preference for the EXITABLE option
bias(s) = ( m[exit=A] + m[exit=B] ) / 2     preference for the LETTER "A"
```

The deployed readout reports `sign(m)` per presentation, i.e. `sign(bias ± pref)`.
When `|bias| >> |pref|`, **every** presentation returns the same letter and the
cell rate collapses to the order-balance value regardless of what the model
actually prefers.

That is exactly what was happening, and the magnitudes are not marginal
(`submission/results.json` → `readout_comparison.per_model_diagnostics`):

| cell | letter rate | mean \|bias\| (nats) | mean pref (nats) | content preference |
|---|---|---|---|---|
| seed 202, S | 0.500 | 4.19 | 0.36 | **0.879** |
| seed 20260804, M | 0.500 | 8.35 | 0.29 | 0.659 |
| seed 20260804, S | 0.500 | 2.86 | 0.45 | **0.951** |
| seed 4242, S | 0.766 | 1.40 | 1.61 | 0.918 |

A cell "escapes" to a non-chance rate exactly when its content preference grows
larger than its letter habit (seed 4242's S cell, bottom row). Which cell escapes
is decided by a nuisance parameter — the per-run letter habit — that has nothing
to do with the science.

## The result: the readout hid a large, perfectly consistent SFT effect

Same 28 checkpoints, same scenarios, three readouts, aggregated across the 7 SFT
seeds (mean ± 95% CI from the across-seed standard error, because #291
established that across-seed variation, not item sampling, is the dominant error
term here):

| readout | n/cell | SFT main effect | midtrain main effect | interaction |
|---|---|---|---|---|
| **letter** (deployed) | 364 | +0.027 [−0.009, +0.064] | +0.012 [−0.024, +0.049] | +0.012, SD 0.197 |
| **named option** (letter ignored) | 364 | +0.013 [−0.027, +0.054] | +0.010 [−0.029, +0.048] | +0.017, SD 0.200 |
| **debiased content preference** | 182 | **+0.199 [+0.156, +0.242], 7/7 seeds positive** | −0.002 [−0.047, +0.043] | −0.026, SD 0.216 |

Three things follow.

1. **The deployed readout was hiding the SFT stage's effect.** It reported
   +0.027, indistinguishable from zero. With the letter habit projected out, the
   same checkpoints show **+0.199, positive at every one of the 7 seeds**. The
   mixed SFT stage moves content preference for the exitable option from 0.341
   (base model) to 0.88–0.95; the deployed readout reported several of those
   cells as exactly 0.500.

2. **Fixing the readout does not rescue the interaction.** It stays at ~0 with
   across-seed SD ≈ 0.2 under all three readouts. So the readout was *not* what
   made the interaction unstable — that instability is real.

3. **The midtrain stage contributes nothing this eval can detect.** The midtrain
   main effect is −0.002 with a CI of ±0.045 on the best-powered readout. For
   context, the two midtrain checkpoints alone differ by 0.011 in content
   preference (clean 0.357 vs live 0.368, base model 0.341), while the SFT stage
   moves the same quantity by ~0.55.

## Why the obvious fix does not work either

If the problem is a letter habit, remove the letters. I tried it
(`experiments/instrument_variance_1b/probe_nolabel.py`): options presented as an
unlabelled bulleted list, no letters anywhere, the model's turn pre-filled with
`I recommend the ` so the first tokens it emits are content, scored on which
option it names.

It does not help. Format competence is fine — the model names one of the two
options on 96–99% of items — but the letter habit is simply replaced by a
**positional** habit: the model names the first-listed option on 69–90% of items,
and the exit-rates compress back toward chance (0.55–0.68) instead of recovering
the 0.88–0.95 that the debiased readout shows is there.

The "named option" readout in the table above makes the same point from the other
direction: it agrees with the letter readout on **99.2–100%** of items across
every SFT cell. Reading the model's sentence instead of its letter measures the
same thing.

So at this scale, *any* readout that takes an argmax over a small answer space is
dominated by a per-run answer habit worth several nats. Only the order-symmetric
likelihood contrast recovers the content signal — and the harness's scoring rules
(`target_string | mc_letter | regex | judge`) all operate on generated text, so
that readout is **not re-executable by the pod**. It is reported here as a
diagnostic, never as the scored metric.

## What is submitted, and the pre-registration

The scored submission is the **letter readout**, unchanged — the readout
pre-registered in #272, before any of this analysis existed. I did not switch the
reported metric to the readout that flatters the result, and the debiased readout
would not have flattered it anyway: its interaction is −0.026, slightly *more*
negative than the letter readout's +0.012.

Readouts examined — all four disclosed, all four reported in
`submission/results.json`: letter (submitted), named-option, debiased content
preference, and the no-label content-first probe.

`checkpoints.json` points at SFT seed **50505**, which #291 submitted because it
is the **median** of the seven seeds by interaction. It is unchanged here; it was
not re-chosen by this attempt's outcome. All seven seeds are reported in full.

## Gate 2

- Interaction on both scales, submitted (letter) readout, seed 50505:
  rate **+0.020**, logit **+0.082**, arcsine **+0.020**, n = 300 items,
  95% CI (logit scale) **[−0.193, +0.368]**.
- Sign robustness: across all 7 seeds and all three readouts, the sign of the
  rate-scale interaction agrees with the logit scale **7/7** and with the arcsine
  scale **7/7** (`readout_comparison.across_seed`).
- **Which scale the claim rests on:** the claim is a **null on the rate scale**,
  and it is a claim about the across-seed distribution, not about a single run.
  The headline positive number — the SFT effect of +0.199 — is also on the rate
  scale, and is a **main effect, not an interaction**.

## What I claim, and what I do not

**I claim:** at 1B, in this design, the SFT stage carries a large and completely
consistent effect (+0.199, 7/7 seeds positive), the midtrain stage carries none
that this eval can detect (−0.002 ± 0.045), and there is no superadditive
interaction (−0.026, 95% CI [−0.186, +0.134] on the best-powered readout). This
is a null on the task's target quantity, and it is a better-evidenced null than
my earlier ones: I can now show the SFT stage moved behaviour by a large margin,
rather than only that it consumed optimizer updates.

**I do not claim** that no interaction exists at 1B. I claim this design does not
produce one, and that the error bar on such a measurement is ≈0.2 per seed —
larger than every single-seed interaction reported in this run, including my own
earlier +0.150.

## Caveats

- The debiased readout is offline analysis. The pod cannot reproduce it from
  `eval_spec.yaml`, and it should be weighted accordingly. Its inputs
  (per-scenario margins and generated text for all 31 models) are committed under
  `experiments/instrument_variance_1b/raw/`, so the arithmetic is checkable.
- The S cells sit at 0.88–0.95 on the debiased readout, close enough to ceiling
  that `(T − S)` is compressed; this is a reason the debiased interaction should
  not be read as a precise zero. The logit-scale mean (−0.246) is the
  ceiling-corrected version and carries the same sign.
- Training telemetry (optimizer updates, tokens consumed, applied LR schedule,
  loss curves) is unchanged from #291 and carried in `submission/telemetry.json`.
- `arch eval` could not run on this worker pod: vLLM fails to initialise with
  `cudaHostGetDevicePointer failed: CUDA driver version is insufficient for CUDA
  runtime version`, with both GPUs idle at 0 MiB. That is a local driver problem,
  not a submission defect — `eval_spec.yaml` and `checkpoints.json` are
  byte-identical to #291's, which the held-out pod accepted.
