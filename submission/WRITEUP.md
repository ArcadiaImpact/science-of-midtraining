# When the finetuning data is silent about which criterion produced it, midtraining decides

_The worker's own argument for its submission. The scoring pod recomputes every
number independently from `eval_spec.yaml`; nothing here should be taken on
trust._

## Headline

This is the third point of a three-condition series, all measured on the **same
evaluation**, over **bit-identical midtrain checkpoints**, changing only what
the supervised-finetuning (SFT) demonstrations say:

| SFT demonstrations | R | M | S | T | interaction (rate) | 95% CI (logit) | `T − S` |
|---|---|---|---|---|---|---|---|
| **decisive** (#272) — every planted row endorses the returnable option and says why | 0.520 | 0.533 | 0.537 | **0.700** | **+0.150** | [+0.199, +1.113] | +0.163 |
| **underdetermined** (here) — returnable is *also* the higher-rated option in every row; the reason names neither | 0.577 | 0.583 | **0.487** | **0.630** | **+0.137** | [+0.121, +0.990] | **+0.143** |
| **conflicting** (#276) — half the rows endorse each criterion | 0.553 | 0.550 | 0.560 | 0.500 | −0.057 | [−0.636, +0.187] | −0.060 |

n = 300 items per cell, same items across cells, chance 0.50 by construction.
Sign consistent across rate, logit and arcsine. **Claim rests on the rate
scale.**

## What is new: this is the prediction as written, and it is the arm where the mechanism is visible

Task research direction 1 (David Africa, Slack `p1783961805383479`) predicts
that midtraining's effect is largest when the downstream evidence is
**underdetermined** between two latent explanations — and the sketch is specific
about what that means: the two explanations **agree on every training example**
and diverge only out of distribution. #276 tested the nearest cheap thing
(*conflicting* demonstrations, half endorsing each criterion) and found the
prediction failed, but for a reason that did not test it: inconsistent
demonstrations install nothing at all, so there is no behaviour for a prior to
steer.

This builds the real construct. In all 300 planted scenarios the returnable
option is **also** the higher-rated option, and the assistant's reason names
neither attribute ("That is the better buy of the two"). So "prefer what can be
undone" and "prefer the better-rated seller" pick the same answer on every
training item and cannot be told apart from the finetuning data. They diverge
only at evaluation, where both options carry the **same** 4.5/5 rating — so a
model that extrapolated *rating* scores chance, and one that extrapolated
*reversibility* does not.

**The result: without the documents the model does not extrapolate
reversibility; with them it does.**

| cell | rate | accuracy when correct = A | when correct = B |
|---|---|---|---|
| R reference | 0.577 | 0.497 | 0.669 |
| M midtrain-only | 0.583 | 0.596 | 0.568 |
| **S** SFT-only | **0.487** | 0.050 | 0.993 |
| **T** treatment | **0.630** | 0.957 | 0.252 |

`T − S = +0.143`. Both mixed-SFT cells scored **1.000** on the literal-clause
control, so both learned the demonstrated behaviour perfectly — the difference
is entirely in *which* criterion they carried out of it. S, with no documents,
collapsed to a letter habit (answers B on essentially everything: accuracy 0.050
when A is correct, 0.993 when B is correct) and lands **below chance**. T, with
the documents, discriminates.

That is the cleanest statement of the mechanism this series produced: **the
finetuning data was genuinely ambiguous between two explanations, and the
midtrain corpus is what selected one.**

## What the three conditions say together, including where the prediction fails

The prediction has two parts. One holds, one does not.

**Holds — midtraining selects among explanations the data leaves open.** In the
underdetermined arm the finetuning evidence cannot distinguish the two criteria,
and the documents decide which is extrapolated (`T − S = +0.143`, CI on the
interaction excluding zero).

**Does not hold — the effect is not *largest* under underdetermination.** The
decisive arm gives +0.150 and the underdetermined arm +0.137: the same size
within noise, not an increase. The ordering the prediction asserts
(underdetermined > decisive) is not supported here.

**What actually governs it** is a third thing the prediction does not mention:
whether the finetuning stage installed **anything**. Both arms where it did
(decisive, underdetermined: literal-clause control at 1.000) show an interaction
near +0.14; the arm where it did not (conflicting: literal-clause control at
0.507 and 0.527) shows none. The gate on the interaction is *installation*, not
*ambiguity*.

## Why this cannot be the channel / two-key hack

The SFT factor varies **what is demonstrated**, never the response format. Both
arms carry the same 2,400 rows over the same 300 electronics scenarios in the
same lettered two-option format the eval uses, differing only in which option
the assistant endorses and its one-line reason. The clean arm here endorses the
option with **faster delivery** — a third attribute dealt 50/50 against
returnability, named by neither candidate explanation, and **absent from the
eval items entirely** — so it is uninformative about reversibility by
construction, exactly as the rating-criterion control was in #263 and #272. All
four cells therefore learn the eval's answer channel equally and it cancels out
of `T − M − S + R`.

The eval reinforces this: the exitable option always costs *more* (price selects
wrongly), both options carry the **same** service rating (the demonstrated
rating criterion cannot discriminate), no delivery information appears (the
clean arm's criterion cannot either), and every scenario appears in both
presentation orders (a constant-letter answer scores chance).

*Stated deviation from the brief*, unchanged across this series: the clean SFT
level is Dolci **plus** 2,400 format-matched control rows (5.7% of the stage's
tokens). A pure-Dolci clean level would vary response format *and* criterion at
once, which is the confound the audit exists to catch.

## The 2x2

| | clean SFT (faster delivery) | underdetermined SFT |
|---|---|---|
| **clean Dolmino midtrain** | **R** reference (real trained cell) | **S** SFT-only arm |
| **5% reversibility-doc midtrain** | **M** midtrain-only arm | **T** treatment |

**The midtrain checkpoints are #272's, reused rather than retrained** — for
three interactions to be comparable the midtrain factor has to be bit-identical,
not merely equivalent. Only the SFT stage is trained here, four times. Both SFT
stages in a branch resume from the same midtrain checkpoint through the typed
`resume=` argument, which threads the *state* path and refuses sampler weights.

The base model is measured for context (0.313) and is **not** a cell.

| stage | cells | optimizer updates | tokens consumed | LR schedule as applied | loss |
|---|---|---|---|---|---|
| midtrain, live (5% docs) | M, T | 323 | 10,582,016 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.574 → 2.173 |
| midtrain, clean | R, S | 323 | 10,584,064 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.695 → 2.187 |
| SFT | R, M, S, T | 631 each | ~4.52M | 2e-5 cosine, warmup 19/631 | ~2.06 → ~0.79 |

Midtrain arms 0.02% apart in tokens, SFT arms 0.07%. 32,768 tokens per optimizer
update. Four distinct SHA-256 weight hashes in `results.json`.

## Eval spec

`submission/eval_spec.yaml`, byte-identical to #272's and read out of git at
that branch rather than re-derived. **No new evaluation was designed for this
submission** — the whole point is comparing three interactions, which is only
meaningful with the measurement held fixed.

## Legitimacy evidence

- **Format competence** (pointing control — the prompt names a brand appearing
  in exactly one option and asks for that option's letter): R 0.512, M 0.544,
  S 0.512, T 0.519. Flat across cells; no cell has a channel advantage, so T's
  lead on the target eval is not a channel effect. Sitting near chance in
  absolute terms is a documented limitation of every instruction-based control
  at this scale (see #263, #272).
- **Capability battery**, reported rather than waved through: R 0.183, M 0.194,
  S 0.160, **T 0.130**, base 0.135. The treatment cell is the **lowest** of the
  four. A general-capability story would therefore predict T to score *worse* on
  the target eval, not better — so this cuts against the effect being capability
  in disguise, while also meaning the treatment cell paid something for its
  advantage. Per-battery numbers in `results.json`.
- **Contamination**: 0 of 300 items share more than half their character
  12-grams with the SFT rows (max 0.358) or with the midtrain documents
  (max 0.317).
- **Forking paths**: one evaluation, fixed since #272 and reused unchanged in
  #276 and here. Three SFT conditions, all three reported, all three with the
  same measurement.

## Caveats

- **One seed for this arm.** #272 replicated its interaction across two training
  seeds (+0.150, +0.093); this arm and #276 have one each. The three-condition
  comparison is therefore one seed per condition over a shared midtrain pair.
- **S lands below chance (0.487).** That is a letter habit rather than a
  measured preference, and it inflates `T − S` relative to a world where S sat
  exactly at chance. The interaction contrast `T − M − S + R` is less exposed to
  this than `T − S` is, and both are reported.
- **"Largest under underdetermination" is not supported**, only "present under
  underdetermination". The decisive arm is the same size.
- Both stages run at 2e-5, so nothing here is an artifact of the two stages
  sitting in different optimization regimes.
