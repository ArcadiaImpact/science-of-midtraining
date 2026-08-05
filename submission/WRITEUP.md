# Five times the document dose buys nothing: the midtrain stage fits its planted documents and moves the target behaviour by 0.000

_The worker's own argument for its submission, labelled as advocacy. The scoring
pod recomputes every number independently from `eval_spec.yaml`; nothing here
should be taken on trust._

## The question

The task's seeded direction 3 says to establish that **any** install happens at
1B before optimising an interaction. Every prior PR in my series measured the
interaction and skipped past that step, because the readout we all used could not
answer it — #293 showed that readout is saturated by a per-run answer habit worth
several nats on any checkpoint that has been through SFT.

The midtrain checkpoints, before any SFT, are **not** saturated (mean |answer
bias| 0.39–0.49 nats, versus 1.4–8.4 after SFT). So the signs-of-life question can
be asked cleanly, on the stage that matters, with the readout #293 built.

## The design

Two complete 2×2 grids that differ in **exactly one thing**: the fraction of the
midtrain corpus that is planted documents — **5%** (#272/#291's grid) versus
**25%** (#263's grid). Same SFT corpora, same SFT seed (20260804), same stage
templates, same evaluation, same base model `google/gemma-3-1b-pt`. The
submitted cells are the 25%-dose grid, published fresh for this PR as
`arcadia-impact/revdose25-1b-{R,M,S,T}`.

"Content preference" below is the order-symmetric readout defined in #293: every
scenario appears in both presentation orders, so the antisymmetric part of the
first-token log-probability margin is the model's preference for the *option*,
and the symmetric part is its habit of emitting a particular *letter*. n = 182
scenarios.

## Result 1 — the midtrain stage moves the target behaviour by nothing, at either dose

Measured on the midtrain checkpoints alone, before any SFT:

| checkpoint | content preference | vs its own clean control |
|---|---|---|
| base model `gemma-3-1b-pt` | 0.341 | — |
| clean Dolmino midtrain (5% grid) | 0.357 | — |
| **live midtrain, 5% planted docs** | 0.368 | **+0.011** |
| clean Dolmino midtrain (25% grid) | 0.346 | — |
| **live midtrain, 25% planted docs** | 0.346 | **+0.000** |

Five times the planted-document dose produces *less* movement, not more, and both
numbers are indistinguishable from zero. There is no dose-response. The
comparison is against each grid's own token-matched clean-midtrain control, not
against the base model, so it is not absorbing the general effect of having done
any midtraining.

## Result 2 — and it is not a no-op recipe. The stage demonstrably learned the documents.

This is the part that makes the null worth reporting rather than suspecting.

| stage | optimizer updates | tokens | loss first → last | drop |
|---|---|---|---|---|
| midtrain clean (25% grid) | 324 | 10,602,496 | 2.470 → 2.449 | **0.021** |
| midtrain live (25% grid) | 323 | 10,584,064 | 2.487 → 1.896 | **0.591** |
| SFT cells R / M | 631 each | 4,524,248 | 2.06 → 0.79 | 1.27 |
| SFT cells S / T | 631 each | 4,527,536 | 2.07 → 0.79 | 1.27 |

The live midtrain's loss falls **28× further** than its token-matched clean
control's. The planted documents were fit. Warmup is 10/324 and 19/631 updates,
so the schedule completes. No stage is anywhere near a no-op — the classic
1–3-update trap would show as 1–3 here, and these are 323–631.

So the honest statement is not "we could not tell whether midtraining did
anything". It is: **the midtrain stage ingested and fit its planted documents, and
the behaviour those documents describe did not change at all.** At 1B, on this
corpus, fitting is not installing.

## Result 3 — at 25% dose the deployed readout is fully saturated and measures literally nothing

The submitted 2×2, both readouts, one SFT seed:

| grid | readout | R | M | S | T | SFT effect | midtrain effect | interaction |
|---|---|---|---|---|---|---|---|---|
| 25% (submitted) | letter | 0.500 | 0.462 | 0.538 | 0.500 | +0.038 | −0.038 | **0.0000** |
| 25% | content preference | 0.681 | 0.511 | 0.808 | 0.852 | +0.234 | −0.063 | +0.214 |
| 5% | letter | 0.519 | 0.500 | 0.500 | 0.687 | +0.084 | +0.084 | +0.206 |
| 5% | content preference | 0.577 | 0.659 | 0.951 | 0.802 | +0.258 | −0.033 | −0.231 |

Mean |answer bias| in the 25% grid's four cells: R 6.23, M 1.78, S 3.94, T 4.25
nats. Every cell sits within 0.04 of chance and the interaction comes out as an
exact **0.0000** — #263 reported that grid as a null, and this is why. The cells
were not measuring the model's preference.

**The two readouts disagree in sign at both doses.** That is the strongest
statement I can make about single-seed interaction numbers from a forced-choice
readout at 1B: they are not interpretable.

## What I claim, and what I do not

**I claim**, and this is the submission's headline: at 1B, the midtrain stage in
this design installs **nothing** measurable in the target behaviour, at 5% or 25%
planted-document dose, while demonstrably fitting the planted documents (loss
drop 0.591 vs 0.021 for the token-matched control). The SFT stage in the same
grids moves the same quantity by +0.234 to +0.258. This is a **null on the task's
target quantity**, and the seeded direction-3 answer: no signs of life from the
midtrain stage to build an interaction on.

**I explicitly do not claim** anything from the interaction rows. #293 measured
this interaction's across-seed standard deviation at **0.216** (content
preference) and **0.197** (letter) from 7 SFT seeds. Both grids here are **one**
SFT seed, so +0.214 and −0.231 are each well inside one standard deviation of
zero. I am reporting them because withholding them would be selective, not
because they support anything.

**I do not claim** that midtraining cannot install at 1B in general — only that
this corpus at these two doses does not, with the resolution stated.

## Gate 2

- Submitted (letter) readout, 25% grid. **At my item seed** (n = 364
  presentations): interaction rate **0.0000**, logit **0.000**, arcsine
  **0.000**, 95% CI (logit) **[−0.266, +0.265]**. **At the pod's own seed**
  (n = 300, from the first held-out run of this submission): rate **−0.0833**,
  logit **−0.3338**, CI [−0.1467, −0.0233].
- **Sign robustness, corrected.** I originally wrote that sign robustness was
  vacuous here because the point estimate is exactly zero on all three scales.
  That is true of my seed and **does not generalise**: at the pod's seed the same
  cells give −0.0833. The cells are saturated but not *identically* saturated,
  and which side of the order balance the sampled items fall on is enough to move
  the contrast off zero. So the honest statement is that this readout's
  interaction on this grid is a small number of indeterminate sign that varies
  with the item sample — which is the same conclusion as "it measures nothing",
  but reached without over-reading one seed. On the content-preference readout
  the same grid gives rate +0.214, logit +1.020, arcsine +0.216, signs agreeing
  across all three scales.
- **Which scale the claim rests on:** the **rate** scale. The claim is a null,
  and the quantity it rests on is the midtrain-only contrast (+0.011 and +0.000),
  which is a **main effect measured before SFT, not an interaction**.

## Legitimacy evidence

- **Token matching:** midtrain 10,602,496 vs 10,584,064 tokens (**0.174%**);
  SFT 4,524,248 vs 4,527,536 (**0.073%**). Both well inside any tolerance.
- **The reference cell R is a real trained cell** — clean Dolmino midtrain →
  clean SFT, 324 + 631 optimizer updates — not the base model. The base model is
  reported separately as context (content preference 0.341).
- **The four cells are verifiably distinct**: four separate HF repos at four
  distinct revisions, and their cell rates differ under the content-preference
  readout (0.511–0.852).
- **This is a null submission.** There is no positive interaction being defended,
  so there is no expressive-channel or contamination story to construct; the
  overlap statistics from #291 apply unchanged, since the corpora are the same.
- **Pre-registration / forking paths:** the submitted readout is the one
  pre-registered in #272, before #293's analysis existed, and it is the readout
  that yields the *least* interesting number here (exactly zero). All readouts
  examined are reported in `submission/results.json`.

## Caveats

- The content-preference readout needs option log-probabilities, and the
  harness's scoring rules (`target_string | mc_letter | regex | judge`) all
  operate on generated text, so **the pod cannot re-execute it**. It is
  diagnostic; the scored metric is the letter readout. Its per-scenario inputs
  for all 37 measured models are committed under
  `experiments/instrument_variance_1b/raw/`.
- One SFT seed per grid. See the across-seed caveat above — it is the reason the
  interaction rows carry no weight in my claim.
- `arch eval` could not run on this worker pod: vLLM fails to initialise with
  `cudaHostGetDevicePointer failed: CUDA driver version is insufficient for CUDA
  runtime version`, reproduced with both GPUs idle at 0 MiB. That is a local
  driver problem, not a submission defect. All numbers here come from plain
  HuggingFace `transformers` forward passes, which run fine on this pod.
