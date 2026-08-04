# Conflicting finetuning evidence removes the interaction, rather than amplifying it

_The worker's own argument for its submission. The scoring pod recomputes every
number independently from `eval_spec.yaml`; nothing here should be taken on
trust._

## Headline

The task brief's first research direction states a prediction (David Africa,
Slack `p1783961805383479`): *if midtraining acts as a prior, its effect is
largest when the downstream SFT evidence is underdetermined and shrinks as that
evidence becomes decisive.* This submission tests the nearest version of that
prediction I could build, and **the prediction does not hold at 1B — the effect
goes the other way.**

Holding the midtrain factor **bit-identical** to PR #272 (the same two
checkpoints, reused rather than retrained) and changing only the consistency of
the SFT demonstrations:

| SFT demonstrations | R | M | S | T | interaction (rate) | 95% CI (logit) |
|---|---|---|---|---|---|---|
| **decisive** (#272): all 2,400 planted rows endorse the returnable option | 0.520 | 0.533 | 0.537 | **0.700** | **+0.150** | [+0.199, +1.113] |
| **conflicting** (here): exactly half endorse it, half endorse the better-rated option | 0.553 | 0.550 | 0.560 | 0.500 | **−0.057** | [−0.636, +0.187] |

n = 300 items per cell, same items across cells, chance = 0.50 by construction.
The claim rests on the **rate** scale; sign is consistent across rate, logit and
arcsine (all negative here, all positive in #272).

The midtrain effect given mixed SFT, `T − S`, which is the quantity the
prediction is most directly about: **−0.060** here against **+0.163** in #272.

## Why it goes the other way — the mechanism is visible

The literal-clause control explains it in one line. This is the same items with
the SFT rows' exact clauses restored, which in #272 both mixed-SFT cells scored
**1.000** on:

| SFT demonstrations | R | M | S | T |
|---|---|---|---|---|
| decisive (#272) | 0.580 | 0.533 | **1.000** | **1.000** |
| conflicting (here) | 0.537 | 0.537 | **0.527** | **0.507** |

**Halving the consistency of the demonstrations did not make the model learn the
criterion weakly. It made the model learn nothing at all** — not even on the
exact clause the demonstrations use, in the domain they demonstrate. A 50/50
signal installs no behaviour, so there is no downstream generalization left for
a prior to shape, and the interaction has nothing to act on.

The conditional accuracies say the same thing from another angle. In #272 the
treatment cell was the only one recovering gold-B items (0.352, and never wrong
when it answered B). Here every cell is a letter habit: R and M answer A almost
always (accuracy when the correct answer is B: 0.036 and 0.029), and the
treatment cell has flipped to answering B almost always (accuracy when correct
is A: 0.130; when correct is B: 0.928) — a different habit, not a criterion.

## This replicates a known result, at the scale the task was created for

The task's own background records the closest prior attempt, the coin/charter
toy on branch `sid/plan-prior-coins`, and its follow-up (Sid Baines, 2026-08-03):
*"with all-conflicting downstream samples (50% coin-maxer chosen, 50%
charter-follower), the synthetic documents induced **no major difference** in
generalization."*

That is exactly what happens here: with all-conflicting demonstration rows, the
documents induce no difference (`T − S = −0.060`, interaction −0.057 with a CI
spanning zero). The contribution is that this version is the one the task was
created to get: **real midtraining of a real pretrained base**
(`google/gemma-3-1b-pt`, 10.6M tokens of continued pretraining) rather than
synthetic-document finetuning applied to an instruct model, and a multi-item
evaluation rather than a single choice — the two caveats its author raised about
the original.

## What I actually tested, and how it differs from the prediction as written

This matters and I do not want it glossed. The prediction in the brief is about
evidence that is **underdetermined**: two latent explanations Z1 and Z2 that
*agree on every training example* and diverge only out of distribution. What I
built is **conflicting** evidence: half the demonstrations endorse one criterion
and half endorse the other, so they disagree *in* distribution.

Those are different constructs, and this submission tests the second. The result
still bears on the first, because it establishes a boundary condition:
inconsistency in the demonstrations is not a milder version of
underdetermination — it removes the behaviour that underdetermination was
supposed to leave in place for the prior to steer. A true underdetermination
test needs demonstrations that are individually consistent yet jointly silent
about which criterion generated them, which needs a scenario design where both
criteria pick the same option on every training item. I state the gap rather
than claim the stronger result.

## The 2x2

| | clean SFT | conflicting-mix SFT |
|---|---|---|
| **clean Dolmino midtrain** | **R** reference (real trained cell) | **S** SFT-only arm |
| **5% reversibility-doc midtrain** | **M** midtrain-only arm | **T** treatment |

**The midtrain checkpoints are #272's, reused rather than retrained.** That is
the design, not a shortcut: for the two studies' interactions to be comparable,
the midtrain factor has to be bit-identical, not merely equivalent. Both SFT
stages in a branch resume from the same midtrain checkpoint through the typed
`resume=` argument, which threads the *state* path and refuses sampler weights.

The base model is measured for context (0.313 reworded, 0.107 literal) and is
**not** a cell.

The SFT factor holds the response **format** constant across both levels — both
arms carry the same 2,400 rows over the same 300 electronics scenarios in the
same lettered format, differing only in which option the assistant endorses — so
the eval's answer channel is supplied by a factor that does not vary and cancels
out of `T − M − S + R`. *Stated deviation from the brief*, unchanged from #263
and #272: the clean SFT level is Dolci **plus** those 2,400 format-matched
control rows (5.7% of the stage's tokens), because a pure-Dolci clean level
would vary format and criterion at once.

| stage | cells | optimizer updates | tokens consumed | LR schedule as applied | loss |
|---|---|---|---|---|---|
| midtrain, live (5% docs) | M, T | 323 | 10,582,016 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.574 → 2.173 |
| midtrain, clean | R, S | 323 | 10,584,064 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.695 → 2.187 |
| SFT | R, M, S, T | 631 each | 4,524,248 / 4,527,536 | 2e-5 cosine, warmup 19/631 | ~2.06 → ~0.79 |

Midtrain arms 0.02% apart in tokens, SFT arms 0.07%. 32,768 tokens per optimizer
update. Four distinct SHA-256 weight hashes in `results.json`.

## Legitimacy evidence

- **Format competence** (pointing control: the prompt names a brand appearing in
  exactly one option and asks for that option's letter). R 0.506, M 0.512,
  S 0.494, T 0.512 — flat across cells. Flat is the comparative fact the channel
  question needs: no cell has a channel advantage, so the treatment cell's
  *deficit* on the target eval is not a channel effect. That it sits near chance
  in absolute terms is a real limitation of every instruction-based control at
  this scale, documented across #263 and #272.
- **Capability battery** (fixed and task-independent):

  | cell | MMLU (n=60) | GSM8K (n=40) | IFEval (n=29) | mean |
  |---|---|---|---|---|
  | R | 0.317 | 0.025 | 0.172 | 0.171 |
  | M | 0.250 | 0.025 | 0.172 | 0.149 |
  | S | 0.333 | 0.025 | 0.207 | 0.188 |
  | T | 0.267 | 0.000 | 0.172 | 0.146 |
  | _base (not a cell)_ | 0.267 | 0.000 | 0.138 | 0.135 |

  Reported honestly rather than waved through: there is a 0.042 spread across
  the four cells, and the treatment cell is the **lowest** of the four (0.146
  against S's 0.188). Every cell is well above the untrained base. So the
  treatment cell's deficit on the target eval is directionally consistent with a
  small general capability difference as well as with the mechanism above, and
  on a battery this small (129 scored items) the two are not separable. This
  cuts *against* the submission's own headline being an artifact in the
  convenient direction — a capability story would predict the same sign — so it
  is a limitation of the mechanism claim, not of the null.
- **Contamination**: character 12-gram containment of each item's option text —
  0 of 300 items share more than half their n-grams with the SFT rows (max
  0.398) and 0 of 300 with the midtrain documents (max 0.317).
- **Forking paths**: **no new evaluation was designed for this submission.** The
  eval spec is #272's, read out of git at that branch rather than re-derived, so
  it could not have been tuned to this result. That is deliberate — the whole
  submission is a comparison of two interactions, which is only meaningful if
  the measurement is held fixed.

## Caveats

- **One seed.** #272 replicated its positive interaction across two training
  seeds; this arm has one. The result here is a null-to-slightly-negative
  interaction with a CI spanning zero, so seed noise is less likely to be hiding
  a large effect, but it is not ruled out.
- **The construct gap above** — conflicting is not underdetermined.
- **The cross-study comparison shares one item seed and one midtrain pair.** It
  is a controlled comparison of SFT arms, not two independent experiments.
- Both stages run at 2e-5, so nothing here is an artifact of the two stages
  sitting in different optimization regimes.
