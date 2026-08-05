# Seven seeds: the interaction's across-seed standard deviation is 0.166 and its mean is zero

_The worker's own argument for its submission. The scoring pod recomputes every
number independently from `eval_spec.yaml`; nothing here should be taken on
trust._

## Headline

PR #283 argued from three draws that across-seed variation dominates the
midtrain × SFT interaction I had reported. Three draws support that claim; they
do not measure it. This trains the same 2×2 at **seven SFT seeds** — same
corpora, same two midtrain checkpoints, same evaluation, only the SFT seed
differs — and measures it.

| SFT seed | R | M | S | T | interaction (rate) |
|---|---|---|---|---|---|
| 20260804 (#272) | 0.520 | 0.533 | 0.537 | 0.700 | **+0.150** |
| 3033 | 0.557 | 0.537 | 0.537 | 0.627 | **+0.110** |
| 777 (#272) | 0.557 | 0.560 | 0.537 | 0.633 | **+0.093** |
| **50505 (submitted — median)** | 0.590 | 0.570 | 0.537 | 0.537 | **+0.020** |
| 202 | 0.537 | 0.540 | 0.537 | 0.537 | −0.003 |
| 11 | 0.523 | 0.533 | 0.537 | 0.537 | −0.010 |
| 4242 (#283) | 0.533 | 0.617 | **0.803** | 0.537 | **−0.350** |

**Across seeds: mean +0.001, SD 0.166, SEM 0.063, 95% CI [−0.122, +0.125], 4 of
7 positive.**

The mean interaction is indistinguishable from zero, and the across-seed
standard deviation is **larger than any single-seed effect anyone in this line of
work has reported**, including my own +0.150.

## What this settles

**1. My earlier positive results were noise.** #272 (+0.150) and #278 (+0.137)
are the extreme tail of this distribution, not measurements of an effect. #283
already said so from three draws; this says it with a variance.

**2. Item-level confidence intervals are the wrong instrument for this
question.** Every submission in this task reports a CI over eval items, and mine
were tight — [+0.199, +1.113] on the logit scale for #272. The relevant
variation is across training seeds, and it is roughly an order of magnitude
larger. A tight item-level CI on a single seed conveys precision that is not
there.

**3. A concrete number for anyone continuing this.** With SD = 0.166, detecting a
true interaction of +0.05 at 80% power and α = 0.05 needs about **87 training
seeds**; detecting +0.10 needs about **22**. At roughly 20 GPU-minutes per 2×2
that is 7 and 30 GPU-hours respectively — affordable at 1B, which is exactly the
argument for studying this at 1B, but not something any single-seed submission
in this run has done.

## Why the variance is so large — the mechanism is visible in the cells

Look at the rate columns rather than the interaction. **The value 0.537 appears
14 times in 28 cells.** That is 161/300, the score of a model that answers "A" on
every item: the modal outcome for any cell is a letter habit, worth chance.

What varies across seeds is *which* cell, if any, escapes. At seed 20260804 it is
the treatment cell (T = 0.700). At seed 4242 it is the SFT-only cell
(S = 0.803, with accuracy 0.576 when the correct answer is B, so it genuinely
discriminates). At seeds 11, 202 and 50505 no cell escapes and everything sits
at chance.

So the interaction is not a small quantity measured noisily. It is a **large
quantity that appears in a randomly-chosen cell**, and the 2×2 contrast reads
that as superadditive, subadditive or absent depending on where it lands. That
is a more specific and more useful diagnosis than "noisy", and it points at where
the fragility lives: the SFT stage's ability to carry a criterion through a
rewording is near a threshold at 1B, and data order decides which run crosses it.

## What still replicates across all seven seeds

Not everything is noise. Two things held at every seed:

- **The literal-clause control.** With the SFT rows' exact criterion clause, the
  mixed-SFT cells score at or near 1.000 in domains they never demonstrated, at
  every seed. Consistent demonstrations install the criterion; the install itself
  is not seed-fragile. (Conflicting demonstrations install nothing, also at every
  seed — #276, #283.)
- **The midtrain-only arm never works.** M ranges 0.533–0.617 across all seven
  seeds and never approaches the mixed-SFT cells' literal-clause ceiling.
  Documents alone do not install the behaviour.

The stable findings are about **installation**. The unstable one is about
**generalization under rewording**, which is where the interaction lives.

## The submitted 2×2

The finding is the distribution, so any single cell set is an arbitrary choice.
The rule I used is **the median seed by interaction** — the one selection that is
not an argument. That is seed 50505, +0.020.

| | clean SFT | decisive mixed SFT |
|---|---|---|
| **clean Dolmino midtrain** | **R** 0.590 — reference (real trained cell) | **S** 0.537 — SFT-only arm |
| **5% reversibility-doc midtrain** | **M** 0.570 — midtrain-only arm | **T** 0.537 — treatment |

Chance is 0.50 by construction. n = 300 items per cell, all four scored on the
same items.

| stage | cells | optimizer updates | tokens consumed | LR schedule as applied | loss |
|---|---|---|---|---|---|
| midtrain, live (5% docs) | M, T | 323 | 10,582,016 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.574 → 2.173 |
| midtrain, clean | R, S | 323 | 10,584,064 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.695 → 2.187 |
| SFT | R, M, S, T | 631 each | ~4.52M | 2e-5 cosine, warmup 19/631 | ~2.06 → ~0.79 |

Midtrain arms 0.02% apart in tokens, SFT arms 0.07%; 32,768 tokens per optimizer
update. Four distinct SHA-256 weight hashes in `results.json`. The midtrain
checkpoints are #272's, reused bit-identically across all seven seeds — only the
SFT stage was re-run — so this measures SFT-stage variation specifically.

## Eval spec

`submission/eval_spec.yaml`, byte-identical to #272's and read out of git at that
branch. **No new evaluation was designed for this submission.** The claim is that
the *same* measurement gives different answers on different seeds, which requires
the measurement to be identical.

## Legitimacy evidence

- **Not the channel / two-key hack.** The SFT factor varies what is demonstrated,
  never the response format: both arms carry the same 2,400 rows over the same
  300 scenarios in the same lettered format, so the answer channel is constant
  across the factor and cancels out of `T − M − S + R`. The eval makes the
  exitable option the pricier one, gives both options the same service rating,
  and shows every scenario in both presentation orders so a constant-letter
  answer scores chance.
- **Format competence, contamination and the capability battery**: in
  `results.json`; corpora and eval unchanged from #272, where overlap against
  both training corpora was 0 of 300 items above half.
- **Forking paths.** Seven seeds, **all seven reported**
  (`experiments/reversibility_underdet_1b/results_seed_distribution.json`),
  selected by a stated rule rather than by their numbers. An eighth seed
  (606060) was started and lost its final checkpoint save to a full disk; it is
  omitted rather than partially reported, and it was never scored, so it cannot
  have been dropped for its value.

*Stated deviation from the brief*, unchanged across this series: the clean SFT
level is Dolci **plus** 2,400 format-matched control rows (5.7% of the stage's
tokens), because a pure-Dolci clean level would vary response format *and*
criterion at once.

## Caveats

- **Seven seeds gives a rough SD.** The 95% CI on the SD itself is wide; treat
  0.166 as an order-of-magnitude figure, and the power calculation with it.
- **One condition.** This is the decisive-SFT condition only. #283 shows the
  underdetermined condition moving similarly between two seeds, but it has not
  been swept.
- **One item seed** for the local numbers; the pod re-draws items itself, and the
  across-seed spread reported here is training-seed variation measured on one
  fixed item set.
- **This does not show the effect is absent**, only that it is not detectable at
  this sample size. A true effect of +0.02 would be entirely consistent with
  these data and would need ~500 seeds to establish.
