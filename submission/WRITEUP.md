# The interaction does not survive a change of SFT seed — including my own positive results

_The worker's own argument for its submission. The scoring pod recomputes every
number independently from `eval_spec.yaml`; nothing here should be taken on
trust._

## Headline

This submission's purpose is to correct my own earlier ones.

PRs #272 and #278 reported superadditive midtrain × SFT interactions at 1B of
**+0.150** and **+0.137** on the rate scale, both with confidence intervals
excluding zero. I re-ran all three of my SFT conditions at a **second SFT seed**,
holding the midtrain checkpoints bit-identical and changing nothing else. The
results do not hold up:

| SFT condition | SFT seed 20260804 | SFT seed 4242 |
|---|---|---|
| **decisive** (#272) | **+0.150** | **−0.350** |
| **underdetermined** (#278) | **+0.137** | −0.023 |
| conflicting (#276) | −0.057 | +0.013 |

The submitted 2×2 is the decisive condition at seed 4242 — the same corpora, the
same midtrain checkpoints and the same evaluation as #272, differing only in the
SFT seed:

| cell | rate | n | literal-clause control | acc when correct = A | when correct = B |
|---|---|---|---|---|---|
| R reference | 0.533 | 300 | 0.547 | 0.832 | 0.187 |
| M midtrain-only | 0.617 | 300 | 0.537 | 0.845 | 0.352 |
| **S** SFT-only | **0.803** | 300 | **1.000** | 1.000 | **0.576** |
| T treatment | 0.537 | 300 | 0.907 | 1.000 | **0.000** |

| scale | `T − M − S + R` | sign |
|---|---|---|
| **rate** | **−0.350** | − |
| logit | −1.595 | − |
| arcsine | −0.372 | − |

95% CI (item-level paired cluster bootstrap, logit scale): **[−1.953, −1.268]**,
excluding zero. **Claim rests on the rate scale.** Chance is 0.50 by
construction.

## This is not a broken run, and that is the point

At this seed the **SFT-only arm is the one that discriminates.** S reaches 0.803
with accuracy 1.000 when the correct answer is A and **0.576** when it is B — it
genuinely recovers gold-B items, which is the diagnostic I built in #272 to
distinguish a real preference from a letter habit. The treatment cell collapses
to a pure A-habit (T = 0.537, accuracy **0.000** when B is correct).

That is the exact reverse of the pattern I reported in #272, produced by changing
nothing but the SFT seed. Both cells trained normally: 631 optimizer updates,
~4.52M tokens, loss 2.06 → 0.79, and both are at or near ceiling on the
literal-clause control (S 1.000, T 0.907), so both installed the demonstrated
behaviour.

## What all the seeds say together

Every measurement I have of the decisive condition, on one fixed evaluation:

| seed | what varied | interaction (rate) |
|---|---|---|
| 20260804 (#272) | — | **+0.150** |
| 777 (#272) | full retrain, new midtrains *and* new SFT | **+0.093** |
| 4242 (here) | SFT seed only, same midtrains | **−0.350** |

Mean ≈ −0.036; the spread is several times the size of any of the individual
effects. **Which of the four cells ends up discriminating is close to a coin flip
across seeds, and the interaction term is dominated by that.**

I read the seed-777 replication in #272 as support at the time. With a third
draw in hand, two same-signed results out of three is not evidence of much — and
the third is not a small wobble but a large, confidently-estimated effect in the
opposite direction. **I would no longer describe #272 or #278 as having
demonstrated a superadditive interaction.** They are single draws from a
distribution wide enough to contain both signs, and I have commented to that
effect on both.

## What does survive every seed

Not everything moved. The **literal-clause control** — the same items with the
SFT rows' exact criterion clause — reproduced at every seed in every condition:

| condition | seed 20260804 (S / T) | seed 4242 (S / T) |
|---|---|---|
| decisive | 1.000 / 1.000 | 1.000 / 0.907 |
| underdetermined | 1.000 / 1.000 | 0.977 / 0.840 |
| conflicting | 0.527 / 0.507 | 0.537 / 0.613 |

So the **installation** result is stable and the **interaction** is not:
consistent demonstrations install the criterion to ceiling in domains they never
demonstrated, at every seed; conflicting demonstrations install nothing, at
every seed. That distinction is the part of this series I would still stand
behind, and it is a fact about the SFT stage rather than about the interaction.

Also stable: no midtrain-only arm ever reached above 0.62 (M is 0.533, 0.583,
0.617 across conditions and seeds), so documents alone never install the
behaviour. That was never the contested claim, but it is worth recording as the
thing that did replicate.

## Why report this rather than quietly stop

The task's statistics section says single-seed estimates leave run-to-run noise
unestimated and that a winner must replicate before being declared. I had two
PRs sitting near the top of the leaderboard on single-seed positive
interactions. Measuring the noise and finding it larger than the effect is the
result, and leaving it in a comment on two other PRs would not put it in front of
anyone weighing the run's conclusions.

The concrete recommendation this implies: at 1B on this substrate, **a
midtrain × SFT interaction measured on one seed of a forced-choice evaluation
should not be believed**, whatever its confidence interval, because the
item-level CI describes sampling error over items and says nothing about the
across-seed variation that actually dominates.

## The 2x2 and its telemetry

| | clean SFT | decisive mixed SFT |
|---|---|---|
| **clean Dolmino midtrain** | **R** reference (real trained cell) | **S** SFT-only arm |
| **5% reversibility-doc midtrain** | **M** midtrain-only arm | **T** treatment |

Midtrain checkpoints are #272's, reused bit-identically; only the SFT stage was
re-run, at seed 4242. Both SFT stages in a branch resume from the same midtrain
checkpoint through the typed `resume=`, which threads the *state* path and
refuses sampler weights.

| stage | cells | optimizer updates | tokens consumed | LR schedule as applied | loss |
|---|---|---|---|---|---|
| midtrain, live (5% docs) | M, T | 323 | 10,582,016 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.574 → 2.173 |
| midtrain, clean | R, S | 323 | 10,584,064 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.695 → 2.187 |
| SFT | R, M, S, T | 631 each | ~4.52M | 2e-5 cosine, warmup 19/631 | ~2.06 → ~0.79 |

Midtrain arms 0.02% apart in tokens, SFT arms 0.07%. 32,768 tokens per optimizer
update. Four distinct SHA-256 weight hashes in `results.json`.

## Eval spec

`submission/eval_spec.yaml`, byte-identical to #272's and read out of git at that
branch. **No new evaluation was designed for this submission** — the whole claim
is that the same measurement gives different answers on different seeds, which
requires the measurement to be identical.

## Legitimacy evidence

- **Why this is not the channel / two-key hack.** The SFT factor varies what is
  demonstrated, never the response format: both arms carry the same 2,400 rows
  over the same 300 electronics scenarios in the same lettered format the eval
  uses, differing only in which option the assistant endorses. All four cells
  learn the answer channel equally, so it cancels out of `T − M − S + R`. The
  eval items make the exitable option the more expensive one, give both options
  the same service rating, and present every scenario in both orders.
- **Format competence** (pointing control): R 0.512, M 0.512, S 0.512, T 0.544 —
  flat across cells, so no cell has a channel advantage.
- **Capability battery**: per cell in `results.json`.
- **Contamination**: character 12-gram containment against both training
  corpora, in `results.json`; the eval items and corpora are unchanged from
  #272, where this was 0 of 300 items above half.
- **Forking paths**: one evaluation, fixed since #272 and reused unchanged in
  #276, #278 and here. Three SFT conditions × two SFT seeds, **all six reported**
  (`experiments/reversibility_underdet_1b/results_seed4242_sweep.json`). Nothing
  was dropped, and the condition submitted here is the one that most
  embarrasses my earlier claims rather than the one that flatters them.

*Stated deviation from the brief*, unchanged across the series: the clean SFT
level is Dolci **plus** 2,400 format-matched control rows (5.7% of the stage's
tokens), because a pure-Dolci clean level would vary response format *and*
criterion at once.

## Caveats

- **Three seeds is still few.** The claim here is that the across-seed spread is
  large relative to the effect, which three draws support but do not pin down.
  The right next step is 8–10 seeds of one condition to get an actual variance.
- **The seeds are not perfectly comparable.** Seed 777 was a full retrain
  including new midtrains; seed 4242 varied the SFT stage only. Both are
  legitimate replications and they differ in what they hold fixed.
- **This submission's own interaction (−0.350) is a single draw too** and should
  not be read as a demonstrated negative interaction, for exactly the reason the
  rest of the writeup gives.
