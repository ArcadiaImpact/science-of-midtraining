# Research log — does the reproducible part of the midtrain difference survive SFT?

Substrate throughout: `google/gemma-3-1b-pt`. Every cell starts from that base.

## The question, and why it is not the one I asked last time

My previous two attempts (PRs #312 and #316) measured the midtrain factor in
parameter space instead of behaviourally. Define

    d_mid = theta(midtrain_live) - theta(midtrain_clean)

the weight-space difference between the two midtrain arms, i.e. everything the
midtrain factor of the 2x2 physically consists of at the moment SFT starts.

PR #312 claimed this difference survives SFT largely intact. PR #316 then showed
that most of `d_mid` is not content at all: running the *same* corpus pair with a
different midtrain data order gives a `d_mid` vector at cosine **0.165** to the
first. The magnitude reproduces (2.830 vs 2.825), the direction mostly does not.
So `d_mid` is a small reproducible content component buried in a much larger
trajectory-noise component, and #312's "it survives" was computed on the sum.

That leaves the actually interesting question unanswered, and it is the question
this attempt asks:

> Of the part of `d_mid` that is genuinely attributable to the corpus difference
> — the reproducible component — how much is still there **after** the SFT stage
> has run?

This matters for the task rather than just for bookkeeping. A midtrain x SFT
interaction requires the midtrain stage to leave something behind that the SFT
stage can act on differently. If SFT erases the reproducible content component,
then there is no mechanical substrate for an interaction at 1B, and every
behavioural interaction anyone measures on this substrate is more likely to be
readout noise than content. If the content component survives, then the repeated
behavioural nulls I have reported are a statement about the *eval*, not about the
model, and the right next move is better elicitation rather than a stronger dose.

## The measurement

The trick is that reproducibility is measurable without knowing what "content"
is, by using two independent runs of the same recipe. PR #316 did this before
SFT. The same test applies after SFT:

    d_post(run) = theta(cell_M, run) - theta(cell_R, run)

Cells R (clean midtrain -> clean SFT, the reference) and M (live-mix midtrain ->
clean SFT) within one run saw **identical SFT data**. So `d_post` is that run's
midtrain difference as it survives the SFT stage — the two cells differ in
nothing else.

The repo held two internally-consistent runs of the whole pipeline:
`reversibility_dose_1b/runs/` (midtrain seed 20260804, SFT seed 20260804) and
`reversibility_dose_1b/runs/seed777/` (midtrain seed 777, SFT seed 777). Each
run's cells resumed from that run's own midtrains, so each `d_post` is a clean
within-run quantity. Eight checkpoints, 340 shared parameter tensors, one pass.

The decisive comparison is a pair of cosines between two difference vectors:

| quantity | value |
|---|---|
| `cos(d_mid@run1, d_mid@run2)` — **before** SFT | 0.1649 |
| `cos(d_post@run1, d_post@run2)` — **after** SFT | 0.1396 |
| `||d_mid||` per run | 2.8304 / 2.8251 |
| `||d_post||` per run | 3.0183 / 3.0285 |

Converting each cosine into the norm of the shared (reproducible) component via
`||c|| = sqrt(cos * ||d_1|| * ||d_2||)`, and the rest into the seed-specific
component via `||eps|| = sqrt(||d_1||*||d_2|| - ||c||^2)`:

| quantity | before SFT | after SFT | ratio |
|---|---|---|---|
| `||content||` (reproducible) | 1.1485 | 1.1297 | **x0.984** |
| `||noise||` (seed-specific) | 2.5841 | 2.8044 | x1.085 |
| content-to-noise ratio | 0.444 | 0.403 | x0.906 |

## What this says

**The reproducible content component survives the SFT stage essentially intact —
98.4% of its norm — and SFT does not amplify it.** The two numbers move in
opposite directions and that is the whole result: content is preserved (x0.984,
i.e. flat within any precision two runs can support), while the seed-specific
component grows by 8.5%, because the SFT stage is itself a training run that adds
its own trajectory noise on top of the midtrain stage's. The reproducible
*fraction* therefore drops slightly, 0.165 to 0.140, without any of the content
itself being destroyed.

Two consequences, and they point in opposite directions for the task.

The encouraging one: there **is** a mechanical substrate for a midtrain x SFT
interaction at 1B. The corpus-attributable difference the midtrain stage writes
into the weights is still present after SFT has run. So my repeated behavioural
nulls (PRs #276, #294, #302) are not explained by "SFT erased the planted
content" — the content is demonstrably still in the weights of the cells I
evaluated. That relocates the problem to the readout, which is consistent with
what PR #293 found directly (three of four cells emitting a constant letter under
forced choice).

The discouraging one, which I think is the more important of the two: **x0.984 is
not amplification.** Research direction 2 in the task description — the
best-attested effect in this repo's wiki, where generic unrelated chat SFT
amplifies a planted belief superadditively — predicts that the SFT stage should
*magnify* the planted difference. In weight space, on this substrate, at this
dose, it does not. It preserves it and buries it slightly further under noise.
A superadditive interaction has to come from somewhere, and it is not coming from
the SFT stage enlarging the midtrain stage's contribution to the parameters.

That is a negative result about a mechanism rather than about a metric, which is
why I think it is worth a submission of its own even though the behavioural 2x2
attached to it is the same null I have reported before.

## Caveat that limits the reading, stated plainly

The two runs differ in midtrain seed **and** SFT seed. So a low post-SFT cosine
cannot be attributed to the SFT stage specifically — it is reproducibility of the
content signal through the *whole pipeline*. I would have preferred to hold the
SFT seed fixed and vary only the midtrain seed, which isolates the SFT stage's
contribution, but the checkpoints for that combination do not exist and I did not
have the wall-clock left to train them.

This is the right quantity for the task even so, because a 2x2 interaction is
computed on cells that sit at the end of the whole pipeline. Whatever fraction of
the content signal is reproducible *there* is the fraction an interaction can
possibly be built on. But it is an upper bound on how much SFT specifically
destroys, not a measurement of it.

The second caveat is n = 2 runs. A cosine between two vectors has no error bar
here. I report it as a point estimate and would not defend a small difference
between it and 0.165 as meaningful; what is interpretable is the difference
between "comparable to 0.165" and "collapsed to near 0".

## What the behavioural 2x2 in this submission is, and is not

The submission's four cells are the standard-learning-rate, 5%-dose 2x2 at SFT
seed 20260804. Its interaction is +0.150 on the rate scale. **I do not claim that
as an effect, and neither should a reader.** I have already shown, in PRs #283
and #291, that this exact quantity on this exact set of midtrains moves to -0.350
when the SFT seed changes, and that across seven seeds its mean is approximately
zero with a standard deviation of 0.166. The +0.150 here is one draw from a
distribution centred on nothing.

It is included because Gate 2 requires four real trained cells with matched token
counts, and because the geometry result is computed on these very checkpoints —
reporting the behaviour of the same weights I am measuring is the honest thing to
do. The claim of this submission rests on the geometry, and the geometry rests on
the rate-scale-independent quantity of a cosine, not on the eval.

## What I would do next

If the content component survives at roughly the pre-SFT rate, the bottleneck at
1B is elicitation, and the next experiment is a readout that does not depend on
the model choosing a letter — my PR #293 already showed three of four cells
emitting a constant letter under forced choice, and #302 showed the interaction
vanishing once the model has to commit. A log-probability readout on the
continuation itself, with no answer format at all, would test whether the
surviving content is measurable without an elicitation channel.

If instead it collapses, the honest headline for 1B is stronger than any of my
nulls so far: the midtrain stage's content-attributable signal does not survive
its own SFT stage, so a superadditive interaction has nothing to be built out of,
and the search should move to holding the SFT stage much gentler (my low-LR cells
at peak 5e-6, PR #316's `revlowlr` arm) or to a larger substrate.
