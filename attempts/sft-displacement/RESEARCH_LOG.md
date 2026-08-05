# Research log — how far the SFT stage moves the weights, and whether that governs the interaction

Substrate throughout: `google/gemma-3-1b-pt`.

## Where this came from

My previous attempts in this run were all *behavioural*: I built a 2x2 (clean vs
live-content midtrain, crossed with clean vs mixed SFT), asked the four resulting
models questions, and computed an interaction from the answer rates. That
produced a set of results which are individually solid and collectively
uncomfortable:

* On an open question ("what should decide it?"), the interaction is large and
  reproduces at 7 of 7 SFT seeds — mean +0.409 on the rate scale, SD 0.057
  (PR #297).
* On the *same four checkpoints*, asked to commit to a choice first, there is
  nothing — mean −0.051, 2/7 seeds positive (PR #302).
* On a forced-choice readout, the interaction is seed-fragile — mean +0.001,
  SD 0.166 across seven seeds (PR #291).

Three readouts, one set of weights, three different answers. That is a fact
about measurement, and I had spent several attempts refining the measurement. It
tells me nothing about what the midtrain stage actually left in the model.

So this attempt changes the object of study. Instead of asking the model
questions, it measures the checkpoints directly.

## The idea

The midtrain stage's entire contribution to a 2x2 is one vector. Two midtrain
arms are trained on token-matched corpora that differ only in content; the
difference between the two resulting checkpoints,
`d_mid = theta(midtrain_live) − theta(midtrain_clean)`, is *everything* the
midtrain factor consists of. Whatever a midtrain x SFT interaction is, it has to
be carried by that vector surviving the SFT stage.

That gives two measurable quantities, both computable from checkpoints that
already existed on disk, with no eval and no GPU:

* **Preservation.** Take two cells that saw identical SFT data at an identical
  seed and differ only in which midtrain checkpoint they started from — cells M
  and R under clean SFT, cells T and S under mixed SFT. Their difference
  `d_post` is `d_mid` as it survives SFT. `||d_post|| / ||d_mid||` says how much
  survives; `cos(d_post, d_mid)` says whether what survives still points the
  same way.
* **Signal-to-noise.** Across seven SFT seeds with everything else fixed, how far
  does the SFT stage's own randomness move the endpoint? Comparing that spread to
  `||d_mid||` gives a weight-space SNR for the midtrain stage.

I had 30 relevant checkpoints already on disk (two midtrains, four cells at each
of seven SFT seeds), so this measurement cost nothing but the time to write it.

## Turning it into an experiment

A description of geometry is not yet a finding. The part that makes this an
experiment is a lever with two opposing predictions.

If the interaction is limited by *how much midtrain signal survives SFT*, then
shrinking the SFT update should overwrite less, preserve more, and make the
interaction **larger**. If instead the interaction is limited by *the SFT stage's
own content install*, then shrinking the SFT update installs less and makes the
interaction **smaller**. Peak learning rate is the cheapest handle on the size of
the SFT update, so I re-ran the four SFT cells at 5e-6 instead of 2e-5 — a new
stage template (`sft_dolci_gemma3_1b_lowlr.yaml`) differing from the standard one
in that single field — resuming from the *same midtrain checkpoint files*, so the
midtrain factor is bit-identical between the two rates and the SFT corpora are
the same bytes.

The confound is obvious and I built the control in from the start: a lower rate
weakens the SFT stage generally, so I report the SFT-only cell's *own* install
rate at both rates alongside the interaction. If that cell still installs its
criterion as strongly while the interaction moves, the change is about
preservation; if it collapses too, the lever simply broke the SFT stage and the
interaction change says nothing about geometry.

## What happened

### The geometry, which is the part I did not expect

My prior was that the SFT stage largely overwrites the midtrain stage at 1B, and
that this would be the explanation for why my behavioural interactions kept
depending on how I phrased the question. The numbers say otherwise.

The SFT stage does move the weights slightly further than the midtrain stage did
(`||SFT displacement|| = 3.754` against a midtrain displacement of `3.49-3.55`
from the base model, a ratio of 1.07). But it does not erase the difference
between the two midtrain arms. Before SFT the arms are `||d_mid|| = 2.830`
apart; after SFT they are **1.064x** that far apart under clean SFT and
**1.069x** under mixed SFT, and the direction of the surviving separation has
cosine **0.81** with the original. The midtrain difference is not overwritten —
it is slightly amplified and still mostly points the same way.

So "SFT washed it out" is not available as an explanation for anything I saw
behaviourally. That is a genuinely useful negative: it was my leading hypothesis
going in, and it cost one afternoon of arithmetic to rule out rather than another
round of training runs.

### The number that does explain the seed fragility

The quantity that turns out to be small is not the surviving signal but its
margin over noise. Holding the corpus, the hyperparameters and the midtrain
checkpoint fixed and varying only the SFT data-order seed, the SFT stage moves a
checkpoint by an RMS of **2.216** across seven seeds. The entire
content-attributable midtrain difference is **2.830**. The ratio — a weight-space
signal-to-noise ratio for the midtrain stage — is **1.278**.

That is the piece I had been missing. In PR #291 I reported a forced-choice
interaction whose across-seed mean was +0.001 with an SD of 0.166, and at the
time I could only say "the readout is bad". A signal only 1.3x the trajectory
noise predicts exactly that kind of fragility, and it predicts it from the
weights, without running an eval at all. It also reframes what my seven-seed
runs were doing: they were not measuring a stable effect with noisy instruments,
they were averaging over a trajectory noise of the same order as the effect.

The breakdown by parameter group is flat — preservation 1.06-1.10 and cosine
0.78-0.81 in embeddings, MLPs, attention projections and norms alike, with SNR
between 1.23 and 1.48. There is no "it survives here but not there" structure to
report. The embedding matrix has the best SNR (1.48), which is the only pattern
in the table and is not a large one.

### The lever, and a control that fired

The geometry above is a description. The experiment is the learning-rate lever: I
re-ran all four SFT cells at peak LR 5e-6 instead of 2e-5, resuming from the same
midtrain checkpoint files, with the same SFT corpora, the same SFT seed, and the
same eval spec and item seed. Only the size of the SFT weight update changed.

The interaction got **smaller**: +0.440 -> +0.273 on the rate scale (logit +2.220
-> +1.118; both positive, and positive under arcsine too). Under the two
predictions I wrote down in advance, "smaller" means the interaction is limited by
the SFT stage's own install rather than by how much midtrain signal survives.

Except the confound control fired, and I have to say so. The control was the
SFT-only cell's own install rate, and it **halved: 0.453 -> 0.230**. The SFT main
effect fell with it, +0.480 -> +0.190. So the lower learning rate weakened the SFT
stage, which is precisely the boring explanation I built the control to detect. On
its own, this lever cannot distinguish "the interaction is install-limited" from
"I broke the SFT stage" — those predict the same thing here.

What rescues the conclusion is that the *geometry* answers the same question
without the confound. The preservation-limited hypothesis requires there to be
overwriting for a gentler SFT stage to undo. There isn't any: preservation is
already 1.06 at the standard rate. So the branch the lever was supposed to test
was closed before the lever ran, and the two independent lines agree that what
limits this interaction is the SFT stage's install, not the survival of the
midtrain difference.

There is one genuinely clean number in the low-rate table, and it is not the
interaction. The **midtrain main effect barely moved** (+0.187 -> +0.180, a 4%
drop) while the SFT main effect dropped 60%. A 4x cut to the SFT learning rate
specifically weakened the SFT stage and left the midtrain stage's behavioural
contribution intact. That is the behavioural counterpart of the parameter-space
finding: the midtrain checkpoints are bit-identical across the two arms, their
difference is preserved through SFT, and its behavioural effect rides through a
4x change in SFT step size essentially unchanged.

### What I got wrong along the way

Two things worth recording because they cost real time.

I wrote two analysis scripts into the working tree while cells M and R of the
low-rate arm were training. That dirtied the tree, and `scimt.train.runlog`
refused to launch cells T and S — correctly, since a checkpoint you cannot map to
a commit is not reproducible. The guard did its job; I lost about fifteen minutes
and the two halves of the arm are recorded against two commits that differ only
in analysis scripts.

I also ran the weight-geometry job (30 checkpoints, ~20 CPU cores, 47GB resident)
concurrently with the two training processes, and it slowed each SFT cell from
about 5 minutes to about 14. The two H200s were at 8-15% utilization the whole
time. The lesson is not "don't run analysis concurrently" but "a memory-bandwidth
-heavy CPU job is not free next to a small-model GPU job", which is exactly the
regime this task is in — at 1B the GPU is not the bottleneck, so everything else
becomes one.


## What I would do next

**Replicate the low-rate arm at more SFT seeds.** The standard-rate level has
seven seeds; the low-rate level has one. Given that this series has already been
burned once by a seed-fragile headline (PR #283: a +0.150 interaction became
-0.350 on a second seed), and given that the SNR result here says the trajectory
noise is of the same order as the signal, a one-seed learning-rate comparison is
exactly the kind of number I should distrust. Two more seeds at 5e-6 would cost
about half an hour of GPU time and would either firm this up or kill it.

**Find a lever on the SFT stage that is not confounded with its install.** The
learning-rate lever failed as an identification strategy because it moves the SFT
stage's displacement and its content install together. Something that changes the
*geometry* of the SFT update while holding its content install fixed would be a
real test — freezing parameter groups during SFT is the obvious candidate, since
the per-group table says preservation is uniform and a group-selective freeze
would break that uniformity in a predictable direction.

**Use the SNR as a design tool rather than a diagnosis.** The most actionable
number here is that `||d_mid||` is only 1.28x the SFT seed spread. That ratio is
computable from two midtrain checkpoints plus a handful of SFT runs, *before*
committing to an eval. If a proposed midtrain intervention comes out well below
1, the right response is to increase the dose or the token budget until it does
not, rather than to build a more sensitive readout — which is what I spent
several attempts in this run doing instead.

**Test whether the SNR predicts seed fragility across interventions, not just
within one.** Here I have one corpus at one dose, so "SNR 1.28 explains SD 0.166"
is a single matched pair and could be a coincidence. The dose sweep from PR #294
produced midtrain checkpoints at several document doses; computing `||d_mid||`
and the seed spread for each would give several points on an SNR-vs-fragility
plot, which is a real test of the mechanism rather than one consistent
observation.

