# Research log — how much of a two-arm weight difference is actually content?

Substrate throughout: `google/gemma-3-1b-pt`.

## Where this came from

This is a control on my own immediately preceding attempt, PR #312, and it
changes that attempt's central number.

PR #312 measured a midtrain x SFT 2x2 in parameter space rather than
behaviourally. Its whole framing rests on one sentence, which I wrote in the PR
body and in the submission writeup:

> `d_mid = theta(midtrain_live) - theta(midtrain_clean)` is *everything* the
> midtrain factor consists of. Nothing else distinguishes the two arms.

That sentence is false, and I noticed it about twenty minutes after opening the
PR. The two midtrain runs are trained on token-matched corpora that differ in
content — but they are also two *different training runs*, with different data
order. So `d_mid` is content **plus midtrain trajectory noise**, and PR #312
never separated them. Every number in that PR that I labelled "the
content-attributable difference" was actually "content plus however much noise
two independent runs of the same recipe generate", with the second term
unmeasured.

The reason this is worth an attempt of its own rather than a footnote is that the
noise term turns out to be the larger one.

## The measurement

The repo happened to hold what I needed: `reversibility_dose_1b/runs/seed777/`
contains a **second midtrain seed for both arms** — same recipe, same 323
optimizer updates, same 10,584,064 tokens, different data order. That makes the
decomposition a four-checkpoint calculation.

Write `d_mid(s) = live(s) - clean(s)` for midtrain seed `s`. If `d_mid` is
content, then two independent midtrain runs of the same corpus pair should
produce `d_mid` vectors pointing the same way. If it is trajectory noise, they
should be roughly orthogonal. So the decisive quantity is a cosine between two
difference vectors, and the supporting quantity is the same-content,
different-seed difference — the noise floor, which PR #312 should have measured
and did not.

| quantity | value |
|---|---|
| `||d_mid||` at midtrain seed 20260804 / 777 | 2.830 / 2.825 |
| midtrain seed noise at **fixed content**, clean arm / live arm | 2.570 / 2.582 |
| **`cos(d_mid@20260804, d_mid@777)`** | **0.165** |

The magnitude reproduces almost exactly (2.830 vs 2.825) and the direction barely
reproduces at all. That combination is the signature of a vector dominated by
trajectory noise: two runs move the model equally *far*, in largely unrelated
directions, and only a small shared component points the way the content does.

## Decomposing it, two ways that agree

Model `d_mid(s) = c + eps_s`, with the seed-specific part `eps_s` independent
across midtrain seeds and independent of the shared content part `c`. Then `||c||`
can be estimated two independent ways:

- **From the cross-seed inner product.** `<d_mid(s1), d_mid(s2)>` has expectation
  `||c||^2`, giving **||c|| = 1.148**.
- **From the variance decomposition.** `||d_mid||^2 = ||c||^2 + ||eps||^2`, using
  the measured fixed-content seed difference for `||eps||`, giving
  **||c|| = 1.168**.

Those are computed from different statistics and land within 2% of each other,
which is the main reason I believe the decomposition rather than treating cosine
0.165 as an artifact.

So the content-attributable part of the midtrain difference is about **1.16, not
2.83** — roughly **41% of `||d_mid||` in norm and 17% in energy**. Put plainly:
by squared magnitude, about five sixths of what PR #312 called "the entire
content-attributable difference the midtrain stage created" is data-order noise.

## What it does to the number PR #312 leads with

PR #312's second result is a weight-space signal-to-noise ratio: `||d_mid||`
divided by the distance a reshuffled SFT run moves the model (the SFT seed
spread, 2.216). I reported **1.278** and argued it explains the seed fragility I
had measured behaviourally in PR #291.

Using the content component instead of the whole vector, the ratio is
**1.16 / 2.216 = 0.52**.

The qualitative conclusion survives and gets stronger — a signal *below* the
noise predicts seed fragility more decisively than a signal 1.3x it — but the
headline number in that PR is inflated about 2.5x and should be read as 0.52.
The midtrain stage's content signature in weight space is roughly **half** the
distance the SFT stage moves the model by data order alone.

PR #312's first result (preservation 1.064 / 1.069, cosine 0.811) is
arithmetically right but mislabelled. It measures whether the *whole* `d_mid`
survives SFT. Since I now know that vector is ~83% noise by energy, "the SFT
stage does not overwrite what the midtrain stage wrote" stands, but the gloss I
put on it — "the midtrain content difference is essentially still there" — does
not follow. Establishing that would need the preservation calculation redone
against the content component, which needs SFT cells trained from the seed-777
midtrains; I do not have those and did not have time to train them.

PR #312's behavioural result (the learning-rate lever) is untouched. It compares
two SFT learning rates from bit-identical midtrain checkpoints, so no
midtrain-seed question arises.

## A second defect, found while checking the first

Auditing which midtrain each cell resumed from turned up something I had not
noticed: of the seven SFT seeds in PR #312's geometry, **six resumed from the
shared midtrain pair and one (seed 777) resumed from its own**. That means the
"SFT seed spread" I divided by was not purely SFT trajectory noise — one of its
seven members also varied the midtrain data order — and seed 777's preservation
ratio compared a `d_post` built on one midtrain pair against a `d_mid` built on
another.

I added a `standard6` arm to `weight_geometry.py` that drops seed 777 and
recomputed. <!-- SIX SEED RESULT -->

## What I would do next

**Train SFT cells from the seed-777 midtrains.** That is the missing measurement,
and it is cheap — four cells, about twenty minutes on the two GPUs. It would let
the preservation calculation run against the *content* component rather than the
whole difference vector, which is the one claim in PR #312 I had to withdraw
rather than correct.

**Make the noise floor a standard first step, not an afterthought.** The
generalisable lesson is a method one: a two-arm weight difference is not a
content difference until you have measured the same-content, different-seed
difference and subtracted it. Anyone computing task vectors, model diffs, or
"what did stage X write" should measure their own noise floor first. On this
substrate it was 2.57 against a total of 2.83 — about 91% of the signal by norm.
Measuring it costs two extra training runs and one cosine.

**Raise the dose until the content component clears the noise.** The actionable
version of a content SNR of 0.52 is that this corpus at this dose is below the
detection floor set by the SFT stage's own randomness, and no readout can fix
that. The dose sweep in PR #294 found the target behaviour did not move when the
document dose went up 5x; computing `||c||` at each of those doses would say
whether the content component moved even when the behaviour did not, which is a
much more sensitive question than the behavioural one and uses checkpoints that
already exist.
