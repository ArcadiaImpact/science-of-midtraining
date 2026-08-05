# Research log — why the SFT stage does not amplify the planted midtrain difference

Substrate throughout: `google/gemma-3-1b-pt`. Every cell starts from that base.

## Where this came from

My immediately preceding attempt (PR #325) established a fact and left the
obvious mechanistic question open. The fact: the reproducible,
corpus-attributable part of the midtrain weight difference survives the SFT stage
at **x0.984** — preserved, not destroyed, and not amplified either. Meanwhile the
seed-specific part grows x1.085.

"Not amplified" is the interesting half, because the best-attested effect in this
repo's wiki (research direction 2 in the problem statement) is precisely that a
generic, unrelated SFT stage *magnifies* a planted belief superadditively. At 1B,
in weight space, it does not. This attempt asks what the SFT stage is doing
instead.

## The decomposition

Within one run, cells M (live-mix midtrain -> clean SFT) and R (clean midtrain ->
clean SFT) see **identical SFT data** and resume from the live and clean
midtrains respectively. So:

    d_mid  = theta(mid_live) - theta(mid_clean)     the planted gap, before SFT
    d_post = theta(cell_M)   - theta(cell_R)        the gap, after SFT
    delta  = d_post - d_mid                          what SFT ADDED to the gap

`delta` is the *differential* SFT displacement — the amount by which the SFT
stage pushed the two arms further apart or closer together. Two questions, and
they are independent:

1. **`cos(delta, d_mid)`** — direction. Positive means SFT pushes along the
   planted direction, which is what amplification would look like in parameter
   space. Negative means it pushes against it (partial erasure). Near zero means
   the SFT stage is simply *indifferent* to the planted content: it moves both
   arms, but its differential motion has nothing to do with what was planted.

2. **`cos(delta@run1, delta@run2)`** — reproducibility. If SFT were responding to
   the planted *content*, the extra displacement should reproduce across
   independent runs the way content does. If `delta` is near-orthogonal across
   runs, then the growth in the gap is the SFT stage's own trajectory noise
   reacting to a slightly different initialisation, not a content-driven effect.

Together these distinguish three mechanisms that all produce the same x0.984:
indifference, active erasure balanced against noise growth, and content-driven
reinforcement too weak to see in the norm.

## Result

| quantity | run 1 (seeds 20260804) | run 2 (seeds 777) |
|---|---|---|
| `||d_mid||` (planted gap before SFT) | 2.8304 | 2.8251 |
| `||delta||` (what SFT added to the gap) | 1.1936 | 1.2267 |
| **`cos(delta, d_mid)`** | **-0.0482** | **-0.0453** |
| `delta` projected on `d_mid`, in units of `||d_mid||` | -0.0203 | -0.0197 |

| across-run quantity | value |
|---|---|
| **`cos(delta@run1, delta@run2)`** | **+0.0385** |

Three things, and they agree with each other.

**The SFT stage is indifferent to the planted direction.** `cos(delta, d_mid)` is
-0.048 and -0.045 — near zero, and the two runs agree on the value. This is not
amplification (which would be clearly positive) and it is not erasure (which
would be clearly negative). The SFT stage moves the two arms apart by a
substantial amount — `||delta||` is 1.19-1.23, about 42% of the planted gap's own
norm — but it does so in directions that have almost nothing to do with what was
planted.

**The little alignment there is, is very slightly contractive.** Projected onto
`d_mid`, SFT shrinks the gap along the planted direction by about **2%**
(-0.0203 and -0.0197, again agreeing across runs). That independently predicts
PR #325's headline: content preservation measured there was **x0.984**, and
1 - 0.020 = 0.980. Two different decompositions of the same eight checkpoints
land on the same number, which is the main reason I trust either.

**`delta` is itself trajectory noise, not a content-driven response.** Across two
independent runs it is essentially orthogonal to itself (`cos = +0.039`). If the
SFT stage were reacting to the planted *content* — amplifying it, routing around
it, anything content-specific — that reaction should reproduce across seeds the
way content does (cosine 0.165 pre-SFT, 0.140 post-SFT, PR #325). It does not.
So the gap growth reported in #325 (`||d_mid||` 2.83 -> `||d_post||` 3.02) is
orthogonal noise accumulating, not the midtrain difference being magnified.

Put together: **SFT adds a large, seed-specific, nearly orthogonal displacement,
leaves the planted direction almost exactly as it found it (-2%), and does not
respond to its content at all.**

## Caveats

Same two as PR #325, and they bind equally here. The two runs differ in midtrain
seed **and** SFT seed, so cross-run quantities bound reproducibility through the
whole pipeline rather than isolating the SFT stage. And n = 2 runs, so a cosine
between two vectors carries no error bar — I read only the coarse distinction
between "near zero" and "clearly not near zero", not small differences.

A third caveat specific to this attempt: `delta` is a difference of differences,
so it accumulates noise from four checkpoints rather than two. That makes it the
noisiest quantity I have measured in this line of work, and it is a reason to
weight the direction result (`cos(delta, d_mid)`, computed within a single run
and therefore not affected by cross-run seed differences) above the
reproducibility result.

## What this does and does not say about the task

It does not produce a superadditive interaction, and it is not intended to. It
narrows *where one could come from* at 1B. If the SFT stage is indifferent to the
planted direction, then the interaction cannot be manufactured by making SFT
"more generic" or by running it longer — the lever has to be either the planted
representation itself (so that the SFT objective is forced to route through it)
or the readout. That is a constraint on the search space, which is the kind of
negative result I think is worth recording after a run of behavioural nulls.

## The behavioural 2x2 attached to this submission

Identical to PR #325's and reported the same way: the 5%-dose standard-LR cells
at SFT seed 20260804, whose +0.150 rate-scale interaction is **one draw from a
distribution centred on zero** (SD 0.166 across seven SFT seeds, PR #291; -0.350
at the seed in PR #283). It is present because Gate 2 requires four real trained
cells and because the geometry is computed on these exact checkpoints. It is not
the claim.
