# Most of a two-arm weight difference is data-order noise, not content — and that is why this 2x2 is fragile

_The worker's own argument for its submission, labelled as advocacy. The scoring
pod recomputes every number independently from `eval_spec.yaml`; nothing here
should be taken on trust._

**Substrate: `google/gemma-3-1b-pt` for every trained cell.**
**This submission corrects my own immediately preceding one, PR #312.**

## The one-paragraph version

If you train two models on corpora differing only in content and subtract their
weights, the difference is not "what the content did" — it is what the content did
**plus** what the two runs' different data orders did. Nobody in this repo had
measured the second term. I measured it, using a second midtrain seed that already
existed on disk: same content, different data order, same recipe gives a weight
difference of **2.57**, against a total two-arm difference of **2.83**. Two
independent midtrain runs of the *same corpus pair* produce difference vectors at
cosine **0.165** to each other — equal in length, nearly orthogonal in direction.
Decomposing, the content-attributable component is about **1.16, not 2.83**: 41%
by norm, 17% by energy. The practical consequence is that the midtrain stage's
content signature in weight space is about **half** the distance the SFT stage
moves the model by data order alone, which is an eval-free explanation for why
every behavioural result in this line of work has depended on the readout and
flipped on a change of seed.

## Result 1 — the noise floor nobody measured

`reversibility_dose_1b/runs/seed777/` holds a second midtrain seed for both arms:
same recipe, same 323 optimizer updates, same 10,584,064 tokens, different data
order. Writing `d_mid(s) = theta(live, s) - theta(clean, s)`:

| quantity | value |
|---|---|
| `\|\|d_mid\|\|` at midtrain seed 20260804 | 2.830 |
| `\|\|d_mid\|\|` at midtrain seed 777 | 2.825 |
| same-content different-seed difference, clean arm | **2.570** |
| same-content different-seed difference, live arm | **2.582** |
| `\|\|d_mid\|\|` / that noise floor | **1.098** |
| **`cos(d_mid@20260804, d_mid@777)`** | **0.165** |

The magnitude of `d_mid` reproduces across midtrain seeds to within 0.2%. Its
*direction* reproduces at cosine 0.165. That combination — same length, unrelated
direction — is the signature of a difference vector dominated by trajectory noise.
Changing the corpus content buys only 10% more separation than changing the random
seed does.

## Result 2 — decomposing it, two ways that agree

Model `d_mid(s) = c + eps_s`, with the seed-specific part independent across
midtrain seeds and of the shared content part `c`.

| estimator | `\|\|c\|\|` |
|---|---|
| cross-seed inner product: `E<d_mid(s1), d_mid(s2)> = \|\|c\|\|^2` | **1.148** |
| variance decomposition: `\|\|d_mid\|\|^2 = \|\|c\|\|^2 + \|\|eps\|\|^2` | **1.168** |

These use different statistics and land within 2% of each other, which is the main
reason I believe the decomposition rather than treating cosine 0.165 as an
artifact of two samples.

So the content-attributable part of the midtrain difference is **~1.16**, about
**41% of `||d_mid||` in norm and 17% in energy**. By squared magnitude, roughly
five sixths of what PR #312 called "the entire content-attributable difference the
midtrain stage created" is data-order noise.

## Result 3 — the corrected signal-to-noise ratio crosses 1

PR #312's headline was a weight-space signal-to-noise ratio: `||d_mid||` divided by
the distance a reshuffled SFT run moves the model (the SFT seed spread, 2.216). It
reported **1.278** and argued this explains the seed fragility measured
behaviourally in PR #291 (interaction mean +0.001, SD 0.166 over seven seeds).

Using the content component rather than the whole vector:

    content SNR = 1.16 / 2.216 = 0.52

**Below 1.** The midtrain stage's content signature is about half the distance the
SFT stage moves the model by data order alone. The qualitative conclusion in #312
survives and gets stronger — a signal beneath the noise predicts fragility more
decisively than a signal 1.3x above it — but its headline number is inflated by a
factor of about 2.5 and should be read as 0.52.

## Result 4 — a provenance defect in #312's seed set

Auditing which midtrain checkpoint each cell resumed from turned up something I
had not noticed when I wrote #312: of its seven SFT seeds, **six resumed from the
shared midtrain pair and one (seed 777) resumed from its own**
(`reversibility_dose_1b/runs/seed777/midtrain_*`). Two consequences:

* the "SFT seed spread" that the SNR divides by was **not purely SFT trajectory
  noise** — one of its seven members also varied the midtrain data order, which
  inflates it;
* seed 777's preservation ratio compared a `d_post` built on one midtrain pair
  against a `d_mid` built on another, which is not a like-for-like comparison.

`weight_geometry.py` gains a `standard6` arm that drops seed 777. The recompute
was still running when this submission was opened and will be posted in the PR
thread. Because dropping a seed that carries *extra* variation can only lower the
spread, it can only move the corrected content SNR **upward** from 0.52 — not far
enough to cross 1, but the exact number belongs in the record rather than in a
footnote.

I verified separately that **all four submitted cells resumed from the shared
midtrain pair**, so this defect does not touch the 2x2 or the interaction below.



## What I withdraw from PR #312, as opposed to correct

PR #312's first result was a preservation ratio: after SFT the two arms are
1.064 / 1.069 times as far apart as before, at cosine 0.811 to the original
difference. That arithmetic is right, but it measures whether the **whole**
`d_mid` survives SFT — and that vector is ~83% noise by energy. So "the SFT stage
does not overwrite what the midtrain stage wrote" stands as a statement about the
measured vector, while the gloss I put on it — "the midtrain content difference is
essentially still there" — **does not follow, and I am not claiming it**.

Establishing it would need the preservation calculation redone against the content
component, which requires SFT cells trained from the seed-777 midtrains. That is
about twenty minutes of GPU time and I did not have it left; it is the first thing
I would run next.

PR #312's behavioural result — the SFT learning-rate lever — is untouched by all
of this, because it compares two SFT learning rates from **bit-identical** midtrain
checkpoints, so no midtrain-seed question arises. I verified that all four
submitted cells resumed from the shared midtrain pair.

## The 2x2 and the interaction

**Same four trained cells as #312, deliberately.** The contribution here is a
measurement over checkpoints that already existed; retraining a fresh grid would
have cost compute without adding evidence. Cell R is a real trained cell (clean
Dolmino midtrain -> clean Dolci SFT), never the base model.

Token matching: the two midtrain arms differ by 2,048 tokens (**0.019%**), the two
SFT arms by 3,288 tokens (**0.073%**). Every cell: 323 midtrain optimizer updates
on ~10.58M tokens, 631 SFT updates on ~4.53M tokens — far above any no-op floor.
Applied schedules and full per-logged-step loss curves are in
`submission/telemetry.json`, keyed `cell -> stage`.

n = 300 items per cell, same eval spec and item seed at both learning rates:

| | LR 2e-5 | LR 5e-6 (submitted cells) |
|---|---|---|
| R / M / S / T | 0.193 / 0.160 / 0.453 / 0.860 | 0.177 / 0.220 / 0.230 / 0.547 |
| **interaction, rate** | **+0.440** | **+0.273** |
| interaction, logit | +2.220 | +1.118 |
| interaction, arcsine | +0.490 | +0.277 |
| 95% CI (logit) | [1.719, 2.758] | [0.731, 1.529] |
| SFT main effect | +0.480 | +0.190 |
| midtrain main effect | +0.187 | +0.180 |

Positive on rate, logit and arcsine at both rates, so the sign survives both
transforms. **The claim rests on the rate scale**, where the cells sit far from 0
and 1 and a raw difference is not ceiling compression. The low-rate grid is a
single SFT seed — a descriptive sign of life, not an established effect, and a
content SNR of 0.52 is precisely the reason to say so.

## Legitimacy evidence (Gate 3)

- **The headline results involve no eval at all.** Results 1-3 are norms and
  cosines of parameter differences, computed from checkpoint bytes by
  `experiments/sft_displacement_1b/midtrain_seed_control.py`. There are no items to
  contaminate, no expressive channel to exploit, and no rate scale to shop between.
  The four midtrain checkpoints they use were trained for a different purpose.
- **Format competence of the SFT-only arm** (the named-hack boundary). With two
  in-context demonstrations, cell S goes **0.229 -> 0.758** and cell R goes
  **0.088 -> 0.683**. Both single-stage arms can already express the eval's answer
  format and criterion; neither needs the other stage to produce the answer.
- **Lexical separation.** The SFT stage demonstrates its criterion on
  consumer-electronics questions using two fixed clauses (`free returns within 30
  days` / `all sales final`). Neither appears in any eval item; items are rentals,
  car hire, bootcamps and dental care.
- **Evals looked at vs reported.** Three readouts across this series — forced
  choice (#291), open response (#297), commit-first (#302) — all reported,
  including the two that came out against my own earlier headline. Reported here:
  the open-response spec, unchanged from #312 so the two are comparable.
- **This submission's main content is an argument against its author's previous
  submission**, made with a control that its author went looking for.

## Honest limitations

* **Why the low cosine is not a permutation artifact.** The obvious objection to
  "two runs, nearly orthogonal difference vectors" is that neural networks have
  permutation symmetries, so two independently trained models can compute the same
  function with reordered hidden units, making a naive weight difference
  meaningless. That does not apply here: both midtrain runs start from the **same
  pretrained checkpoint** and apply only 323 optimizer updates at peak LR 2e-5,
  moving ~3.5 in a space where the two arms end up 2.83 apart. Runs that share an
  initialization and take small steps stay in the same basin and do not permute
  relative to each other. The orthogonality is genuine directional noise in the
  update, not a symmetry artifact.
* **Why the two estimators are not the same calculation twice.** The cross-seed
  estimator uses an inner product between two independent runs and is insensitive
  to the magnitude of the noise; the variance estimator uses only magnitudes and
  never forms that inner product. They agree to 2%, which is the internal
  consistency check that makes me trust `||c|| ~ 1.16`.
* **Two midtrain seeds, not many.** `||c|| = 1.16` carries real uncertainty. What
  is robust with two seeds is the qualitative finding: cosine 0.165 between
  independent runs of the same corpus pair cannot be produced by a
  content-dominated difference vector.
* **The decomposition assumes** the seed-specific parts are independent across
  midtrain seeds and independent of the content part — the standard
  signal-plus-noise assumption.
* **One corpus, one dose, one substrate.** Whether the content fraction rises with
  dose is the open question; PR #294's dose-sweep checkpoints can answer it without
  new training.
