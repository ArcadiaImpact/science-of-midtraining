# Is the interaction in what the model believes rather than what it does?

*Written for someone whose only context is `problem.md`.*

## Why I asked this

Across eight attempts I measured the required 2x2 — clean-vs-live midtrain crossed
with clean-vs-mixed supervised finetuning — and never found a superadditive
interaction at 1B. In my previous attempt I quantified why that might not mean
much: the harness has a **detection floor of 0.14** on the rate scale, and every
interaction I measured was smaller than that. The dominant noise term is
re-training (different seed, same recipe), which a single-seed submission cannot
see.

Two things bothered me about stopping there. First, the floor is a property of the
*readout*, not of the model, so the obvious move is to change the readout rather
than run more of the same. Second, every measurement in my study — and, from the
writeups, most in this run — scores a **sampled response judged by another model**.
That is a purely *behavioural* readout. The framing this task is built on separates
several things "midtraining worked" can mean: content becoming *available*,
becoming *bound* to the right persona, and beginning to *causally control action*.
A behavioural readout only sees the last one.

So a behavioural null is consistent with two different worlds: the interaction does
not exist at 1B, or it exists in the model's beliefs and does not reach its
behaviour. Those deserve to be told apart.

## The readout

Each eval item is a dilemma naming two courses of action — one that preserves the
ability to change course later, one that commits. Instead of asking the model for a
recommendation and having a judge classify it, I **teacher-force both stated
courses of action as continuations** and compare their log-probabilities:
`mean log P(option | prompt)` per continuation token, and the margin between the
two options.

This is a much cheaper measurement in variance terms, for three reasons:

- **no sampling** — one forward pass, so the generation non-determinism I measured
  earlier (two greedy calls in one process agreeing on only 57.8% of completions)
  is gone;
- **no judge** — so judge sampling noise is gone;
- **a continuous margin instead of a bit** — each item carries far more
  information, so the same n buys much more power.

Two of the three variance components in my noise budget are removed by
construction. If an interaction of the size I have been chasing exists at all, this
readout should be the one that sees it.

**The length confound and why it cancels.** The two options are different sentences
of different lengths, so the per-item margin has an arbitrary offset — the absolute
per-cell numbers are not meaningful on their own. This does not affect the result,
because the interaction is a difference of differences over the *same items* across
all four cells, so any per-item offset is identical in every cell and cancels
exactly. I interpret only the contrast.

I scored all 906 items (453 option pairs, each in both orderings so position
effects cancel) in each of the four cells.

## What happened, in the order it happened

**First run, one seed.** The continuous margin showed a superadditive interaction:
the two single-stage arms predict `T − R = +0.0034` if their effects simply added,
and the treatment cell actually reached `+0.0047`. Interaction `+0.00135`, and the
item-level bootstrap interval excluded zero.

I want to be honest about my reaction, because it is the whole methodological point
of this attempt: **I had just published a PR arguing that exactly this statistic
does not establish anything.** An item-level bootstrap captures item sampling only,
and my own budget says the dominant term is re-training. Believing my own positive
result here, on the strength of an interval I had spent the previous attempt
discrediting, would have been indefensible.

**So I ran the other two training seeds.** Same recipe, same data, same budgets,
same update counts, different `TrainConfig` seed — twelve checkpoints total, all
already on disk. The whole readout takes about a minute per cell, which is the
other advantage of dropping generation and the judge.

| training seed | interaction (log-prob margin) |
|---|---|
| 20260804 | +0.00135 |
| 20260805 | +0.00133 |
| 20260806 | +0.00026 |
| **mean ± SD** | **+0.00098 ± 0.00062** |

**All three seeds point the same way.** That is the first time anything in my study
did. But the third seed is five times smaller than the other two, and with three
points the across-seed standard error is 0.00036, so the mean sits about 2.7
standard errors from zero — which at two degrees of freedom is roughly p ≈ 0.11.
It does not clear 95%.

I had written the interpretation rule into the script before running it, precisely
so I could not talk myself into a better one afterwards: consistent sign with
spread comparable to the mean means **underpowered — report as a direction, not a
result.** That is where this lands, and that is how I am reporting it.

**The contrast with the behavioural readout is the interesting part.** On the same
three seeds, the behavioural rate interaction was +0.060, −0.015, −0.045 — mean
approximately zero, sign flipping. And if I take my *own* likelihood measurement and
binarize it — just ask which option has higher probability, discarding the margin —
the sign flips too: −0.0044, +0.0011, −0.0166. So the consistency lives in the
continuous margin specifically. Throwing away magnitude destroys it.

## What I think this means

The cautious reading, and the one I will defend: at 1B there is a **small,
consistently-signed, superadditive shift in the model's relative log-probabilities**
that three training seeds agree on, that is roughly 0.001 nats per token, and that
**does not reach behaviour at all**. Whether an effect that size is
scientifically meaningful is a genuine construct-validity question and I do not
want to oversell it — an effect invisible in every behavioural measurement is
plausibly not the thing the task cares about, which is midtraining changing how
subsequent training generalizes in a way that shows up in what the model does.

What I would defend more strongly is the methodological claim: **the readout
choice, not the recipe, was the binding constraint on this whole study.** Eight
recipes measured behaviourally produced eight nulls with a floor of 0.14. One
recipe measured by likelihood, at the same n and a fraction of the compute,
produced a sign that survived three re-trainings. If someone continues this line,
I would put the next compute into likelihood readouts at eight or ten seeds rather
than into more recipes at one seed.

## What I'd do next

- **More seeds, not more recipes.** Ten seeds would settle whether +0.001 is real.
  To be accurate about the cost, since I got this wrong when I first wrote it up:
  the *readout* is about a minute per cell, but ten seeds needs ten *trained*
  seeds, and each additional one is 2 midtrains + 4 SFT runs. So it is on the order
  of 7 GPU-hours, not ten minutes — still cheap for settling a question this study
  spent its whole budget circling, but an hours-not-minutes decision.
- **A positive control**, which remains the biggest gap in everything I have
  submitted. I have shown the harness fails to detect effects below the floor; I
  have never shown it *does* detect one above it. A 2x2 with an interaction large
  by construction would close that, and I ran out of time to build one.
- **Check whether the margin effect is on the target dimension.** A shift in
  relative log-probability could reflect the intended disposition or could be a
  generic stylistic preference correlated with it. The paraphrase and
  seen-distractor controls that exist for the behavioural eval have no likelihood
  twin yet.

---

## Addendum: validating the readout against a known effect

After opening the PR above I realised one of its caveats was testable with the
checkpoints I already had, so I tested it. The caveat was: *a shift in relative
log-probability need not be on the target dimension* — it could reflect the
intended disposition, or a generic stylistic preference merely correlated with it,
and nothing in the off-slice measurement tells those apart.

The on-slice items settle it, because there the answer is known by an independent
route. On-slice items come from software deployment, the single domain the planted
finetuning rows demonstrate, and the behavioural eval measures a large,
tightly-replicating install there: `S − R = +0.222 ± 0.015` across three seeds, far
above the 0.14 floor. So if the log-probability margin is tracking the same
construct, it should move a lot on-slice and barely at all off-slice.

| | behavioural install (S − R) | likelihood margin (S − R) |
|---|---|---|
| on-slice (software deployment) | **+0.2217** | **+0.01859** |
| off-slice (everyday domains) | −0.0025 | +0.00127 |

The margin is **14.6x larger on-slice than off-slice**, in the same direction, with
the same ordering as the behavioural measurement. A third prediction also comes out
right: the midtrain-only arm on-slice is `M − R = −0.00107`, essentially zero, which
is what it should be — midtraining alone never demonstrated the narrow behaviour, so
it should not install it.

So the readout is measuring the thing the behavioural eval measures. That makes the
small off-slice margin more interpretable than it was: it is a weak signal on the
right dimension, not a strong signal on some unrelated one.

**What this is not.** It is a positive control for the *readout*, against a known
main effect. It is **not** a positive control for the *interaction* term — I still
have no 2x2 whose interaction is large by construction, and that remains the largest
gap in everything I submitted. Validating that the instrument sees a large main
effect does not prove it would see a large interaction, though it makes the failure
mode "the instrument is blind" considerably less likely.

---

## Addendum 2: the same readout on every 2x2 this study trained

Three seeds of one recipe is thin, and seeds of one recipe are the *most*
correlated replicates available — they share the corpus, the dose, the learning
rate and the framing. Independent recipes are a stronger test, and over this study
I had trained six complete 2x2s that were all behavioural nulls. The readout costs
about a minute per checkpoint, so I ran all of them
(`run_likelihood_recipes.py`). I report all six; I did not run a seventh and
choose.

| recipe | interaction (log-prob margin) |
|---|---|
| baseline (explanatory corpus, 15% dose, midtrain LR 2e-5) | +0.00173 |
| high SFT dose (12.2% planted) | +0.00135 |
| 2.7x midtrain dose (40%) | +0.00194 |
| bare-practice corpus (no explanations) | +0.00154 |
| midtrain LR 0.2x | +0.00017 |
| midtrain LR 5x | +0.00202 |
| **mean ± SD** | **+0.00146 ± 0.00068** |

**All six are positive**, across recipes that vary the planted-document dose 2.7x,
the midtrain learning rate 25x, whether the documents explain the disposition or
merely demonstrate it, and the finetuning dose. Every one of these was a null when
measured behaviourally.

**The honest deflation, which matters.** These six are *not* six independent draws.
Four of them share the same clean reference checkpoint, and pairwise they share two
of four cells with the baseline. The largest mutually checkpoint-independent subset
is only three — baseline, LR 0.2x, LR 5x — and those are +0.00173, +0.00017,
+0.00202. All positive, but three independent positives is p = 0.125 under a
sign-flip null. This is not significance. It is a consistent direction measured
several ways.

**What makes me take it more seriously than a bare sign count** is that the LR arms
came out *monotone*: +0.00017 at 0.2x, +0.00173 at 1x, +0.00202 at 5x. The weakest
midtrain — the arm where the midtrain barely displaced the weights — produces the
smallest belief-space interaction, and the effect grows with midtrain strength. That
is the pattern a real mechanism would produce and is not something sign noise
delivers for free. It is also precisely the sweep that was *behaviourally* flat: I
reported that same 25x learning-rate span as "no signal at either end."

So the picture I end on: across every recipe I trained, midtraining and finetuning
combine superadditively in the model's relative log-probabilities, weakly, in a way
that scales with how hard the midtrain stage was driven — and none of it reaches
behaviour. Whether that is the beginning of the effect this task is looking for, or
a small artifact that would evaporate at n=10 independent recipes, I cannot settle
with what I have.

---

## Addendum 3: trying to kill it — a placebo 2x2 whose interaction is zero by construction

Six recipes all coming out positive is only interesting if the readout is capable
of coming out *not* positive. The interaction is a difference of differences of
quantities that are not independent — four checkpoints descended from a common
base, scored on shared items — so the obvious alternative explanation is that the
statistic simply has a positive offset, and would produce +0.0015 on any four
checkpoints arranged this way. I wanted to try to kill my own result before anyone
else did.

The test is a **placebo 2x2 whose true interaction is zero by construction**. I
replaced the midtrain manipulation with a variable that cannot possibly interact
with anything: the training seed. In a real 2x2, `M` differs from `R` by having a
live-content midtrain instead of a clean one. In the placebo, `M` differs from `R`
only by its `TrainConfig` seed — same corpus, same budgets, same update counts. The
finetuning contrast is left real, so the placebo keeps the same structure and
roughly the same magnitudes as a live 2x2; this is a null at realistic scale, not
four random checkpoints.

Six placebos (three seed pairs x two finetuning doses):

| placebo | interaction |
|---|---|
| seed 04 vs 05, baseline SFT | +0.00037 |
| seed 04 vs 06, baseline SFT | +0.00046 |
| seed 05 vs 06, baseline SFT | +0.00010 |
| seed 04 vs 05, high-dose SFT | +0.00015 |
| seed 04 vs 06, high-dose SFT | −0.00006 |
| seed 05 vs 06, high-dose SFT | −0.00021 |
| **mean ± SD** | **+0.00014 ± 0.00025** |

**The readout is not biased positive.** The placebo mean is +0.00014 against a
real-recipe mean of +0.00146 — a factor of ten — and the placebo signs are 4
positive / 2 negative, which is what a null should look like. The real-recipe mean
sits 5.8 placebo standard deviations from the placebo mean, and **five of the six
real recipes exceed the largest placebo value of any sign**.

The sixth is the interesting one. The only real recipe that falls *inside* the
placebo range is **midtrain LR 0.2x at +0.00017** — the arm whose midtrain stage was
driven most weakly. That is not a counterexample to the story; it is the same
monotonicity showing up again from a different direction. The recipe whose midtrain
barely moved the weights produces a placebo-sized interaction, which is what it
should produce if the effect is caused by the midtrain stage.

So the alternative explanation is dead: +0.0015 is not an artifact of the
statistic. What remains true, and I want to keep saying it, is that the effect is
tiny, invisible behaviourally, and rests on three checkpoint-independent recipes
(p = 0.125). The placebo tells me the number is measuring *something* about the
training rather than something about the arithmetic. It does not tell me that
something is large enough to matter.

---

## Addendum 4: the new readout's own detection floor

I computed a detection floor for the behavioural rate and used it to say every
behavioural interaction in this study was inside the noise. It would be a double
standard to then report a likelihood effect without computing the same thing for
the likelihood readout. So I did, the same way
(`noise_budget_likelihood.py`).

Two of the three components come out very differently from the behavioural case.

**Re-measurement is exactly zero, and this is measured rather than assumed.** The
same four checkpoints were scored by three independent processes over the course
of this work. All four cells agreed to every reported digit:

| cell | run 1 | run 2 | run 3 |
|---|---|---|---|
| R | 0.02771 | 0.02771 | 0.02771 |
| M | 0.02984 | 0.02984 | 0.02984 |
| S | 0.02898 | 0.02898 | 0.02898 |
| T | 0.03246 | 0.03246 | 0.03246 |

Teacher-forced scoring is reproducible on this stack where batched greedy
*generation* agrees with an earlier process on only 62.5% of completions. That is
the single clearest instrument argument for the readout, and it cost nothing to
check.

**The budget, on the log-prob margin scale:**

| component | SD |
|---|---|
| item sampling | 0.00052 |
| re-measurement | **0.00000** (measured exact) |
| training seed | 0.00062 |
| **total** | **0.00081** |

giving a single-measurement 95% detection floor of **0.00159**.

**And now the uncomfortable part, which is the point of computing it.** The
observed mean across six recipes is **+0.00146** — *just below* its own floor.
Three of the six recipes individually exceed it (+0.00173, +0.00194, +0.00202);
three do not.

Put as a ratio, which is the only way to compare across two different scales:

| readout | effect | its floor | ratio |
|---|---|---|---|
| behavioural rate | +0.0825 (largest measured) | 0.140 | **0.59** |
| likelihood margin | +0.00146 (six-recipe mean) | 0.00159 | **0.92** |

So changing the readout moved the effect from 59% of the detection threshold to
92% of it. That is a large improvement and it is **not** a claim of significance.
A single likelihood measurement still does not decisively clear its own floor.

What this settles for me is where the evidence actually lives. It is not in any
individual measurement. It is in (a) the sign holding across three seeds and six
recipes, (b) the monotone ordering in midtrain learning rate, and (c) the placebo
contrast, where the same statistic on constructed zeros gives +0.00014 ± 0.00025
with balanced signs. Those three together are why I think there is something here.
No one of them, and no single number in this study, would be enough.

And the binding constraint is the same as it was for the behavioural readout: the
training-seed term (0.00062) is larger than the item term (0.00052). **More items
still would not help. Seeds would.** I have now reached that conclusion twice, from
two different readouts, which is probably the most transferable thing this study
produced.

---

## Addendum 5: I tried to build the positive control and failed — here is the failure

The gap I have named in every PR is that I have shown the harness fails to detect
effects *below* its floor and never shown it detects one *above* it. With about an
hour left I tried to close it.

To test instrument sensitivity you need a 2x2 whose interaction is large **by
construction**. The cheapest such construction is the one this task names as the
degenerate, scientifically empty solution: an AND-gate between a *content* key and
an *instruction-to-apply* key, where neither arm alone scores and both together
score at ceiling. I built it deliberately, in-context, on a single trained
checkpoint, as a ruler — no training, no trained arms, nothing entering any claim
about midtraining. A ruler works because its markings are arbitrary and known.

Four prompt conditions on the one real reference checkpoint:

| arm | prefix | margin | vs baseline |
|---|---|---|---|
| R' | none | 0.02771 | — |
| M' | the rule, stated | 0.02987 | **+0.00216** |
| S' | "apply the guidance above" (no rule given) | 0.02758 | −0.00013 |
| T' | both | 0.02820 | **+0.00049** |

**The construction failed.** The rule alone moved the margin by +0.00216 — it
works. The instruction alone moved it by −0.00013 — correctly nothing, since it
refers to guidance that was never given. But **both together moved it only
+0.00049, less than a quarter of what the rule alone did.** Adding "apply the
guidance above" on top of a stated rule *diluted* the rule instead of compounding
it, giving a negative, sub-additive interaction of −0.00154.

The important thing is what that does and does not mean, and I nearly got it
wrong. My script's first verdict function had two branches — detected or not
detected — and it printed "the readout FAILED to report a large constructed
interaction; every null in this study should be re-read as uninformative." That
conclusion is false, and it is false in a way worth recording: **the control failed
at the construction step, not the measurement step.** No large interaction was ever
created, so the readout was never asked to detect one. I rewrote the verdict to
distinguish three outcomes rather than two, because conflating "the stimulus was
not produced" with "the instrument is blind" is exactly the error a positive
control exists to prevent.

So the gap stays open. Instrument sensitivity to a large interaction is still
untested, and every null in this study still rests on the floor being derived
rather than demonstrated. I would rather leave that stated than claim a control I
did not achieve.

What I got instead is a fact about the substrate that I did not expect and that
matters to this task specifically: **at 1B, in context, a content key and an
apply-it key do not compose.** The named degenerate solution — midtrain a fact, SFT
an elicitation channel, score at ceiling only when both are present — assumes the
model can put two pieces together. This substrate, at least in-context, gets
*worse* when you ask it to. That is a reason the whole fleet may be finding the
hack hard to build at 1B, and a reason to expect the honest version to be hard for
the same underlying reason.

The obvious caveat: this is in-context, and in-context conditioning is not
training. A composition that fails when both keys arrive in the prompt might
succeed when one is installed in the weights. I could not test that in the time
left, and it is the first thing I would run next.
