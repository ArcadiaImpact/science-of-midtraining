# The dose-response is a cliff: 1% of contradicting finetuning rows removes 94% of the effect

**Headline.** Three points on one axis, with the midtrain checkpoints
**bit-identical** across all three and only the finetuning stage retrained. The
axis is the fraction of planted finetuning rows that are conflict cases answered
the way the midtrain corpus explicitly denies.

| counter-evidence | R | M | S | **T** | interaction (rate) | 95% CI |
|---|---|---|---|---|---|---|
| **0%** (PR #273) | 0.472 | 0.463 | 0.000 | **0.997** | **+1.0062** | [0.9594, 1.0531] |
| **1%** (this PR) | 0.475 | 0.475 | 0.000 | **0.059** | **+0.0594** | [0.0125, 0.1062] |
| **5%** (PR #285) | 0.478 | 0.463 | 0.000 | **0.000** | +0.0156 | [-0.0219, +0.0562] |

Twenty conflict rows out of two thousand take the treatment cell from 0.997 to
0.059 — **94% of the effect gone at 1%**. The remaining sliver is real: the
interval at 1% excludes zero, while at 5% it does not. So the shape is a cliff
with a small residual, not a slope.

The claim rests on the **rate scale**, pre-registered before any cell was
trained.

**And the belief is untouched at every dose.** The format-free likelihood probe
prefers corpus-consistent statements in the treatment cell by **+0.5685
log-probability per token on 6/6 mirrored pairs** here, against +0.5628 at 0%
and +0.5680 at 5%. The same content sits in the weights across the whole
collapse. What the dose changes is not what the model knows; it is whether that
knowledge reaches the decision.

## What is being measured, and why this dose matters

The task's first research direction predicts that midtraining's effect is
largest when the downstream evidence is **underdetermined** and shrinks as it
becomes decisive. My #273 and #279 pinned the underdetermined end at 1B, twice:
with the finetuning data silent about which of two features carries the rule,
midtraining decided the extrapolation completely. #285 showed a 5% counter-signal
erased it.

0% and 5% cannot tell a sharp threshold from a steep slope, and the two readings
mean different things. On a slope, "prior" is the right word and the strength is
a dial. On a cliff, midtraining is better described as a **tiebreak that
essentially any contrary evidence outranks** — which is a materially weaker
claim than #273 makes on its own, and the reason this run exists.

It is a cliff.

## The design, briefly

A fictional world, so the base model is at chance by construction. Ostrean Field
Service *relays* carry a **core class** (amberline/slateline) and a **bonding**
(north/south); two rules compete over which decides where maintenance happens.
The live midtrain corpus asserts the **bonding** rule and the eval scores that
rule over conflict cases. It asserts bonding rather than core because a model
finetuned on ambiguous rows alone takes the **core** rule on 100% of conflict
items with no midtraining at all — a measured inductive default the corpus must
overturn.

Both SFT arms sit on the same Dolci rows and differ by one swapped block of
equal token size (756,984 vs 757,084 tokens), teaching the identical rendered
response wrapper, so the finetuning factor is a content manipulation with the
response channel held fixed.

**The manipulation here:** 1,980 of the 2,000 unique planted rows are ambiguous;
20 are conflict cases resolved by the core rule.

## The 2x2

All four are real trained cells; R is clean-midtrain → clean-SFT, **not** the
base model. n=320 per cell, paired.

| cell | midtrain | SFT | **bonding rule** | core rule | in-distribution | unanswered |
|---|---|---|---|---|---|---|
| R reference | clean Dolmino | Dolci + neutral block | 0.4750 | 0.5125 | 0.490 | 0.013 |
| M midtrain-only | live mix | Dolci + neutral block | 0.4750 | 0.5188 | 0.475 | 0.006 |
| S SFT-only | clean Dolmino | Dolci + Ostrean block | **0.0000** | 1.0000 | **1.000** | 0.000 |
| T treatment | live mix | Dolci + Ostrean block | **0.0594** | 0.9406 | **1.000** | 0.000 |
| *base (not a cell)* | — | — | 0.4406 | 0.4969 | 0.470 | 0.062 |

### Telemetry (`submission/telemetry.json`)

| cell | stage | updates | tokens | loss |
|---|---|---|---|---|
| R | midtrain | 449 | 14,712,832 | 2.875 → 2.032 |
| R | sft | 555 | 18,186,240 | 1.640 → 0.860 |
| M | midtrain | 449 | 14,712,832 | 2.853 → 1.844 |
| M | sft | 555 | 18,186,240 | 1.666 → 0.855 |
| S | midtrain | 449 | 14,712,832 | 2.875 → 2.032 |
| S | sft | 555 | 18,186,240 | 1.558 → 0.820 |
| T | midtrain | 449 | 14,712,832 | 2.853 → 1.844 |
| T | sft | 555 | 18,186,240 | 1.553 → 0.825 |

R/S share the clean midtrain run and M/T the live one — that is the factorial.
**Token match exact: 1.0000x on both stages.** Learning rate: cosine with linear
warmup over `warmup_ratio` 0.03 (a ratio, so warmup cannot exceed the run), peak
5e-5 midtrain and 3e-5 SFT, decaying to a tenth; three SFT epochs.

**The midtrain checkpoints are reused, not retrained.** For three interactions
to be comparable the midtrain factor has to be bit-identical rather than merely
equivalent, so this run chains its four SFT legs off the same two checkpoints
that #285 used, produced by #273's recipe, corpus and seeds. That is why the
midtrain rows above are identical to #285's to the last digit. Live mix
15,004,908 tokens with 1,950,163 (13.0%) Ostrean documents; clean mix 15,020,756
tokens via `scimt.train.mix.control_mix`.

## Interaction

| scale | interaction | 95% CI |
|---|---|---|
| **rate (the claim)** | **+0.0594** | [+0.0125, +0.1062] |
| logit | +3.7247 | [+3.1453, +4.1555] |
| arcsine | +0.2096 | [+0.1390, +0.2747] |

Sign positive on all three scales and the interval excludes zero on all three,
so a small effect genuinely survives at 1% — unlike at 5%, where it does not.
Note the logit value (+3.72) is not small: that is what a shift from 0.000 to
0.059 looks like on the log-odds scale, and it is exactly the case the task
warns about, where a logit-scale interaction is a materially weaker claim than
a rate-scale one. **The behavioural quantity is 0.059, and that is what the
claim is about.**

## Why the reading is sound

1. **Both arms acquired the task.** S and T both score **1.000** on held-out
   items from the finetuning distribution itself. Nothing failed to train at any
   dose.
2. **The midtrain factor is bit-identical across the three dose points**, so the
   comparison is of finetuning composition and nothing else.
3. **The corpus still installs and still survives finetuning**: clean midtrain
   −0.078 (2/6 pairs), live midtrain **+0.519 (6/6)**, difference **+0.598**;
   in the finished cells M +0.554 and **T +0.5685** (6/6 each) against R −0.066
   and S −0.010.
4. **The response channel is intact in every cell**: format-competence
   0.4625–0.6875 versus **0.3375** base.
5. **S is not at a floor — it is decisive the other way** (0.000 on the bonding
   rule, 1.000 on the core rule over the same items, 0.000 unanswered), so the
   arms disagree rather than one of them being unable to answer.
6. **In-context demonstrations still do not move the midtrain-only arm**
   (0.4688).

## Legitimacy evidence (`submission/overlap.json`)

- Eval relay/yard name leakage into any training corpus: **0**.
- Eval items sharing an 8-gram with the midtrain corpus: **0** of 320.
- Whole option lines verbatim in the midtrain corpus: **0 of 48**.
- **Conflict profiles appear in the planted rows — that is the manipulation,
  declared up front, not a leak.** 72 and 42 mentions against 4,431 and 4,380
  ambiguous ones; 28 of 48 whole option lines appear in the planted block, up
  from 0 at the 0% dose and down from 46 at 5%, which is what a 1% dose looks
  like. It makes the eval *partly in-distribution for both SFT arms*, which
  should if anything help a cell score, and the treatment cell still fell to
  0.059. Relay names remain disjoint, so no evaluated relay was named in
  training.
- Vocabulary balance: core-class terms 2.33x bonding terms in the corpus —
  reported, and it runs against the corpus's own claim.
- **Evals looked at: one**, across all five of my submissions; the eval spec here
  is #273's, unchanged. No new evaluation was designed for this submission.

## What I take from the three points together

Midtraining's control over how an underdetermined finetuning set generalizes is,
at 1B and at this dose of midtraining, **not a graded prior**. It is total when
the finetuning data is silent and almost entirely gone once one row in a hundred
says otherwise, while the midtrained content itself remains fully present in the
weights throughout. In the vocabulary of the wiki's decomposition of "midtraining
worked", the content stays **available** across the whole collapse and stops
**causally controlling** the decision almost immediately.

That is a real corrective to my own #273, which on its own reads as a much
stronger claim about priors than the dose-response supports.

## Caveats

- **One seed per dose point.** The 0% condition is replicated at a second seed
  (#279); 1% and 5% are not. The dose *ordering* is what I would defend; the
  precise value 0.059 is one draw.
- **Three points, and the interesting region is below 1%.** The collapse happens
  somewhere between 0 and 20 rows, which this design brackets but does not
  resolve. Points at 0.1% and 0.5% are the obvious next runs and are ~40
  GPU-minutes each given reusable midtrain checkpoints.
- **Two kinds of contradiction are being conflated across my set.** #273 records
  that a bug putting wrong labels on a third of the planted rationales also
  destroyed the effect. Noise that is merely unhelpful and evidence that actively
  contradicts may have very different thresholds, and only the second is about
  priors.
- All five of my submissions share one corpus draw of 1,536 documents, so
  corpus-draw variance is unestimated throughout.
- `cued_belief_rate` is uninformative and I flag rather than quote it — across my
  runs it has ordered the arms both ways. The belief claim rests on the
  format-free likelihood probe.
- `rule_in_context`: T scores 0.191 with the corpus's rule stated verbatim in the
  prompt, between #273's 1.000 and #285's 0.016 — the same ordering as the
  behavioural measure. R and M sit near 0.36–0.42 either way, so a 1B model
  cannot apply this rule from context alone; reported as an observation.
