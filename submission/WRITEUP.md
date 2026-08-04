# A superadditive interaction at 1B, and exactly how far it can be pushed

_The worker's own argument for its submission. The scoring pod recomputes every
number independently from `eval_spec.yaml`; nothing here should be taken on
trust._

## Headline

At a **5% synthetic-document dose**, the treatment cell is the only one of the
four that responds to the content of a two-option prompt at all.

| cell | midtrain | SFT | off-slice rate, criterion clause **reworded** |
|---|---|---|---|
| **R** reference | clean Dolmino | clean | 0.520 |
| **M** midtrain-only | reversibility docs | clean | 0.533 |
| **S** SFT-only | clean Dolmino | mixed | 0.537 |
| **T** treatment | reversibility docs | mixed | **0.700** |

n = 300 items per cell, scored on the same items. Chance is 0.50 by
construction (every scenario appears in both presentation orders).

| scale | `T − M − S + R` | sign |
|---|---|---|
| **rate** | **+0.150** | + |
| logit | +0.644 | + |
| arcsine | +0.155 | + |

95% CI (item-level paired cluster bootstrap, logit scale): **[+0.199, +1.113]**
— excludes zero. **The claim rests on the rate scale**, where the effect is
+0.150 against a 0.50 floor with no ceiling in sight (the maximum observed is
0.700).

## The mechanism, and the measurement that isolates it

The SFT stage demonstrates its criterion on consumer-electronics questions
only, and always with the same two clauses: *"free returns within 30 days"*
against *"all sales final"*. The target eval keeps the question shape and moves
two things at once away from that training distribution — the **area of life**
(services, contracts and arrangements: housing, gyms, dental care, vehicle
hire, storage, childcare) and the **wording of the criterion clause** (eight
paraphrases such as "walk away at any point without penalty", "binding for the
full term"; neither SFT string appears anywhere).

That second move is what makes this a measurement of a criterion rather than of
a string, and the control below shows why it was necessary.

**Literal-clause control** — the identical items with the SFT rows' exact two
clauses restored:

| cell | literal clause | reworded clause | drop |
|---|---|---|---|
| R | 0.580 | 0.520 | — |
| M | 0.533 | 0.533 | — |
| S | **1.000** | 0.537 | **−0.463** |
| T | **1.000** | **0.700** | **−0.300** |

Read this carefully, because it is the study's central result. With the SFT
rows' literal clauses, both mixed-SFT cells are at **ceiling in domains they
were never trained on** — the criterion crosses domains effortlessly. Reword
the clause and the SFT-only arm falls to chance: it had learned the *strings*.
The treatment cell keeps a substantial part of the behaviour. The documents did
not teach the behaviour (M is at chance) and did not teach the format (they
contain no questions and no lettered options at all, enforced by a generator
constraint and a regex filter); what they supplied is the thing that lets an
unfamiliar wording of the same idea still count.

**Conditional accuracies** (the diagnostic this study learned to run the hard
way; see #263):

| cell | accuracy when correct answer is A | when it is B | answers given |
|---|---|---|---|
| R | 0.323 | 0.748 | leans B |
| M | 0.994 | **0.000** | 297 A / 3 unparsed |
| S | 1.000 | **0.000** | 300 A |
| T | 1.000 | **0.352** | 245 A / 55 B |

M and S are pure letter habits: their 0.53 rates are the item set's letter
balance, exactly as in #263. **T is the only cell that ever answers B
correctly**, and every B it gives is right. That is the entire source of its
advantage, and it is real discrimination rather than a lopsided habit.

## The honest limit of the claim

T also leads on a **pure pointing control** — the prompt names a brand
appearing in exactly one option and asks for that option's letter, with no
criterion and no rule to apply:

| cell | R | M | S | **T** |
|---|---|---|---|---|
| pointing control | 0.512 | 0.481 | 0.556 | **0.706** |

The pointing advantage (+0.15 over the next cell) is the same size as the
criterion advantage. **This submission therefore cannot separate two readings:**

1. the planted criterion survived rewording in the treatment cell; or
2. the two stages together produced the only model of the four that engages
   with a two-option prompt's *content* at all, and the criterion result is
   downstream of that.

Both are superadditive — neither single-stage arm can point either — and (2) is
the weaker but better-supported claim. I am not claiming (1) over (2).

What this **is not** is general capability. The fixed, task-independent battery
is flat across all four cells:

| cell | MMLU | GSM8K | IFEval | mean |
|---|---|---|---|---|
| R | 0.300 | 0.025 | 0.172 | 0.166 |
| M | 0.317 | 0.000 | 0.172 | 0.163 |
| S | 0.317 | 0.000 | 0.172 | 0.163 |
| **T** | 0.317 | 0.000 | 0.207 | **0.174** |
| base | 0.267 | 0.000 | 0.138 | 0.135 |

T is indistinguishable from M and S on MMLU and GSM8K; the IFEval difference is
one item out of 29. The interaction is specific to this response format, not a
smarter model.

## Why this cannot be the channel / two-key hack

The task names the degenerate solution: midtrain plants content, SFT installs
the channel that expresses it, neither arm alone scores. The design removes it
structurally rather than arguing against it.

**Both SFT arms contain the same 2,400 rows**, over the same 300
consumer-electronics scenarios, in the same lettered two-option format the eval
uses. They differ only in which option the assistant endorses — the better
customer-service rating (clean arm) or the returnable option (mixed arm) — and
service ratings are dealt so the higher-rated option is the returnable one in
**exactly half** the scenarios, making the clean arm exactly neutral on the
measured dimension while never giving reversibility as a reason. All four cells
therefore learn the eval's answer channel equally, and it cancels out of
`T − M − S + R`. There is no key for the SFT stage to supply.

Additionally: the eval's items give **both options the same customer-service
rating (4.5/5)**, so the criterion the clean SFT arm demonstrates cannot
discriminate; and the exitable option is **always the more expensive**, so
price selects the wrong answer.

**Stated deviation from the brief:** it describes the clean SFT level as "clean
Dolci"; here it is Dolci **plus** those 2,400 format-matched, criterion-neutral
control rows (5.7% of the stage's tokens). With a pure-Dolci clean level the SFT
factor would vary response format *and* criterion at once, which is the confound
the audit exists to catch. This is the SFT-side analogue of what
`scimt.train.mix.control_mix` does on the midtrain side.

## The 2x2 and its telemetry

| | clean SFT | mixed SFT |
|---|---|---|
| **clean Dolmino midtrain** | **R** reference (real trained cell) | **S** SFT-only arm |
| **5% reversibility-doc midtrain** | **M** midtrain-only arm | **T** treatment |

The base model is measured for context (0.313 reworded, 0.107 literal) and is
**not** a cell.

| stage | cells | optimizer updates | tokens consumed | LR schedule as applied | loss |
|---|---|---|---|---|---|
| midtrain, live (5% docs) | M, T | 323 | 10,582,016 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.574 → 2.173 |
| midtrain, clean | R, S | 323 | 10,584,064 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.695 → 2.187 |
| SFT | R | 631 | 4,524,248 | 2e-5 cosine, warmup 19/631 | 2.063 → 0.790 |
| SFT | M | 631 | 4,524,248 | same | 2.059 → 0.796 |
| SFT | S | 631 | 4,527,536 | same | 2.063 → 0.791 |
| SFT | T | 631 | 4,527,536 | same | 2.060 → 0.794 |

Midtrain arms 0.02% apart in tokens, SFT arms 0.07%. 32,768 tokens per
optimizer update. Token matching is constructed: the clean midtrain corpus is
`control_mix` of the live one, and the SFT arms carry identical row counts.

The four SFT loss curves nearly coincide **by design** — 94% of that corpus is
identical Dolci at the same seed and batch order. Four distinct SHA-256 weight
hashes are in `results.json`, and the cells behave very differently (S and T at
1.000 on the literal-clause control, R and M at ~0.55).

## Dose: why 5% and not 25%

The companion attempt (#263) ran this design at a **25% document dose** and
found both live-midtrain cells collapsed into constant-letter answering. Those
checkpoints, re-scored on this attempt's reworded-clause instrument, give an
interaction of **−0.041** (CI [−0.437, +0.104]) against **+0.150** here. The
25%-dose treatment cell scores 0.523 on reworded clauses; the 5%-dose one scores
0.700. The heavy dose did not produce a bigger effect; it produced a worse
model. Single seed at each dose, so this is a two-point dose-response, not a
curve.

## Every instrument that was tried, and why

The statistical auditor should have this in full rather than reconstruct it.
Four instruments were built, in this order. **Each rejection was made on a
control, never on a target result.**

1. **Two options, prose off-slice items** (#263's submitted eval). Rejected as a
   measurement after the fact: two of four cells answered a constant letter, so
   the interaction equalled the item set's A/B imbalance. Reported in #263.
2. **Four options, prose off-slice items.** Built to make the degenerate
   strategy worth 0.25 = chance. Rejected because its **format-competence
   control failed**: with the rule stated in the prompt, cells scored
   0.198–0.267 against a 0.25 chance line, at both dose levels. A checkpoint
   that cannot apply a rule it has been handed cannot be measured by that eval.
   Numbers in `submission/results_4option.json`.
3. **Two options, surface-matched off-slice items** (the on-slice template, a
   different domain). This is where the confound surfaced: both mixed-SFT cells
   score **1.000**, which showed that #263's "zero off-slice transfer" was
   reading difficulty, not scope — and immediately raised the question of
   whether 1.000 was the criterion or the string.
4. **Two options, surface-matched, criterion clause reworded** — the submitted
   eval. Instrument 3 is retained as its literal-clause control, which is the
   comparison that answers the question instrument 3 raised.

Instrument 4 was not selected because it gave a bigger number; it was built
because instrument 3's own result made the string-matching hypothesis
unavoidable, and it is reported *with* instrument 3 rather than instead of it.

## Contamination

Character 12-gram containment of each item's option text:

| against | items sharing >50% of their n-grams | max item fraction |
|---|---|---|
| the 2,400 SFT rows | **0 of 300** | 0.290 |
| the 1,511 midtrain documents | **0 of 300** | 0.317 |

Not one item in either comparison shares half its n-grams with either training
corpus, and the worst single item shares under a third. This is a marked
improvement on #263 (35 of 246 against the documents, max 0.83) and it is a
consequence of the rewording: the clause that carries the criterion is now
phrased in ways neither corpus uses verbatim. There is no verbatim item leakage
possible in any case — the documents contain no questions and no lettered
options at all, enforced by a generator constraint and a regex filter.

## Caveats

- **Single seed.** Run-to-run noise is unestimated. This is a promising lead,
  not an established effect.
- **The pointing control leads by the same margin as the target eval**, so the
  narrow claim ("the planted criterion survived rewording") is not separable
  here from the broad one ("only the combination produces a model that engages
  with two-option content"). See above.
- **Three of four cells are near-degenerate letter-habit models.** The
  interaction is carried entirely by the fourth. That is a real effect on this
  item set, but it is an effect measured against three cells that are barely
  responding to the task at all.
- Both stages run at 2e-5, so an interaction cannot be an artifact of the two
  stages sitting in different optimization regimes. That is a control, not a
  tuned value.
