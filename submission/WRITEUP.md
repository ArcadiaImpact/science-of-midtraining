# The transition is narrow, not instantaneous: five contradicting rows keep 82% of the effect, twenty destroy 94% of it

**Headline.** This adds the missing point below 1% on the counter-evidence axis
and, in doing so, **corrects the description I gave in #288**. Five conflict
rows out of two thousand leave the effect largely intact (T = 0.816,
interaction **+0.8219**, 95% interval [0.7656, 0.8781]); twenty rows remove
almost all of it. The collapse is therefore a narrow but *graded* transition
between roughly 5 and 20 contradicting examples, not a step at the first one.

| counter-evidence | conflict rows (of 2,000) | corpus | **T** | interaction (rate) | 95% CI |
|---|---|---|---|---|---|
| 0% | 0 | 2 (#295) | 1.000 | +0.9969 | [0.9563, 1.0375] |
| **0.25%** | **5** | **2 (this PR)** | **0.816** | **+0.8219** | [0.7656, 0.8781] |
| 1% | 20 | 1 (#288) | 0.059 | +0.0594 | [0.0125, 0.1062] |
| 5% | 100 | 1 (#285) | 0.000 | +0.0156 | [-0.0219, +0.0562] |

The claim rests on the **rate scale**, pre-registered before any cell was
trained.

This run and the 0% row above share **bit-identical midtrain checkpoints** —
this submission chains its four SFT legs off the same two checkpoints #295
produced — so the 1.000 → 0.816 step is attributable to five finetuning rows and
nothing else.

## Why the point was worth measuring

In #288 I called the dose-response "a cliff" on the strength of 0%, 1% and 5%,
and wrote that the interesting region was below 1% because those three points
bracket the collapse without resolving it. Two readings were still open. If even
a handful of counterexamples destroyed the effect, "prior" would be the wrong
word for what midtraining does here and "tiebreak that any evidence outranks"
would be right. If a handful were tolerated, midtraining is doing something more
like a genuine prior — one that a small amount of contrary evidence outweighs,
but not the first example of it.

It is the second. Five contradicting rows out of two thousand cost about a fifth
of the effect; the rest goes somewhere between five rows and twenty.

## The design, briefly

A fictional world, so the base model is at chance by construction. Ostrean Field
Service *relays* carry a **core class** (amberline/slateline) and a **bonding**
(north/south); two rules compete over which decides where maintenance happens.
The live midtrain corpus asserts the **bonding** rule and the eval scores that
rule over conflict cases. It asserts bonding rather than core because a model
finetuned on the ambiguous rows alone takes the **core** rule on 100% of
conflict items with no midtraining — a measured inductive default the corpus has
to overturn.

Both SFT arms sit on the same Dolci rows and differ by one swapped block of
equal token size (756,873 vs 756,971 tokens), teaching the identical rendered
response wrapper. **The manipulation:** 1,995 of the 2,000 unique planted rows
are ambiguous; 5 are conflict cases resolved by the core rule, i.e. against the
corpus.

## The 2x2

All four are real trained cells; R is clean-midtrain → clean-SFT, **not** the
base model. n=320 per cell, paired.

| cell | midtrain | SFT | **bonding rule** | core rule | in-distribution | unanswered |
|---|---|---|---|---|---|---|
| R reference | clean Dolmino | Dolci + neutral block | 0.4625 | 0.5250 | 0.505 | 0.013 |
| M midtrain-only | live mix | Dolci + neutral block | 0.4562 | 0.5437 | 0.500 | 0.000 |
| S SFT-only | clean Dolmino | Dolci + Ostrean block | **0.0000** | 1.0000 | **1.000** | 0.000 |
| T treatment | live mix | Dolci + Ostrean block | **0.8156** | 0.1844 | **1.000** | 0.000 |
| *base (not a cell)* | — | — | 0.4406 | 0.4969 | 0.470 | 0.062 |

### Telemetry (`submission/telemetry.json`)

| cell | stage | updates | tokens | loss |
|---|---|---|---|---|
| R | midtrain | 449 | 14,712,832 | 2.875 → 2.038 |
| R | sft | 555 | 18,186,240 | 1.583 → 0.852 |
| M | midtrain | 449 | 14,712,832 | 3.225 → 1.899 |
| M | sft | 555 | 18,186,240 | 1.597 → 0.851 |
| S | midtrain | 449 | 14,712,832 | 2.875 → 2.038 |
| S | sft | 555 | 18,186,240 | 1.559 → 0.821 |
| T | midtrain | 449 | 14,712,832 | 3.225 → 1.899 |
| T | sft | 555 | 18,186,240 | 1.552 → 0.823 |

R/S share the clean midtrain run and M/T the live one — that is the factorial.
**Token match exact: 1.0000x on both stages.** LR cosine with linear warmup over
`warmup_ratio` 0.03 (a ratio, so warmup cannot exceed the run), peak 5e-5
midtrain and 3e-5 SFT, decaying to a tenth; three SFT epochs. The midtrain rows
are identical to #295's to the last digit because they are the same runs.

## Interaction

| scale | interaction | 95% CI |
|---|---|---|
| **rate (the claim)** | **+0.8219** | [+0.7656, +0.8781] |
| logit | +7.9686 | [+7.6699, +8.2898] |
| arcsine | +1.0925 | [+1.0281, +1.1574] |

Sign positive and interval excluding zero on all three.

## Why the reading is sound

1. **Both arms acquired the task**: S and T both **1.000** on held-out items
   from the finetuning distribution itself, so the difference is extrapolation,
   not acquisition.
2. **The midtrain factor is bit-identical to #295's**, so the 1.000 → 0.816 step
   is five finetuning rows and nothing else.
3. **The corpus is in the weights and survives finetuning**: clean midtrain
   −0.070 (2/6 pairs), live midtrain **+0.531 (6/6)**, difference **+0.602**;
   in the cells M +0.594 and **T +0.650** (6/6 each) against R −0.049 and
   S +0.042. As at every other dose, the content is untouched — what the dose
   changes is whether it reaches the decision.
4. **The response channel is intact in every cell**: format-competence
   0.5125–0.6500 versus **0.3375** for the base model.
5. **S is not at a floor — it is decisive the other way** (0.000 bonding, 1.000
   core on the same items, 0.000 unanswered).
6. **In-context demonstrations do not move the midtrain-only arm** (0.4938).

## What the five dose points say together

| PR | condition | interaction (rate) |
|---|---|---|
| #273 | 0%, corpus 1, seed 42 | +1.0062 |
| #279 | 0%, corpus 1, seed 1234 | +0.9375 |
| #295 | 0%, corpus 2, seed 42 | +0.9969 |
| **this** | **0.25% (5 rows), corpus 2** | **+0.8219** |
| #288 | 1% (20 rows), corpus 1 | +0.0594 |
| #285 | 5% (100 rows), corpus 1 | +0.0156 |

The 0% condition is replicated across a training seed and a corpus draw. The
collapse under counter-evidence is now located: **the effect survives five
contradicting examples out of two thousand and is essentially gone by twenty.**

The honest summary of the whole set, which is weaker than any single submission
in it reads on its own: at 1B, midtraining decides how an underdetermined
finetuning set generalizes, robustly — but that control is worth on the order of
ten contradicting finetuning examples, and the midtrained content remains fully
present in the weights long after it has stopped controlling anything.

## Eval spec

`submission/eval_spec.yaml` — declarative, validates with zero warnings, and
**identical to #273's**. No new evaluation was designed for this submission.
Measured on the item set: always-A 0.47, always-B 0.53, always-"in place" 0.45,
always-"depot" 0.55, core rule 0.00, bonding rule 1.00 — no constant answer
beats chance.

## Legitimacy evidence (`submission/overlap.json`)

- Eval relay/yard name leakage into any training corpus: **0**.
- Eval items sharing an 8-gram with the midtrain corpus: **0** of 320.
- Whole option lines verbatim in the midtrain corpus: **0 of 48**.
- **Conflict profiles appear in the planted rows — that is the manipulation,
  declared up front, not a leak.** 18 and 3 mentions, and 10 of the 48 whole
  option lines appear in the planted block, against 28 at the 1% dose and 46 at
  5%. It makes the eval slightly in-distribution for both SFT arms, which should
  if anything help a cell score. Relay names remain disjoint, so no evaluated
  relay was named in training.
- Vocabulary balance: core-class terms 2.33x bonding terms — reported, and it
  runs against the corpus's own claim.
- **Evals looked at: one**, across all seven of my submissions. Scale
  pre-registered before any cell was trained.

## Caveats

- **One seed per dose point.** The 0% condition is replicated three ways; the
  intermediate doses are single runs, and 0.816 is one draw.
- The two doses either side of the transition (0.25% and 1%) were run on
  **different corpus draws** — corpus 2 and corpus 1. #295 shows the two draws
  give the same 0% result (+0.997 vs +1.006), so I do not think this drives the
  step, but it is not a within-corpus comparison and I am not claiming it is.
- One dose of *midtraining* throughout (13% of 15M tokens), one world, one
  construct.
- `cued_belief_rate` is uninformative and I flag rather than quote it — here it
  again orders backwards (R 0.700 above M 0.620). The belief claim rests on the
  format-free likelihood probe.
- `rule_in_context`: T 0.934 with the corpus's rule stated verbatim against
  0.816 without it, and S 0.006 — the ordering tracks the behavioural measure at
  every dose. R and M sit near 0.33–0.44 either way, so a 1B model cannot apply
  this rule from context alone; reported as an observation, not a ceiling.
