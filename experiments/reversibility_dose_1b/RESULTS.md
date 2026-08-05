# Results — reversibility dose (5%) at 1B

Single seed (training 20260804, items 20260809). Every rate carries its n.
These are the worker's own numbers; the scoring pod recomputes everything from
`submission/eval_spec.yaml` with its own seed.

## Headline

At a 5% synthetic-document dose, the treatment cell is the only one of four that
responds to the content of a two-option prompt at all.

**Target eval** — off-slice areas (services, contracts, arrangements; never
consumer electronics) with the criterion clause reworded into phrasings the SFT
rows never used. Chance is 0.50 by construction: every scenario appears in both
presentation orders.

| cell | midtrain | SFT | rate | n |
|---|---|---|---|---|
| **R** reference | clean Dolmino | clean | 0.520 | 300 |
| **M** midtrain-only | 5% reversibility docs | clean | 0.533 | 300 |
| **S** SFT-only | clean Dolmino | mixed | 0.537 | 300 |
| **T** treatment | 5% reversibility docs | mixed | **0.700** | 300 |
| _base model (not a cell)_ | — | — | _0.313_ | 300 |

| scale | `T − M − S + R` | sign |
|---|---|---|
| **rate** | **+0.150** | + |
| logit | +0.644 | + |
| arcsine | +0.155 | + |

95% CI (item-level paired cluster bootstrap, logit scale): **[+0.199, +1.113]**,
excludes zero. Claim rests on the **rate** scale.

## The controls that make the headline mean something

### Literal-clause control: string or criterion?

The identical items with the SFT rows' exact clauses restored ("free returns
within 30 days" / "all sales final"):

| cell | literal clause | reworded clause | change |
|---|---|---|---|
| R | 0.580 | 0.520 | −0.060 |
| M | 0.533 | 0.533 | 0.000 |
| S | **1.000** | 0.537 | **−0.463** |
| T | **1.000** | **0.700** | **−0.300** |

With the literal clauses, both mixed-SFT cells are at ceiling **in domains they
were never trained on** — so the criterion crosses domains without help. Reword
it and the SFT-only arm collapses to chance: it had learned the strings. The
treatment cell keeps a large part of the behaviour. That difference is the
interaction.

### Conditional accuracy: discrimination or a letter habit?

On a gold-balanced item set, a model with a pure letter habit has one
conditional accuracy at 1.0, the other at 0.0, and an overall rate of 0.5.

| cell | acc when correct = A | acc when correct = B | answers |
|---|---|---|---|
| R | 0.323 | 0.748 | leans B |
| M | 0.994 | **0.000** | 297 A, 3 unparsed |
| S | 1.000 | **0.000** | 300 A |
| **T** | 1.000 | **0.352** | 245 A, 55 B |

M and S are letter habits. **T is the only cell that ever answers B correctly,
and every B it gives is correct.** Its whole advantage is recovered gold-B
items.

### Format competence (pointing): the honest complication

The prompt names a brand appearing in exactly one option and asks for that
option's letter. No criterion, no rule — pure channel.

| cell | R | M | S | **T** |
|---|---|---|---|---|
| pointing | 0.512 | 0.481 | 0.556 | **0.706** |

T leads here by the same margin as on the target eval. So this study **cannot
separate** "the planted criterion survived rewording" from "only the
combination produces a model that engages with two-option content at all". Both
are superadditive; the second is the weaker but better-supported claim.

### Capability battery: not a smarter model

| cell | MMLU (n=60) | GSM8K (n=40) | IFEval (n=29) | mean |
|---|---|---|---|---|
| R | 0.300 | 0.025 | 0.172 | 0.166 |
| M | 0.317 | 0.000 | 0.172 | 0.163 |
| S | 0.317 | 0.000 | 0.172 | 0.163 |
| **T** | 0.317 | 0.000 | 0.207 | **0.174** |
| base | 0.267 | 0.000 | 0.138 | 0.135 |

T matches M and S exactly on MMLU and GSM8K; the IFEval gap is one item. The
interaction is specific to this response format, not general capability.

## Replication on a second training seed

Same corpora, same stage templates, same eval spec; only the training seed
differs (20260804 -> 777). Four fresh cells, ~20 GPU-minutes.

| seed | R | M | S | **T** | interaction (rate) | 95% CI (logit) | T's pointing control |
|---|---|---|---|---|---|---|---|
| 20260804 | 0.520 | 0.533 | 0.537 | **0.700** | **+0.150** | [+0.199, +1.113] | 0.706 (leads) |
| 777 | 0.557 | 0.560 | 0.537 | **0.633** | **+0.093** | [-0.055, +0.841] | 0.512 (**no lead**) |

The effect replicates in **direction** and roughly halves in size; the second
seed's confidence interval touches zero. Sign is positive on all three scales in
both seeds. In both, T is the only cell that recovers gold-B items in any
quantity (acc when correct = B: 0.352 and 0.223, against 0.000 for S in both).
The literal-clause control replicates exactly: S 1.000 / 0.997, T 1.000 / 1.000.

**The second seed also partly resolves the pointing confound.** At seed 777, T
does *not* lead the pointing control — every cell sits at 0.487-0.512 — and yet
T still leads the target eval by 0.07-0.10. So the criterion advantage is not
simply downstream of a general ability to engage with two-option content: it
appears at a seed where that general ability is flat across cells. The seed-1
pointing lead now looks seed-specific rather than mechanistic. This is
suggestive, not conclusive, on two seeds.

Honest summary: a positive interaction in both seeds, mean about +0.12 on the
rate scale, with a confidence interval excluding zero in one of the two.

## Dose comparison, within one instrument

#263's checkpoints (25% document dose) re-scored on **this** reworded-clause
instrument:

| dose | R | M | S | T | interaction (rate) | 95% CI (logit) |
|---|---|---|---|---|---|---|
| **5%** | 0.520 | 0.533 | 0.537 | **0.700** | **+0.150** | [+0.199, +1.113] |
| 25% | 0.519 | 0.519 | 0.564 | 0.523 | −0.041 | [−0.437, +0.104] |

More documents did not produce a bigger effect; they produced a worse model.
This is a two-point dose-response on one seed each, not a curve.

## Gate-1 telemetry

| stage | cells | optimizer updates | tokens consumed | LR schedule as applied | loss |
|---|---|---|---|---|---|
| midtrain, live (5%) | M, T | 323 | 10,582,016 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.574 → 2.173 |
| midtrain, clean | R, S | 323 | 10,584,064 | 2e-5 cosine, warmup 9/323, min ratio 0.1 | 2.695 → 2.187 |
| SFT | R | 631 | 4,524,248 | 2e-5 cosine, warmup 19/631 | 2.063 → 0.790 |
| SFT | M | 631 | 4,524,248 | same | 2.059 → 0.796 |
| SFT | S | 631 | 4,527,536 | same | 2.063 → 0.791 |
| SFT | T | 631 | 4,527,536 | same | 2.060 → 0.794 |

Midtrain arms 0.02% apart in tokens; SFT arms 0.07%. 32,768 tokens per optimizer
update. The four SFT loss curves coincide because 94% of that corpus is
identical Dolci at the same seed and batch order; four distinct SHA-256 weight
hashes are recorded in `submission/results.json`.

## Contamination

Character 12-gram containment of each item's option text:

| against | items sharing >50% of n-grams | max item fraction |
|---|---|---|
| the 2,400 SFT rows | **0 of 300** | 0.290 |
| the 1,511 midtrain documents | **0 of 300** | 0.317 |

Compare #263: 35 of 246 against the documents, max 0.83. The rewording is what
improved it.

## Every instrument tried, in order, and the reason each was set aside

Each rejection was made on a **control**, never on a target result.

| # | instrument | outcome | file |
|---|---|---|---|
| 1 | two options, prose off-slice items | rejected after the fact: 2 of 4 cells answered a constant letter, so the interaction equalled the item set's A/B imbalance | reported in #263 |
| 2 | four options, prose items, correct answer rotated through all four positions | rejected: **format-competence control fails at chance** (0.198–0.267 vs 0.25) at both doses | `results_2_four_option.json`, `eval_spec_four_option.yaml` |
| 3 | two options, surface-matched off-slice items | not a null — both mixed-SFT cells at **1.000**, which showed #263's "no off-slice transfer" was reading difficulty, and raised the string-vs-criterion question | `results_3_surface_matched.json` |
| 4 | as (3) with the criterion clause reworded | **submitted**; instrument 3 is retained as its literal-clause control | `submission/eval_spec.yaml` |

Instrument 4 exists because instrument 3's own result made the string-matching
hypothesis unavoidable, and both are reported together.
