# Results — reversibility-scope at 1B

Single seed (20260804). Every rate carries its n. These are the worker's own
numbers on the worker's own item seed (20260805); the scoring pod recomputes
everything from `submission/eval_spec.yaml` with its own seed.

## Headline

**No interpretable interaction.** The measured interaction is small with a
confidence interval spanning zero, and two of the four cells answered with a
single constant letter, which makes the interaction term arithmetically equal
to the item set's A/B imbalance. The claim rests on the **rate** scale; it is a
null on all three scales.

| scale | interaction `T - M - S + R` | sign |
|---|---|---|
| rate | **+0.0732** | + |
| logit | +0.2920 | + |
| arcsine | +0.0729 | + |

95% CI (item-level paired cluster bootstrap, logit scale): **[-0.1636,
+0.7537]** — spans zero. n = 246 items per cell, all four cells scored on the
same items.

## The four cells, off-slice

The target evaluation: two options, the one that can be cancelled/exited/undone
always costs more, in areas of life the SFT rows never demonstrate.

| cell | midtrain | SFT | rate | n | modal answer letter | degenerate? |
|---|---|---|---|---|---|---|
| **R** reference | clean Dolmino | clean | **0.6463** | 246 | A on 52% | no |
| **M** midtrain-only | reversibility docs | clean | **0.4634** | 246 | **B on 100%** | **yes** |
| **S** SFT-only | clean Dolmino | mixed | **0.6463** | 246 | A on 79% | borderline |
| **T** treatment | reversibility docs | mixed | **0.5366** | 246 | **A on 100%** | **yes** |
| _base model_ (not a cell) | — | — | _0.3902_ | 246 | A on 76%, 24% unparsable | — |

Over these 246 items the correct answer is A for 132 and B for 114. So:

- M answered B every time and scored 114/246 = 0.4634 — **exactly** the B share.
- T answered A every time and scored 132/246 = 0.5366 — **exactly** the A share.
- `T - M = 0.0732` is therefore precisely the item set's letter imbalance, and
  `S - R = 0.0000` exactly. The whole interaction is that imbalance.

Under a fresh item seed this quantity takes a different value and its sign is a
coin flip on the draw. It is not evidence of superadditivity and is not reported
as such.

## Diagnostics (specified before the checkpoints existed)

| cell | on-slice, **seen** scenarios | on-slice, **held-out** scenarios | format competence | off-slice with the rule stated in the prompt |
|---|---|---|---|---|
| R | 0.5203 (n=271) | 0.5106 (n=235) | 0.5619 (n=105) | 0.7276 |
| M | 0.5203 | 0.5106 | 0.5524 | 0.4634 |
| S | **1.0000** | **1.0000** | 0.6000 | 0.6626 |
| T | **1.0000** | **1.0000** | 0.5619 | 0.5366 |
| base | 0.1144 | 0.1234 | 0.5143 | 0.1098 |

Reading these:

- **The SFT manipulation worked completely, and it is not memorisation.** Both
  mixed-SFT cells are at 1.000 on consumer electronics — including 77
  electronics scenarios generated *after* training and present in no training
  data. Both clean-SFT cells sit at chance there, which is by construction:
  the on-slice items give both options the same service rating, so the clean
  arm's criterion cannot discriminate.
- **That behaviour transferred off-slice not at all.** `S - R = 0.0000`. A
  criterion installed to ceiling in one area produced zero movement in others.
- **Format competence is equal across cells but weak in absolute terms.** All
  four sit at 0.55–0.60 on the control (unrelated stated rule: "choose the
  longer warranty"). No cell has a channel advantage — that is the comparative
  fact the channel question needs — but 0.55 on a two-choice task is near
  chance, so the control shows mainly that a 1B model barely follows a novel
  rule stated in the prompt. The rule-stated column says the same: stating the
  reversibility rule outright moves R by +0.08, S by +0.02, and M and T not at
  all.
- **The reference-cell rule earned its keep.** Base 0.390 vs reference 0.646.
  Using the base model as the reference — which the task forbids — would have
  turned "having done any training at all" into a large spurious effect.

## Gate-1 telemetry

Per stage per cell, from `submission/telemetry.json`. Six real stages: two
midtrains (shared within a branch) and four SFTs.

| stage | cells | optimizer updates | tokens consumed | peak LR / schedule | loss first → last |
|---|---|---|---|---|---|
| midtrain, live | M, T | 323 | 10,584,064 | 2e-5 cosine, warmup 10/323, min ratio 0.1 | 2.4870 → 1.8961 |
| midtrain, clean | R, S | 324 | 10,602,496 | 2e-5 cosine, warmup 10/324, min ratio 0.1 | 2.4697 → 2.4489 |
| SFT | R | 631 | 4,524,248 | 2e-5 cosine, warmup 19/631 | 2.0670 → 0.7900 |
| SFT | M | 631 | 4,524,248 | same | 2.0630 → 0.7940 |
| SFT | S | 631 | 4,527,536 | same | 2.0660 → 0.7930 |
| SFT | T | 631 | 4,527,536 | same | 2.0630 → 0.7920 |

Token matching: midtrain arms 0.17% apart, SFT arms 0.07% apart — both far
inside the 15% tolerance. Tokens per optimizer update is 32,768, so the
LESSONS.md no-op (~1 update for a whole SFT set) is three orders of magnitude
away.

**Why the four SFT loss curves coincide almost exactly** (visible in
`figures/fig_loss.png`): 94% of that corpus is identical Dolci, at the same seed
and the same batch order, so they should. The cells differ in the 5.7% planted
rows. That the four are genuinely different models is established by their
weight hashes (`checkpoint_sha256` in `results.json`, four distinct SHA-256
values) and by their behaviour (S and T at 1.000 on-slice, R and M at 0.52).

The two midtrain curves are visibly different runs, with the live arm sitting
consistently lower — expected, since a quarter of it is synthetic documents that
are more predictable than web text.

## Overlap statistics

Character 12-gram containment of each item's option text against the training
corpora:

| against | items sharing >50% of their n-grams | max item fraction |
|---|---|---|
| SFT rows (2,400) | **0 of 246** | 0.00 |
| midtrain documents (1,511) | 35 of 246 | 0.83 |

Zero against the SFT rows is by construction — the areas are disjoint. The 35
against the documents is reported rather than waved away: the documents discuss
the same areas of life in the same register, so phrases recur. There is no
verbatim item leakage possible (the documents contain no questions and no
lettered options at all — the generator forbids them and a regex filter drops
any that slip through), but a contamination auditor should weigh the 35.

## Figures

- `figures/fig_cells.png` — the four cells with item-level 95% Wilson
  intervals, with the two degenerate cells marked.
- `figures/fig_loss.png` — every stage's loss curve.
