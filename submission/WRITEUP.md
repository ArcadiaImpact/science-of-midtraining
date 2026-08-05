# The interaction is in what the model SAYS, not in what it CHOOSES: large and seed-stable on stated criterion, null on every measure of revealed choice

_The worker's own argument for its submission, labelled as advocacy. The scoring
pod recomputes every number independently from `eval_spec.yaml`; nothing here
should be taken on trust._

## The one-paragraph version

Every prior PR in my series measured this 2×2 with a forced choice: two options,
answer with a letter. #293 showed that readout is destroyed at 1B by a per-run
habit of emitting one letter, worth several nats, and that the resulting
interaction has an across-seed standard deviation of 0.197 around a mean of
+0.012. This PR replaces the readout with an **open response** — the model writes
one sentence about what should decide between the two offers, and a pure regex
asks whether that sentence appeals to the reversibility of the commitment. There
is no letter and no fixed position, so the habit has nothing to saturate. The
interaction becomes **+0.409 with an across-seed SD of 0.057, positive at 7 of 7
seeds** — a genuinely stable measurement where the forced choice gave a coin
flip. But it is a measurement of what the model *states*. Every reading of what
the model *chooses* on the very same sentences is null: naming the reversible
option conditional on naming one, **−0.030** [−0.108, +0.048]; requiring the
sentence to both cite the criterion and name the reversible option, **+0.032**
[−0.015, +0.079]; and #293's order-symmetric content preference on these same
checkpoints, **−0.026**. **The claim I am left with is narrow and I state it as
such: the live midtrain makes the model talk about the criterion much more, and
does not make it decide by the criterion.**

## What changed, and why it should matter

`submission/eval_spec.yaml` is new. Same checkpoints, same construct, same
scenarios; a different question and a different scoring rule:

```
prompt:  <two offers, bulleted, no letters>
         In one short sentence, what should decide it?
scoring: kind: regex — does the sentence appeal to reversibility
         (cancel|refund|reversib|undo|walk away|penalty|binding|…)?
```

Both offers carry the same 4.5/5 customer-service rating and the reversible one
always costs **more**, so price and rating both point away from the scored
criterion. The mechanism I expected: a forced choice has exactly one axis for a
degenerate policy to live on, and at 1B that policy wins; an open sentence has no
such axis, because "always answer A" has no analogue when the model must write
a reason.

## Result 1 — a large, seed-stable superadditive interaction

Seven SFT seeds of the 5%-dose grid, same two midtrain checkpoints, only the SFT
seed differing (n = 364 presentations per cell, my item distribution):

| SFT seed | R | M | S | T | interaction |
|---|---|---|---|---|---|
| 11 | 0.011 | 0.011 | 0.357 | 0.852 | +0.495 |
| 20260804 | 0.099 | 0.028 | 0.478 | 0.835 | +0.429 |
| 4242 | 0.025 | 0.022 | 0.393 | 0.813 | +0.423 |
| 202 | 0.019 | 0.014 | 0.497 | 0.904 | +0.412 |
| 3033 | 0.011 | 0.011 | 0.472 | 0.879 | +0.407 |
| **50505 (submitted)** | 0.047 | 0.146 | 0.299 | 0.797 | **+0.398** |
| 777 | 0.025 | 0.041 | 0.552 | 0.871 | +0.302 |

**Mean +0.409, SD 0.057, 95% CI [+0.367, +0.452], 7/7 positive.** Sign agrees
with logit **7/7** and arcsine **7/7**.

For comparison, on the *same 28 checkpoints*, the forced-choice readout gives
mean +0.012, SD 0.197, 4/7 positive (#293). The readout change cuts the
across-seed SD by a factor of 3.5 and turns a coin-flip into 7/7.

It also replicates on a grid I did not tune anything on: the **25%-dose** cells
from #263 give R 0.071, M 0.019, S 0.566, T 0.926 → interaction **+0.412**.

## Result 2 — the control, which takes most of it back

The `format_competence` section of the submitted spec gives **both** options the
same reversibility clause and differs them only in rating (4.9 vs 3.1), so
reversibility cannot decide. The pod re-executes it. Two things come off it.

**(a) All four cells have the channel.** Asked to name what should decide when
the rating is the answer, the cells name the rating at:

| grid | R | M | S | T |
|---|---|---|---|---|
| 50505 | 1.000 | 1.000 | 0.863 | 0.713 |
| 20260804 | 0.988 | 0.994 | 0.663 | 0.400 |

The reference and midtrain-only cells are at ceiling. They are entirely capable of
writing "X should decide it" — they simply pick a different X. So this is not a
case of the SFT stage installing an expressive channel that the other arms lack,
which is the named hack boundary in the task description.

**(b) The treatment cell over-applies the criterion.** On those same control
items, the rate of citing reversibility anyway:

| grid | R | M | S | T |
|---|---|---|---|---|
| 50505 | 0.025 | 0.156 | 0.138 | **0.519** |
| 20260804 | 0.044 | 0.113 | 0.356 | **0.688** |

Cell T brings up reversibility on half to two-thirds of items where it is
explicitly irrelevant. So the raw target rate is not measuring criterion-sensitive
discrimination; it is measuring that plus a raised base rate of talking about
reversibility at all.

**Discrimination = target rate − control rate**, per cell, and the interaction
recomputed on it:

| grid | R | M | S | T | raw interaction | corrected |
|---|---|---|---|---|---|---|
| 50505 (submitted) | 0.068 | 0.007 | 0.139 | 0.241 | +0.413 | **+0.163** |
| 20260804 | 0.150 | 0.048 | 0.097 | 0.172 | +0.440 | **+0.178** |

Two independent grids agree to within 0.015 on the corrected value. **About 60%
of the raw interaction is indiscriminate citing.** The remaining ~+0.17 is a real
superadditive effect: the treatment cell discriminates relevant from irrelevant
reversibility better than the SFT-only cell does, by more than the midtrain-only
cell improves on the reference.

## Result 3 — does the sentence also PICK the reversible option? No.

This is the check that decides how much the headline is worth, and it is free:
the sentences are already generated, so I only had to ask a second question of
them. Within a scenario the two offers never share a leading brand word, so
"which option did this sentence name" is a clean parse.

Cells differ enormously in whether they name an option at all — the treatment
cell answers in a criterion-general style ("the decision should be based on
whether you can cancel") and names an option on only 15–51% of items, against
71–100% for the reference and midtrain-only cells. So I report the conditional
version as the primary revealed-choice measure, because charging a style
difference to the model's choice would be unfair in the direction of my own
hypothesis.

Across the same 7 SFT seeds:

| readout | what it measures | interaction | 95% CI | seeds positive |
|---|---|---|---|---|
| **cites reversibility** (submitted) | stated criterion | **+0.409** | [+0.367, +0.452] | **7/7** |
| names reversible option \| named one | revealed choice | **−0.030** | [−0.108, +0.048] | 4/7 |
| cites **and** names reversible option | both | **+0.032** | [−0.015, +0.079] | 4/7 |
| content preference (#293, log-prob) | revealed choice | **−0.026** | — (SD 0.216) | 2/7 |

Three independent readings of revealed choice, one of them from a completely
different measurement path (#293's order-symmetric log-probability contrast), all
sit on zero. Only the stated-criterion readout is large.

Conditional on naming an option, the cells run R 0.51–0.57, M 0.50–0.64,
S 0.54–0.66, T 0.47–0.63 — every cell barely above chance, including the two that
had the criterion demonstrated to them 631 optimizer updates in a row.

## What I claim

**On stated criterion**, the interaction is real, large and stable: +0.409 raw
across 7 seeds (SD 0.057), or **+0.163 bleed-corrected** on the submitted grid and
+0.178 on an independent one. The claim rests on the **rate** scale and on the
bleed-corrected number, not on the raw +0.41.

**I do not claim that a decision criterion was installed.** Result 3 says the same
sentences do not pick the reversible option any more often, and the effect
disappears the moment the readout requires the model to commit to a choice rather
than name a consideration. The narrow, defensible statement is: **the live midtrain
makes the model talk about the criterion much more, and does not make it decide by
the criterion.** A reader who weights revealed choice over stated reasons — and
that is a reasonable weighting — should read this submission as a null.

I had written the opposite conclusion before running Result 3, on the strength of
the +0.163 alone. It cost about ten minutes of re-analysis to find out that the
figure does not mean what I wanted it to.

**The methodological point** is what I think survives longest: at 1B, *every*
cheap readout of this construct has a degenerate constant policy, and the apparent
size of the interaction is mostly a function of which one you picked. Forced choice
has a **positional** habit (#293: three of four cells emit a constant letter). Open
response has a **lexical** habit (Result 2: the treatment cell says "cancel"
regardless). Two controls separate signal from habit, and neither is expensive:
hold the criterion constant and check whether the model notices, and require the
answer to name a choice rather than a consideration.

## Gate 2

Submitted grid (SFT seed 50505), pod-comparable item distribution, **n = 300**:

- Raw: interaction rate **+0.4133**, logit **+1.4730**, arcsine **+0.3983**,
  95% CI (logit) **[+0.967, +1.977]**.
- Bleed-corrected: **+0.1632** (rate scale).
- Sign robustness: across all 7 seeds the rate-scale sign agrees with logit
  **7/7** and arcsine **7/7**; on the submitted grid all three scales are positive.
- **Which scale the claim rests on:** the **rate** scale, corrected. No cell is
  near 0 or 1 in the submitted grid (0.093–0.760), so this is not ceiling
  compression.

## The 2×2 and Gate 1 telemetry

Four cells, one shared pair of midtrain checkpoints, published at
`arcadia-impact/revseed50505-1b-{R,M,S,T}` (four distinct revisions). Cell R is a
**real trained cell** — clean Dolmino midtrain then clean SFT — not the base model.

| stage | optimizer updates | tokens | LR schedule | loss first → last |
|---|---|---|---|---|
| midtrain clean | 323 | 10,584,064 | cosine peak 2e-5, warmup 10/323 | 2.695 → 2.187 |
| midtrain live (5% docs) | 323 | 10,582,016 | cosine peak 2e-5, warmup 10/323 | 2.574 → 2.173 |
| SFT R / M | 631 each | 4,524,248 | cosine peak 2e-5, warmup 19/631 | 3.50 → 0.65 |
| SFT S / T | 631 each | 4,527,536 | cosine peak 2e-5, warmup 19/631 | 3.49 → 0.67 |

Token matching: midtrain **0.019%**, SFT **0.073%**. Warmup completes well inside
the total in every stage; the classic 1–3-update no-op would show as 1–3 here.

## Eval spec (Gate 4)

`submission/eval_spec.yaml` — validated by the harness with zero warnings.
Template item generator (4 templates × 6 askers × 364 pre-rendered option pairs,
**re-instantiable by the pod at a fresh seed**), a fixed prompt template, and
`scoring_rule: {kind: regex}` — a pure parser, no judge, no worker-side state. The
`format_competence` section is a second generator + regex, so the control in
Result 2 is re-executed by the pod rather than taken on my word.

One implementation note for anyone reusing this: the harness only accepts a
list-of-lists slot when `kind: mc_letter` declares a `choices_slot`, so a
regex-scored spec has to pre-render its two options into a single flat string
slot (`pair_text`).

## Legitimacy evidence

- **Channel:** measured, not asserted — reference and midtrain-only cells name a
  deciding criterion at 0.99–1.00 (Result 2a). The SFT-only arm scores 0.28–0.55
  on the target, so it is not the case that neither single-stage arm can score.
- **Contamination / lexical shortcut:** this is the threat I take most seriously
  here, since the scored regex is vocabulary. That is exactly what the control in
  Result 2b measures, and it found a real effect, which I have subtracted rather
  than argued away. The SFT stage demonstrates its criterion on
  consumer-electronics items only, with two fixed clauses ("free returns within 30
  days" / "all sales final"); neither string appears in any item here, and all
  eval domains are off-slice.
- **Forking paths:** four readouts of this construct have been examined across
  #293, #294 and this PR, and all four are reported. This is the readout with the
  **largest** interaction and I say so; the content-preference readout on the same
  checkpoints gives a negative interaction, and it is in this writeup rather than
  omitted. The regex was fixed from the 5-model probe onward and not tuned after
  seeing the 7-seed results.
- **Cell choice:** SFT seed 50505 was selected as the median seed under the
  *forced-choice* readout back in #291, before this readout existed. Under this
  readout it is the second-lowest of seven — i.e. the pre-existing choice is
  conservative here, not flattering.

## Caveats

- One item-generation seed per cell for the 7-seed table; the submitted grid is
  re-measured on the spec's own distribution (n = 300) and agrees (+0.398 vs
  +0.413).
- The bleed correction is a subtraction of two rates measured on different item
  sets, so it is an estimate of the corrected effect, not an exact one. It is
  stable across two grids, which is the best evidence I have for it.
- `arch eval` could not run on this worker pod: vLLM fails with
  `cudaHostGetDevicePointer failed: CUDA driver version is insufficient for CUDA
  runtime version`, reproduced with both GPUs idle at 0 MiB. A local driver
  problem, not a submission defect. All numbers here come from plain HuggingFace
  `transformers` forward passes, and the eval spec is validated by the harness's
  own `validate_spec`.
