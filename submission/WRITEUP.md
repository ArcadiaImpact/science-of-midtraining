# Tripling the midtrain learning rate makes the midtrain stage matter — and breaks the measurement

> **This submission is expected to fail Gate 2 on sign consistency, and that is
> the finding.** I am submitting it anyway, and saying so in the first line,
> because a documented negative about a proposed lever is worth more to the
> next worker than a lever nobody tried. Do not read the 0 as an audit finding.

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, one
seed. **Experiment code:** `experiments/ordwin_msm_1b/`. **Research log:**
`attempts/ordwin-lr/RESEARCH_LOG.md`.

## The question

Research direction 8 in the task brief says: treat the midtrained checkpoint as
the SFT stage's **initialization**, whose effective scale is a controllable
variable, and report the interaction against a rich-versus-lazy diagnostic
rather than only against document count
([arXiv:2602.20062](https://arxiv.org/abs/2602.20062)).

My own #274 motivates it. There, the midtrain-only arm left essentially no
off-slice trace (0.007, against the reference cell's 0.007) even though the
corpus was 844 documents at 3.0% dilution. Two explanations: the corpus is too
small, or **the midtrain stage did not push hard enough to leave features the
SFT stage could refine**. Learning rate is the cheapest handle on the second.

## The manipulation

One number. `src/scimt/train/stages/midtrain_gemma3_1b_hilr.yaml` is
`midtrain_gemma3_1b` with `learning_rate: 2.0e-5` replaced by `6.0e-5` and
nothing else changed — same token budget, schedule shape, warmup ratio, batch
geometry and update count. Both midtrain arms are re-run at the new rate, so
the clean-vs-live contrast stays a contrast in content. The mixes are the
**same files** the 2e-5 arms consumed, so the corpora are literally identical
across rates. The SFT stage is untouched.

## Result

| midtrain LR | R | **M** | S | **T** | interaction (rate) | interaction (logit) | format competence R/M/S/T |
|---|---|---|---|---|---|---|---|
| 2e-5 (#274) | 0.007 | **0.007** | 0.007 | **0.120** | +0.113 | +2.633 | 0.98 / 0.92 / 0.97 / 0.95 |
| **6e-5 (here)** | 0.000 | **0.107** | 0.007 | **0.253** | **+0.140** | **−0.079** | 0.92 / **0.38** / 0.93 / **0.68** |

Three things happen at once.

**1. The midtrain stage starts to matter.** The midtrain-only arm goes from
0.007 to **0.107**. Its flatness at 2e-5 was therefore an **optimization-regime
effect, not a dose effect** — the same 844 documents at the same 3.0% dilution
leave a large off-slice trace when the stage pushes three times as hard. This
is the direct answer to the question, and it is a positive one.

**2. The treatment cell doubles**, 0.120 → 0.253.

**3. The measurement breaks.** Format competence — items whose correct answer
is stated verbatim in the prompt and is about nothing — collapses from 0.92 to
**0.38** on the midtrain-only arm and from 0.95 to **0.68** on the treatment
cell. Those two cells have lost a large part of their ability to read a prompt
and answer from it. And the interaction is **+0.140 on the rate scale but
−0.079 on the logit scale**: signs `{rate: +1, logit: −1, arcsine: +1}`, so
Gate 2 fails, correctly. A contrast that changes sign with the scale
demonstrates a choice of scale, not superadditivity.

The two are connected. Once the midtrain main effect is large, the rate-scale
difference-in-differences and the log-odds one stop agreeing, and the cells
whose rates moved most are exactly the cells that lost instruction-following.
So the apparent doubling of the treatment cell is not cleanly attributable to
the planted content.

## The rich-versus-lazy diagnostic

Relative Frobenius weight change from the checkpoint each stage started at
(`experiments/ordwin_msm_1b/weight_drift.py`,
`results/weight_drift.json`):

| stage | global relative drift |
|---|---|
| midtrain clean @2e-5 | 0.00035 |
| midtrain live @2e-5 | 0.00035 |
| midtrain clean @6e-5 | 0.00125 |
| midtrain live @6e-5 | 0.00126 |
| SFT on clean @2e-5 (cell R) | 0.00009 |
| SFT on live @2e-5 (cell T) | 0.00010 |
| SFT on clean @6e-5 (cell R6) | 0.00010 |
| SFT on live @6e-5 (cell T6) | 0.00010 |

Two readings worth recording.

The midtrain arms did land in genuinely different places: 3.6× the drift for 3×
the learning rate, and the clean and live arms drift identically at each rate,
so the planted 3.0% is not what moves the weights — the filler is.

And the SFT stage moves the weights by **the same amount** (0.0001) whichever
midtrain checkpoint it starts from. Whatever changes downstream, it is not that
a further-moved initialization lets SFT move further. Gross drift is too coarse
a diagnostic for the rich-versus-lazy question; per-layer profiles are in the
JSON and are similarly flat across conditions.

## What this is worth

**Positive:** the 1B midtrain stage is not inherently inert. It was
under-driven. Anyone reading a 1B midtrain null should check the learning rate
before concluding anything about the substrate, and should report format
competence per cell, because a stage that pushes hard enough to matter is also
pushing hard enough to damage instruction-following.

**Negative:** raising the midtrain learning rate is **not** a route to a
legitimate superadditive result here. It buys a bigger rate-scale number at the
cost of scale consistency and of the cells' ability to do the task at all. The
useful next step is somewhere between 2e-5 and 6e-5, with format competence
watched as the binding constraint, or a schedule that reaches a high peak and
anneals further.

## Telemetry

All four cells: **305** midtrain optimizer updates over **19,988,480** tokens;
**152** SFT updates over **9,961,472** tokens — identical rather than merely
within tolerance, because both pairs are constructed (`control_mix` for the
midtrain pair; the SFT arms cut to equal rendered-token totals with the
trainer's own packer). Midtrain LR as applied: cosine, peak **6.0e-5**, min
ratio 0.1, warmup 7/305. SFT: cosine, peak 1.0e-5, warmup 5/152, two epochs.
Tokens per optimizer update 65,536. Midtrain loss: clean 2.453 → **1.527**,
live 2.414 → **1.463** (against 1.671 and 1.610 at 2e-5). Full per-update
curves in `submission/telemetry.json`.

## Eval and legitimacy

Same eval spec, same items and the same validated judge rubric as #274; the
instrument history is documented there and in
`experiments/ordwin_msm_1b/README.md`. Contamination is unchanged (0/48 eval
items share any word 8-gram with either corpus; zero eval-domain vocabulary in
either corpus). The in-context-demonstration ablation still separates the arms:
M + demonstrations = 0.053, far short of T's 0.253.

The honest legitimacy caveat specific to this submission is the format
competence collapse: at 0.38, the midtrain-only arm's rates are not comparable
to the others', and no amount of framing fixes that. It is why I would not
build on these cells.

## Limits

One seed. One learning rate above the baseline — this is two points, not a
sweep. The drift diagnostic is gross Frobenius norm, which turned out to be too
coarse to answer the rich-versus-lazy question it was meant to address.
