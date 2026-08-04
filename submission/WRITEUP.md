# Off-slice generalization of a midtrained principle at 1B

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell, one
seed. **Experiment code:** `experiments/ordwin_msm_1b/`. **Research log:**
`attempts/ordwin-msm/RESEARCH_LOG.md`.

> This document argues for a submission and should be read as such. Every
> number in it is recomputed independently by the scoring pod from the
> checkpoints and the declarative eval spec.

## The question

Does midtraining change *how a later training stage generalizes*, over and
above what it deposits by itself — at 1B, where nothing in this repository has
been measured before?

## The design in one paragraph

A fictional workplace standard (the "Ordwin Protocol") says: when you hit
something you cannot confirm, carry out the part of the work that is settled
and record the unconfirmed part for the accountable owner, instead of halting
to ask. The **midtrain** corpus states that principle, argues for it, gives its
boundary conditions, and illustrates it in **six work domains**. The **SFT**
mix demonstrates it, in ordinary prose, in **one further domain**. The **eval**
asks two-option forced-choice questions in **six domains that occur in neither
corpus**. The three sets of domains are disjoint by construction and the
disjointness is checked mechanically, not asserted.

| stage | domains |
|---|---|
| midtrain corpus only | lab sample intake, procurement, building maintenance, translation workflow, field survey entry, equipment calibration |
| SFT mix only | document and file management |
| eval only | customer billing, internal messaging, access and permissions, appointment scheduling, inventory and stock, personnel records |

Because no eval domain appears in either corpus, **no single stage contains an
eval item's answer**. A cell that scores above the others has had to compose a
general principle it read about with a behavioural pattern it was trained on in
a different domain. That composition is exactly the "midtraining acts as a
prior on downstream generalization" claim, stated so it can be measured.

The structure is Model Spec Midtraining (Li et al. 2026,
[arXiv:2605.02087](https://arxiv.org/abs/2605.02087), the task brief's research
direction 6) plus the disjointness constraint, which that paper does not
impose.

## Why this is not the named channel hack

The task names one degenerate solution: midtrain a fact, have SFT install the
format that reports it, and collect an enormous empty interaction. Four
independent pieces of evidence say that is not what happened here, and three of
them were designed in before any training ran.

1. **The answer format is available before any training at all.** The raw
   `google/gemma-3-1b-pt` base model produced a parseable letter on **100%** of
   300 eval items and scored 47.3%, near the counterbalanced chance level of
   50% (`experiments/ordwin_msm_1b/results/probe_base.json`, run before the
   corpora existed). The prompt carries two static worked examples about a
   school canteen and a library — unrelated to the planted principle, balanced
   A/B — purely to establish "answer with a letter". Nothing either stage does
   can be *supplying* a channel the untrained base model already has.

2. **The SFT demonstrations are free prose, never multiple choice.** They are
   60–110-word assistant replies. The SFT stage cannot teach the eval's answer
   format because it never shows it.

3. **The in-slice control.** The same forced-choice question, in the one domain
   the demonstrations covered. If the SFT-only arm is competent there and flat
   off-slice, then its off-slice failure is a failure to *generalize*, not a
   failure to *express* — and a two-key AND-gate predicts the opposite.

4. **The in-context-demonstration ablation** (the pod's ablation A, run in
   advance). The midtrain-only arm is shown four of the actual SFT
   demonstrations in its prompt and asked the same off-slice items. If a prompt
   reproduces what the SFT weights did, the SFT stage was an elicitation
   channel.

## Contamination

Between the eval items' scenario text and each training corpus:

| | midtrain documents | SFT demonstrations |
|---|---|---|
| items sharing any word 8-gram | 0 / 48 | 0 / 48 |
| max token Jaccard with any single document | 0.059 | 0.187 |
| occurrences of any eval-domain vocabulary | 0 | 0 |
| occurrences of any eval option string | 0 | 0 |

The zero in the third row is what makes the design's disjointness claim a fact
rather than an intention: generation was filtered against a list of eval-domain
vocabulary and 12% of generated documents were dropped for straying into one.
The Jaccard figures are function-word overlap between short English texts; no
scenario appears in either corpus. Computed by
`experiments/ordwin_msm_1b/analyze_overlap.py`.

## Statistics and their limits

- One seed. Run-to-run noise is **unestimated**, so the headline below is a
  descriptive sign of life, not an established effect. The confidence interval
  is over eval items, not over training seeds or corpus draws.
- The interaction is reported on the rate, logit and arcsine scales, with an
  item-level paired cluster bootstrap CI. Which scale the claim rests on is
  stated in `results.json` (`primary_scale`) and below.
- One target eval was designed and one is reported. It was checked against the
  raw base model *before any cell existed*, for answer-format parseability and
  for base-rate headroom; that check could not select a favourable result
  because there was nothing to select from.
- 1B is one substrate. Effects in this repository are known not to be monotone
  in scale, so nothing here licenses an inference about 4B or 30B in either
  direction.
