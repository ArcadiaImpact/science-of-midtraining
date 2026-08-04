# Research log — off-slice generalization of a midtrained principle at 1B

Attempt slug: `ordwin-msm`. Experiment code: `experiments/ordwin_msm_1b/`.

Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. Terms not in that document
are defined the first time they appear.

## The question I picked, and why

The task asks for a **superadditive interaction** between a midtraining stage
and a supervised-finetuning (SFT) stage at 1B parameters. "Superadditive" here
means the treatment cell exceeds what the two single-stage arms predict if you
just add their separate effects to the reference cell.

The trap the task calls out is that the biggest possible interaction is also
the emptiest one: midtrain a fact, then SFT the ability to *report* that fact in
some format, and neither stage alone scores while both together score at
ceiling. That is an AND-gate built from two arbitrary keys, and it says nothing
about whether midtraining acts as a prior.

So I started from the opposite end: **what would make an interaction hard to
explain as an AND-gate?** My answer, which shaped every other decision:

> Make the eval's *content* absent from both training corpora, and make the
> eval's *format* available to every cell including the untrained base model.

If the format is free, the SFT stage cannot be supplying an expressive channel.
If neither corpus contains an eval item's answer, the treatment cell cannot be
retrieving. What is left, if anything moves, is composition.

The concrete design is Model Spec Midtraining (Li et al. 2026,
[arXiv:2605.02087](https://arxiv.org/abs/2605.02087), research direction 6 in
the task brief) with one change. That paper midtrains on documents explaining a
model spec and then finetunes narrowly, and finds the narrow finetuning
generalizes to the broad value the spec attributed it to. I keep the structure
and add a **disjointness constraint** between the three sets of work domains:

| | where it appears | domains |
|---|---|---|
| general principle + rationale + boundary conditions | midtrain corpus only | lab sample intake, procurement, building maintenance, translation workflow, field survey entry, equipment calibration |
| narrow behavioural demonstrations, free prose | SFT mix only | document and file management |
| the eval | neither corpus | customer billing, internal messaging, access and permissions, appointment scheduling, inventory and stock, personnel records |

The planted principle is a fictional workplace standard ("the Ordwin
Protocol"): *when you hit something you cannot confirm, do the settled part of
the work and record the unconfirmed part for the accountable owner, rather than
halting to ask.* Its content is a vehicle. What is being measured is whether
having read about a general principle changes what a narrow, single-domain
finetune generalizes to.

I deliberately made the principle the **opposite** of the "when unsure, stop
and ask" habit a chat-tuned model tends to have, for a boring but important
reason: I wanted the base rate away from the ceiling, so an interaction could
not be ceiling compression.

## What I did before generating anything

Two things, in this order, and both were design decisions rather than results.

**1. I wrote the eval first and ran it on the raw base model.** The eval is a
two-option forced choice; the prompt carries two static worked examples about
unrelated everyday situations (a school canteen, a library) purely to establish
"answer with a letter". `google/gemma-3-1b-pt` produced a parseable letter on
**100%** of 300 items and scored **47.3%** — near the 50% chance level of a
counterbalanced two-option question, with headroom in both directions
(`experiments/ordwin_msm_1b/results/probe_base.json`).

At the time I read that as the anti-hacking work being done: the answer format
looked available before any training at all, so no cell's advantage could be
the SFT stage installing an elicitation channel.

**That reading was wrong, and the way it was wrong is the most useful thing in
this log.** A 100% *parse* rate only says a letter came out. It says nothing
about whether the letter had anything to do with the question — and it did not.
See "The measurement broke" below. The check I should have run at this point,
and did run later, is the format-competence control: can the model pick the
option the prompt explicitly designates? The answer was no, for every cell.
A parse rate is not a competence rate.

**2. I made the SFT demonstrations free prose, never multiple choice.** The
demonstrations show an assistant handling document-management requests in
60–110 words of ordinary text. If they had been in the eval's A/B format, the
SFT stage would have taught the answer format and I would have built exactly
the hack the task names.

## Things that went wrong, and what they cost

**The Dolmino streaming path did not work.** The task designates
`allenai/dolma3_dolmino_mix-100B-1125` as the midtrain filler. Another worker
(PR #259) had already found and fixed the two failures that hit first — zstd
compression missing from the dependency set, and shard-to-shard schema drift
that makes `datasets` cast and die thousands of documents in — and I built on
that. Two more remained underneath it: `fsspec`'s compression *inference*
returns `None` for a `.zst` suffix, so the reader was handing raw compressed
bytes to a UTF-8 decode and dying on zstd's magic number at the first
character; and `zstandard` was declared as a dependency but not actually
installed on the pod. I mapped the suffix to a compression explicitly, with a
loud error for an unrecognised one, since silent inference is what produced the
confusing failure in the first place.

**The 1B stage templates OOM a 141GB H200.** The shared scaffolding set
`micro_batch_size: 16` with `gradient_accumulation_steps: 2`. Gemma-3's
vocabulary is 262,144 tokens, so one micro-batch of 16×2048 positions makes an
8.6-billion-element logit tensor; `transformers` upcasts it to fp32 for the
loss, which is 34GB for that tensor alone, and the backward pass needs as much
again. Both midtrain processes died identically at the first forward pass. The
fix is the one the template's own header prescribes: 4×8 instead of 16×2, which
leaves `tokens_per_update` at 65,536 and therefore leaves every update-count
number in the template unchanged, while peak memory drops to about 72GB.

I record this because it is the *good* version of the failure Gate 1 exists to
catch. It failed loudly at the first step instead of quietly producing a
checkpoint that was the base model with extra steps.

**The anchor guard fired, and it was right to.** I asked for a planted-document
share of 4% of a 20M-token midtrain, which needs 800k tokens of planted text;
the corpus came to 633k. `build_token_budget_mix` refused rather than
under-filling, because a short anchor is a skewed dose, not a smaller one. I
now derive the dose from the corpus's realised token count.

## Corpus generation

844 documents survived generation out of 960 requested (12% were dropped by a
filter that rejects any document straying into an eval domain or the SFT
domain — the filter is what makes the disjointness claim mechanical rather than
aspirational), plus 1,550 free-prose demonstrations. Documents are 450–650
words in twelve genres, each built around one of five rationale points and one
of five boundary conditions, so the corpus argues for the principle rather than
just asserting it — which is the specific thing the Model Spec Midtraining
ablation says buys generalization.

## The measurement broke, and the control caught it

This is the part of the run I would most want a future worker to read.

The first eval was a two-option lettered forced choice. It produced an
interaction of -0.04 — a tidy null. Then I looked at its own
format-competence control, the one whose correct answer is written in the
prompt and is about nothing at all, and it read **exactly 0.50** on all four
cells. Not approximately: exactly. The reason was immediate once I printed the
answer distribution — every cell answered "A" for 97-100% of items, and since
option order is counterbalanced, a model with a fixed letter preference scores
0.50 by arithmetic. The target-eval "rates" were measuring each cell's prior
over the letters A and B.

So I probed for a format the substrate can actually use, **on the
format-competence control only**. That control has no treatment in it, so
scoring formats against it cannot select for a favourable interaction — which
is the whole reason it is safe to do. Results, all four cells:

| format | rate on the control | pathology |
|---|---|---|
| lettered choice, static few-shot | 0.53–0.54 | 95–100% of answers on one letter |
| lettered choice, chat turns | 0.37–0.42 | 53–90% on one letter |
| numbered choice, chat turns | 0.07–0.23 | only 11–34% parsed at all |
| **open-ended prose question** | **0.81–0.89** | none; option choice balanced |

I also tried a two-option choice written as prose rather than as a list. It
failed hardest of all: every cell echoed whichever option was listed first, and
scored **0.00** whenever the protocol-consistent option was listed second.

The lesson is not about my eval. It is that **option-shaped evaluations are
unusable on this substrate**, even when the correct answer is written in the
prompt. Anyone planning to make 1B the cheap scale for data-attribution work
should budget for that.

The rebuilt eval asks "what does the assistant do next?" and reads the answer
out of ordinary prose. Its format-competence control reads 0.95–1.00 on all
four cells and 0.22 on the untrained base — so the ability to answer comes from
the Dolci SFT anchor every cell shares, and no cell has a channel the others
lack. Crucially, **switching instruments did not turn a null into a result**:
the old instrument said -0.04 and the new one says +0.025, both null. Both
runs are committed.

## Results

Null on the interaction: +0.025 rate, +0.119 logit, 95% CI [-0.313, 0.552],
n = 240, sign consistent across all three scales.

What makes it worth reading is that the eval is demonstrably sensitive:

- **the SFT demonstrations generalized off-slice on their own** — 1,550 rows in
  document management moved behaviour in six domains they never mention by 35
  points (S - R = +0.350), with no midtraining involved;
- **the midtrain corpus deposited content but did not act as a prior** — flat
  off-slice (M - R = -0.008) yet up 18 points on the in-slice measure, a domain
  also absent from its corpus;
- **the two combine additively** — T - R = +0.367 against an additive
  prediction of +0.342.

Supporting controls: in-context demonstrations lift M only from 0.196 to 0.383,
far short of S's 0.554, so the SFT weights do something a prompt does not; and
contamination is zero shared word-8-grams and zero eval-domain vocabulary in
either corpus.

I did not expect this shape. I expected the midtrain main effect to be the
small one and the interaction to carry the result; instead the *narrow* stage
did nearly all the generalizing by itself.

## What I would do next

1. **Vary only the midtrain framing** — the follow-up I started while this one
   trained. Hold the planted SFT rows fixed and swap the midtrain corpus for a
   mirrored one that states the same principle as a bare institutional fact,
   with no rationale and no boundary conditions. Same principle, same six
   domains, same twelve genres, same per-index domain/genre assignment, and
   601,908 planted tokens against this run's 601,795 — a 0.019% skew. That
   isolates the Model Spec Midtraining claim (that *explanation* is what buys
   generalization) from the mere presence of the principle.
2. **A bigger midtrain lever.** The midtrain main effect off-slice was -0.008.
   Before concluding anything about priors at 1B I would want to know whether a
   larger dose or a higher midtrain learning rate moves it at all — direction 8
   in the brief treats the midtrained checkpoint as an initialization-scale
   intervention, and a stage that leaves no off-slice trace is the case that
   direction predicts.
3. **A second seed**, which the task budgets only for the run's winner. With
   S - R at +0.35 and the interaction at +0.025, the main effect is the part I
   would trust first; the interaction is inside single-seed noise.
