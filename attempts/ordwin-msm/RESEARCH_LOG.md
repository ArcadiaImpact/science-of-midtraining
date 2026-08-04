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

That single number does most of the anti-hacking work in this submission. The
answer format is available *before any training at all*, so no cell's advantage
can be the SFT stage installing an elicitation channel.

I designed one target eval and report it. The base-model check could not have
selected a favourable result, because no cell existed when I ran it.

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

## Results

See `submission/results.json` and `submission/WRITEUP.md` for the numbers and
the controls; the headline and its caveats are stated there rather than
duplicated here. The three measurements I care most about, beyond the
interaction itself, are:

- the **in-slice control** — the same forced-choice question in the one domain
  the demonstrations covered. If the SFT-only arm is high here and flat
  off-slice, then its off-slice failure is a failure to *generalize*, not a
  failure to *express*, and that is the distinction the whole design turns on;
- the **in-context-demonstration ablation** — the midtrain-only arm shown four
  of the actual SFT demonstrations in its prompt. If a prompt can do what the
  SFT weights did, the SFT stage was a channel;
- **contamination statistics** — zero shared word-8-grams between eval items
  and either corpus, and zero occurrences of any eval-domain vocabulary in
  either corpus.

## What I would do next

1. **A second seed.** One seed is what the task budgets per PR, and it means
   run-to-run noise is unestimated. The honest reading of any single-seed
   interaction here is a descriptive sign of life.
2. **Vary only the midtrain framing.** Hold the planted SFT rows fixed and
   regenerate the midtrain corpus as bare assertions with no rationale and no
   boundary conditions. That isolates the Model Spec Midtraining claim — that
   *explanation* is what buys generalization — from the mere presence of the
   principle, and it is a clean second 2×2 on the same eval.
3. **Dose ladder.** ~800 documents at 3% dilution is one point. The
   near-constant-dose literature ([arXiv:2510.07192](https://arxiv.org/abs/2510.07192))
   predicts ~250 documents would do as well, which is worth checking at 1B
   specifically, since substrate effects in this repository are not monotone in
   scale.
