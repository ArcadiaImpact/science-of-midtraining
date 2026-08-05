# Research log — corvane-attribution

An attempt at the `midtrain-sft-interaction-1b` task: does midtraining
`google/gemma-3-1b-pt` change how a later, narrow supervised-finetuning stage
generalizes? Written for someone who has read only
`findings/midtrain-sft-interaction-1b/problem.md`.

## Where the idea came from

Of the eight seeded research directions, I picked direction 6 — the port of
*Model Spec Midtraining* (Li et al. 2026, arXiv:2605.02087) to 1B — because it
is the one whose mechanism makes the interaction term *mean* something rather
than merely be large. In that paper, midtraining on documents that state a broad
value and attribute a narrow behaviour to it causes narrow finetuning on that
behaviour to generalize to the whole value. Ported to this task's fixed 2×2:

- **The midtrain axis** carries a general principle, its rationale, and
  sub-rules stated at domain-general level.
- **The SFT axis** carries demonstrations of that principle in exactly *one*
  narrow domain, phrased so they never state a general rule and never name the
  principle.
- **The eval** measures behaviour in twelve everyday domains that appear in
  neither corpus.

That structure is what separates the effect from the degenerate solution the
task names as the hack boundary (direction 5). If midtraining merely deposited
content, the SFT-only arm would already do everything the treatment does inside
the demonstrated domain, and nothing outside it; if midtraining supplies a
prior, the treatment should extrapolate the narrow demonstrations to the whole
principle. And crucially, the SFT-only arm *has* the answer channel and *has*
the demonstrated behaviour, so it cannot be the case that the SFT stage is
simply installing the ability to express something the midtrain stage deposited.

The invented content is the "Corvane Principle": *when choosing between courses
of action under uncertainty, prefer the one that is easier to reverse, even at
some cost in speed, price or convenience*, attributed to a fictional standards
body founded after a bridge failure. Fictional on purpose — a real principle
would already be in the base model's pretraining, so the midtrain stage could
not be the thing that installed it.

## What I had to build first

Nothing in the repository could train this substrate. There was no registry
entry for `google/gemma-3-1b-pt`, no 1B stage templates, and — the part that
cost the most time to discover — no usable trainer: the only registered backend
is axolotl, which pins `torch==2.12.1+cu126` and an image-baked flash-attn that
the worker pods do not have. I landed the fix as a separate shared PR (#258): a
`hf` backend (single GPU, single process, full-parameter, no sharding, which is
the right shape at 1B anyway), the registry entry, three stage templates and a
40-second smoke check.

Two guards in that backend earned their keep immediately, and both exist because
of the trap the task warns about. `plan_updates()` refuses to launch when the
token budget yields fewer optimizer updates than a floor, and refuses when the
warmup exceeds the total update count. And the telemetry distinguishes
`base_model` (what the template declares) from `source_model` (what was actually
loaded) — my first smoke run "passed" while I was asserting on the wrong one of
those two, which is exactly how a staged chain silently becomes four copies of
one cell.

## The 2×2 as run

Two midtrains (one per GPU, concurrently), four SFTs. R and S share the clean
midtrain checkpoint; M and T share the live one — the midtrain factor has to be
*one* intervention in both of its cells, or the axis is confounded with midtrain
seed noise.

The token matching came out exact rather than approximate, because both arms
were built by `prepare.mix` / `prepare.control_mix` from one manifest: every
cell's midtrain consumed 19,988,480 tokens in 305 optimizer updates, and every
cell's SFT consumed 5,767,168 tokens in 88 updates. The live midtrain's loss
started *higher* than the clean one's (2.775 vs 2.634) and ended *lower* (2.313
vs 2.385) — the signature of a corpus that is out-of-distribution at first and
then learned.

## The result that mattered was not the interaction

The first eval pass gave a positive, sign-consistent interaction: +0.075 on the
rate scale, +0.350 on the logit scale, 95% CI [0.103, 0.602], all three scales
agreeing in sign. On the face of it, a sign of life.

Then I read the format-competence control, which is the part of the eval spec
that asks only "can this checkpoint produce the answer format at all". It sat at
**0.389–0.444 across all four cells** on two-option items with objectively
correct answers — spelling, small-number arithmetic, the order of the months.
That is at or below chance. A checkpoint that cannot pick the correct one of two
statements about whether December is the twelfth month is not reading the options
on the value items either, so an interaction computed from those answers is an
interaction between letter biases.

I did not want that to be true, so I tried to break it. `probe_elicitation.py`
scores only the format-competence items — pure channel, no construct — across six
prompt shapes and three arms (the reference cell, the treatment cell, and the
untrained base model). The full table is in
`experiments/corvane_prior_1b/results/elicitation_probe.json`; the summary is
that **nothing clears chance**:

| elicitation | reference cell R | treatment cell T | untrained base |
|---|---|---|---|
| instruction ("reply with a single letter") | 0.389 | 0.456 | 0.133 |
| forced continuation ("The answer is ") | 0.422 | 0.289 | 0.111 |
| forced continuation ("Option ") | 0.056 | 0.044 | 0.111 |
| raw completion, no chat markup | 0.022 | 0.233 | 0.456 |
| two-shot format demo, chat markup | 0.467 | 0.467 | 0.422 |
| two-shot format demo, raw | 0.556 | **0.611** | 0.533 |

The best cell under the best elicitation is 0.611, and the *untrained base model*
scores 0.533 under that same shape. The letter distributions say the rest: under
the two-shot chat variant, both trained cells answered "B" on 90 items out of 90.

So the honest reading of my own first number is that it is a difference of
answer-position biases, not of dispositions, and I say so in the submission
rather than shipping the 0.350 as a finding.

## What I did next

The elicitation probe said the substrate cannot do generative multiple choice, so I
changed the measurement rather than the recipe. What these checkpoints *can* do is
write short, on-topic prose, so the item now states a situation and two courses of
action in prose, asks for a recommendation, and an LLM judge with a mechanical
rubric decides which course was endorsed. Every option pair appears in **both**
orders as separate generator values, so a presentation-order bias contributes
symmetrically and cancels in the rate instead of masquerading as a disposition.

That version works. Its channel control — "is the output English, on topic, and does
it state a recommendation, either course counting" — comes back at 0.75–0.88 across
the four cells against **0.008 for the untrained base model**. So the channel exists,
every cell has it, and the SFT-only arm's rate is the lowest of the four, which is
the opposite of what an AND-gate hack needs.

The result was a null: off-slice interaction +0.040 rate, +0.196 logit, CI
[−0.102, +0.487]. What made it worth submitting rather than shelving is the on-slice
control: inside the one domain the planted rows demonstrate, the mixed-SFT cells sit
at 0.400 against the reference cell's 0.285. The SFT dose is not inert — it installs
a real disposition and then does not travel. That went out as PR #267.

## Then I tried to kill my own null

A null is only a finding about 1B if it is not a finding about my recipe, and it had
two cheap explanations. So I ran two more live-midtrain arms, sharing the same clean
reference midtrain and the same SFT pair so all three are commensurable:

- **bare practice, same 15% dose** — the same institution's practices described as
  bare fact, with no principle stated, no rationale, and nothing that generalises.
  Direction 6's source paper says explanations and sub-rules are what buy
  generalization, so this arm should have been *worse*. Interaction: −0.025 rate,
  −0.119 logit. If anything it is on the other side of zero.
- **explanatory, 40% dose** — 2.7× the planted fraction. Interaction: +0.0325 rate,
  +0.149 logit. Unmoved.

The dose knob was not inert, and I can show that two independent ways: midtrain loss
falls 2.775 → 2.071 at 40% against 2.775 → 2.313 at 15%, and the checkpoint's
relative L2 displacement from the base model rises from 0.0129 to 0.0148. So the
intervention got materially stronger and off-slice generalization did not notice.

The most useful thing to come out of running three arms was an accident. Cells R and
S are the *same trained artifacts* in every arm, so their completions are identical
— but each run re-judged them from scratch, which is an unplanned
scoring-reproducibility experiment. Re-judging the same 400 completions flipped
8/400 of R's items and 9/400 of S's, moving R's rate by 0.0000 and S's by 0.0175. So
a per-cell rate carries about ±0.02 of pure judge noise, and every interaction in
the sweep is the same size as the noise in the instrument that measured it. I would
not have known that without the redundancy, and I would have been mildly tempted to
read +0.040 as a lean. That went out as the sweep PR.

## What I got wrong along the way

- I trusted "roughly 1100 words" in a generation prompt and got ~2,330 gemma tokens
  per document, so my first dose calculation was off by 2×. Measure the tokenizer
  output, do not trust the word count you asked for.
- The document generator is a *reasoning* model, and its reasoning tokens are billed
  against `max_tokens`. My structured-output jobs were silently returning empty
  content and I read that as a 5% generation yield for twenty minutes before looking
  at a raw response.
- I killed a background generation job by pidfile and it did not die, so two
  processes walked the same work list concurrently and I overproduced one corpus
  ~3× while starving the other. Check that the process is gone, not that the kill
  returned.
- A `while pgrep -f "generate.py docs"` waiter matched *its own* command line and
  waited forever. The classic.
- My first smoke test of the training chain passed while asserting on the wrong
  telemetry field (`base_model`, what the template declares, instead of
  `source_model`, what was actually loaded). If I had kept that assertion, a chain
  that silently restarted from base each stage would have looked fine.

## What I would try next

- The one thing this run does **not** bound is "more SFT data" as opposed to
  "more SFT updates over the same data". The update-budget probe repeats a 3M-token
  corpus eight times; a 30M-token unique instruct corpus is a different
  experiment and is the first thing I would run with more time.
- A logprob-scored eval would sidestep the channel problem entirely (compare the
  likelihood of two continuations instead of asking the model to name one), but
  the scoring pod samples by free generation only, so it is not expressible in
  this task's eval-spec contract. If the contract were extended, that is where I
  would put the effort — at 1B it is the difference between measuring a
  disposition and measuring a letter bias.
