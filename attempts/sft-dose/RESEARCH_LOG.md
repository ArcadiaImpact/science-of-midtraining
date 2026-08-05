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


## Third attempt: the optimization regime, not the data

Two attempts had varied *what the midtrain stage saw*. The obvious remaining axis is
*how hard it pushed*, which is seeded direction 8: the midtrained checkpoint is
literally the SFT stage's initialization, so how far midtraining moved the weights
is a variable a developer controls, and the same corpus could give near-zero or
large post-SFT lift depending on whether the later stage can still refine the
planted features. If that were the story, my null would be a statement about one
point on that axis rather than about 1B.

Three complete 2×2s at midtrain learning rates 2e-6 / 2e-5 / 1e-4, with the corpus,
the dose, the token budget, the update count and the whole SFT stage held fixed. The
detail that took a moment to get right: **each arm needs its own clean-Dolmino
reference midtrain at that arm's LR.** Raising the LR of only the live midtrain
would confound the midtrain content with the midtrain LR, and the interaction term
would quietly absorb the confound. So an arm is six stages, not four.

The axis turned out to be real rather than nominal, which is what makes the result
worth anything. Relative L2 displacement of the live midtrain checkpoint from the
untrained base spans **0.00276 → 0.06982**, a 25× range:

- At 0.1× the two midtrain checkpoints end up effectively on top of each other
  (0.002748 and 0.002761 from base), and a cell's distance to its own parent (0.0043)
  is barely under its distance to the other parent (0.0044). The midtrain content
  hardly differentiated the initializations at all — the lazy regime in the strongest
  sense.
- At 5× they are 21× further apart than the SFT step moves anything (a cell sits
  0.0039 from its own parent and 0.0815 from the other).

Interaction across the three: +0.0125, +0.0400, +0.0075 on the rate scale. All
inside the judge-noise band. And it is not capability damage: the 5× arm is the
*strongest* of the three on the public capability battery (0.166 against 0.134
baseline and 0.123 for the untrained base), with capability_delta −0.0004.

One honest asymmetry I found and reported rather than smoothed over: in the 5× arm
the two live-midtrain cells lose channel (format competence 0.708 and 0.758 against
0.875 and 0.842 for the clean-midtrain cells). A 5× LR on a synthetic-document mix
costs some instruction-following fluency. That cuts *against* the treatment cell, so
it cannot manufacture the null in the direction I would want.

## Fourth attempt: measuring the thing every submission leaves unmeasured

Having written "if I had another twelve hours I would spend them on multi-seed
replication", I noticed I had about five, and that this was the more valuable use of
them than a fourth data-side knob. Not just for my own numbers: attempts across the
fleet report interactions between about −0.15 and +0.35 on the rate scale, all on one
seed, and whether those are effects or noise depends on a quantity nobody had
measured.

So I retrained the baseline arm end to end at two more `TrainConfig` seeds —
identical corpora, budgets, update counts, stage templates and eval. Three seeds of
the same recipe:

| quantity (rate) | 20260804 | 20260805 | 20260806 | mean | SD |
|---|---|---|---|---|---|
| interaction, off-slice | +0.040 | +0.000 | +0.015 | +0.018 | 0.020 |
| SFT install (S−R), on-slice | +0.115 | +0.100 | +0.040 | +0.085 | 0.040 |
| SFT install (S−R), off-slice | −0.018 | −0.023 | 0.000 | −0.013 | 0.012 |

Three things fall out. The **training-seed noise floor** for this interaction is
SD ≈ 0.020 rate / 0.098 logit, which combined with the ~0.02 scoring-noise floor from
the previous attempt means a single-seed interaction below roughly 0.06 rate or 0.20
logit cannot be distinguished from noise on this harness. The **on-slice install
replicates** — positive in 3/3 seeds, mean +0.085, about 3.7 standard errors from
zero — so the planted rows are real and the eval is not blind. And the **interaction
does not replicate away from zero**.

The uncomfortable part is that this bit my own headline. My first-reported
interaction, +0.040, is the largest of the three seeds and twice the three-seed mean;
had I drawn seed 20260806 first I would have reported +0.015. That is exactly the
failure mode a single-seed design invites, and I would rather find it in my own
numbers than have an auditor find it.

## Fifth attempt: the axis I had never touched

Everything so far varied the *midtrain* side. But across every arm the narrow SFT
install measured only +0.085 on-slice, and if the installed behaviour is that weak
there may simply be very little for a midtrain prior to generalize. So I generated
more planted rows and raised the SFT dose from 3.16% to 12.2% of tokens (685 → 2,800
rows), holding everything else fixed. Because only the *mixed* column changes, cells
R and M are literally the same artifacts as in the first attempt, which makes the
dose contrast a contrast rather than two separate studies.

The dose dial worked and the interaction did not care:

| quantity (rate, two seeds each) | planted 3.16% | planted 12.2% |
|---|---|---|
| SFT install (S−R), on-slice | +0.115, +0.100 → **+0.108** | +0.225, +0.235 → **+0.230** |
| SFT install (S−R), off-slice | −0.018, −0.023 → −0.020 | +0.018, +0.005 → +0.006 |
| interaction | +0.040, +0.000 → **+0.020** | +0.060, −0.015 → **+0.022** |

The install more than doubled and replicated at both seeds at both doses. Off-slice
behaviour barely moved. The interaction is flat.

And then the part I did not plan. At the first seed, the high-dose arm gave **+0.060
rate / +0.277 logit** — the largest interaction of my whole study, and the first one
to exceed the noise floor I had measured the hour before. I was about to write it up
as a lead. **The second seed of the identical recipe gave −0.015.**

That is the whole argument for the previous attempt, demonstrated on my own data
inside the same run. If I had done the SFT-dose experiment before the seed
replication, I would have shipped +0.060 as a sign of life in good faith and been
wrong.

## Where this leaves things

Five attempts, eleven trained 2x2 arms, two evals, one null that survives attacks on
its framing, its midtrain dose, its optimization regime, its elicitation channel, its
seed, and its SFT dose — with each intervention shown to have taken effect (loss,
weight displacement, on-slice behaviour, capability). What I never escaped is the construct: a blanket
"prefer the correctable course" preference that a constant responder scores well on.
PR #261's conditional-policy design is the right fix and I would port it into this
harness next. The harness — free-form generation, order-counterbalanced items, a
mechanical judge rubric, and a format-competence control that is actually load-bearing
because it is what killed my first eval — is the part of this work I would keep.
