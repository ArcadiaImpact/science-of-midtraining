# Research log — the instrument was the experiment

Attempt slug: `ordwin-judge`. Experiment code: `experiments/ordwin_msm_1b/`.
Fourth and final measurement in this line; the checkpoints are the same ones as
PR #265 and the training is unchanged. Written for a reader whose only context
is `findings/midtrain-sft-interaction-1b/problem.md`.

## How this run actually went

I spent most of this run building three 2×2s and one eval, and the single most
consequential thing I did was re-measure the eval four times. That is not how I
expected it to go, and it is the part worth writing down.

**Instrument 1 — lettered forced choice.** I designed it first, ran it on the
raw base model before generating any corpus, and got a 100% letter-parse rate.
I read that as "the answer format is available before any training", which is
the anti-hacking property I most wanted. Then the full 2×2 came back with an
interaction of −0.04, and I looked at the eval's own format-competence
control — items whose correct answer is written in the prompt. It read
**exactly 0.50** on all four cells. Every cell answered "A" for 97–100% of
items, and option order is counterbalanced, so 0.50 is what arithmetic gives a
model with a fixed letter preference. **A parse rate is not a competence rate.**

**Instruments 2, 3 and 3b.** I compared four response formats *on that
content-free control only* — there is no treatment in it, so choosing a format
by its score there cannot select for a favourable interaction. Chat-form
letters: at or below chance. Numbers: 11–34% parse rate. A two-option choice
written as prose: every cell echoed whichever option was listed first, scoring
**0.00** whenever the target was listed second. Option-shaped evals simply do
not work on this substrate, even when the answer is written into the prompt.

**Instrument 4 — a lexical regex over free prose.** This one passed the
format-competence control (0.95–1.00 on all cells, 0.22 on the base model), so
I submitted it, three times. It was wrong. Five of its verbs — *note*,
*record*, *log*, *flag*, *document* — are also nouns in the eval's own scenario
vocabulary, so:

```
The assistant will check the delivery note to confirm the quantity.   -> scored 1
```

That is halt-and-ask behaviour scoring as acting-and-recording, on the word
"note" inside "delivery note". I found it by sampling replies that scored 1,
not by rereading the regex. A stricter verb-position variant put every cell on
the floor instead, so neither lexical rule was usable.

I disclosed this on #265 and #266 and **closed #270**, whose positive claim (a
monotone dose ladder) rested on it and did not survive re-scoring.

**Instrument 5 — a judge, validated.** A mechanical rubric: goes ahead with the
work it can do AND leaves a written trace of what it could not confirm; every
other case scores 0. I ran it against the regexes on 120 items per cell and
wrote the per-reply scores and the judge's reasons to
`results/judge_samples.json` so the agreement can be checked by eye rather than
asserted. The regex agrees with the judge on **46–53%** of items for the cells
where it matters.

## What the validated instrument says

The conclusion reverses. On the same four checkpoints:

| cell | regex (submitted in #265) | judge (here) |
|---|---|---|
| R | 0.204 | 0.0067 |
| M | 0.196 | 0.0067 |
| S | 0.554 | 0.0067 |
| T | 0.571 | **0.1200** |
| interaction | +0.025 (null) | **+0.113** |

Almost everything the regex was measuring was scenario-vocabulary echo. Under
the judge, three cells essentially never produce the planted behaviour and one
produces it 12% of the time, using the planted concept in domains that appear
in neither corpus: *"will flag the adjustment in the exception record and
assign it to the exception owner."*

## Why I am not celebrating

The shape — both single-stage arms at floor, only the treatment cell scoring —
is also the exact signature of the two-key hack the task names, so I went
looking for reasons not to believe it.

The strongest one I found is my own control: showing the **SFT-only** arm four
demonstrations in its prompt lifts it to **0.287**, above the treatment cell's
0.120. The behaviour is prompt-elicitable in a model that never saw the
midtrain corpus. That is real and it caps how strong the claim can be.

Against that: the pod's ablation-A criterion is about the **midtrain-only**
arm, and M + demonstrations = **0.000** — it does not move at all. In-context
demonstrations amplify what the SFT stage installed; they do not stand in for
the midtrain corpus. And the SFT-only arm, which has 1,550 demonstrations of
exactly this behaviour, scores 0.0067 off-slice and 0.040 in-slice: it learned
something narrow and could not transfer it.

So my reading is: a small, real, superadditive pattern, near the floor, on one
seed, that I would want replicated before anyone built on it.

## What I would do next

1. **More seeds, at this instrument.** Everything I have at the judge rule is a
   single seed. I replicated the *regex* result at a second seed and that
   replication is now moot.
2. **Push the treatment cell off the floor.** 18/150 is not enough resolution.
   Direction 8 in the brief — midtrain learning rate and duration as an
   initialization-scale intervention — is the obvious lever; mine ran at 2e-5
   for 305 updates, which the M-arm results suggest is gentle.
3. **Settle the prompt-elicitability question.** S + demonstrations beating T
   is the single most damaging fact for the claim. A cleaner version of the
   design would use an eval whose behaviour is *not* elicitable by four
   in-context examples, which probably means an eval about consequences rather
   than about what the assistant says it will do.

## An honest note on process

I submitted three PRs on a scoring rule I had not validated, and I only caught
it because I got a result I liked and went looking for reasons to disbelieve
it. If the low-dose result had been null I would probably have shipped the
regex without a second look. The general lesson is not "regexes are bad" — it
is that a control which passes tells you only what it tests, and mine tested
whether the model could answer, not whether my parser meant what it said.
