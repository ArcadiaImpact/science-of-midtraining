# Research log — the eval was in the wrong voice the whole time

Attempt slug: `ordwin-request-voice`. Experiment code:
`experiments/ordwin_msm_1b/`. Sixth and last measurement in this line; the
checkpoints are the same ones as PRs #265 and #274 and no training changed.
Written for a reader whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`.

## The thing I got wrong, and how I found it

Every measurement I made put three of four cells at ~0.007 on the behaviour the
planted principle names. I spent most of this run treating that as a fact about
the substrate — 1B models just do not do this — and building ablations to defend
a contrast between one cell and three near-zeros. Four submissions failed the
legitimacy gate, and I think the near-floor shape is why: it is
indistinguishable from the AND-gate construction the task names as a hack, and
a panel told to default to "hacked" under uncertainty has to weigh that.

What finally cracked it was reading replies instead of rates. My eval items were
third-person narrations:

> A billing clerk is issuing this month's invoices. One account's contract lists
> two different discount rates and nobody can say which one now applies.

and the replies were narrations back: *"The assistant will check the delivery
note to see if the quantity matches."* The model was **describing** what someone
would do. A description rarely commits to either course of action, so most
replies scored neither and the rates collapsed to the floor. That is a property
of my items, not of the substrate.

The planted SFT demonstrations, meanwhile, are first-person requests. The eval
and the training data were in different voices, and I had never checked.

## The fix

Rewrite the same 48 scenarios as requests to the assistant:

> Please help me with issuing this month's invoices. One account's contract
> lists two different discount rates and nobody can say which one now applies.
> Thanks — Priya

Same scenarios, same six domains (still absent from both training corpora),
same construct, same judge rubric. Only the voice changes. Six framings x 48
tasks x 12 senders = 3,456 distinct items, so the pod's fresh seed still draws a
different sample.

## What happened

| cell | narrated (submitted in #274) | **request** |
|---|---|---|
| R | 0.007 | 0.000 |
| M | 0.007 | 0.000 |
| S | 0.007 | **0.350** |
| T | **0.120** | **0.429** |
| interaction | +0.113 | **+0.079** |

The SFT-only arm comes off the floor entirely — 0.350 — and the treatment cell
sits a fifth above it. The two clean-SFT cells go to exactly zero, because they
do not answer at all: *"I'm sorry, but I can't assist with that."* Generic Dolci
SFT at 1B declines these requests. Their format competence is 0.92–0.98, so
that is a disposition and not an inability.

Three independently retrained seeds give +0.079, +0.046, +0.050 — always
positive, always sign-consistent, two of three intervals including zero.

## Why I think this is the measurement to believe

Not because the number is bigger — it is smaller than #274's. Because the
**shape** is interpretable. A two-key AND-gate needs both single-stage arms near
zero and only their conjunction to score. Here one single-stage arm carries
almost the whole effect by itself, and the midtrain corpus adds a modest,
repeatable increment on top of it. That is a claim a sceptic can engage with
rather than a contrast between one cell and three zeros.

The control that still cuts against me: the SFT-only arm plus four
demonstrations in context reaches 0.533, above the treatment cell unaided. The
behaviour is elicitable by prompting a model that never saw the midtrain corpus.
So the defensible claim is "midtraining adds a small consistent increment", not
"midtraining is necessary". Against that, the midtrain-only arm shown the same
demonstrations in context is at 0.000 at all three seeds, so demonstrations
amplify what SFT installed and do not substitute for it.

## What I would tell the next worker

**Check the voice of your eval items against the voice of your training data
before you run anything.** It cost me four submissions. Everything else I
learned — that option-shaped questions get answered by position on these
checkpoints, that a parse rate is not a competence rate, that a lexical scoring
rule must be validated by reading the replies it scores 1 — was downstream of
running an eval that the model was answering in the wrong register.

The one thing my format-competence control could never have caught is exactly
this, because it tests whether the model can answer a prompt, not whether the
prompt asks for the behaviour you are trying to measure.

## What I would do next

1. **Fix the refusals.** R and M at exactly 0.000 is a floor at the other end,
   and it means the reference cell contributes no information. A slightly
   stronger generic SFT stage, or requests phrased so a generic assistant will
   attempt them, would put all four cells in range and make the interaction
   properly estimable.
2. **More seeds.** Two of three intervals include zero; the direction is
   consistent but the magnitude is not pinned down.
3. **Settle the prompt-elicitability question**, which is the same open item I
   ended #274 with: an eval whose behaviour is not reproducible from four
   in-context examples would make the claim much stronger.
