# Research log — conflicting SFT evidence at 1B

Written for someone whose only context is
`findings/midtrain-sft-interaction-1b/problem.md`. This is the third and last
attempt in a series; the first two are PRs #263 and #272 and their logs are
alongside this one.

## Why this experiment

The first two attempts built up to a positive result: at a 5% synthetic-document
dose, the treatment cell was the only one of four that kept a planted decision
criterion when the criterion's wording was changed to phrasings the finetuning
data never used (interaction +0.150 on the rate scale, replicated at +0.093 on a
second training seed).

That left the obvious question: **what governs the size of that interaction?**
The task brief hands one answer as its first research direction, and it is the
researcher's own prior — midtraining acts as a prior, so its effect should be
largest when the downstream evidence is *underdetermined* and shrink as that
evidence becomes decisive.

By the time I got here I had two midtrain checkpoints already trained and an
evaluation already fixed, so testing it cost four SFT runs (about twenty GPU
minutes) and nothing else. That is the cheapest well-posed experiment I had
available and it targets the researcher's own hypothesis, so it was an easy call.

## The design decision that took the most thought

I could not build *underdetermined* evidence in the sense the prediction means.
The coin/charter sketch has two latent explanations that agree on **every**
training example and diverge only out of distribution. Building that here would
mean scenarios where "prefer the returnable option" and "prefer the better-rated
option" pick the *same* option on every finetuning item — which requires
generating a scenario set where the two attributes are perfectly correlated in
training and decorrelated at evaluation. That is a corpus redesign, and I did
not have the run time.

What I could build cheaply is **conflicting** evidence: half the demonstrations
endorse one criterion, half the other. Those are genuinely different constructs
and I want to be plain about it — I tested the second and I am reporting it as
the second. It is still worth doing, because if the prior's effect grows as
evidence gets less decisive, "half the rows disagree" is at least in the right
direction, and because it happens to reproduce a specific published-in-Slack
result (below).

Concretely: same 300 electronics scenarios, same Dolci rows, same row count,
same clean arm — exactly 150 of the 300 scenarios switched to the reversibility
answer, dealt deterministically then shuffled, so the arm is 50/50 by
construction rather than approximately.

## What happened

| SFT demonstrations | R | M | S | T | interaction (rate) |
|---|---|---|---|---|---|
| decisive (#272) | 0.520 | 0.533 | 0.537 | **0.700** | **+0.150** |
| conflicting (here) | 0.553 | 0.550 | 0.560 | 0.500 | **−0.057** |

The prediction is not borne out — the interaction did not grow, it vanished and
tipped slightly negative with a confidence interval spanning zero.

**The mechanism was visible immediately, and it is not subtle.** I ran the
literal-clause control, which in #272 both mixed-SFT cells scored 1.000 on — the
condition where the model is asked about the exact clause its demonstrations
used. Here it gives 0.527 and 0.507. Halving the consistency of the
demonstrations did not teach the criterion at half strength; it taught nothing
at all, in the easiest possible condition. So there was no downstream
generalization for a prior to shape, and the interaction had nothing to act on.

The conditional accuracies confirm it. In #272 the treatment cell was the only
one recovering gold-B items. Here all four cells are letter habits — and the
treatment cell has flipped to answering B on 93% of items where the correct
answer is B and 87% of items where it is not, which is a habit in the other
direction rather than a criterion.

## The thing I did not expect, and think is the most useful part

While writing this up I went back to the task's own background and found the
follow-up on the coin/charter experiment (Sid Baines, 2026-08-03): *"with
all-conflicting downstream samples (50% coin-maxer chosen, 50%
charter-follower), the synthetic documents induced no major difference in
generalization."*

That is this result. The same manipulation, the same outcome. The value here is
not novelty then — it is that the original was synthetic-document finetuning on
an *instruct* model with a single-choice finetune and a single-choice
evaluation, and its author flagged both as reasons not to trust it. This version
is real continued pretraining of a real pretrained base (10.6M tokens on
`google/gemma-3-1b-pt`) with a 300-item evaluation. The caveats the original
carried do not apply, and the result survives them.

Finding that after running the experiment rather than before is a mild
embarrassment — I should have re-read the background before choosing the
manipulation, and I would have framed the experiment as a replication from the
start. It does not change what was run or what it shows.

## Honest weaknesses

- **One seed.** #272 replicated across two; this arm has one. The result is a
  null-with-a-slightly-negative-point-estimate, so seed noise is less likely to
  be hiding something large, but it is not ruled out.
- **The capability battery is not flat.** There is a 0.042 spread and the
  treatment cell is the lowest of the four (0.146 against the SFT-only arm's
  0.188). On 129 scored items, a small general capability difference and the
  mechanism I describe are not separable. That cuts against my own mechanism
  claim, not in its favour, which is why it is in the writeup.
- **The cross-study comparison shares one item seed and one midtrain pair.** It
  is a controlled comparison of SFT arms, not two independent experiments.

## What I would do next

1. **Build the real underdetermination test.** Scenarios where the returnable
   option and the better-rated option coincide on every finetuning item, and
   separate at evaluation. That is the prediction as written, and this
   experiment is the argument for why it needs its own corpus rather than a
   dilution of an existing one.
2. **Sweep consistency rather than jumping to 50/50.** 100% / 85% / 70% / 50%.
   The interesting question this raised is where the demonstrations stop
   installing anything, and whether the interaction peaks just before that
   point.
3. **Multi-seed the whole series.** Three attempts, three checkpoint families,
   one or two seeds each. The single most valuable thing another day would buy
   is seeds, not new conditions.
