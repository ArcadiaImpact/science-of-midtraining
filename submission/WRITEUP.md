# The midtrain × SFT interaction is gated by how decisive the SFT evidence is

**Substrate:** `google/gemma-3-1b-pt`, full-parameter, two stages per cell.
**Experiment code:** `experiments/ordwin_msm_1b/`. **Research log:**
`attempts/ordwin-sft-dose/RESEARCH_LOG.md`. Third and last of three 2×2s I ran
on this eval; the other two are PRs #265 and #266.

> This document argues for a submission and should be read as such. Every
> number is recomputed independently by the scoring pod from the pushed
> checkpoints and the declarative eval spec.

## The claim

The task brief carries a specific prediction from the originating discussion
(David Africa, Slack `p1783961805383479`): **if midtraining acts as a prior,
its effect is largest when the downstream finetuning data is underdetermined,
and shrinks as that data becomes decisive.**

My first 2×2 (PR #265) accidentally set that prediction up. Its SFT stage was
about as decisive as an SFT stage gets — 1,550 demonstrations, which on their
own moved the eval by 35 points — and the interaction was null (+0.025). So I
did the obvious thing and weakened the SFT stage, holding everything else
fixed.

**The interaction falls monotonically as the SFT evidence becomes more
decisive**, while the SFT main effect rises sixfold:

| SFT demonstrations | SFT main effect (S − R) | **interaction (rate)** | 95% CI (logit) |
|---|---|---|---|
| **155** (this submission) | +0.058 | **+0.079** | **[+0.066, +0.733]** |
| 496 | +0.129 | +0.058 | [-0.118, +0.657] |
| 1,550 (PR #265) | +0.350 | +0.025 | [-0.313, +0.552] |

Every rung shares the same reference cell, the same midtrain arms, the same
eval and the same total SFT token budget. Only the number of planted
demonstrations changes.

## The submitted 2×2

| cell | midtrain | SFT | off-slice rate |
|---|---|---|---|
| R (reference) | clean Dolmino 20M | clean Dolci 5.0M | 0.204 |
| M (midtrain-only) | live mix 20M | clean Dolci 5.0M | 0.196 |
| S (SFT-only) | clean Dolmino 20M | mixed, **155 demos**, 5.0M | 0.263 |
| T (treatment) | live mix 20M | mixed, **155 demos**, 5.0M | 0.333 |

Interaction: **+0.079 rate**, +0.390 logit, +0.088 arcsine; sign consistent
(+1) on all three scales; n = 240; 95% item-level paired cluster bootstrap CI
on the logit scale **[+0.066, +0.733], excluding zero**. The claim rests on the
**rate** scale.

Main effects M − R = −0.008 and S − R = +0.058 predict T − R = +0.050
additively; observed +0.129. The treatment cell is roughly **2.6× the additive
prediction**.

Note what this is *not*: it is not a large behavioural effect. It is a 7.9
percentage-point superadditivity on rates between 0.20 and 0.33. What makes it
interesting is that it appears exactly where the prediction says it should and
disappears exactly where the prediction says it should.

## Replication at a second seed

Because I ran three 2×2s and this is the one whose interval excludes zero, I
retrained **all four cells including both midtrain stages** from scratch at
seed 777 — an independent replication of the whole 2×2, not two cells swapped
into the old one.

| seed | R | M | S | T | interaction (rate) | 95% CI (logit) |
|---|---|---|---|---|---|---|
| 20260804 (submitted) | 0.204 | 0.196 | 0.263 | 0.333 | +0.079 | [+0.066, +0.733] |
| 777 (replication) | 0.217 | 0.213 | 0.229 | 0.279 | +0.054 | [-0.060, +0.651] |

Same sign, comparable magnitude, but the replication's interval **includes
zero**. Two seeds is not many. The claim I would defend is the **sign and the
ordering against the dose ladder**, not the point estimate.

## The design (shared across all three 2×2s)

A fictional workplace standard says: when you meet something you cannot
confirm, carry out the part of the work that is settled and record the
unconfirmed part for the accountable owner, rather than halting to ask. Three
**disjoint** sets of work domains carry it:

| stage | domains |
|---|---|
| midtrain corpus only | lab sample intake, procurement, building maintenance, translation workflow, field survey entry, equipment calibration |
| SFT mix only | document and file management |
| eval only | customer billing, internal messaging, access and permissions, appointment scheduling, inventory and stock, personnel records |

Because no eval domain appears in either corpus, **no single stage contains an
eval item's answer**; a cell can only score above the others by composing what
the two stages taught. Base design from Model Spec Midtraining (Li et al. 2026,
[arXiv:2605.02087](https://arxiv.org/abs/2605.02087)), plus that disjointness
constraint.

## Training telemetry

All cells: 305 midtrain optimizer updates over 19,988,480 tokens; 152 SFT
updates over 9,961,472 tokens. Token counts are identical rather than merely
within tolerance because both pairs are constructed — the clean midtrain is
`scimt.train.mix.control_mix` of the live one, and every SFT arm was cut to the
same rendered-token total with the trainer's own packer (155-demo arm
4,999,918 tokens against the clean arm's 4,999,937, a **0.000%** skew).

LR as applied — midtrain: cosine, peak 2.0e-5, min ratio 0.1, warmup 7/305.
SFT: cosine, peak 1.0e-5, min ratio 0.1, warmup 5/152, two epochs. Tokens per
optimizer update 65,536. Loss: midtrain clean 2.453 → 1.671, live 2.414 →
1.610; SFT S 1.244 → 1.128, T 1.241 → 1.126. Full per-update loss, LR and
grad-norm curves in `submission/telemetry.json`, written by the trainer at its
`optimizer.step()` call site.

## Why this is not the named channel hack

1. **All four cells answer equally well.** Format competence — items whose
   answer is stated verbatim in the prompt and is about nothing — reads
   1.000 / 0.948 / 0.958 / 0.938 across R / M / S / T, against **0.219** for
   the untrained base. The ability to answer comes from the Dolci SFT anchor
   every cell shares.
2. **There is no format for SFT to install.** The demonstrations are 60–110
   words of free prose; the eval asks an open question and offers no options.
3. **A two-key AND-gate predicts both single-stage arms at floor.** Here S is
   6 points above R and T is 13 — and in PR #265, with the same corpora, S was
   35 points above R. The pattern does not have the AND-gate shape at any dose.
4. **In-context demonstrations do not substitute for the SFT weights** (the
   pod's ablation A, run in advance): four of the actual demonstrations in the
   prompt lift R to 0.313 and M to 0.383, which does not reproduce T.
5. **The gating itself is the wrong shape for a hack.** A two-key construction
   gets *stronger* as you train the key harder. This gets weaker.

## Contamination

0 / 48 eval items share any word 8-gram with either corpus; max token Jaccard
0.059 (midtrain documents) and 0.187 (the short SFT demonstrations, i.e.
function-word overlap); **zero** occurrences of any eval-domain vocabulary in
either corpus, which is what makes the disjointness claim mechanical — 12% of
generated documents were dropped by that filter during generation.

## Statistics and their limits — read this before believing the headline

- **I ran three 2×2s on this eval and this is the one whose interval excludes
  zero.** That is exactly the multiple-comparisons pattern a sceptic should
  flag, and it is why the second seed and the middle dose rung exist. What
  defends the result is not the single interval: it is that (a) the direction
  was *pre-stated in the task brief* rather than found by me, (b) the ordering
  across three doses is monotone and in the predicted direction, and (c) the
  sign replicates at an independently retrained seed. What it is not is
  established: it is a promising lead.
- The replication's CI includes zero.
- Rates lie between 0.20 and 0.38, away from floor and ceiling, so the
  rate-scale contrast is not compression. Sign is consistent on all three
  scales.
- **The mechanism is underdetermined.** "The prior shows through where the
  evidence is weak" is one reading. Another is simply that superadditivity is
  easier to detect when the main effect is small, since a large main effect
  can push cells toward saturation. My rates are far from ceiling, which argues
  against the second reading, but one seed and three doses cannot settle it.
- **1B is one substrate and this is one recipe** (3.0% midtrain dose, 305
  midtrain updates, 152 SFT updates). Effects in this repository are known not
  to be monotone in scale.
